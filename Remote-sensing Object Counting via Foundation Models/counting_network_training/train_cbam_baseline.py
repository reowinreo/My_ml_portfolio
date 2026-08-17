# train_with_pseudo.py
# integrateafter completetrainingprogram, useGD+SAM3generate pseudo labelas"real label"

import argparse
import os
import glob
import time
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from functools import partial
from timm.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg

from CBAM import CBAM

# ============== model definition (fromALTGVT.pyintegrate) ==============
class Regression(nn.Module):
    def __init__(self, clipnum=16, chan=256):
        super(Regression, self).__init__()
        self.v1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.v2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.v3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.res = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.ReLU()
        )
        self.output = nn.Sequential(  # globalregression
            nn.Linear(4096 + clipnum, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.c16 = nn.Sequential(  # local16gridregression
            nn.Linear(chan, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.clipnum = clipnum
        self.init_param()

    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def block(self, y):
        n = int(math.sqrt(self.clipnum))
        b, h, w = y.size(0), y.size(-2) // n, y.size(-1) // n
        y_c16 = torch.zeros(y.size(0), self.clipnum).to(y.device)
        num = 0
        for i in range(0, y.size(-2) - h + 1, h):
            for j in range(0, y.size(-1) - w + 1, w):
                sub_y = y[:, :, i:i + h, j:j + w]
                sub_y_output = self.c16(sub_y.contiguous().view(sub_y.size(0), -1))
                y_c16[:, num:num + 1] = sub_y_output
                num += 1
        return y_c16

    def forward(self, x1, x2, x3):
        x1 = self.v1(x1)
        x2 = self.v2(x2)
        x3 = self.v3(x3)
        x = x1 + x2 + x3
        y = self.res(x)
        y_c16 = self.block(y)  # local16gridcount
        y = y.view(y.size(0), -1)
        y_concat = torch.cat([y, y_c16], dim=1)
        y_concat = self.output(y_concat)  # global count
        return y_c16, y_concat

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GroupAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ws=1):
        super(GroupAttention, self).__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, H, W):
        B, N, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws
        total_groups = h_group * w_group
        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)
        qkv = self.qkv(x).reshape(B, total_groups, -1, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, ws=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio) if ws == 1 else GroupAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, ws=ws)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim))
        self.s = s

    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    return  (H, W) mustis int, notcanuse sqrt(N) push. 
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # convolutionpatchify
        x = self.proj(x)  # [B, embed_dim, H', W']
        H_out, W_out = x.shape[2], x.shape[3]  # realinteger H', W'

        # flattenbecometoken
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = self.norm(x)

        return x, (H_out, W_out)  # mustis int


class CPVTV2(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], block_cls=Block):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.patch_embeds = nn.ModuleList()
        self.pos_drops = nn.ModuleList()
        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for k in range(len(depths)):
            if k == 0:
                _patch_embed = PatchEmbed(
                    img_size=img_size,
                    patch_size=patch_size,
                    in_chans=in_chans,
                    embed_dim=embed_dims[k]
                )
            else:
                _patch_embed = PatchEmbed(
                    img_size=img_size // (2 ** k),
                    patch_size=2,
                    in_chans=embed_dims[k - 1],
                    embed_dim=embed_dims[k]
                )

            self.patch_embeds.append(_patch_embed)
            self.pos_drops.append(nn.Dropout(p=drop_rate))

            _block = nn.ModuleList([
                block_cls(
                    dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k],
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i], norm_layer=norm_layer,
                    sr_ratio=sr_ratios[k]
                )
                for i in range(depths[k])
            ])
            self.blocks.append(_block)
            cur += depths[k]

        self.norm = norm_layer(embed_dims[-1])
        self.regression = Regression(clipnum=16)  # fixed16grid
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def forward_features(self, x):
        outputs = []
        B = x.shape[0]
        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            x = self.pos_drops[i](x)
            for blk in self.blocks[i]:
                x = blk(x, H, W)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outputs.append(x)
        return outputs

    def forward(self, x):
        x = self.forward_features(x)
        y_c16, y_global = self.regression(x[1], x[2], x[3])  # localandglobal output
        return y_c16, y_global

class PCPVT(CPVTV2):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256],
                 num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[4, 4, 4], sr_ratios=[4, 2, 1], block_cls=Block):
        super(PCPVT, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dims, num_heads,
                                    mlp_ratios, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate,
                                    norm_layer, depths, sr_ratios, block_cls)

class ALTGVT(PCPVT):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256],
                 num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[4, 4, 4], sr_ratios=[4, 2, 1], block_cls=Block, wss=[7, 7, 7]):
        super(ALTGVT, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dims, num_heads,
                                     mlp_ratios, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate,
                                     norm_layer, depths, sr_ratios, block_cls)
        del self.blocks
        self.wss = wss
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.blocks = nn.ModuleList()
        for k in range(len(depths)):
            _block = nn.ModuleList([block_cls(
                dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                sr_ratio=sr_ratios[k], ws=1 if i % 2 == 1 else wss[k]) for i in range(depths[k])])
            self.blocks.append(_block)
            cur += depths[k]
        self.apply(self._init_weights)

def alt_gvt_small(pretrained=False, **kwargs):
    model = ALTGVT(
        patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 10, 4], wss=[8, 8, 8, 8], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        # belowloadalt_gvt_small.pthandspecifypath
        checkpoint = torch.load('alt_gvt_small.pth', map_location='cpu')  # fakesetalreadybelowload
        model.load_state_dict(checkpoint, strict=False)
    return model



# ============== CBAM Backbone (replaceALTGVTasbackbone, resttraininglogicnotchange) ==============
class _ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, k=3, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CBAMBackbone(nn.Module):
    """
    Description:
    - thisisonelightquantityCNN + CBAM  backbone, output4stagefeature map: 
        stage0: [B,  64, 128, 128]
        stage1: [B, 128,  64,  64]   -> Regression   x1
        stage2: [B, 256,  32,  32]   -> Regression   x2
        stage3: [B, 512,  16,  16]   -> Regression   x3
    - thissamplecaninnotchange Regression/loss/trainingpipeline  situationbelow, takeALTGVTbackbonereplaceforCBAMbackbone. 
    """
    def __init__(self, clipnum=16, in_chans=3):
        super().__init__()
        # 512x512 -> 128x128
        self.stage0 = nn.Sequential(
            _ConvBNReLU(in_chans, 64, stride=4, k=7, p=3),
            _ConvBNReLU(64, 64, stride=1, k=3, p=1),
            CBAM(64),
        )
        # 128x128 -> 64x64
        self.stage1 = nn.Sequential(
            _ConvBNReLU(64, 128, stride=2, k=3, p=1),
            _ConvBNReLU(128, 128, stride=1, k=3, p=1),
            CBAM(128),
        )
        # 64x64 -> 32x32
        self.stage2 = nn.Sequential(
            _ConvBNReLU(128, 256, stride=2, k=3, p=1),
            _ConvBNReLU(256, 256, stride=1, k=3, p=1),
            CBAM(256),
        )
        # 32x32 -> 16x16
        self.stage3 = nn.Sequential(
            _ConvBNReLU(256, 512, stride=2, k=3, p=1),
            _ConvBNReLU(512, 512, stride=1, k=3, p=1),
            CBAM(512),
        )

        # regression headkeepnotchange
        self.regression = Regression(clipnum=clipnum)

    def forward_features(self, x):
        outs = []
        x = self.stage0(x)
        outs.append(x)
        x = self.stage1(x)
        outs.append(x)
        x = self.stage2(x)
        outs.append(x)
        x = self.stage3(x)
        outs.append(x)
        return outs

    def forward(self, x):
        feats = self.forward_features(x)
        y_c16, y_global = self.regression(feats[1], feats[2], feats[3])
        return y_c16, y_global


def cbam_small(pretrained=False, **kwargs):
    # reserveAPI, and alt_gvt_small callwayconsistent
    model = CBAMBackbone(**kwargs)
    return model

# ============== loss function (fromranking_loss.pyintegrate) ==============
T = 1

def torch_dcg_at_k(batch_sorted_labels, cutoff=None):
    if cutoff is None:
        cutoff = batch_sorted_labels.size(1)
    batch_numerators = torch.pow(2.0, batch_sorted_labels[:, 0:cutoff]) - 1.0
    batch_discounts = torch.log2(torch.arange(cutoff).type(torch.FloatTensor).to(batch_sorted_labels.device) + 2.0)
    batch_dcg_at_k = torch.sum(batch_numerators / batch_discounts, dim=1, keepdim=True)
    return batch_dcg_at_k

def get_approx_ranks(input, alpha=10):
    batch_pred_diffs = torch.unsqueeze(input, dim=2) - torch.unsqueeze(input, dim=1)
    batch_indicators = torch.sigmoid(alpha * torch.transpose(batch_pred_diffs, 1, 2))
    batch_hat_pis = torch.sum(batch_indicators, dim=2) + 0.5
    return batch_hat_pis

def ranking_loss(batch_preds=None, batch_stds=None, alpha=10):
    batch_hat_pis = get_approx_ranks(batch_preds, alpha=alpha)
    batch_idcgs = torch_dcg_at_k(batch_sorted_labels=batch_stds, cutoff=None)
    batch_gains = torch.pow(2.0, batch_stds) - 1.0
    batch_dcg = torch.sum(torch.div(batch_gains, torch.log2(batch_hat_pis + 1)), dim=1).unsqueeze(dim=1)
    batch_approx_nDCG = torch.div(batch_dcg, batch_idcgs)
    batch_loss = 1 - torch.mean(batch_approx_nDCG)
    return batch_loss

class Ranking_loss(nn.Module):
    def __init__(self, alpha=10, eps=1e-6):
        super(Ranking_loss, self).__init__()
        self.alpha = alpha
        self.eps = eps

    def getCorrel(self, labels, Max_index, Min_index):
        y_true = torch.zeros(2, len(labels) - 1).to(labels.device)
        if Max_index == 0:
            labels_del = labels[1:].squeeze()
            y_true[0, :] = labels[Max_index] - labels_del
        else:
            labels_del = torch.cat((labels[0:Max_index], labels[Max_index + 1:]), dim=0).squeeze()
            y_true[0, :] = labels[Max_index] - labels_del

        if Min_index == 0:
            labels_del = labels[1:].squeeze()
            y_true[1, :] = labels_del - labels[Min_index]
        else:
            labels_del = torch.cat((labels[0:Min_index], labels[Min_index + 1:]), dim=0).squeeze()
            y_true[1, :] = labels_del - labels[Min_index]
        return y_true

    def getSimilar(self, pred, Max_index, Min_index):
        y_pred = torch.zeros(2, len(pred) - 1).to(pred.device)
        if Max_index == 0:
            labels_del = pred[1:].squeeze()
            y_pred[0, :] = pred[Max_index] - labels_del
        else:
            labels_del = torch.cat((pred[0:Max_index], pred[Max_index + 1:]), dim=0).squeeze()
            y_pred[0, :] = pred[Max_index] - labels_del

        if Min_index == 0:
            labels_del = pred[1:].squeeze()
            y_pred[1, :] = labels_del - pred[Min_index]
        else:
            labels_del = torch.cat((pred[0:Min_index], pred[Min_index + 1:]), dim=0).squeeze()
            y_pred[1, :] = labels_del - pred[Min_index]
        return y_pred

    def forward(self, pred_in, labels_in, c4=False, cn=16):
        eps = self.eps

        # ======= local loss: image by imagecompute, skip"allmutualetc."sample =======
        if c4:
            batch_losssum = 0.0
            valid = 0
            for idx in range(len(pred_in)):
                pred = pred_in[idx:idx + 1].view(cn, 1)
                labels = labels_in[idx:idx + 1].view(cn, 1)

                MaxValue, Max_index = torch.max(labels, dim=0)
                MinValue, Min_index = torch.min(labels, dim=0)
                denom = (MaxValue - MinValue).abs()

                # 16cellallsame -> rankinginformationnotsavein, directskip
                if denom.item() < eps:
                    continue

                batch_stds = self.getCorrel(labels, Max_index, Min_index)
                batch_stds = 1 - (batch_stds / (denom + eps))
                batch_stds = batch_stds * T

                batch_preds = self.getSimilar(pred, Max_index, Min_index)
                batch_preds = 1 - batch_preds

                target_batch_stds, batch_sorted_inds = torch.sort(batch_stds, dim=1, descending=True)
                target_batch_preds = torch.gather(batch_preds, dim=1, index=batch_sorted_inds)

                batch_losssum += ranking_loss(target_batch_preds, target_batch_stds, self.alpha)
                valid += 1

            # ifthis batch insidesohassampleall skipped, return 0(no backprop)
            if valid == 0:
                return pred_in.sum() * 0.0
            return batch_losssum / valid

        # ======= global loss: batch insidecompute, denominator add eps =======
        else:
            MaxValue, Max_index = torch.max(labels_in, dim=0)
            MinValue, Min_index = torch.min(labels_in, dim=0)
            denom = (MaxValue - MinValue).abs()

            # batch insideallsame -> notrankinginformation, directreturn 0
            if denom.item() < eps:
                return pred_in.sum() * 0.0

            batch_stds = self.getCorrel(labels_in, Max_index, Min_index)
            batch_stds = 1 - (batch_stds / (denom + eps))
            batch_stds = batch_stds * T

            batch_preds = self.getSimilar(pred_in, Max_index, Min_index)
            batch_preds = 1 - batch_preds

            target_batch_stds, batch_sorted_inds = torch.sort(batch_stds, dim=1, descending=True)
            target_batch_preds = torch.gather(batch_preds, dim=1, index=batch_sorted_inds)

            batch_loss = ranking_loss(target_batch_preds, target_batch_stds, self.alpha)
            return batch_loss


# ============== datasetclass (frommake_dataset.pyintegrate, modifyforloadpseudo label) ==============
ROOT_DATA_DIR = "dataset"  # unifypath

def _build_allow_set_for_rsoc_mixed(dataset_label, split):
    cat_token = {
        "RSOC-Ship": "ship",
        "RSOC-S-Vehicle": "small-vehicle",
        "RSOC-L-Vehicle": "large-vehicle",
    }.get(dataset_label, None)
    if cat_token is None:
        return set()
    dota_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "DOTA_data")
    txt_path = os.path.join(dota_dir, f"{split}_{cat_token}.txt")
    if os.path.isfile(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            allow = {line.strip() for line in f if line.strip()}
        return allow
    label_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", split, "labelTxt-v1.0", f"{split}set_reclabelTxt")
    allow = set()
    for lbl in glob.glob(os.path.join(label_dir, "*.txt")):
        base = os.path.splitext(os.path.basename(lbl))[0]
        try:
            hit = False
            with open(lbl, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 9:
                        continue
                    cls = parts[-2] if len(parts) >= 10 else parts[-1]
                    if cls == cat_token:
                        hit = True
                        break
            if hit:
                allow.add(base)
        except Exception:
            continue
    return allow

def _list_images_for_dataset_split(dataset_label, split):
    if dataset_label == "RSOC-Building":
        rsoc_root = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "RSOC_building", "building")
        sub = "train_data" if split == "train" else ("test_data" if split == "test" else "val_data" if split == "val" else None)
        if sub is None:
            return []
        img_dir = os.path.join(rsoc_root, sub, "images")
        return sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if dataset_label in ["RSOC-Ship", "RSOC-S-Vehicle", "RSOC-L-Vehicle"]:
        img_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", split, "images")
        if not os.path.isdir(img_dir):
            return []
        allow = _build_allow_set_for_rsoc_mixed(dataset_label, split)
        if not allow:
            return []
        out = [p for p in glob.glob(os.path.join(img_dir, "*.jpg")) if os.path.splitext(os.path.basename(p))[0] in allow]
        return sorted(out)
    if dataset_label in ["VD-People", "VD-Vehicle"]:
        vd_root = os.path.join(ROOT_DATA_DIR, "VisDrone-People" if dataset_label == "VD-People" else "VisDrone-Vehicle")
        img_dir = os.path.join(vd_root, split, "images") or os.path.join(vd_root, split, "Images")
        if not os.path.isdir(img_dir):
            return []
        return sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    return []

class PseudoRankDataset(Dataset):
    def __init__(self, dataset_label, split, pseudo_txt_path, transform=None):
        self.dataset_label = dataset_label
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.img_paths = _list_images_for_dataset_split(dataset_label, split)
        # loadpseudo labeltxt
        self.pseudo_labels = {}
        with open(pseudo_txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                img_name = parts[0]
                counts = list(map(int, parts[1:]))
                if len(counts) != 16:
                    continue
                self.pseudo_labels[img_name] = counts  # local16count, totalcount=sum(counts)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_name = os.path.basename(img_path)
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if img_name not in self.pseudo_labels:
            raise ValueError(f"No pseudo label for {img_name}")
        local_counts = np.array(self.pseudo_labels[img_name])  # local16count (as"real"locallabel)
        global_count = np.sum(local_counts)  # global"real"label
        return img, global_count, local_counts, img_name

# ============== trainingentry (fromtrain_epoch.pyintegrate) ==============
def parse_args():
    parser = argparse.ArgumentParser(description='Train with Pseudo Labels')
    parser.add_argument('--dataset', default="RSOC-Building", help='Dataset: RSOC-Building, RSOC-Ship, etc.')
    parser.add_argument('--lr', type=float, default=1e-5, help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--resume', default="", type=str, help='resume model path')
    parser.add_argument('--clipnum', type=int, default=16, help='grid num')
    parser.add_argument('--min_mae', type=float, default=np.inf, help='min MAE for saving')
    parser.add_argument('--epoch', type=int, default=500, help='max epochs')
    parser.add_argument('--val-start', type=int, default=1, help='start val epoch')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='device')
    parser.add_argument('--num-workers', type=int, default=4, help='num workers')
    args = parser.parse_args()
    return args

def train_epoch(epoch, model, optimizer, train_dataloader, NDCGLoss, args):
    model.train()
    loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    loss_total, loss_global, loss_local, num = 0, 0, 0, 0
    for i, (image, target_global, target_local, _) in loop:
        if len(torch.unique(target_global)) < 2:  # skipnodifferencebatch
            continue
        image, target_global, target_local = image.to(args.device), target_global.to(args.device).float(), target_local.to(args.device).float()
        with torch.set_grad_enabled(True):
            output_local, output_global = model(image)
            loss_global_val = NDCGLoss(output_global.squeeze(), target_global)
            loss_local_val = NDCGLoss(output_local, target_local, c4=True, cn=args.clipnum)
            loss = loss_global_val + 0.5 * loss_local_val
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_total += loss.item()
        loss_global += loss_global_val.item()
        loss_local += loss_local_val.item()
        num += 1
        loop.set_postfix(epoch=epoch, loss=loss_total / num, loss_global=loss_global / num, loss_local=loss_local / num)

def val_epoch(model, val_dataloader, args):
    model.eval()
    alpha_list, beta_list = [], []
    loop = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
    for i, (image, target_global, _, _) in loop:
        image = image.to(args.device)
        with torch.no_grad():
            _, output_global = model(image)
        alpha_list.append(output_global.item())
        beta_list.append(target_global.item())
    # linearityregressioncalibration
    num = len(alpha_list)
    alpha_mean = sum(alpha_list) / num
    top = sum([beta * (alpha - alpha_mean) for alpha, beta in zip(alpha_list, beta_list)])
    bot = sum([(alpha - alpha_mean) ** 2 for alpha in alpha_list])
    k = top / bot if bot != 0 else 1
    b = sum([beta - k * alpha for alpha, beta in zip(alpha_list, beta_list)]) / num
    # computeMAE/RMSE
    mae_all, rmse_all, numx = 0.0, 0.0, 0
    for i, (image, target_global, _, _) in enumerate(val_dataloader):
        numx += 1
        image = image.to(args.device)
        with torch.no_grad():
            _, output_global = model(image)
        target = target_global.item()
        output = k * output_global.item() + b
        mae_all += abs(output - target)
        rmse_all += (output - target) ** 2
    mae = mae_all / numx
    rmse = math.sqrt(rmse_all / numx)
    print(f'Val MAE: {mae:.2f}, RMSE: {rmse:.2f}')
    return mae, rmse

if __name__ == '__main__':
    #np.random.seed(42)
    #torch.manual_seed(42)
    #torch.cuda.manual_seed(42)
    args = parse_args()
    min_mae = args.min_mae
    model = cbam_small(pretrained=False)
    model.to(args.device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=args.device))
    NDCGLoss = Ranking_loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # loadpseudo labelanddataset
    pseudo_dir = os.path.join(ROOT_DATA_DIR, "local pseudo")
    pseudo_train_txt = os.path.join(pseudo_dir, f"{args.dataset}_grid16_train.txt")
    pseudo_val_txt = os.path.join(pseudo_dir, f"{args.dataset}_grid16_val.txt") if os.path.exists(os.path.join(pseudo_dir, f"{args.dataset}_grid16_val.txt")) else pseudo_train_txt  # ifnoval, usetrain
    train_dataset = PseudoRankDataset(args.dataset, "train", pseudo_train_txt)
    val_dataset = PseudoRankDataset(args.dataset, "val", pseudo_val_txt)

    # ======= keyfix: novalidation setthenusetraining setfillwhenvalidation set =======
    if len(val_dataset) == 0:
        print("[WARN] current dataset has no val split, automatically use train as val performcalibrationandevaluation. ")
        val_dataset = train_dataset

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers
    )

    os.makedirs("my_model", exist_ok=True)
    for epoch in range(1, args.epoch + 1):
        train_epoch(epoch, model, optimizer, train_dataloader, NDCGLoss, args)
        if epoch >= args.val_start:
            mae, rmse = val_epoch(model, val_dataloader, args)
            if mae < min_mae:
                min_mae = mae
                torch.save(model.state_dict(), os.path.join("my_model", f"mae{mae:.2f}_rmse{rmse:.2f}_epoch{epoch}_{args.dataset}.pth"))
    print("Training done!")