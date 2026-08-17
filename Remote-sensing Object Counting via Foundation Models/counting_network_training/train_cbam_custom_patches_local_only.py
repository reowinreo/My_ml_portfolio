"""train_custom_grid_cbam.py

Objective:
  based on CBAM backbone + Regression head + Ranking Loss. 
  modifylogic: 
  1. support N x M arbitrarygridsplit, andautomatichandlenotrulepatch(if 64exceptwith3  remainderpartmatch). 
  2. removeglobal branch, onlyuselocal(Local)predict, validationwhenthroughsumgetglobal count. 

Configuration:
  modify CONFIG in  "grid_rows" and "grid_cols" i.e.canchangesplitway. 
"""

from __future__ import annotations

import os
import glob
import math
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# =========================
# configuration sectiondomain
# =========================
CONFIG = {
    "dataset_root": "dataset",
    # optional: "RSOC-Ship" | "RSOC-Building" | "VD-People" etc.
    "dataset": "RSOC-Building",

    # ===modify here N x M===
    "grid_rows": 3,      # N (highsidetocutseveralknife)
    "grid_cols": 4,      # M (widesidetocutseveralknife)
    # =====================

    "img_size": 512,
    "batch_size": 8,
    "epochs": 200,
    "val_start": 1,
    "lr": 1e-5,
    "weight_decay": 1e-4,
    "num_workers": 4,
    "seed": 42,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "save_dir": "my_model",
    "resume": "",
    "min_mae": float("inf"),
}


# =========================
# common utilities: computesplitboundary
# =========================
def get_boundaries(total_len: int, n: int) -> List[int]:
    """
    willtotallength total_len partbecome n portion. 
    ifnotcanwholeexcept, remainderpartmatchtobeforeseveralblock. 
    returnboundarylist, for example 64part3portion -> [0, 22, 43, 64] (correspondblocklong 22, 21, 21)
    """
    base = total_len // n
    rem = total_len % n
    boundaries = [0]
    current = 0
    for i in range(n):
        # ifalsohasremainder, currentblocklength+1
        step = base + 1 if i < rem else base
        current += step
        boundaries.append(current)
    return boundaries


# =========================
# CBAM module
# =========================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


# =========================
# Regression Head (Only Local)
# =========================
class Regression(nn.Module):
    def __init__(self, rows: int, cols: int, chan: int = 256):
        super().__init__()
        self.rows = rows
        self.cols = cols

        # featurefusecombinelayer
        self.v1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.v2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(256, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.v3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Conv2d(512, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, dilation=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.ReLU(),
        )

        # localpredicthead: toeverypatchperformpredict
        # inputis flatten after feature, output 1 (thisblock count)
        self.local_head = nn.Sequential(
            nn.Linear(chan, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # remove  Global Head (self.output)

        self._init_param()

    def _init_param(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _block_slicing(self, y: torch.Tensor) -> torch.Tensor:
        """
        in 64x64  feature map y on, according to rows, cols performsplit. 
        supportnotrulesplit (if 64/3 -> 22, 21, 21). 
        """
        B, C, H, W = y.size()

        # getsplitboundary
        h_bounds = get_boundaries(H, self.rows)
        w_bounds = get_boundaries(W, self.cols)

        # resulttoleratedevice: [B, rows*cols]
        total_blocks = self.rows * self.cols
        y_local_pred = torch.zeros(B, total_blocks, device=y.device)

        idx = 0
        for r in range(self.rows):
            h_start, h_end = h_bounds[r], h_bounds[r+1]
            for c in range(self.cols):
                w_start, w_end = w_bounds[c], w_bounds[c+1]

                # slice [B, C, h_sub, w_sub]
                sub_y = y[:, :, h_start:h_end, w_start:w_end]

                # flattenfeed local_head
                # Note:local_head inputneedfixeddimension? 
                # originalcode: self.c16(sub_y.contiguous().view(sub_y.size(0), -1))
                # originalcodein self.c16  inputis chan(256). 
                # wait, inoriginalcodeinside, self.res output channelis 1 (Regression.res finallyis Conv2d(64, 1, 1)). 
                # so y   shape is [B, 1, 64, 64]. 
                # sub_y is [B, 1, h', w'] -> view -> [B, h'*w']. 
                # but Linear(chan, 256) in chan=256 isnotto , unless y notis res  output. 

                # backlookoriginalcode: 
                # self.res output is [B, 1, 64, 64]. 
                # originalcode _block inside: sub_y.view(sub_y.size(0), -1) -> become [B, h*w]. 
                # originalcode self.c16 define: Linear(chan, 256). default chan=256. 
                # thismean originalcodefakeseteveryblock size h*w justgoodequal to 256 (i.e. 16x16). 
                # because 64x64 partbecome 4x4block, everyblock 16x16, so 16*16*1 = 256. 
                # **ifweneedsupportarbitrary N x M, Linear  input dimensionthenchange ! **

                # repairjustsidecase: 
                # in order tosupportdynamicsize, wenotcandirect flatten feedfixed Linear. 
                # shouldthisto sub_y perform Global Average Pooling become 1x1, orone whouse AdaptiveAvgPool tofixedsize. 
                # butoriginallogicis Regression, usepixelinformation. 
                # considertooriginalasone whomeaningimageis "Regression on density map patch". 
                # simple changemethod: infeed Linear previous, will sub_y interpolateorone who Padding toonefixedsize? 
                # orone who: in view ofthisis Density Map regression, directto sub_y sumitsrealtheniscount, buthereuse  MLP. 
                #
                # **keepchangeminimumandlogicthroughsmooth sidecase**: 
                # use AdaptiveAvgPool2d takearbitrarysize patchbecomefixedsize (for example 16x16), 
                # thissamplethencanreuse Linear(256, ...). 

                # 1. unify Pool to 16x16 (keepandoriginal 4x4splitwhen featurequantityconsistent)
                sub_y_fixed = torch.nn.functional.adaptive_avg_pool2d(sub_y, (16, 16))

                # 2. Flatten: [B, 1, 16, 16] -> [B, 256]
                feat = sub_y_fixed.view(B, -1)

                # 3. Predict
                out = self.local_head(feat) # [B, 1]
                y_local_pred[:, idx] = out.squeeze(1)
                idx += 1

        return y_local_pred

    def forward(self, x1, x2, x3):
        x1 = self.v1(x1)
        x2 = self.v2(x2)
        x3 = self.v3(x3)
        x = x1 + x2 + x3
        y = self.res(x)  # [B, 1, 64, 64] density mapfeature

        # onlyexecutelocalsplitpredict
        y_local = self._block_slicing(y)

        return y_local


class CBAMBackbone(nn.Module):
    def __init__(self, rows: int, cols: int, in_chans: int = 3):
        super().__init__()
        self.stage0 = nn.Sequential(
            _ConvBNReLU(in_chans, 64, stride=4, k=7, p=3),
            _ConvBNReLU(64, 64, stride=1, k=3, p=1),
            CBAM(64),
        )
        self.stage1 = nn.Sequential(
            _ConvBNReLU(64, 128, stride=2, k=3, p=1),
            _ConvBNReLU(128, 128, stride=1, k=3, p=1),
            CBAM(128),
        )
        self.stage2 = nn.Sequential(
            _ConvBNReLU(128, 256, stride=2, k=3, p=1),
            _ConvBNReLU(256, 256, stride=1, k=3, p=1),
            CBAM(256),
        )
        self.stage3 = nn.Sequential(
            _ConvBNReLU(256, 512, stride=2, k=3, p=1),
            _ConvBNReLU(512, 512, stride=1, k=3, p=1),
            CBAM(512),
        )
        # propagatepass rows, cols
        self.regression = Regression(rows=rows, cols=cols)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage0(x)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        # onlyreturn local
        y_local = self.regression(x1, x2, x3)
        return y_local


class _ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, k=3, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


# =========================
# Ranking Loss (Simplified for Only Local)
# =========================
T = 1
def torch_dcg_at_k(batch_sorted_labels, cutoff=None):
    if cutoff is None: cutoff = batch_sorted_labels.size(1)
    numerators = torch.pow(2.0, batch_sorted_labels[:, 0:cutoff]) - 1.0
    discounts = torch.log2(torch.arange(cutoff, device=batch_sorted_labels.device, dtype=torch.float32) + 2.0)
    dcg = torch.sum(numerators / discounts, dim=1, keepdim=True)
    return dcg

def get_approx_ranks(input, alpha=10):
    diffs = torch.unsqueeze(input, dim=2) - torch.unsqueeze(input, dim=1)
    indicators = torch.sigmoid(alpha * torch.transpose(diffs, 1, 2))
    hat_pis = torch.sum(indicators, dim=2) + 0.5
    return hat_pis

def ranking_loss_func(batch_preds, batch_stds, alpha=10):
    hat_pis = get_approx_ranks(batch_preds, alpha=alpha)
    idcgs = torch_dcg_at_k(batch_sorted_labels=batch_stds, cutoff=None)
    gains = torch.pow(2.0, batch_stds) - 1.0
    dcg = torch.sum(gains / torch.log2(hat_pis + 1), dim=1).unsqueeze(dim=1)
    approx_ndcg = dcg / (idcgs + 1e-6)
    return 1 - torch.mean(approx_ndcg)

class RankingLoss(nn.Module):
    def __init__(self, alpha=10, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def _get_diff_matrix(self, vec, max_idx, min_idx):
        # constructtworowmatrix, partdo nottableshowandmaximumvalueandminimumvalue difference
        res = torch.zeros(2, len(vec) - 1, device=vec.device)
        # Max Row
        if max_idx == 0:
            vec_del = vec[1:].squeeze()
            res[0, :] = vec[max_idx] - vec_del
        else:
            vec_del = torch.cat((vec[0:max_idx], vec[max_idx + 1 :]), dim=0).squeeze()
            res[0, :] = vec[max_idx] - vec_del
        # Min Row
        if min_idx == 0:
            vec_del = vec[1:].squeeze()
            res[1, :] = vec_del - vec[min_idx]
        else:
            vec_del = torch.cat((vec[0:min_idx], vec[min_idx + 1 :]), dim=0).squeeze()
            res[1, :] = vec_del - vec[min_idx]
        return res

    def forward(self, pred_in: torch.Tensor, labels_in: torch.Tensor) -> torch.Tensor:
        """
        pred_in, labels_in: [Batch, N*M]
        """
        loss_sum = 0.0
        valid = 0
        batch_size, num_patches = pred_in.shape

        for idx in range(batch_size):
            pred = pred_in[idx].view(num_patches, 1)
            labels = labels_in[idx].view(num_patches, 1)

            max_val, max_idx = torch.max(labels, dim=0)
            min_val, min_idx = torch.min(labels, dim=0)
            denom = (max_val - min_val).abs()

            # ifthisoneimageinsidesohascellchildcountallsame(such asallis0), notmethodarrangeRanking
            if denom.item() < self.eps:
                continue

            # 1. normalization Label difference
            stds = self._get_diff_matrix(labels, max_idx, min_idx)
            stds = 1 - (stds / (denom + self.eps))
            stds = stds * T

            # 2. compute Pred difference
            preds = self._get_diff_matrix(pred, max_idx, min_idx)
            preds = 1 - preds # herekeeporiginallogic

            # 3. Sort & Rank
            target_stds, inds = torch.sort(stds, dim=1, descending=True)
            target_preds = torch.gather(preds, dim=1, index=inds)

            loss_sum += ranking_loss_func(target_preds, target_stds, self.alpha)
            valid += 1

        if valid == 0:
            return pred_in.sum() * 0.0
        return loss_sum / valid


# =========================
# datahandle
# =========================
def _safe_read_lines(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [ln.strip() for ln in f if ln.strip()]

def _list_images(root, dataset_label, split):
    # (keeporiginalhas readlogicnotchange, omitomitpartredundantcode, onlyretaincorepathcheckfind)
    # herein order tocodecomplete-ity, simplerestorewritecorelogic
    if dataset_label == "RSOC-Building":
        rsoc_root = os.path.join(root, "ASPDNet_dataset", "RSOC_building", "building")
        sub = "train_data" if split == "train" else "test_data"
        img_dir = os.path.join(rsoc_root, sub, "images")
        if not os.path.isdir(img_dir): return []
        return sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    # simple-ize  VisDrone / RSOC logic
    if "VisDrone" in dataset_label:
        vd_root = os.path.join(root, "VisDrone-People" if "People" in dataset_label else "VisDrone-Vehicle")
        img_dir = os.path.join(vd_root, split, "images")
        if not os.path.isdir(img_dir): img_dir = os.path.join(vd_root, split, "Images")
        return sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    return [] # otherdatasettemporarilyomit, needretainoriginalfunctioncantakeonsurface copybelowcome

def _counts_from_points_nxm(points_xy: np.ndarray, width: int, height: int, rows: int, cols: int) -> np.ndarray:
    """
    according to rows x cols statisticspointfallinwhichcellchild. 
    Note:here boundarylogicmustand Model inside _block_slicing strictcellconsistent. 
    Model isin 64x64 onsplit, hereisin ImgSize(512) onsplit. 
    because 512 = 8 * 64, sodirecttake Model  boundary * 8 i.e.can. 
    """
    # 1. get 64 scalebelow boundary
    feat_h, feat_w = 64, 64
    h_bounds_64 = get_boundaries(feat_h, rows)
    w_bounds_64 = get_boundaries(feat_w, cols)

    # 2. map to 512 (orother img_size)
    # scale_h = height / 64.0
    # scale_w = width / 64.0
    # butin order toabsolutelytoalign, suggestforcefakesetinputalreadyby resize to img_size
    scale = width // 64  # usuallyis 512//64 = 8

    h_bounds = [b * scale for b in h_bounds_64]
    w_bounds = [b * scale for b in w_bounds_64]

    counts = np.zeros((rows * cols,), dtype=np.float32)

    for (x, y) in points_xy:
        # Clamp
        x = max(0.0, min(x, width - 1e-4))
        y = max(0.0, min(y, height - 1e-4))

        # Find row index
        r_idx = -1
        for i in range(rows):
            if h_bounds[i] <= y < h_bounds[i+1]:
                r_idx = i
                break
        if r_idx == -1: r_idx = rows - 1 # Fallback

        # Find col index
        c_idx = -1
        for i in range(cols):
            if w_bounds[i] <= x < w_bounds[i+1]:
                c_idx = i
                break
        if c_idx == -1: c_idx = cols - 1

        flat_idx = r_idx * cols + c_idx
        counts[flat_idx] += 1.0

    return counts

# read GT point functionneedkeeporiginalhaslogic, onlyisfinallycall _counts_from_points_nxm
def _read_gt_points_wrapper(path, w, h, out_size, mode="txt"):
    # heresimple-izehandle, youneedaccording tooriginalcompletecodetake _read_dota, _read_mat etc.logicintegrate
    # fakesetherealreadythroughtaketo points list [(x, y), ...] andalreadynormalizationto out_size
    # herewritedeadonefake implementdemoAPI, actualuseyouoriginalcode readlogici.e.can
    # !!! pleasemustmustretainoriginalcode  _read_dota_label_points etc.toolvolumeimplement !!!
    return np.zeros((0, 2))

class RealRankDataset(Dataset):
    def __init__(self, root, dataset_label, split, img_size=512, rows=4, cols=4):
        self.root = root
        self.dataset_label = dataset_label
        self.split = split
        self.img_size = img_size
        self.rows = rows
        self.cols = cols
        self.img_paths = _list_images(root, dataset_label, split)
        self._mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def _transform(self, img_pil):
        img = img_pil.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - self._mean) / self._std
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_name = os.path.basename(img_path)
        base = os.path.splitext(img_name)[0]

        img_pil = Image.open(img_path).convert("RGB")
        w0, h0 = img_pil.size
        img = self._transform(img_pil)

        # === coremodify: readpointcoordinateafter, callnew  NxM statistics ===
        # Note:hereneedyoutakeoriginalcodeinread mat/txt getcoordinatelist codemove come
        # fakesetweget  pts (N, 2), coordinatealreadymap to self.img_size

        # <simulatecode> actualpleasereuseyouoriginalhas  _read_building_mat_points etc.logic, onlychangefinallyonerow
        # belowsurfacethissegmentisoriginalcodelogic simple-izereproduce: 
        pts = np.zeros((0, 2), dtype=np.float32)
        if self.dataset_label == "RSOC-Building":
             # hereshouldcall _read_building_mat_points return points array
             # in order todemo, wetemporarilywhenfakesetreadto point
             pass
        elif "VisDrone" in self.dataset_label:
             pass
        # </simulatecode>

        # ! ! ! keypoint: use Real Logic readandmapcoordinate! ! ! 
        # herein order toguaranteecodecanrun, Itakeoriginalcodekeyreadfunctionpastebackslightlyasmodify
        # ---------------------------------------------------------
        local = self._load_real_gt(base, img_name, w0, h0)
        # ---------------------------------------------------------

        return img, torch.from_numpy(local), img_name

    def _load_real_gt(self, base, img_name, w0, h0):
        # thisisone helper, setbecome originalcodereadlogic, finallydo NxM map
        pts = []
        # 1. Building
        if self.dataset_label == "RSOC-Building":
            import scipy.io as sio
            rsoc_root = os.path.join(self.root, "ASPDNet_dataset", "RSOC_building", "building")
            sub = "train_data" if self.split == "train" else "test_data"
            mat_path = os.path.join(rsoc_root, sub, "ground_truth", f"GT_{base}.mat")
            if not os.path.isfile(mat_path):
                 mat_path = os.path.join(rsoc_root, sub, "ground_truth", f"GT_{img_name}".replace(".jpg", ".mat"))

            if os.path.isfile(mat_path):
                try:
                    mat = sio.loadmat(mat_path)
                    if "center" in mat:
                        p = mat["center"][0, 0] # viewtoolvolumestructure
                        if p.ndim == 2:
                            # mapping
                            p[:, 0] = p[:, 0] / max(w0, 1) * self.img_size
                            p[:, 1] = p[:, 1] / max(h0, 1) * self.img_size
                            pts = p
                except: pass

        # 2. VisDrone (omitomit txt readdetail, logicsameon, onlyis scale to img_size)
        # ...

        pts = np.asarray(pts, dtype=np.float32)
        if len(pts) == 0:
            return np.zeros((self.rows * self.cols,), dtype=np.float32)

        # corecall
        return _counts_from_points_nxm(pts, self.img_size, self.img_size, self.rows, self.cols)


# =========================
# trainingmainloop
# =========================
def train_epoch(model, loader, optimizer, loss_fn, device, epoch):
    model.train()
    loop = tqdm(loader, desc=f"Ep {epoch}", ncols=100)
    loss_avg = 0.0
    cnt = 0
    for img, target_local, _ in loop:
        img = img.to(device)
        target_local = target_local.to(device) # [B, N*M]

        # 1. Forward (onlyhas local)
        out_local = model(img) # [B, N*M]

        # 2. Loss
        loss = loss_fn(out_local, target_local)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_avg += loss.item()
        cnt += 1
        loop.set_postfix(loss=loss_avg/cnt)

@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    preds_sum = []
    gts_sum = []

    for img, target_local, _ in tqdm(loader, desc="Val"):
        img = img.to(device)
        out_local = model(img) # [B, N*M]

        # validationlogic: becauseno Global branch, wesum Local comedocountevaluation
        pred_count = out_local.sum(dim=1).item()
        gt_count = target_local.sum(dim=1).item()

        preds_sum.append(pred_count)
        gts_sum.append(gt_count)

    # compute MAE / RMSE
    mae = np.mean(np.abs(np.array(preds_sum) - np.array(gts_sum)))
    rmse = np.sqrt(np.mean((np.array(preds_sum) - np.array(gts_sum))**2))
    print(f"Val MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return mae, rmse

def main():
    # Setup
    random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    rows = CONFIG["grid_rows"]
    cols = CONFIG["grid_cols"]

    # Model
    model = CBAMBackbone(rows=rows, cols=cols)
    model.to(CONFIG["device"])

    if CONFIG["resume"]:
        model.load_state_dict(torch.load(CONFIG["resume"]))

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    loss_fn = RankingLoss()

    # Dataset
    # Note:youneedensure Dataset classinside _load_real_gt completeimplement youoriginalcode readlogic
    train_ds = RealRankDataset(CONFIG["dataset_root"], CONFIG["dataset"], "train",
                               rows=rows, cols=cols)
    val_ds = RealRankDataset(CONFIG["dataset_root"], CONFIG["dataset"], "val",
                             rows=rows, cols=cols)
    if len(val_ds) == 0: val_ds = train_ds

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    min_mae = float("inf")

    for ep in range(1, CONFIG["epochs"]+1):
        train_epoch(model, train_loader, optimizer, loss_fn, CONFIG["device"], ep)
        if ep >= CONFIG["val_start"]:
            mae, _ = val_epoch(model, val_loader, CONFIG["device"])
            if mae < min_mae:
                min_mae = mae
                torch.save(model.state_dict(), f"{CONFIG['save_dir']}/best_nxm.pth")

if __name__ == "__main__":
    main()