import os
import sys
import math
import torch
import torch.nn as nn
from typing import Tuple

# 引入 ALTGVT 核心结构
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
from ALTGVT import alt_gvt_small

class Regression(nn.Module):
    """
    回归头模块：用于处理 Backbone 提取的特征并输出局部及全局计数。
    虽然 CBAM 类被删除，但 HSRankALTGVT 内部可能会引用含有类似结构的 Regression 层。
    """
    def __init__(self, clipnum: int = 16, chan: int = 256):
        super().__init__()
        # 这里的 v1, v2, v3 是为了融合不同尺度的特征
        self.v1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.v2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(256, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.v3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Conv2d(512, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.ReLU(),
        )
        # 全局输出层
        self.output = nn.Sequential(
            nn.Linear(4096 + clipnum, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )
        # 局部 16 网格输出层
        self.c16 = nn.Sequential(
            nn.Linear(chan, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )
        self.clipnum = clipnum
        self._init_param()

    def _init_param(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _block(self, y: torch.Tensor) -> torch.Tensor:
        n = int(math.sqrt(self.clipnum))
        h, w = y.size(-2) // n, y.size(-1) // n
        y_c16 = torch.zeros(y.size(0), self.clipnum, device=y.device)
        num = 0
        for i in range(0, y.size(-2) - h + 1, h):
            for j in range(0, y.size(-1) - w + 1, w):
                sub_y = y[:, :, i : i + h, j : j + w]
                sub_y_out = self.c16(sub_y.contiguous().view(sub_y.size(0), -1))
                y_c16[:, num : num + 1] = sub_y_out
                num += 1
        return y_c16

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = self.v1(x1)
        x2 = self.v2(x2)
        x3 = self.v3(x3)
        x = x1 + x2 + x3
        y = self.res(x)
        y_c16 = self._block(y)
        y_flat = y.view(y.size(0), -1)
        y_concat = torch.cat([y_flat, y_c16], dim=1)
        y_global = self.output(y_concat)
        return y_c16, y_global

class HSRankALTGVT(nn.Module):
    """
    这是你目前在 train.py 中实际调用的模型类。
    它封装了 ALT-GVT Backbone。
    """
    def __init__(self, clipnum: int = 16, pretrained_backbone: bool = False):
        super().__init__()
        # 实际的计算逻辑和层都在 ALTGVT.py 的 alt_gvt_small 中
        self.backbone = alt_gvt_small(pretrained=pretrained_backbone)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rank_local, rank_global = self.backbone(x)
        # 映射到 0-1 之间
        rank_local = torch.sigmoid(rank_local)
        rank_global = torch.sigmoid(rank_global)
        return rank_local, rank_global
