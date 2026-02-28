import os
import sys
import math
import torch
import torch.nn as nn
from typing import Tuple

# Import ALTGVT core architecture
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
from ALTGVT import alt_gvt_small

class Regression(nn.Module):
    """
    Regression head for backbone features; outputs local and global count scores.
    Although CBAM was removed, HSRankALTGVT may still reference a similar Regression layer.
    """
    def __init__(self, clipnum: int = 16, chan: int = 256):
        super().__init__()
        # v1, v2, and v3 fuse multi-scale features
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
        # Global output head
        self.output = nn.Sequential(
            nn.Linear(4096 + clipnum, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )
        # Local 16-grid output head
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
    This is the model class currently used in train.py.
    It wraps the ALT-GVT backbone.
    """
    def __init__(self, clipnum: int = 16, pretrained_backbone: bool = False):
        super().__init__()
        # Core computations/layers are implemented in ALTGVT.py::alt_gvt_small
        self.backbone = alt_gvt_small(pretrained=pretrained_backbone)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rank_local, rank_global = self.backbone(x)
        # Map outputs to [0, 1]
        rank_local = torch.sigmoid(rank_local)
        rank_global = torch.sigmoid(rank_global)
        return rank_local, rank_global
