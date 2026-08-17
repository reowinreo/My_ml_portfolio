

from __future__ import annotations

import os
import glob
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


import sys
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from ALTGVT import alt_gvt_small
import ALTGVT as _altgvt_mod




CONFIG = {
    # datasetroot directory
    "dataset_root": "dataset",

    #   "RSOC-Ship" | "RSOC-S-Vehicle" | "RSOC-L-Vehicle" | "RSOC-Building"
    #   "VD-People" | "VD-Vehicle"
    "dataset": "RSOC-Building",

    "img_size": 512,
    "clipnum": 16,
    "batch_size": 8,
    "epochs": 200,
    "val_start": 1,
    "lr": 1e-5,
    "weight_decay": 1e-4,
    "num_workers": 4,
    "seed": 85,


    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "save_dir": "my_model",
    "resume": "",
    "min_mae": float("inf"),

    #hyperparameter
    "lambda_local": 1,          # λ
    "pretrained_backbone": True,
    "fast_lr": 1e-2,
}



# Regression head
class Regression(nn.Module):
    def __init__(self, clipnum: int = 16, chan: int = 256):
        super().__init__()
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
        self.output = nn.Sequential(
            nn.Linear(4096 + clipnum, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            #nn.Sigmoid(),
        )
        self.c16 = nn.Sequential(
            nn.Linear(chan, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            #nn.Sigmoid(),
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


# paper backbone(onlyranking output)
class HSRankALTGVT(nn.Module):
    """usepaperin  ALTGVT-Small backbone(output rank_local / rank_global). """
    def __init__(self, clipnum: int = 16, pretrained_backbone: bool = False):
        super().__init__()
        # Note:ALTGVT.py   pretrained=True willtryreadlocal 'alt_gvt_small.pth'
        self.backbone = alt_gvt_small(pretrained=pretrained_backbone)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # backbone originalOutput: (rank_local[ B,16 ], rank_global[ B,1 ])
        rank_local, rank_global_raw = self.backbone(x)

        # ====== modifypoint: remove"global ANN branch" output, change touse c16(=rank_local)   16 valuesumasglobal output ======
        # Description:inyou setsetinside, rank_local thenis c_16(16 dimensionvector), thereforeglobalscore = sum(c_16). 
        rank_global = rank_local.sum(dim=1, keepdim=True)

        # youraise to"originalprograminside ann afterisor nothastakenumbercompressto 0.x  operation": 
        # - thisscriptinsidedefine  Regression head tail Sigmoid iscommented out (see Regression.output / Regression.c16). 
        # - butyoulocal  ALTGVT.py insidemaybeenable  Sigmoid(orothercompress). 
        # heredoone"automaticalign": iforiginal rank_global_raw looks likealreadythroughbycompressto [0,1], that new  rank_global do the same  Sigmoid. 
        with torch.no_grad():
            _rg = rank_global_raw.detach()
            _minv = float(_rg.min().item())
            _maxv = float(_rg.max().item())
            _looks_sigmoid = (_minv >= -1e-3) and (_maxv <= 1.0 + 1e-3)

        if _looks_sigmoid:
            rank_global = torch.sigmoid(rank_global)

        return rank_local, rank_global

# Ranking Loss
T = 2


def torch_dcg_at_k(batch_sorted_labels: torch.Tensor, cutoff: Optional[int] = None) -> torch.Tensor:
    if cutoff is None:
        cutoff = batch_sorted_labels.size(1)
    numerators = torch.pow(2.0, batch_sorted_labels[:, 0:cutoff]) - 1.0
    discounts = torch.log2(torch.arange(cutoff, device=batch_sorted_labels.device, dtype=torch.float32) + 2.0)
    dcg = torch.sum(numerators / discounts, dim=1, keepdim=True)
    return dcg


def get_approx_ranks(input: torch.Tensor, alpha: float = 10) -> torch.Tensor:
    diffs = torch.unsqueeze(input, dim=2) - torch.unsqueeze(input, dim=1)
    indicators = torch.sigmoid(alpha * torch.transpose(diffs, 1, 2))
    hat_pis = torch.sum(indicators, dim=2) + 0.5
    return hat_pis


def ranking_loss(batch_preds: torch.Tensor, batch_stds: torch.Tensor, alpha: float = 10) -> torch.Tensor:
    hat_pis = get_approx_ranks(batch_preds, alpha=alpha)
    idcgs = torch_dcg_at_k(batch_sorted_labels=batch_stds, cutoff=None)
    gains = torch.pow(2.0, batch_stds) - 1.0
    dcg = torch.sum(gains / torch.log2(hat_pis + 1), dim=1).unsqueeze(dim=1)
    approx_ndcg = dcg / (idcgs + 1e-6)
    return 1 - torch.mean(approx_ndcg)


class RankingLoss(nn.Module):
    def __init__(self, alpha: float = 10, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    @staticmethod
    def _get_correl(labels: torch.Tensor, max_idx: torch.Tensor, min_idx: torch.Tensor) -> torch.Tensor:
        y_true = torch.zeros(2, len(labels) - 1, device=labels.device)
        if max_idx == 0:
            labels_del = labels[1:].squeeze()
            y_true[0, :] = labels[max_idx] - labels_del
        else:
            labels_del = torch.cat((labels[0:max_idx], labels[max_idx + 1 :]), dim=0).squeeze()
            y_true[0, :] = labels[max_idx] - labels_del

        if min_idx == 0:
            labels_del = labels[1:].squeeze()
            y_true[1, :] = labels_del - labels[min_idx]
        else:
            labels_del = torch.cat((labels[0:min_idx], labels[min_idx + 1 :]), dim=0).squeeze()
            y_true[1, :] = labels_del - labels[min_idx]
        return y_true

    @staticmethod
    def _get_similar(pred: torch.Tensor, max_idx: torch.Tensor, min_idx: torch.Tensor) -> torch.Tensor:
        y_pred = torch.zeros(2, len(pred) - 1, device=pred.device)
        if max_idx == 0:
            pred_del = pred[1:].squeeze()
            y_pred[0, :] = pred[max_idx] - pred_del
        else:
            pred_del = torch.cat((pred[0:max_idx], pred[max_idx + 1 :]), dim=0).squeeze()
            y_pred[0, :] = pred[max_idx] - pred_del

        if min_idx == 0:
            pred_del = pred[1:].squeeze()
            y_pred[1, :] = pred_del - pred[min_idx]
        else:
            pred_del = torch.cat((pred[0:min_idx], pred[min_idx + 1 :]), dim=0).squeeze()
            y_pred[1, :] = pred_del - pred[min_idx]
        return y_pred

    def forward(self, pred_in: torch.Tensor, labels_in: torch.Tensor, c4: bool = False, cn: int = 16) -> torch.Tensor:
        eps = self.eps
        if c4:
            loss_sum = 0.0
            valid = 0
            for idx in range(len(pred_in)):
                pred = pred_in[idx : idx + 1].view(cn, 1)
                labels = labels_in[idx : idx + 1].view(cn, 1)
                max_val, max_idx = torch.max(labels, dim=0)
                min_val, min_idx = torch.min(labels, dim=0)
                denom = (max_val - min_val).abs()
                if denom.item() < eps:
                    continue
                stds = self._get_correl(labels, max_idx, min_idx)
                stds = 1 - (stds / (denom + eps))
                stds = stds * T

                preds = self._get_similar(pred, max_idx, min_idx)
                preds = 1 - preds

                target_stds, inds = torch.sort(stds, dim=1, descending=True)
                target_preds = torch.gather(preds, dim=1, index=inds)
                loss_sum += ranking_loss(target_preds, target_stds, self.alpha)
                valid += 1

            if valid == 0:
                return pred_in.sum() * 0.0
            return loss_sum / valid

        max_val, max_idx = torch.max(labels_in, dim=0)
        min_val, min_idx = torch.min(labels_in, dim=0)
        denom = (max_val - min_val).abs()
        if denom.item() < eps:
            return pred_in.sum() * 0.0

        stds = self._get_correl(labels_in, max_idx, min_idx)
        stds = 1 - (stds / (denom + eps))
        stds = stds * T

        preds = self._get_similar(pred_in, max_idx, min_idx)
        preds = 1 - preds

        target_stds, inds = torch.sort(stds, dim=1, descending=True)
        target_preds = torch.gather(preds, dim=1, index=inds)
        return ranking_loss(target_preds, target_stds, self.alpha)


# datasetandreal labelread
def _safe_read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _build_allow_set_for_rsoc_mixed(root: str, dataset_label: str, split: str) -> set[str]:
    cat_token = {
        "RSOC-Ship": "ship",
        "RSOC-S-Vehicle": "small-vehicle",
        "RSOC-L-Vehicle": "large-vehicle",
    }.get(dataset_label)
    if cat_token is None:
        return set()

    dota_dir = os.path.join(root, "ASPDNet_dataset", "DOTA_data")
    txt_path = os.path.join(dota_dir, f"{split}_{cat_token}.txt")
    if os.path.isfile(txt_path):
        # has listwillwrite as "P0000", has willwrite as "P0000.jpg"; unifybecomenotwithextension
        allow = set()
        for line in _safe_read_lines(txt_path):
            allow.add(os.path.splitext(line)[0])
        return allow

    # fallback: scan labelTxt
    label_dir = os.path.join(
        root,
        "ASPDNet_dataset",
        split,
        "labelTxt-v1.0",
        f"{split}set_reclabelTxt",
    )
    allow = set()
    for lbl in glob.glob(os.path.join(label_dir, "*.txt")):
        base = os.path.splitext(os.path.basename(lbl))[0]
        try:
            hit = False
            for line in _safe_read_lines(lbl):
                parts = line.split()
                if len(parts) < 9:
                    continue
                cls = parts[8] if len(parts) >= 10 else parts[-1]
                if cls == cat_token:
                    hit = True
                    break
            if hit:
                allow.add(base)
        except Exception:
            continue
    return allow


def _list_images(root: str, dataset_label: str, split: str) -> List[str]:
    if dataset_label == "RSOC-Building":
        rsoc_root = os.path.join(root, "ASPDNet_dataset", "RSOC_building", "building")
        if split == "train":
            sub = "train_data"
        elif split in ("val", "test"):
            sub = "test_data"
        else:
            return []
        img_dir = os.path.join(rsoc_root, sub, "images")
        if not os.path.isdir(img_dir):
            return []
        return sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    if dataset_label in ("RSOC-Ship", "RSOC-S-Vehicle", "RSOC-L-Vehicle"):
        img_dir = os.path.join(root, "ASPDNet_dataset", split, "images")
        if not os.path.isdir(img_dir):
            return []
        allow = _build_allow_set_for_rsoc_mixed(root, dataset_label, split)
        if not allow:
            return []
        out = [
            p
            for p in glob.glob(os.path.join(img_dir, "*.jpg"))
            if os.path.splitext(os.path.basename(p))[0] in allow
        ]
        return sorted(out)

    if dataset_label in ("VD-People", "VD-Vehicle"):
        vd_root = os.path.join(root, "VisDrone-People" if dataset_label == "VD-People" else "VisDrone-Vehicle")
        img_dir1 = os.path.join(vd_root, split, "images")
        img_dir2 = os.path.join(vd_root, split, "Images")
        img_dir = img_dir1 if os.path.isdir(img_dir1) else img_dir2
        if not os.path.isdir(img_dir):
            return []
        exts = ("*.jpg", "*.png", "*.jpeg")
        out = []
        for e in exts:
            out.extend(glob.glob(os.path.join(img_dir, e)))
        return sorted(out)

    return []


def _grid_index(x: float, y: float, w: float, h: float, n: int = 4) -> int:
    # clamp
    x = max(0.0, min(x, w - 1e-6))
    y = max(0.0, min(y, h - 1e-6))
    gx = int(x / (w / n))
    gy = int(y / (h / n))
    gx = min(max(gx, 0), n - 1)
    gy = min(max(gy, 0), n - 1)
    return gy * n + gx


def _counts_from_points(points_xy: np.ndarray, width: int, height: int, n: int = 4) -> np.ndarray:
    counts = np.zeros((n * n,), dtype=np.float32)
    for (x, y) in points_xy:
        idx = _grid_index(float(x), float(y), float(width), float(height), n=n)
        counts[idx] += 1.0
    return counts


def _read_dota_label_points(label_path: str, width: int, height: int, class_token: str, out_size: int = 512) -> np.ndarray:

    pts = []
    for line in _safe_read_lines(label_path):
        parts = line.split()
        if len(parts) < 9:
            continue
        # typical: 8 coords + class + diff
        cls = parts[8] if len(parts) >= 10 else parts[-1]
        if cls != class_token:
            continue
        try:
            coords = list(map(float, parts[:8]))
        except Exception:
            continue
        xs = coords[0::2]
        ys = coords[1::2]
        cx = sum(xs) / 4.0
        cy = sum(ys) / 4.0
        # map to 512
        cx = cx / max(width, 1) * out_size
        cy = cy / max(height, 1) * out_size
        pts.append((cx, cy))
    if len(pts) == 0:
        return np.zeros((16,), dtype=np.float32)
    return _counts_from_points(np.asarray(pts, dtype=np.float32), out_size, out_size, n=4)


def _read_building_mat_points(mat_path: str, width: int, height: int, out_size: int = 512) -> np.ndarray:
    import scipy.io as sio

    mat = sio.loadmat(mat_path)
    if "center" not in mat:
        return np.zeros((16,), dtype=np.float32)
    center = mat["center"]
    # youto example: center is object array, first itemis Nx2
    try:
        pts = center[0, 0]
        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            return np.zeros((16,), dtype=np.float32)
    except Exception:
        return np.zeros((16,), dtype=np.float32)

    # map to 512
    pts2 = np.zeros_like(pts, dtype=np.float32)
    pts2[:, 0] = pts[:, 0] / max(width, 1) * out_size
    pts2[:, 1] = pts[:, 1] / max(height, 1) * out_size
    return _counts_from_points(pts2, out_size, out_size, n=4)


def _read_visdrone_density_map(npy_path: str) -> Tuple[np.ndarray, float]:
    dm = np.load(npy_path)
    dm = np.asarray(dm, dtype=np.float32)
    h, w = dm.shape[:2]
    n = 4
    counts = np.zeros((n * n,), dtype=np.float32)
    for gy in range(n):
        y0 = int(round(gy * h / n))
        y1 = int(round((gy + 1) * h / n))
        for gx in range(n):
            x0 = int(round(gx * w / n))
            x1 = int(round((gx + 1) * w / n))
            idx = gy * n + gx
            counts[idx] = float(dm[y0:y1, x0:x1].sum())
    return counts, float(dm.sum())


def _read_visdrone_txt_points(txt_path: str, width: int, height: int, out_size: int = 512) -> np.ndarray:

    pts = []
    for line in _safe_read_lines(txt_path):
        parts = line.replace(",", " ").split()
        if len(parts) < 6:
            continue
        try:
            x, y, bw, bh = map(float, parts[:4])
            cat = int(float(parts[5]))
        except Exception:
            continue
        if cat <= 0:
            continue
        cx = x + bw / 2.0
        cy = y + bh / 2.0
        cx = cx / max(width, 1) * out_size
        cy = cy / max(height, 1) * out_size
        pts.append((cx, cy))
    if len(pts) == 0:
        return np.zeros((16,), dtype=np.float32)
    return _counts_from_points(np.asarray(pts, dtype=np.float32), out_size, out_size, n=4)


class RealRankDataset(Dataset):
    def __init__(self, root: str, dataset_label: str, split: str, img_size: int = 512):
        self.root = root
        self.dataset_label = dataset_label
        self.split = split
        self.img_size = img_size
        self.n = 4

        self.img_paths = _list_images(root, dataset_label, split)

        # in order toavoidsomeenvironment torchvision importfail(commonin nms/op compilenotmatch), 
        # herenotuse torchvision.transforms, changeusepure PIL + numpy + torch implement: Resize + ToTensor + Normalize. 
        self._mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def _transform(self, img_pil: Image.Image) -> torch.Tensor:
        img = img_pil.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC, 0..1
        arr = (arr - self._mean) / self._std
        arr = np.transpose(arr, (2, 0, 1))  # CHW
        return torch.from_numpy(arr)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int):
        img_path = self.img_paths[index]
        img_name = os.path.basename(img_path)
        base = os.path.splitext(img_name)[0]

        img_pil = Image.open(img_path).convert("RGB")
        w0, h0 = img_pil.size
        img = self._transform(img_pil)

        local = None
        global_count = None

        if self.dataset_label in ("RSOC-Ship", "RSOC-S-Vehicle", "RSOC-L-Vehicle"):
            class_token = {
                "RSOC-Ship": "ship",
                "RSOC-S-Vehicle": "small-vehicle",
                "RSOC-L-Vehicle": "large-vehicle",
            }[self.dataset_label]
            label_dir = os.path.join(
                self.root,
                "ASPDNet_dataset",
                self.split,
                "labelTxt-v1.0",
                f"{self.split}set_reclabelTxt",
            )
            label_path = os.path.join(label_dir, base + ".txt")
            if not os.path.isfile(label_path):
                raise FileNotFoundError(f"Label not found: {label_path}")
            local = _read_dota_label_points(label_path, w0, h0, class_token, out_size=self.img_size)
            global_count = float(local.sum())

        elif self.dataset_label == "RSOC-Building":
            rsoc_root = os.path.join(self.root, "ASPDNet_dataset", "RSOC_building", "building")
            sub = "train_data" if self.split == "train" else "test_data"
            gt_dir = os.path.join(rsoc_root, sub, "ground_truth")
            mat_path = os.path.join(gt_dir, f"GT_{base}.mat")
            if not os.path.isfile(mat_path):
                # compatible: hassomedatawilluse GT_IMG_xxx.mat / IMG_xxx.jpg
                mat_path2 = os.path.join(gt_dir, f"GT_{img_name}".replace(".jpg", ".mat"))
                mat_path = mat_path2
            if not os.path.isfile(mat_path):
                raise FileNotFoundError(f"Building GT not found: {mat_path}")
            local = _read_building_mat_points(mat_path, w0, h0, out_size=self.img_size)
            global_count = float(local.sum())

        elif self.dataset_label in ("VD-People", "VD-Vehicle"):
            vd_root = os.path.join(self.root, "VisDrone-People" if self.dataset_label == "VD-People" else "VisDrone-Vehicle")
            gt_dir1 = os.path.join(vd_root, self.split, "Ground Truth")
            gt_dir2 = os.path.join(vd_root, self.split, "ground_truth")
            gt_dir = gt_dir1 if os.path.isdir(gt_dir1) else gt_dir2
            if not os.path.isdir(gt_dir):
                raise FileNotFoundError(f"VisDrone GT dir not found: {gt_dir1} / {gt_dir2}")

            npy_path = os.path.join(gt_dir, base + ".npy")
            if os.path.isfile(npy_path):
                local, global_count = _read_visdrone_density_map(npy_path)
            else:
                txt_path = os.path.join(gt_dir, base + ".txt")
                if not os.path.isfile(txt_path):
                    raise FileNotFoundError(f"VisDrone GT not found: {npy_path} or {txt_path}")
                local = _read_visdrone_txt_points(txt_path, w0, h0, out_size=self.img_size)
                global_count = float(local.sum())

        else:
            raise ValueError(f"Unknown dataset label: {self.dataset_label}")

        local = np.asarray(local, dtype=np.float32).reshape(-1)
        if local.shape[0] != 16:
            raise ValueError(f"local_counts must be 16, got {local.shape} for {img_name}")

        return img, torch.tensor(global_count, dtype=torch.float32), torch.from_numpy(local), img_name


# Train / Val
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




def train_epoch(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    loader: DataLoader,
    loss_fn: RankingLoss,
    device: str,
    lambda_local: float = 0.5,
):
    model.train()
    loop = tqdm(loader, desc=f"train {epoch}", ncols=110)
    loss_total = 0.0
    loss_g = 0.0
    loss_l = 0.0
    num = 0

    for image, target_global, target_local, _ in loop:
        image = image.to(device)
        target_global = target_global.to(device)
        target_local = target_local.to(device)

        # Output:
        #   rank_local: [B,16]   rank_global: [B,1]
        rank_local, rank_global = model(image)

        denom_g = (target_global.max() - target_global.min()).abs().item()
        if denom_g < 1e-6:
            lg = rank_global.sum() * 0.0
        else:
            lg = loss_fn(rank_global.squeeze(), target_global)

        # ifsomeimage 16 blockallsame, willin RankingLoss insidereturn 0
        ll = loss_fn(rank_local, target_local, c4=True, cn=16)

        loss = lg + float(lambda_local) * ll

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_total += float(loss.item())
        loss_g += float(lg.item())
        loss_l += float(ll.item())
        num += 1
        loop.set_postfix(
            loss=loss_total / max(num, 1),
            global_rank=loss_g / max(num, 1),
            local_rank=loss_l / max(num, 1),
        )



@torch.no_grad()
def val_epoch(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    """invalidation setonevaluation"ranking output (rank_global)" fittingeffect: 
    - firstuseleast squarestake rank_global linearitymap to count
    - againcompute MAE / RMSE
    return (MAE, RMSE), used forsave"rankingtablenowbest" model. 
    """
    model.eval()

    preds_rank = []
    gts = []

    for image, target_global, _, _ in tqdm(loader, desc="val", ncols=110):
        image = image.to(device)
        target_global = target_global.to(device)

        _, rank_global = model(image)

        preds_rank.append(float(rank_global.item()))
        gts.append(float(target_global.item()))

    n = len(gts)
    if n == 0:
        return float("inf"), float("inf")

    # ranking output -> count  linearityregressioncalibration(originallogic)
    a_mean = sum(preds_rank) / n
    top = sum(gt * (a - a_mean) for a, gt in zip(preds_rank, gts))
    bot = sum((a - a_mean) ** 2 for a in preds_rank)
    k = top / bot if bot != 0 else 1.0
    b = sum(gt - k * a for a, gt in zip(preds_rank, gts)) / n

    mae = 0.0
    rmse = 0.0
    for a, gt in zip(preds_rank, gts):
        p = k * a + b
        mae += abs(p - gt)
        rmse += (p - gt) ** 2
    mae /= n
    rmse = math.sqrt(rmse / n)

    print(f"Val (RANK+LS) MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return mae, rmse


def main():
    set_seed(int(CONFIG["seed"]))
    root = CONFIG["dataset_root"]
    dataset_label = CONFIG["dataset"]
    device = CONFIG["device"]
    _altgvt_mod.device = torch.device(device)

    pretrained_backbone = bool(CONFIG.get("pretrained_backbone", False))
    model = HSRankALTGVT(clipnum=int(CONFIG["clipnum"]), pretrained_backbone=pretrained_backbone).to(device)
    if CONFIG["resume"]:
        model.load_state_dict(torch.load(CONFIG["resume"], map_location=device))

    loss_fn = RankingLoss()

    base_lr = float(CONFIG["lr"])
    fast_lr = float(CONFIG.get("fast_lr", base_lr * 1000))
    weight_decay = float(CONFIG["weight_decay"])

    fast_params = []
    if hasattr(model.backbone, "regression"):
        try:
            fast_params += list(model.backbone.regression.output[-2].parameters())
            fast_params += list(model.backbone.regression.c16[-2].parameters())
        except Exception:
            # fallback: takewhole output/c16 putto fast group
            fast_params += list(model.backbone.regression.output.parameters())
            fast_params += list(model.backbone.regression.c16.parameters())

    fast_ids = {id(p) for p in fast_params}
    base_params = [p for p in model.parameters() if p.requires_grad and id(p) not in fast_ids]

    optimizer = optim.Adam(
        [
            {"params": base_params, "lr": base_lr, "weight_decay": weight_decay},
            {"params": fast_params, "lr": fast_lr, "weight_decay": weight_decay},
        ],
        betas=(0.9, 0.999),
    )
    train_ds = RealRankDataset(root, dataset_label, "train", img_size=int(CONFIG["img_size"]))
    val_ds = RealRankDataset(root, dataset_label, "val", img_size=int(CONFIG["img_size"]))

    # somedatasetno val(for example building onlyhas train/test), thenuse train fillwhen val
    if len(val_ds) == 0:
        print("[WARN] current dataset has no val split, automatically use train as val performcalibrationandevaluation. ")
        val_ds = train_ds

    if len(train_ds) == 0:
        raise RuntimeError(f"No training images found for {dataset_label} under {root}")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(CONFIG["batch_size"]),
        shuffle=True,
        num_workers=int(CONFIG["num_workers"]),
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, int(CONFIG["num_workers"]) // 2),
        pin_memory=True,
    )

    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    min_mae = float(CONFIG["min_mae"])
    epochs = int(CONFIG["epochs"])

    for epoch in range(1, epochs + 1):
        train_epoch(epoch, model, optimizer, train_loader, loss_fn, device, lambda_local=float(CONFIG.get("lambda_local", 0.5)))
        if epoch >= int(CONFIG["val_start"]):
            mae, rmse = val_epoch(model, val_loader, device)
            if mae < min_mae:
                min_mae = mae
                out_path = os.path.join(
                    CONFIG["save_dir"], f"mae{mae:.2f}_rmse{rmse:.2f}_epoch{epoch}_{dataset_label}.pth"
                )
                torch.save(model.state_dict(), out_path)
                print(f"[SAVE] {out_path}")

    print("Training done!")


if __name__ == "__main__":
    main()
