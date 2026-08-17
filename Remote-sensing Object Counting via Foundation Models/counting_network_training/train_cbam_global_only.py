"""train_cbam_global_only.py

Objective:
  inbased onreal label trainingbasison, [removelocal(Local)branch], onlyretainglobal(Global)regression. 

modifypoint: 
  1. Regression head: remove  _block cutand c16 branch, y feature mapdirectenter output fully connectedlayer. 
  2. trainingloop: onlycompute global ranking loss, notagaincompute local loss. 
  3. modelOutput:forward onlyreturn global_count. 

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
# CONFIG
# =========================
CONFIG = {
    "dataset_root": "dataset",

    # optional: "RSOC-Ship", "RSOC-Building", "VD-People", "VD-Vehicle" etc.
    "dataset": "RSOC-Building",

    "img_size": 512,
    # Note:althoughnotagainuse clipnum split, butin order tocompatibilityoraftercontinueextension, retainvariabledefine, butinnetworkinnotagainassplitaccordingaccording
    "clipnum": 16,
    "batch_size": 8,
    "epochs": 200,
    "val_start": 1,
    "lr": 1e-5,
    "weight_decay": 1e-4,
    "num_workers": 4,
    "seed": 42,

    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "save_dir": "my_model_global_only",
    "resume": "",
    "min_mae": float("inf"),
}


# =========================
# CBAM Components
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
# Regression head (Modified: Global Only)
# =========================
class Regression(nn.Module):
    def __init__(self, chan: int = 256):
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

        # modifypoint: input dimensionnotagainadd clipnum(16), onlyonlyis 64*64=4096
        self.output = nn.Sequential(
            nn.Linear(4096, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        # remove  self.c16 (local branch)

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
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # remove  _block function

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        x1 = self.v1(x1)
        x2 = self.v2(x2)
        x3 = self.v3(x3)
        x = x1 + x2 + x3

        y = self.res(x)  # Shape: [Batch, 1, 64, 64]

        # modifypoint: direct Flatten, notcut, notspellconnect
        y_flat = y.view(y.size(0), -1)  # Shape: [Batch, 4096]

        y_global = self.output(y_flat)
        return y_global


# =========================
# CBAM Backbone (Modified Interface)
# =========================
class _ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, k: int = 3, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CBAMBackbone(nn.Module):
    def __init__(self, in_chans: int = 3):
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
        self.regression = Regression()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage0(x)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        y_global = self.regression(x1, x2, x3)
        return y_global


# =========================
# Ranking Loss (UNCHANGED logic, but only used for global)
# =========================
T = 1


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
            labels_del = torch.cat((labels[0:max_idx], labels[max_idx + 1:]), dim=0).squeeze()
            y_true[0, :] = labels[max_idx] - labels_del
        if min_idx == 0:
            labels_del = labels[1:].squeeze()
            y_true[1, :] = labels_del - labels[min_idx]
        else:
            labels_del = torch.cat((labels[0:min_idx], labels[min_idx + 1:]), dim=0).squeeze()
            y_true[1, :] = labels_del - labels[min_idx]
        return y_true

    @staticmethod
    def _get_similar(pred: torch.Tensor, max_idx: torch.Tensor, min_idx: torch.Tensor) -> torch.Tensor:
        y_pred = torch.zeros(2, len(pred) - 1, device=pred.device)
        if max_idx == 0:
            pred_del = pred[1:].squeeze()
            y_pred[0, :] = pred[max_idx] - pred_del
        else:
            pred_del = torch.cat((pred[0:max_idx], pred[max_idx + 1:]), dim=0).squeeze()
            y_pred[0, :] = pred[max_idx] - pred_del
        if min_idx == 0:
            pred_del = pred[1:].squeeze()
            y_pred[1, :] = pred_del - pred[min_idx]
        else:
            pred_del = torch.cat((pred[0:min_idx], pred[min_idx + 1:]), dim=0).squeeze()
            y_pred[1, :] = pred_del - pred[min_idx]
        return y_pred

    def forward(self, pred_in: torch.Tensor, labels_in: torch.Tensor) -> torch.Tensor:
        # onlyretain Global Ranking Loss  logic
        eps = self.eps
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


# =========================
# Data Utilities (Mostly UNCHANGED, just helpers)
# =========================
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
        allow = set()
        for line in _safe_read_lines(txt_path):
            allow.add(os.path.splitext(line)[0])
        return allow
    label_dir = os.path.join(root, "ASPDNet_dataset", split, "labelTxt-v1.0", f"{split}set_reclabelTxt")
    allow = set()
    for lbl in glob.glob(os.path.join(label_dir, "*.txt")):
        base = os.path.splitext(os.path.basename(lbl))[0]
        try:
            hit = False
            for line in _safe_read_lines(lbl):
                parts = line.split()
                if len(parts) < 9: continue
                cls = parts[8] if len(parts) >= 10 else parts[-1]
                if cls == cat_token:
                    hit = True
                    break
            if hit: allow.add(base)
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
        if not os.path.isdir(img_dir): return []
        return sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    if dataset_label in ("RSOC-Ship", "RSOC-S-Vehicle", "RSOC-L-Vehicle"):
        img_dir = os.path.join(root, "ASPDNet_dataset", split, "images")
        if not os.path.isdir(img_dir): return []
        allow = _build_allow_set_for_rsoc_mixed(root, dataset_label, split)
        if not allow: return []
        out = [p for p in glob.glob(os.path.join(img_dir, "*.jpg")) if
               os.path.splitext(os.path.basename(p))[0] in allow]
        return sorted(out)

    if dataset_label in ("VD-People", "VD-Vehicle"):
        vd_root = os.path.join(root, "VisDrone-People" if dataset_label == "VD-People" else "VisDrone-Vehicle")
        img_dir1 = os.path.join(vd_root, split, "images")
        img_dir2 = os.path.join(vd_root, split, "Images")
        img_dir = img_dir1 if os.path.isdir(img_dir1) else img_dir2
        if not os.path.isdir(img_dir): return []
        exts = ("*.jpg", "*.png", "*.jpeg")
        out = []
        for e in exts: out.extend(glob.glob(os.path.join(img_dir, e)))
        return sorted(out)
    return []


# auxiliaryfunction: althoughmodelnotuselocal, butreadlabel logicalsoisdependencygridproductpartorcenter pointcomputegettotalnumber
def _grid_index(x, y, w, h, n=4):
    x = max(0.0, min(x, w - 1e-6))
    y = max(0.0, min(y, h - 1e-6))
    gx = int(x / (w / n))
    gy = int(y / (h / n))
    gx = min(max(gx, 0), n - 1)
    gy = min(max(gy, 0), n - 1)
    return gy * n + gx


def _counts_from_points(points_xy, width, height, n=4):
    counts = np.zeros((n * n,), dtype=np.float32)
    for (x, y) in points_xy:
        idx = _grid_index(float(x), float(y), float(width), float(height), n=n)
        counts[idx] += 1.0
    return counts


def _read_dota_label_points(label_path, width, height, class_token, out_size=512):
    pts = []
    for line in _safe_read_lines(label_path):
        parts = line.split()
        if len(parts) < 9: continue
        cls = parts[8] if len(parts) >= 10 else parts[-1]
        if cls != class_token: continue
        try:
            coords = list(map(float, parts[:8]))
        except:
            continue
        xs = coords[0::2];
        ys = coords[1::2]
        cx = sum(xs) / 4.0;
        cy = sum(ys) / 4.0
        cx = cx / max(width, 1) * out_size
        cy = cy / max(height, 1) * out_size
        pts.append((cx, cy))
    if len(pts) == 0: return np.zeros((16,), dtype=np.float32)
    return _counts_from_points(np.asarray(pts, dtype=np.float32), out_size, out_size, n=4)


def _read_building_mat_points(mat_path, width, height, out_size=512):
    import scipy.io as sio
    mat = sio.loadmat(mat_path)
    if "center" not in mat: return np.zeros((16,), dtype=np.float32)
    center = mat["center"]
    try:
        pts = center[0, 0]
        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2: return np.zeros((16,), dtype=np.float32)
    except:
        return np.zeros((16,), dtype=np.float32)
    pts2 = np.zeros_like(pts, dtype=np.float32)
    pts2[:, 0] = pts[:, 0] / max(width, 1) * out_size
    pts2[:, 1] = pts[:, 1] / max(height, 1) * out_size
    return _counts_from_points(pts2, out_size, out_size, n=4)


def _read_visdrone_density_map(npy_path):
    dm = np.load(npy_path)
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


def _read_visdrone_txt_points(txt_path, width, height, out_size=512):
    pts = []
    for line in _safe_read_lines(txt_path):
        parts = line.replace(",", " ").split()
        if len(parts) < 6: continue
        try:
            x, y, bw, bh = map(float, parts[:4])
            cat = int(float(parts[5]))
        except:
            continue
        if cat <= 0: continue
        cx = x + bw / 2.0;
        cy = y + bh / 2.0
        cx = cx / max(width, 1) * out_size
        cy = cy / max(height, 1) * out_size
        pts.append((cx, cy))
    if len(pts) == 0: return np.zeros((16,), dtype=np.float32)
    return _counts_from_points(np.asarray(pts, dtype=np.float32), out_size, out_size, n=4)


class RealRankDataset(Dataset):
    def __init__(self, root: str, dataset_label: str, split: str, img_size: int = 512):
        self.root = root
        self.dataset_label = dataset_label
        self.split = split
        self.img_size = img_size
        self.n = 4
        self.img_paths = _list_images(root, dataset_label, split)
        self._mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def _transform(self, img_pil: Image.Image) -> torch.Tensor:
        img = img_pil.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - self._mean) / self._std
        arr = np.transpose(arr, (2, 0, 1))
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

        # even ifonlyuseglobal, wealsoisreuseprevious readlogiccomeget global_count
        if self.dataset_label in ("RSOC-Ship", "RSOC-S-Vehicle", "RSOC-L-Vehicle"):
            class_token = {"RSOC-Ship": "ship", "RSOC-S-Vehicle": "small-vehicle", "RSOC-L-Vehicle": "large-vehicle"}[
                self.dataset_label]
            label_dir = os.path.join(self.root, "ASPDNet_dataset", self.split, "labelTxt-v1.0",
                                     f"{self.split}set_reclabelTxt")
            label_path = os.path.join(label_dir, base + ".txt")
            if not os.path.isfile(label_path): raise FileNotFoundError(f"Label not found: {label_path}")
            local = _read_dota_label_points(label_path, w0, h0, class_token, out_size=self.img_size)
            global_count = float(local.sum())
        elif self.dataset_label == "RSOC-Building":
            rsoc_root = os.path.join(self.root, "ASPDNet_dataset", "RSOC_building", "building")
            sub = "train_data" if self.split == "train" else "test_data"
            gt_dir = os.path.join(rsoc_root, sub, "ground_truth")
            mat_path = os.path.join(gt_dir, f"GT_{base}.mat")
            if not os.path.isfile(mat_path): mat_path = os.path.join(gt_dir, f"GT_{img_name}".replace(".jpg", ".mat"))
            if not os.path.isfile(mat_path): raise FileNotFoundError(f"Building GT not found: {mat_path}")
            local = _read_building_mat_points(mat_path, w0, h0, out_size=self.img_size)
            global_count = float(local.sum())
        elif self.dataset_label in ("VD-People", "VD-Vehicle"):
            vd_root = os.path.join(self.root,
                                   "VisDrone-People" if self.dataset_label == "VD-People" else "VisDrone-Vehicle")
            gt_dir1 = os.path.join(vd_root, self.split, "Ground Truth")
            gt_dir2 = os.path.join(vd_root, self.split, "ground_truth")
            gt_dir = gt_dir1 if os.path.isdir(gt_dir1) else gt_dir2
            if not os.path.isdir(gt_dir): raise FileNotFoundError(f"VisDrone GT not found")
            npy_path = os.path.join(gt_dir, base + ".npy")
            if os.path.isfile(npy_path):
                local, global_count = _read_visdrone_density_map(npy_path)
            else:
                txt_path = os.path.join(gt_dir, base + ".txt")
                if not os.path.isfile(txt_path): raise FileNotFoundError(f"VisDrone GT not found: {npy_path}")
                local = _read_visdrone_txt_points(txt_path, w0, h0, out_size=self.img_size)
                global_count = float(local.sum())
        else:
            raise ValueError(f"Unknown dataset label: {self.dataset_label}")

        # return local onlyascompatible, trainingwhennotuse
        local = np.asarray(local, dtype=np.float32).reshape(-1)
        return img, torch.tensor(global_count, dtype=torch.float32), torch.from_numpy(local), img_name


# =========================
# Train / Val (Modified Loop)
# =========================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(epoch: int, model: nn.Module, optimizer: optim.Optimizer, loader: DataLoader, loss_fn: RankingLoss,
                device: str):
    model.train()
    loop = tqdm(loader, desc=f"train {epoch}", ncols=110)
    loss_total = 0.0
    num = 0
    # Note:hereweconnectreceive target_local butignoreit
    for image, target_global, _, _ in loop:
        if (target_global.max() - target_global.min()).abs().item() < 1e-6:
            continue
        image = image.to(device)
        target_global = target_global.to(device)

        # modifypoint: modelonlyreturnone global output
        out_global = model(image)

        # modifypoint: onlycompute global ranking loss
        loss = loss_fn(out_global.squeeze(), target_global)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_total += float(loss.item())
        num += 1
        loop.set_postfix(loss=loss_total / max(num, 1))


@torch.no_grad()
def val_epoch(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    preds = []
    gts = []
    for image, target_global, _, _ in tqdm(loader, desc="val", ncols=110):
        image = image.to(device)
        out_global = model(image)  # onlyreturnonevalue
        preds.append(float(out_global.item()))
        gts.append(float(target_global.item()))

    n = len(preds)
    if n == 0: return float("inf"), float("inf")

    # simple linear calibration (Linear Regression Calibration)
    a_mean = sum(preds) / n
    top = sum(gt * (a - a_mean) for a, gt in zip(preds, gts))
    bot = sum((a - a_mean) ** 2 for a in preds)
    k = top / bot if bot != 0 else 1.0
    b = sum(gt - k * a for a, gt in zip(preds, gts)) / n

    mae = 0.0
    rmse = 0.0
    for a, gt in zip(preds, gts):
        p = k * a + b
        mae += abs(p - gt)
        rmse += (p - gt) ** 2
    mae /= n
    rmse = math.sqrt(rmse / n)
    print(f"Val MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return mae, rmse


def main():
    set_seed(int(CONFIG["seed"]))
    root = CONFIG["dataset_root"]
    dataset_label = CONFIG["dataset"]
    device = CONFIG["device"]

    model = CBAMBackbone()
    model.to(device)
    if CONFIG["resume"]:
        model.load_state_dict(torch.load(CONFIG["resume"], map_location=device))

    loss_fn = RankingLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(CONFIG["lr"]), weight_decay=float(CONFIG["weight_decay"]))

    train_ds = RealRankDataset(root, dataset_label, "train", img_size=int(CONFIG["img_size"]))
    val_ds = RealRankDataset(root, dataset_label, "val", img_size=int(CONFIG["img_size"]))

    if len(val_ds) == 0:
        print("[WARN] No val set found, using train as val.")
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
        train_epoch(epoch, model, optimizer, train_loader, loss_fn, device)
        if epoch >= int(CONFIG["val_start"]):
            mae, rmse = val_epoch(model, val_loader, device)
            if mae < min_mae:
                min_mae = mae
                out_path = os.path.join(CONFIG["save_dir"],
                                        f"mae{mae:.2f}_rmse{rmse:.2f}_epoch{epoch}_{dataset_label}.pth")
                torch.save(model.state_dict(), out_path)
                print(f"[SAVE] {out_path}")

    print("Training done!")


if __name__ == "__main__":
    main()