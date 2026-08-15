# main_dual_branch_density_map.py
# RSOC-Building ONLY
# Input is immediately split into two branches:
#   Branch-A: ALTGVT (rank_global + rank_local)
#   Branch-B: Density Map Regression (CSRNet-lite)
#
# Total loss:
#   L_altgvt = L_global + 0.5 * L_local
#   L = a1 * L_altgvt + a2 * L_density
# where a_1/a_2 are manual weights (set in CONFIG)
# ------------------------------------------------------------

from __future__ import annotations
import os, glob, math, random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# ========= Your ALTGVT backbone =========
import ALTGVT as _altgvt_mod
from ALTGVT import alt_gvt_small

# =========================
# Config
# =========================
CONFIG = {
    "dataset_root": "dataset",
    "dataset": "RSOC-Building",

    "label_dir": os.path.join("dataset", "local pseudo"),
    "train_label": "rsoc-building_grid16_train.txt",
    "test_label": "rsoc-building_grid16_test.txt",

    "img_size": 512,
    "grid_n": 4,  # 4x4 -> 16 cells

    "truncate": 3.0,
    "sigma_area_ranges": {
        "a1": 1024,
        "a2": 5355,
        "a3": 16129,
        "s1": 1.0,
        "s2": 15.0,
        "s3": 32.0,
    },

    "density_scale": 1.0,

    "a_1": 1.0,  # ranking branch loss weight
    "a_2": 1,  # density map branch loss weight (ignored when dynamic weighting is enabled)
    "use_dynamic_weighting": True,  # use uncertainty to dynamically weight the two branch losses
    "dyn_init_log_vars": (0.0, 0.0),  # initial values for (log_var_rank, log_var_density)

    "batch_size": 8,
    "epochs": 150,
    "lr": 1e-4,
    "fast_lr": None,
    "weight_decay": 1e-5,
    "num_workers": 4,
    "seed": 42,

    "device": "cuda:0" if torch.cuda.is_available() else "cpu",

    "save_dir": "my_model_dual_branch",
    "resume": "",
}


# =========================
# Utils
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_boxes(label_txt: str) -> Dict[str, List[Tuple[float, float, float, float]]]:
    mapping: Dict[str, List[Tuple[float, float, float, float]]] = {}
    if not os.path.isfile(label_txt):
        print(f"[WARN] label file not found: {label_txt}")
        return mapping

    for ln in safe_read_lines(label_txt):
        parts = ln.split()
        if len(parts) != 5:
            continue
        name = parts[0]
        x1, y1, x2, y2 = map(float, parts[1:])
        mapping.setdefault(name, []).append((x1, y1, x2, y2))
    print(f"[INFO] loaded boxes from {label_txt}, images with boxes: {len(mapping)}")
    return mapping


def map_area_to_sigma(area: float, cfg: dict) -> float:
    a1 = cfg["a1"]
    a2 = cfg["a2"]
    a3 = cfg["a3"]
    s1 = cfg["s1"]
    s2 = cfg["s2"]
    s3 = cfg["s3"]

    if area <= 0:
        return s1

    area = float(area)
    if area < a1:
        return s1
    if area <= a2:
        t = (area - a1) / (a2 - a1)
        return s1 + t * (s2 - s1)
    if area < a3:
        return s2
    # area >= a3
    return s3


def gaussian_kernel2d(sigma: float, truncate: float = 3.0) -> np.ndarray:
    radius = int(math.ceil(truncate * sigma))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    ys = np.arange(-radius, radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    ker = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma * sigma))
    s = float(ker.sum())
    if s > 0:
        ker /= s
    return ker.astype(np.float32)


def boxes_to_density_map_dynamic_sigma(
        boxes_xyxy: List[Tuple[float, float, float, float]],
        orig_w: int,
        orig_h: int,
        out_size: int,
        truncate: float,
) -> np.ndarray:
    density = np.zeros((out_size, out_size), dtype=np.float32)
    if len(boxes_xyxy) == 0:
        return density

    cfg = CONFIG["sigma_area_ranges"]
    scale_x = out_size / float(orig_w)
    scale_y = out_size / float(orig_h)

    for (x1, y1, x2, y2) in boxes_xyxy:
        x1_r = max(0.0, min(out_size - 1.0, x1 * scale_x))
        x2_r = max(0.0, min(out_size - 1.0, x2 * scale_x))
        y1_r = max(0.0, min(out_size - 1.0, y1 * scale_y))
        y2_r = max(0.0, min(out_size - 1.0, y2 * scale_y))
        cx = 0.5 * (x1_r + x2_r)
        cy = 0.5 * (y1_r + y2_r)

        area = max((x2_r - x1_r) * (y2_r - y1_r), 1.0)
        sigma = map_area_to_sigma(area, cfg)
        ker = gaussian_kernel2d(sigma, truncate=truncate)
        kr, kc = ker.shape
        rad_r = kr // 2
        rad_c = kc // 2

        cx_i = int(round(cx))
        cy_i = int(round(cy))

        r1 = max(0, cy_i - rad_r)
        c1 = max(0, cx_i - rad_c)
        r2 = min(out_size, cy_i + rad_r + 1)
        c2 = min(out_size, cx_i + rad_c + 1)

        k_r1 = r1 - (cy_i - rad_r)
        k_c1 = c1 - (cx_i - rad_c)
        k_r2 = k_r1 + (r2 - r1)
        k_c2 = k_c1 + (c2 - c1)

        density[r1:r2, c1:c2] += ker[k_r1:k_r2, k_c1:k_c2]

    return density


def local_counts_from_boxes(
        boxes_xyxy: List[Tuple[float, float, float, float]],
        orig_w: int,
        orig_h: int,
        img_size: int,
        n: int,
) -> np.ndarray:
    counts = np.zeros((n, n), dtype=np.float32)
    if len(boxes_xyxy) == 0:
        return counts.reshape(-1)

    scale_x = img_size / float(orig_w)
    scale_y = img_size / float(orig_h)
    cell_w = img_size / float(n)
    cell_h = img_size / float(n)

    for (x1, y1, x2, y2) in boxes_xyxy:
        x1_r = max(0.0, min(img_size - 1.0, x1 * scale_x))
        x2_r = max(0.0, min(img_size - 1.0, x2 * scale_x))
        y1_r = max(0.0, min(img_size - 1.0, y1 * scale_y))
        y2_r = max(0.0, min(img_size - 1.0, y2 * scale_y))
        cx = 0.5 * (x1_r + x2_r)
        cy = 0.5 * (y1_r + y2_r)

        j = int(cx // cell_w)
        i = int(cy // cell_h)
        j = max(0, min(n - 1, j))
        i = max(0, min(n - 1, i))
        counts[i, j] += 1.0

    return counts.reshape(-1)


# =========================
# Dataset
# =========================
class RSOCDualDataset(Dataset):
    def __init__(self, root: str, split: str = "train"):
        assert split in ["train", "test"]
        self.root = root
        self.split = split

        dataset_name = CONFIG["dataset"]
        assert dataset_name == "RSOC-Building", "This script is only for RSOC-Building"

        img_dir = os.path.join(root, "ASPDNet_dataset", "RSOC_building", "building", f"{split}_data", "images")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"image dir not found: {img_dir}")

        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

        label_dir = CONFIG["label_dir"]
        if split == "train":
            label_txt = os.path.join(label_dir, CONFIG["train_label"])
        else:
            label_txt = os.path.join(label_dir, CONFIG["test_label"])

        self.box_map = load_boxes(label_txt)

        self.img_size = int(CONFIG["img_size"])
        self.n = int(CONFIG["grid_n"])
        self.truncate = float(CONFIG["truncate"])

        self.mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        img_name = os.path.basename(img_path)

        img_pil = Image.open(img_path).convert("RGB")
        w0, h0 = img_pil.size

        img_resized = img_pil.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        arr = np.asarray(img_resized, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        arr = np.transpose(arr, (2, 0, 1))
        image = torch.from_numpy(arr).float()

        boxes = self.box_map.get(img_name, [])

        local = local_counts_from_boxes(boxes, w0, h0, self.img_size, self.n)
        global_cnt = float(local.sum())
        target_local = torch.from_numpy(local).float()
        target_global = torch.tensor(global_cnt, dtype=torch.float32)

        dm = boxes_to_density_map_dynamic_sigma(boxes, w0, h0, self.img_size, self.truncate)
        dm = torch.from_numpy(dm).unsqueeze(0).float()

        return image, target_local, target_global, dm, img_name


# =========================
# Ranking Loss (consistent with your main program, slightly more numerically robust)
# =========================
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

    def forward(self, pred_in: torch.Tensor, labels_in: torch.Tensor, c4: bool = False, cn: int = 16) -> torch.Tensor:
        eps = self.eps
        if c4:
            loss_sum = 0.0
            valid = 0
            for idx in range(len(pred_in)):
                pred = pred_in[idx: idx + 1].view(cn, 1)
                labels = labels_in[idx: idx + 1].view(cn, 1)
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


# =========================
# Density model (CSRNet-lite)
# =========================
class CSRNetLite(nn.Module):
    def __init__(self, imagenet_pretrained: bool = True):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=5, dilation=5),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Conv2d(512, 1, 1)

        for m in list(self.backend.modules()) + list(self.out.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        f = self.frontend(x)
        y = self.backend(f)
        y = self.out(y)
        h, w = y.shape[-2], y.shape[-1]

        if (h, w) != (H, W):
            y_up = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
            scale = float(H * W) / float(max(h * w, 1))
            y_up = y_up * scale
            return y_up
        return y


# =========================
# HSRank ALT-GVT wrapper
# =========================
class HSRankALTGVT(nn.Module):
    def __init__(self, pretrained_backbone: bool = False, ckpt_path: str | None = None):
        super().__init__()
        self.backbone = alt_gvt_small(pretrained=pretrained_backbone)

        if ckpt_path is None:
            ckpt_path = os.path.join(os.path.dirname(__file__), "alt_gvt_small.pth")

        if ckpt_path and os.path.isfile(ckpt_path):
            try:
                state = torch.load(ckpt_path, map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                new_state = {}
                for k, v in state.items():
                    if k.startswith("module."):
                        new_state[k[7:]] = v
                    else:
                        new_state[k] = v
                missing, unexpected = self.backbone.load_state_dict(new_state, strict=False)
                print(f"[HSRankALTGVT] Loaded pretrained weights from {ckpt_path}")
                if missing:
                    print(f"[HSRankALTGVT] Missing keys: {len(missing)}")
                if unexpected:
                    print(f"[HSRankALTGVT] Unexpected keys: {len(unexpected)}")
            except Exception as e:
                print(f"[HSRankALTGVT] Failed to load {ckpt_path}: {e}")
        else:
            print(f"[HSRankALTGVT] Pretrained checkpoint not found: {ckpt_path}")

    def forward(self, x: torch.Tensor):
        rank_local, rank_global = self.backbone(x)
        return rank_local, rank_global


# =========================
# Dynamic loss weighting (Uncertainty Weighting)
# -------------------------
# Let the model learn the weights of the two branch losses instead of manual a_1/a_2.
# Common practice: introduce two learnable log_vars (corresponding to task noise/uncertainty), total loss:
#   L = exp(-s1)*L1 + exp(-s2)*L2 + (s1 + s2)
# where s1=log_var1, s2=log_var2. Update network parameters and s1/s2 simultaneously during training.
# =========================
class UncertaintyWeighting(nn.Module):
    def __init__(self, init_log_vars=(0.0, 0.0)):
        super().__init__()
        init = torch.tensor(list(init_log_vars), dtype=torch.float32)
        if init.numel() != 2:
            raise ValueError("init_log_vars must have length 2")
        self.log_vars = nn.Parameter(init)

    def forward(self, loss_rank: torch.Tensor, loss_density: torch.Tensor):
        s1 = self.log_vars[0]
        s2 = self.log_vars[1]
        w1 = torch.exp(-s1)
        w2 = torch.exp(-s2)
        loss = w1 * loss_rank + w2 * loss_density + (s1 + s2)
        return loss

    @torch.no_grad()
    def current_weights(self):
        s1 = float(self.log_vars[0].item())
        s2 = float(self.log_vars[1].item())
        w1 = math.exp(-s1)
        w2 = math.exp(-s2)
        return w1, w2, s1, s2

# =========================
# Dual-branch model
# =========================
class DualBranchModel(nn.Module):
    def __init__(self, pretrained_altgvt: bool = False, use_dynamic_weighting: bool = True):
        super().__init__()
        self.altgvt = HSRankALTGVT(pretrained_backbone=pretrained_altgvt)
        self.density = CSRNetLite(imagenet_pretrained=True)

        # Learnable dynamic weights (uncertainty weighting)
        self.loss_balancer = UncertaintyWeighting(init_log_vars=CONFIG.get("dyn_init_log_vars", (0.0, 0.0))) if use_dynamic_weighting else None

    def forward(self, x: torch.Tensor):
        rank_local, rank_global = self.altgvt(x)
        density = self.density(x)
        return rank_local, rank_global, density


# =========================
# Train / Eval
# =========================
def train_one_epoch(epoch: int, model: DualBranchModel, loader: DataLoader,
                    optimizer: optim.Optimizer, rank_loss_fn: RankingLoss,
                    mse_fn: nn.Module, device: str):
    model.train()
    pbar = tqdm(loader, desc=f"train {epoch}", ncols=110)

    loss_avg = 0.0
    altgvt_avg = 0.0
    density_avg = 0.0
    for image, target_local, target_global, gt_density, _ in pbar:
        image = image.to(device)
        target_local = target_local.to(device)
        target_global = target_global.to(device)
        gt_density = gt_density.to(device)

        optimizer.zero_grad(set_to_none=True)

        rank_local, rank_global, pred_density = model(image)

        lg = rank_loss_fn(rank_global.view(-1), target_global.view(-1))
        ll = rank_loss_fn(rank_local, target_local, c4=True, cn=16)
        l_altgvt = lg + 0.5 * ll

        l_density = mse_fn(pred_density, gt_density)
        # -------- total loss --------
        # L_rank = (Lg + 0.5*Ll)
        # L_total uses dynamic weights: L = exp(-s1)*L_rank + exp(-s2)*L_density + (s1+s2)
        if getattr(model, "loss_balancer", None) is not None:
            loss = model.loss_balancer(l_altgvt, l_density)
        else:
            a_1 = float(CONFIG.get("a_1", 1.0))
            a_2 = float(CONFIG.get("a_2", 1.0))
            loss = a_1 * l_altgvt + a_2 * l_density
        loss.backward()
        optimizer.step()

        loss_avg = 0.9 * loss_avg + 0.1 * float(loss.item()) if loss_avg > 0 else float(loss.item())
        altgvt_avg = 0.9 * altgvt_avg + 0.1 * float(l_altgvt.item()) if altgvt_avg > 0 else float(l_altgvt.item())
        density_avg = 0.9 * density_avg + 0.1 * float(l_density.item()) if density_avg > 0 else float(l_density.item())
        pbar.set_postfix({
            "L": f"{loss_avg:.4f}",
            "L_rank": f"{altgvt_avg:.4f}",
            "L_dens": f"{density_avg:.4f}",
            "Lg": f"{float(lg.item()):.3f}",
            "Ll": f"{float(ll.item()):.3f}",
        })

    return loss_avg, altgvt_avg, density_avg


import math


@torch.no_grad()
def eval_density_count_mae_rmse(model, loader, device):
    model.eval()
    density_scale = float(CONFIG.get("density_scale", 1.0))
    if density_scale <= 0:
        density_scale = 1.0

    total_abs = 0.0
    total_sq = 0.0
    total_n = 0

    for image, _, target_global, _, _ in tqdm(loader, desc="eval_density(test)", ncols=110):
        image = image.to(device)
        target_global = target_global.to(device).view(-1)

        _, _, pred_density = model(image)
        pred_cnt = pred_density.sum(dim=(1, 2, 3)) / density_scale
        diff = (pred_cnt - target_global)

        total_abs += diff.abs().sum().item()
        total_sq += (diff ** 2).sum().item()
        total_n += diff.numel()

    mae = total_abs / max(total_n, 1)
    rmse = math.sqrt(total_sq / max(total_n, 1))
    return mae, rmse


@torch.no_grad()
def eval_rank_branch_metrics(model, loader, device, rank_loss_fn):
    """
    Use the "ranking branch" to evaluate two types of metrics on the test set:
      - rloss: average (Lg + 0.5*Ll), used to observe the convergence of the ranking loss itself (lower is better)
      - mae/rmse: reuse the "pure ranking program" approach: perform a least-squares linear calibration on rank_global
        count_hat = k * rank_global + b, then compute MAE / RMSE on the test set (lower is better)
    """
    model.eval()

    # 1) ranking loss (Lg + 0.5*Ll)
    loss_sum = 0.0
    loss_cnt = 0

    # 2) Cache needed for rank_global -> count linear regression calibration
    preds_rank = []
    gts = []

    for image, target_local, target_global, _, _ in tqdm(loader, desc="eval_rank(test)", ncols=110):
        image = image.to(device)
        target_local = target_local.to(device)
        target_global = target_global.to(device).view(-1)

        rank_local, rank_global, _ = model(image)
        rank_global = rank_global.view(-1)

        # --- ranking loss (same as training) ---
        lg = rank_loss_fn(rank_global, target_global)
        ll = rank_loss_fn(rank_local, target_local, c4=True, cn=16)
        l = lg + 0.5 * ll

        loss_sum += float(l.item())
        loss_cnt += 1

        # --- Collect rank_global & gt_count for MAE/RMSE ---
        preds_rank.extend(rank_global.detach().cpu().tolist())
        gts.extend(target_global.detach().cpu().tolist())

    rloss = loss_sum / max(loss_cnt, 1)

    n = len(gts)
    if n == 0:
        return rloss, float("inf"), float("inf")

    # rank_global -> count least-squares linear fit (closed-form, replicating the pure ranking script)
    x_mean = sum(preds_rank) / n
    top = sum(gt * (x - x_mean) for x, gt in zip(preds_rank, gts))
    bot = sum((x - x_mean) ** 2 for x in preds_rank)
    k = top / bot if bot != 0 else 1.0
    b = sum(gt - k * x for x, gt in zip(preds_rank, gts)) / n

    mae = 0.0
    rmse = 0.0
    for x, gt in zip(preds_rank, gts):
        p = k * x + b
        mae += abs(p - gt)
        rmse += (p - gt) ** 2
    mae /= n
    rmse = math.sqrt(rmse / n)

    return rloss, mae, rmse


def build_optimizer(model: DualBranchModel) -> optim.Optimizer:
    base_lr = float(CONFIG["lr"])
    fast_lr = float(CONFIG["fast_lr"]) if CONFIG["fast_lr"] is not None else base_lr * 1000.0
    wd = float(CONFIG["weight_decay"])

    fast_params = []
    if hasattr(model.altgvt.backbone, "regression"):
        try:
            fast_params += list(model.altgvt.backbone.regression.output[-2].parameters())
            fast_params += list(model.altgvt.backbone.regression.c16[-2].parameters())
        except Exception:
            fast_params += list(model.altgvt.backbone.regression.output.parameters())
            fast_params += list(model.altgvt.backbone.regression.c16.parameters())

    fast_ids = {id(p) for p in fast_params}
    base_params = [p for p in model.parameters() if p.requires_grad and id(p) not in fast_ids]

    opt = optim.Adam(
        [
            {"params": base_params, "lr": base_lr, "weight_decay": wd},
            {"params": fast_params, "lr": fast_lr, "weight_decay": wd},
        ],
        betas=(0.9, 0.999),
    )
    return opt


def main():
    set_seed(int(CONFIG["seed"]))
    device = CONFIG["device"]
    root = CONFIG["dataset_root"]

    _altgvt_mod.device = torch.device(device)

    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    train_ds = RSOCDualDataset(root, "train")
    test_ds = RSOCDualDataset(root, "test")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(CONFIG["batch_size"]),
        shuffle=True,
        num_workers=int(CONFIG["num_workers"]),
        drop_last=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=8,
        shuffle=False,
        num_workers=max(1, int(CONFIG["num_workers"]) // 2),
        pin_memory=True,
    )

    # HSRankALTGVT will automatically try to load alt_gvt_small.pth internally
    model = DualBranchModel(pretrained_altgvt=False, use_dynamic_weighting=bool(CONFIG.get("use_dynamic_weighting", True))).to(device)
    print("[Device]", device)
    print("[Model]", model.__class__.__name__)

    if CONFIG.get("resume"):
        sd = torch.load(CONFIG["resume"], map_location=device)
        model.load_state_dict(sd, strict=False)
        print("[Resume]", CONFIG["resume"])

    rank_loss_fn = RankingLoss()
    mse_fn = nn.MSELoss()
    optimizer = build_optimizer(model)

    best_density_mae = float("inf")

    for epoch in range(1, int(CONFIG["epochs"]) + 1):
        train_loss, train_rank_loss, train_density_loss = train_one_epoch(epoch, model, train_loader, optimizer,
                                                                          rank_loss_fn, mse_fn, device)
        print(
            f"[Epoch {epoch}] Train L_rank={train_rank_loss:.6f} | Train L_density={train_density_loss:.6f} | Train L_total={train_loss:.6f}")

        density_mae, density_rmse = eval_density_count_mae_rmse(model, test_loader, device)
        rank_rloss, rank_mae, rank_rmse = eval_rank_branch_metrics(model, test_loader, device, rank_loss_fn)

        # Print current weights (each epoch)
        if getattr(model, "loss_balancer", None) is not None:
            w1, w2, s1, s2 = model.loss_balancer.current_weights()
            weight_str = f"w_rank={w1:.4f} w_density={w2:.4f} | log_var_rank={s1:.4f} log_var_density={s2:.4f}"
        else:
            a_1 = float(CONFIG.get("a_1", 1.0))
            a_2 = float(CONFIG.get("a_2", 1.0))
            weight_str = f"a_1(rank)={a_1:.3f} a_2(density)={a_2:.3f}"

        print(
            f"[Test] Density MAE={density_mae:.3f} RMSE={density_rmse:.3f} | "
            f"Rank RLoss={rank_rloss:.3f} MAE={rank_mae:.3f} RMSE={rank_rmse:.3f} | "
            + weight_str
        )

        if density_mae < best_density_mae:
            best_density_mae = density_mae
            save_name = (
                f"best_dual_branch_"
                f"DMAE{density_mae:.2f}_DRMSE{density_rmse:.2f}_"
                f"RLOSS{rank_rloss:.3f}_RMAE{rank_mae:.3f}_RRMSE{rank_rmse:.3f}.pth"
            )
            torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], save_name))

    density_mae, density_rmse = eval_density_count_mae_rmse(model, test_loader, device)
    rank_rloss, rank_mae, rank_rmse = eval_rank_branch_metrics(model, test_loader, device, rank_loss_fn)

    # Final weights output
    if getattr(model, "loss_balancer", None) is not None:
        w1, w2, s1, s2 = model.loss_balancer.current_weights()
        weight_str = f"w_rank={w1:.4f} w_density={w2:.4f} | log_var_rank={s1:.4f} log_var_density={s2:.4f}"
    else:
        a_1 = float(CONFIG.get("a_1", 1.0))
        a_2 = float(CONFIG.get("a_2", 1.0))
        weight_str = f"a1(rank)={a_1:.3f} a2(density)={a_2:.3f}"

    print(
        f"[Test] Density MAE={density_mae:.3f} RMSE={density_rmse:.3f} | "
        f"Rank RLoss={rank_rloss:.3f} MAE={rank_mae:.3f} RMSE={rank_rmse:.3f} | "
        + weight_str
    )

    if density_mae < best_density_mae:
        best_density_mae = density_mae
        save_name = (
            f"last_dual_branch_"
            f"DMAE{density_mae:.2f}_DRMSE{density_rmse:.2f}_"
            f"RLOSS{rank_rloss:.3f}_RMAE{rank_mae:.3f}_RRMSE{rank_rmse:.3f}.pth"
        )
        torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], save_name))


if __name__ == "__main__":
    main()
