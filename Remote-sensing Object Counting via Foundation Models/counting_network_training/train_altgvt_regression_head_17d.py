"""rank_then_regress_freeze_rank.py

in"onlytrainingrankingpart" scriptbasison, newly addedone"regression head"performsecondstagetraining: 
- readyoualreadythroughtraininggood "ranking model"weight(directory/fileallcanmatch). 
- buildnewmodel: beforesurfaceisoldranking model(freezeparameter), aftersurfacenewly addedregression module. 
- regression moduleInput:spellconnect [rank_local(16dimension), rank_global(1dimension)] => 17dimensionvector. 
- trainingwhenonlytrainingregression module; ranking modelonlyforwardprovidefeature, notupdate. 
- save  .pth: include"ranking model + regression module" completestructureparameter. 

Usage:
  1) ensuresamedirectorybelowhas ALTGVT.py, withandyou datasetdirectory dataset/...
  2) modify CONFIG in  dataset / dataset_root / rank_ckpt_path or rank_ckpt_dir
  3) directrun: python rank_then_regress_freeze_rank.py

Note:
  - thisscriptreuseyoupreviousscript  RealRankDataset(readreal label, output global_count and 16dimension local_counts). 
  - validation/savelogic: with"regressionpredict count   MAE minimum"forbestmodel. 
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

# ===== Paper backbone (ALTGVT / Twins-SVT) =====
import sys
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from ALTGVT import alt_gvt_small
import ALTGVT as _altgvt_mod


# =========================
# only modify here
# =========================
CONFIG = {
    "dataset_root": "dataset",
    "dataset": "RSOC-Building",   # "RSOC-Ship" | "RSOC-S-Vehicle" | "RSOC-L-Vehicle" | "RSOC-Building" | "VD-People" | "VD-Vehicle"

    "img_size": 512,
    "clipnum": 16,
    "batch_size": 8,
    "epochs": 70,
    "val_start": 1,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "num_workers": 4,
    "seed": 85,

    "device": "cuda:0" if torch.cuda.is_available() else "cpu",

    "rank_ckpt_path": "my_model/mae5.96_rmse9.35_epoch184_RSOC-Building.pth",                 # for example "my_model/mae12.34_rmse20.11_epoch50_RSOC-Building.pth"
    "rank_ckpt_dir": "my_model",
    "rank_ckpt_glob": "*.pth",

    # ====== save ======
    "save_dir": "my_model",
    "min_mae": float("inf"),

    # ====== regression headstructure ======
    # input 17dimension(16 local + 1 global), output 1dimension count
    "reg_hidden": [128, 64, 16],          # MLP hidden layer, can be changed/can be kept
    "reg_dropout": 0.0,                   # can be changed 0.1~0.3
    "use_smooth_l1": False,               # True=SmoothL1Loss, False=L1Loss
}


# =========================
# utility functions
# =========================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_rank_checkpoint(path: str, ckpt_dir: str, pattern: str) -> str:
    """support: directtofile, ortodirectoryautomaticselectlatest .pth"""
    if path and os.path.isfile(path):
        return path
    if ckpt_dir and os.path.isdir(ckpt_dir):
        cands = sorted(glob.glob(os.path.join(ckpt_dir, pattern)))
        if not cands:
            raise FileNotFoundError(f"No checkpoint found in dir={ckpt_dir}, pattern={pattern}")
        # selectmodifywhenintervallatest 
        cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return cands[0]
    raise FileNotFoundError("Please set CONFIG['rank_ckpt_path'] to a valid file or CONFIG['rank_ckpt_dir'] to a valid dir.")


# =========================
# onlyranking model(fromyoupreviousscript)
# =========================
class HSRankALTGVT(nn.Module):
    """usepaperin  ALTGVT-Small backbone(output rank_local / rank_global). """
    def __init__(self, clipnum: int = 16, pretrained_backbone: bool = False):
        super().__init__()
        self.backbone = alt_gvt_small(pretrained=pretrained_backbone)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rank_local, rank_global = self.backbone(x)  # [B,16], [B,1]
        eps = 1e-8
        rank_local = torch.log(torch.clamp(rank_local, min=eps))
        rank_global = torch.log(torch.clamp(rank_global, min=eps))
        return rank_local, rank_global


# =========================
# newly added: regression head(onlytrainthis)
# =========================
class RankToCountRegressor(nn.Module):
    """input 17dimension(16 local + 1 global), output 1dimension count"""
    def __init__(self, in_dim: int = 17, hidden: Optional[List[int]] = None, dropout: float = 0.0):
        super().__init__()
        hidden = hidden or [128, 64, 16]
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, feat17: torch.Tensor) -> torch.Tensor:
        return self.mlp(feat17)


class RankFrozenPlusReg(nn.Module):





    
    def __init__(self, rank_model: nn.Module, regressor: nn.Module):
        super().__init__()
        self.rank_model = rank_model
        self.regressor = regressor

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rank_local, rank_global = self.rank_model(x)  # [B,16], [B,1]
        # spell 17dimension: local(16) + global(1)
        feat17 = torch.cat([rank_local, rank_global], dim=1)  # [B,17]
        pred_count = self.regressor(feat17)  # [B,1]
        return rank_local, rank_global, pred_count


# =========================
# dataset: reuseyoupreviousscript real labelread
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

    label_dir = os.path.join(
        root, "ASPDNet_dataset", split, "labelTxt-v1.0", f"{split}set_reclabelTxt"
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
        sub = "train_data" if split == "train" else "test_data"
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
            p for p in glob.glob(os.path.join(img_dir, "*.jpg"))
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
        out: List[str] = []
        for e in exts:
            out.extend(glob.glob(os.path.join(img_dir, e)))
        return sorted(out)

    return []


def _grid_index(x: float, y: float, w: float, h: float, n: int = 4) -> int:
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
    try:
        pts = center[0, 0]
        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            return np.zeros((16,), dtype=np.float32)
    except Exception:
        return np.zeros((16,), dtype=np.float32)

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
        self.img_paths = _list_images(root, dataset_label, split)

        # pure PIL + numpy + torch: Resize + ToTensor + Normalize
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

        if self.dataset_label in ("RSOC-Ship", "RSOC-S-Vehicle", "RSOC-L-Vehicle"):
            class_token = {
                "RSOC-Ship": "ship",
                "RSOC-S-Vehicle": "small-vehicle",
                "RSOC-L-Vehicle": "large-vehicle",
            }[self.dataset_label]
            label_dir = os.path.join(
                self.root, "ASPDNet_dataset", self.split, "labelTxt-v1.0", f"{self.split}set_reclabelTxt"
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


# =========================
# Train / Val: onlytrainingregression
# =========================
def freeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False


def unfreeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = True


def train_epoch(epoch: int, model: RankFrozenPlusReg, optimizer: optim.Optimizer, loader: DataLoader, loss_fn: nn.Module, device: str):
    model.train()
    loop = tqdm(loader, desc=f"train_reg {epoch}", ncols=110)
    total = 0.0
    num = 0
    for image, target_global, _, _ in loop:
        image = image.to(device)
        target_global = target_global.to(device).view(-1, 1)

        # onlytrain regressor; rank_model alreadyfreeze
        _, _, pred = model(image)
        loss = loss_fn(pred, target_global)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total += float(loss.item())
        num += 1
        loop.set_postfix(loss=total / max(num, 1))


@torch.no_grad()
def val_epoch(model: RankFrozenPlusReg, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    preds = []
    gts = []
    for image, target_global, _, _ in tqdm(loader, desc="val_reg", ncols=110):
        image = image.to(device)
        target_global = target_global.to(device).view(-1, 1)
        _, _, pred = model(image)
        preds.append(float(pred.item()))
        gts.append(float(target_global.item()))

    n = len(gts)
    if n == 0:
        return float("inf"), float("inf")

    mae = 0.0
    rmse = 0.0
    for p, gt in zip(preds, gts):
        mae += abs(p - gt)
        rmse += (p - gt) ** 2
    mae /= n
    rmse = math.sqrt(rmse / n)
    print(f"Val (REG) MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return mae, rmse


def main():
    set_seed(int(CONFIG["seed"]))
    device = CONFIG["device"]
    _altgvt_mod.device = torch.device(device)  # compatible ALTGVT internaltoglobal device  use

    # 1) readranking model ckpt
    ckpt = resolve_rank_checkpoint(CONFIG.get("rank_ckpt_path", ""), CONFIG.get("rank_ckpt_dir", ""), CONFIG.get("rank_ckpt_glob", "*.pth"))
    print(f"[LOAD] rank checkpoint: {ckpt}")

    rank_model = HSRankALTGVT(clipnum=int(CONFIG["clipnum"]), pretrained_backbone=False).to(device)
    state = torch.load(ckpt, map_location=device)
    rank_model.load_state_dict(state, strict=True)

    # 2) freezeranking modelparameter
    freeze_module(rank_model)
    rank_model.eval()  # freezeaftercankeep eval(training regressor whenalsonotwillaffectgradient)

    # 3) newbuildregression module(onlytrainthis)
    reg = RankToCountRegressor(in_dim=17, hidden=list(CONFIG.get("reg_hidden", [128, 64, 16])), dropout=float(CONFIG.get("reg_dropout", 0.0))).to(device)
    unfreeze_module(reg)

    # 4) spellbecomecompletemodel(savewhenthenisthisstructure)
    model = RankFrozenPlusReg(rank_model=rank_model, regressor=reg).to(device)

    # loss & optimizer(onlyto regressor parameter)
    if bool(CONFIG.get("use_smooth_l1", False)):
        loss_fn = nn.SmoothL1Loss()
    else:
        loss_fn = nn.L1Loss()

    optimizer = optim.Adam(
        model.regressor.parameters(),
        lr=float(CONFIG["lr"]),
        weight_decay=float(CONFIG["weight_decay"]),
        betas=(0.9, 0.999),
    )

    # 5) data
    root = CONFIG["dataset_root"]
    dataset_label = CONFIG["dataset"]
    train_ds = RealRankDataset(root, dataset_label, "train", img_size=int(CONFIG["img_size"]))
    val_ds = RealRankDataset(root, dataset_label, "val", img_size=int(CONFIG["img_size"]))

    if len(val_ds) == 0:
        print("[WARN] current dataset has no val split, automatically use train as val. ")
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
                out_path = os.path.join(
                    CONFIG["save_dir"], f"reg_mae{mae:.2f}_rmse{rmse:.2f}_epoch{epoch}_{dataset_label}.pth"
                )
                torch.save(model.state_dict(), out_path)
                print(f"[SAVE] {out_path}")

    print("Stage-2 regression training done!")


if __name__ == "__main__":
    main()
