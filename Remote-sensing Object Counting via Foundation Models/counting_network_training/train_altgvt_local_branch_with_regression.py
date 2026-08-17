from __future__ import annotations

import os
import glob
import math
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


# =========================
# only modify here configurationi.e.can
# =========================
CONFIG = {
    # 1) traininggood modelweight(youinherechangepathi.e.can)
    #    For example: r"my_model\mae10.23_rmse13.88_epoch80_RSOC-Building.pth"
    "model_path": r"my_model\mae6.47_rmse10.22_epoch167_RSOC-Building.pth",

    # 2) datasetroot directory(andyoutraining scriptkeepconsistent)
    "dataset_root": "dataset",

    #   "RSOC-Ship" | "RSOC-S-Vehicle" | "RSOC-L-Vehicle" | "RSOC-Building"
    #   "VD-People" | "VD-Vehicle"
    "dataset": "RSOC-Building",

    "img_size": 512,
    "num_workers": 4,

    # 3) usewhich split comefitting a,b; usewhich split comeevaluation MAE/RMSE
    #    Note:RSOC-Building usuallyonlyhas train/test; youcan fit=train, eval=test
    "fit_split": "train",
    "eval_split": "test",

    # 4) compute"globalinput" = sum(c16) after, isor notagaindo once Sigmoid takeitcompressto 0~1(0.x)
    #    - "auto": according to backbone original global outputisor notin [0,1] automaticdecide(recommend)
    #    - True / False: forceopen/close
    "apply_sigmoid_to_sum": "auto",

    # 5) predictvalueisor notcropfor >=0(countnotshouldfornegative)
    "clip_pred_min0": True,

    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "pretrained_backbone": False,  # onlydoinference/evaluation, onegeneral False
}


# =========================
# letthisscriptcandirectin PyCharm insiderun: guaranteecan import samedirectory  ALTGVT.py
# =========================
import sys
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from ALTGVT import alt_gvt_small
import ALTGVT as _altgvt_mod


# =========================
# model: readtraininggood ranking model, output c16, andconstruct s = sum(c16) aslinearityfittinginput
# =========================
class HSRankALTGVT_SumC16(nn.Module):
    """
    Output:
      - c16: [B,16]
      - s:   [B,1]  = sum(c16) (optional Sigmoid)
    """
    def __init__(self, pretrained_backbone: bool = False, apply_sigmoid_to_sum: str | bool = "auto"):
        super().__init__()
        self.backbone = alt_gvt_small(pretrained=pretrained_backbone)
        self.apply_sigmoid_to_sum = apply_sigmoid_to_sum

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c16, global_raw = self.backbone(x)  # c16:[B,16] global_raw:[B,1] (originalmodel  global output)

        s = c16.sum(dim=1, keepdim=True)  # [B,1]

        # isor notto s do Sigmoid: alignyoutraining scriptinside"global output 0.x" habit
        mode = self.apply_sigmoid_to_sum
        if mode == "auto":
            with torch.no_grad():
                gr = global_raw.detach()
                mn = float(gr.min().item())
                mx = float(gr.max().item())
                looks_sigmoid = (mn >= -1e-3) and (mx <= 1.0 + 1e-3)
            if looks_sigmoid:
                s = torch.sigmoid(s)
        elif isinstance(mode, bool):
            if mode:
                s = torch.sigmoid(s)

        return c16, s


# =========================
# dataset: reuseyoutraining script "real label"readlogic
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
        out = []
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
    # need scipy: pip install scipy
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
        self.n = 4
        self.img_paths = _list_images(root, dataset_label, split)

        # PIL + numpy + torch: Resize + ToTensor + Normalize(aligntraining script)
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
# linearityfitting(closed-form solution)andevaluation
# =========================
def fit_linear_closed_form(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    """
    closed-form solution(least squares)fitting y = a*x + b
    """
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.size == 0:
        return 0.0, 0.0

    xm = float(x.mean())
    ym = float(y.mean())
    var = float(((x - xm) ** 2).sum())
    if var < 1e-12:
        a = 0.0
        b = ym
    else:
        cov = float(((x - xm) * (y - ym)).sum())
        a = cov / var
        b = ym - a * xm
    return float(a), float(b)


def mae_rmse(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    mae = float(np.mean(np.abs(pred - gt)))
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    return mae, rmse


@torch.no_grad()
def collect_sumc16_and_gt(model: nn.Module, loader: DataLoader, device: str) -> Tuple[List[float], List[float]]:
    model.eval()
    xs: List[float] = []
    ys: List[float] = []

    for img, gt_global, _, _ in tqdm(loader, desc="collect", ncols=110):
        img = img.to(device, non_blocking=True)
        gt_global = gt_global.to(device, non_blocking=True)

        _, s = model(img)  # s: [B,1]
        xs.append(float(s.squeeze().item()))
        ys.append(float(gt_global.squeeze().item()))

    return xs, ys


def main():
    device = CONFIG["device"]
    _altgvt_mod.device = torch.device(device)

    # 1) buildmodelandloadweight
    model = HSRankALTGVT_SumC16(
        pretrained_backbone=bool(CONFIG["pretrained_backbone"]),
        apply_sigmoid_to_sum=CONFIG["apply_sigmoid_to_sum"],
    ).to(device)

    model_path = CONFIG["model_path"]
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"model_path not found: {model_path}")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    print(f"[OK] Loaded model: {model_path}")

    # 2) dataset
    root = CONFIG["dataset_root"]
    dataset_label = CONFIG["dataset"]
    img_size = int(CONFIG["img_size"])

    fit_split = str(CONFIG["fit_split"])
    eval_split = str(CONFIG["eval_split"])

    fit_ds = RealRankDataset(root, dataset_label, fit_split, img_size=img_size)
    eval_ds = RealRankDataset(root, dataset_label, eval_split, img_size=img_size)

    if len(fit_ds) == 0:
        raise RuntimeError(f"No images found for fit_split='{fit_split}' under {root} ({dataset_label})")
    if len(eval_ds) == 0:
        raise RuntimeError(f"No images found for eval_split='{eval_split}' under {root} ({dataset_label})")

    fit_loader = DataLoader(
        fit_ds, batch_size=1, shuffle=False,
        num_workers=int(CONFIG["num_workers"]), pin_memory=True
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=1, shuffle=False,
        num_workers=int(CONFIG["num_workers"]), pin_memory=True
    )

    # 3) collect x = sum(c16) and y = count(real label)
    print(f"[INFO] Collecting for fitting on split='{fit_split}' ...")
    x_fit, y_fit = collect_sumc16_and_gt(model, fit_loader, device)

    # 4) closed-form solutionfitting y = a*x + b
    a, b = fit_linear_closed_form(x_fit, y_fit)
    print(f"[FIT] y = a*x + b  |  a={a:.6f}, b={b:.6f}, n={len(x_fit)}")

    # 5) in eval_split onevaluation MAE / RMSE
    print(f"[INFO] Evaluating on split='{eval_split}' ...")
    x_eval, y_eval = collect_sumc16_and_gt(model, eval_loader, device)
    x_eval = np.asarray(x_eval, dtype=np.float64)
    y_eval = np.asarray(y_eval, dtype=np.float64)

    pred = a * x_eval + b
    if bool(CONFIG.get("clip_pred_min0", True)):
        pred = np.maximum(pred, 0.0)

    mae, rmse = mae_rmse(pred, y_eval)
    print(f"[RESULT] MAE={mae:.4f} | RMSE={rmse:.4f}  (split={eval_split})")

    # optional: meanwhiletoone"takewholeafter"  MAE/RMSE(ifyoumorecloseheartintegercount)
    pred_round = np.rint(pred)
    mae_r, rmse_r = mae_rmse(pred_round, y_eval)


if __name__ == "__main__":
    main()
