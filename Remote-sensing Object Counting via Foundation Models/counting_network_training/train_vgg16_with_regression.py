# -*- coding: utf-8 -*-
"""
linearityfittingcount-real labelevaluation_vgg16ranking.py

Objective:
1) readyou"mainprogram-vgg16-pseudo label.py"trainingout ranking model(HSRankVGG16   .pth weight). 
2) usereal label(and"mainprogram-addregression.py"  RealRankDataset readwayconsistent)takeouteach image  global_count. 
3) to"ranking output rank_global"dooneelementlinearityleast squaresfitting: count = a * rank_global + b(closed-form solution). 
4) intest setonoutputfinal MAE / RMSE. 

description(veryimportant): 
- thisnotneeduse SGD "trainingwholenetwork", but **stillthenneed** inhasreal label dataon"fitting" a,b(onlyiscloseformula solution, 2 parameter). 
- fittingusewhichsplit: defaultuse train fitting; ifyou datasethas val, alsocan be changedbecome val fittingagainmeasure test. 
"""

from __future__ import annotations

import os
import glob
import json
import math
import random
import importlib.util
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


# =========================
# only modify here
# =========================
CONFIG = {
    # datasetroot directory(andyou workprocessconsistent: dataset/ belowputeachkinddataset)
    "dataset_root": "dataset",
    # "RSOC-Ship" | "RSOC-S-Vehicle" | "RSOC-L-Vehicle" | "RSOC-Building" | "VD-People" | "VD-Vehicle"
    "dataset": "RSOC-Building",
    "img_size": 512,

    # ranking modelstructure(fromyoutraining scriptinsidedynamic import)
    "rank_script_path": "mainprogram-VGG16-pseudo labeltraining-noregression.py",   # needandthisscriptsamedirectory, orfillabsolutelytopath
    "rank_model_class": "HSRankVGG16",
    "clipnum": 16,

    # ====== read"ranking model" weight ======
    # choose one of two: prefer rank_ckpt_path; if emptythenfrom rank_ckpt_dir inautomaticselectlatest  .pth
    "rank_ckpt_path": "mae6.96_rmse9.18_epoch60_RSOC-Building.pth",                 # for example "my_model/xxx_epoch50_RSOC-Building.pth"
    "rank_ckpt_dir": "my_model",
    "rank_ckpt_glob": "*.pth",

    # fitting a,b usewhich split(default train). ifyouhas val, alsocanchange to "val"
    "fit_split": "train",
    # evaluationusewhich split(RSOC-Building   test_data propagate "test" or "val" allwillmap to test_data)
    "eval_split": "test",

    "batch_size": 8,
    "num_workers": 4,
    "seed": 85,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",

    # predictisor notcroptonon-negative(countusually >=0). suggest True
    "clamp_nonneg": True,

    # takefittingout  a,b saveoneportion, sidethenyouaftercontinuedirectuse
    "save_ab_json": "my_model/ls_ab_rankglobal_to_count.json",
}


# =========================
# utility functions
# =========================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_checkpoint(path: str, ckpt_dir: str, pattern: str) -> str:
    """support: directtofile, ortodirectoryautomaticselectlatest .pth"""
    if path and os.path.isfile(path):
        return path
    if ckpt_dir and os.path.isdir(ckpt_dir):
        cands = sorted(glob.glob(os.path.join(ckpt_dir, pattern)))
        if not cands:
            raise FileNotFoundError(f"No checkpoint found in dir={ckpt_dir}, pattern={pattern}")
        cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return cands[0]
    raise FileNotFoundError("Please set CONFIG['rank_ckpt_path'] to a valid file or CONFIG['rank_ckpt_dir'] to a valid dir.")


def import_from_file(py_path: str, module_name: str = "rank_module"):
    """from .py filedynamic import, avoidintextfilenamecause import fail. """
    py_path = os.path.abspath(py_path)
    if not os.path.isfile(py_path):
        raise FileNotFoundError(f"rank_script_path not found: {py_path}")
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import module from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# =========================
# real labelread(reuse"mainprogram-addregression.py" logic)
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
    """Returns:(image_tensor, global_count, local_counts(16), img_name)"""
    def __init__(self, root: str, dataset_label: str, split: str, img_size: int = 512):
        self.root = root
        self.dataset_label = dataset_label
        self.split = split
        self.img_size = img_size
        self.img_paths = _list_images(root, dataset_label, split)

        # PIL + numpy + torch: Resize + ToTensor + Normalize (ImageNet)
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
# linearityleast squares(closeformula solution)
# =========================
def fit_linear_ls(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """oneelementlinearityregressioncloseformula solution: y = a*x + b"""
    x = x.astype(np.float64).reshape(-1)
    y = y.astype(np.float64).reshape(-1)
    if x.size == 0:
        return 1.0, 0.0
    xm = float(x.mean())
    ym = float(y.mean())
    top = float(((x - xm) * (y - ym)).sum())
    bot = float(((x - xm) ** 2).sum())
    a = top / bot if bot != 0 else 1.0
    b = ym - a * xm
    return float(a), float(b)


def compute_mae_rmse(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    pred = pred.reshape(-1).astype(np.float64)
    gt = gt.reshape(-1).astype(np.float64)
    if pred.size == 0:
        return float("inf"), float("inf")
    mae = float(np.mean(np.abs(pred - gt)))
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    return mae, rmse


@torch.no_grad()
def collect_rank_global_and_gt(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """runonetimesdataset: take rank_global(ranking output)and gt_count"""
    model.eval()
    xs: List[float] = []
    ys: List[float] = []

    for images, gt_global, _, _ in tqdm(loader, desc="collect", ncols=110):
        images = images.to(device)
        gt_global = gt_global.to(device).view(-1, 1)

        _, rank_global = model(images)  # rank_global: [B,1]
        rank_global = rank_global.detach().float().view(-1).cpu().numpy()
        gt_global = gt_global.detach().float().view(-1).cpu().numpy()

        xs.extend([float(v) for v in rank_global])
        ys.extend([float(v) for v in gt_global])

    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def main():
    set_seed(int(CONFIG["seed"]))
    device = str(CONFIG["device"])
    root = str(CONFIG["dataset_root"])
    dataset_label = str(CONFIG["dataset"])
    img_size = int(CONFIG["img_size"])

    # 1) loadranking modelstructure(fromtraining scriptdynamic import)
    rank_script_path = str(CONFIG["rank_script_path"])
    rank_mod = import_from_file(rank_script_path, module_name="rank_module_vgg16")
    cls_name = str(CONFIG["rank_model_class"])
    if not hasattr(rank_mod, cls_name):
        raise AttributeError(f"{rank_script_path} does not define class {cls_name}")
    RankModel = getattr(rank_mod, cls_name)

    # Note:heretake pretrained_backbone=False, avoidthisscriptrunwhengobelowload VGG16 ImageNet weight. 
    # becauseyoutraininggood  checkpoint insidealreadythroughinclude backbone weight , load_state_dict willoverwriteentergo. 
    model = RankModel(
        clipnum=int(CONFIG["clipnum"]),
        pretrained_backbone=False,
        vgg16_ckpt="",
    ).to(device)

    # 2) readranking model ckpt
    ckpt = resolve_checkpoint(
        str(CONFIG.get("rank_ckpt_path", "")),
        str(CONFIG.get("rank_ckpt_dir", "")),
        str(CONFIG.get("rank_ckpt_glob", "*.pth")),
    )
    print(f"[LOAD] rank checkpoint: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=True)

    # 3) dataset(real label)
    fit_split = str(CONFIG.get("fit_split", "train"))
    eval_split = str(CONFIG.get("eval_split", "test"))

    fit_ds = RealRankDataset(root, dataset_label, fit_split, img_size=img_size)
    if len(fit_ds) == 0 and fit_split != "train":
        print(f"[WARN] fit_split='{fit_split}' nodata, automaticfallbackto train. ")
        fit_split = "train"
        fit_ds = RealRankDataset(root, dataset_label, fit_split, img_size=img_size)
    if len(fit_ds) == 0:
        raise RuntimeError(f"No images found for fit_split={fit_split} dataset={dataset_label} under {root}")

    eval_ds = RealRankDataset(root, dataset_label, eval_split, img_size=img_size)
    if len(eval_ds) == 0 and eval_split != "test":
        print(f"[WARN] eval_split='{eval_split}' nodata, automaticfallbackto test. ")
        eval_split = "test"
        eval_ds = RealRankDataset(root, dataset_label, eval_split, img_size=img_size)
    if len(eval_ds) == 0:
        raise RuntimeError(f"No images found for eval_split={eval_split} dataset={dataset_label} under {root}")

    fit_loader = DataLoader(
        fit_ds,
        batch_size=int(CONFIG["batch_size"]),
        shuffle=False,
        num_workers=int(CONFIG["num_workers"]),
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=int(CONFIG["batch_size"]),
        shuffle=False,
        num_workers=max(1, int(CONFIG["num_workers"]) // 2),
        pin_memory=True,
    )

    # 4) collect rank_global andreal count, anddocloseformula solutionfitting
    x_fit, y_fit = collect_rank_global_and_gt(model, fit_loader, device)
    a, b = fit_linear_ls(x_fit, y_fit)
    print(f"[LS FIT] split={fit_split} | a={a:.6f}, b={b:.6f}  (count = a*rank_global + b)")

    # saveoneportion(optional)
    save_path = str(CONFIG.get("save_ab_json", "")).strip()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": dataset_label,
                    "fit_split": fit_split,
                    "rank_ckpt": ckpt,
                    "a": a,
                    "b": b,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[SAVE] a,b -> {save_path}")

    # 5) inevaluationsetoncompute MAE / RMSE
    x_eval, y_eval = collect_rank_global_and_gt(model, eval_loader, device)
    pred = a * x_eval.astype(np.float64) + b
    if bool(CONFIG.get("clamp_nonneg", True)):
        pred = np.maximum(pred, 0.0)

    mae, rmse = compute_mae_rmse(pred.astype(np.float32), y_eval.astype(np.float32))
    print(f"[EVAL] split={eval_split} | MAE: {mae:.3f} | RMSE: {rmse:.3f}")

    # (optional)looklooknotdolinearitymap "original rank_global"error, thenintothan
    mae_raw, rmse_raw = compute_mae_rmse(x_eval.astype(np.float32), y_eval.astype(np.float32))
    print(f"[DEBUG] raw rank_global vs count | MAE: {mae_raw:.3f} | RMSE: {rmse_raw:.3f}")


if __name__ == "__main__":
    main()
