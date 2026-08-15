# -*- coding: utf-8 -*-
"""dualbranch_eval_linear_fusion.py

Evaluation script:
- Load the dual-branch (ranking + density map) model checkpoint (state_dict)
- Use closed-form least squares to map ranking branch rank_global to count: y = a*x + b
- Fusion: final = w_rank * y_rank + w_density * y_density
- Output MAE and RMSE for ranking, density, and fused results

Default: fit (a,b) on train set, evaluate on test set.
To fit and evaluate on test (more optimistic), set FIT_SPLIT to "test".

Note: this script dynamically loads your previous training script to reuse exactly the same:
- DualBranchModel
- RSOCDualDataset
- CONFIG (and density_scale, etc.)

Make sure TRAIN_SCRIPT points to your training script file name (usually in the same directory as this script).
"""

from __future__ import annotations

import os
import math
import importlib.util
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# =========================
# Only modify this section
# =========================
EVAL_CONFIG = {
    # Training script (the manualWeights version I gave you before)
    "train_script": "main_dual_branch_density_map.py",

    # Path to the trained .pth (state_dict) checkpoint
    # e.g. "my_model_dual_branch/best_dual_branch_....pth"
    "checkpoint": "my_model_dual_branch/best_dual_branch_DMAE12.80_DRMSE19.26_RLOSS0.284_RMAE8.682_RRMSE12.518.pth",

    # Dataset root directory (same as training script)
    "dataset_root": "dataset",

    # Which split to use for linear fitting: "train" or "test"
    "fit_split": "train",

    # batch_size (can be larger for evaluation; reduce if OOM)
    "batch_size": 8,

    # Strongly recommend num_workers=0 on Windows (avoid DataLoader multiprocessing pickling errors)
    "num_workers": 0,

    # Fusion weights: final = w_rank * rank_calib + w_density * density_count
    "w_rank": 0.5,
    "w_density": 0.5,

    # Device
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
}


def load_training_module(train_script_path: str):
    """Dynamically load module from .py file to avoid duplicating model/dataset code."""
    if not os.path.isfile(train_script_path):
        raise FileNotFoundError(f"TRAIN_SCRIPT not found: {train_script_path}")

    # Use a fixed, reusable module name; register it in sys.modules,
    # so even if you manually increase num_workers, imports are more likely to work in subprocesses.
    module_name = "dualbranch_train_module"
    spec = importlib.util.spec_from_file_location(module_name, train_script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load spec from: {train_script_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Register in sys.modules to improve Windows multiprocessing compatibility
    import sys

    sys.modules[module_name] = mod
    return mod


@torch.no_grad()
def collect_rank_global_and_gt(model, loader, device: str, density_scale: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect:
    - x: rank_global (B,)
    - y: gt_count (B,)
    - d: density_count (B,)
    """
    model.eval()
    xs = []
    ys = []
    ds = []

    for image, _, target_global, _, _ in tqdm(loader, desc="collect", ncols=110):
        image = image.to(device)
        target_global = target_global.to(device).view(-1)

        _, rank_global, pred_density = model(image)
        rank_global = rank_global.view(-1)

        density_cnt = pred_density.sum(dim=(1, 2, 3)) / density_scale

        xs.append(rank_global.detach().cpu())
        ys.append(target_global.detach().cpu())
        ds.append(density_cnt.detach().cpu())

    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    d = torch.cat(ds, dim=0)
    return x, y, d


def fit_linear_closed_form(x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    """Closed-form least squares: y ≈ a*x + b
    Returns (a, b).

    a = cov(x,y) / var(x)
    b = mean(y) - a*mean(x)
    """
    x = x.float()
    y = y.float()

    mx = x.mean()
    my = y.mean()

    vx = ((x - mx) ** 2).mean()
    if float(vx.item()) < 1e-12:
        # Edge case: x is nearly constant, cannot fit slope
        a = 0.0
        b = float(my.item())
        return a, b

    cov = ((x - mx) * (y - my)).mean()
    a = float((cov / vx).item())
    b = float((my - (cov / vx) * mx).item())
    return a, b


def mae_rmse(pred: torch.Tensor, gt: torch.Tensor) -> Tuple[float, float]:
    diff = (pred - gt).float()
    mae = float(diff.abs().mean().item())
    rmse = math.sqrt(float((diff ** 2).mean().item()))
    return mae, rmse


def main():
    # resolve paths relative to this file
    this_dir = os.path.dirname(os.path.abspath(__file__))
    train_script_path = EVAL_CONFIG["train_script"]
    if not os.path.isabs(train_script_path):
        train_script_path = os.path.join(this_dir, train_script_path)

    ckpt_path = EVAL_CONFIG["checkpoint"]
    if not ckpt_path:
        raise ValueError(
            "Please set EVAL_CONFIG['checkpoint'] to your trained .pth (state_dict) path."
        )
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(this_dir, ckpt_path)

    device = EVAL_CONFIG["device"]

    # load training module
    train_mod = load_training_module(train_script_path)

    # override dataset_root if user sets it here
    if hasattr(train_mod, "CONFIG") and isinstance(train_mod.CONFIG, dict):
        train_mod.CONFIG["dataset_root"] = EVAL_CONFIG["dataset_root"]
        # label_dir is usually based on dataset_root; update if your training script hard-codes it
        if "label_dir" in train_mod.CONFIG:
            # If label_dir is a relative path, derive it from dataset_root to dataset/local pseudo
            # You can also manually set your own absolute path
            train_mod.CONFIG["label_dir"] = os.path.join(EVAL_CONFIG["dataset_root"], "local pseudo")

    density_scale = float(getattr(train_mod, "CONFIG", {}).get("density_scale", 1.0))
    if density_scale <= 0:
        density_scale = 1.0

    # build datasets/loaders
    dataset_root = train_mod.CONFIG["dataset_root"]

    fit_split = EVAL_CONFIG["fit_split"]
    if fit_split not in ("train", "test"):
        raise ValueError("fit_split must be 'train' or 'test'")

    fit_ds = train_mod.RSOCDualDataset(dataset_root, fit_split)
    test_ds = train_mod.RSOCDualDataset(dataset_root, "test")

    fit_loader = DataLoader(
        fit_ds,
        batch_size=int(EVAL_CONFIG["batch_size"]),
        shuffle=False,
        num_workers=int(EVAL_CONFIG.get("num_workers", 0)),
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(EVAL_CONFIG["batch_size"]),
        shuffle=False,
        num_workers=int(EVAL_CONFIG.get("num_workers", 0)),
        pin_memory=True,
    )

    # build model and load checkpoint
    model = train_mod.DualBranchModel(pretrained_altgvt=False).to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    print(f"[OK] Loaded checkpoint: {ckpt_path}")
    print(f"[Device] {device}")
    print(f"[density_scale] {density_scale}")

    # collect fit data
    print(f"\n[1/3] Collecting fit data from split={fit_split} ...")
    x_fit, y_fit, _ = collect_rank_global_and_gt(model, fit_loader, device, density_scale)

    # fit a,b
    a, b = fit_linear_closed_form(x_fit, y_fit)
    print(f"\n[2/3] Linear fit (closed-form) for rank_global -> count: y = a*x + b")
    print(f"  a = {a:.6f}")
    print(f"  b = {b:.6f}")

    # evaluate on test
    print("\n[3/3] Evaluating on test split ...")
    x_test, y_test, d_test = collect_rank_global_and_gt(model, test_loader, device, density_scale)

    rank_pred = a * x_test + b
    dens_pred = d_test

    w_rank = float(EVAL_CONFIG["w_rank"])
    w_density = float(EVAL_CONFIG["w_density"])
    final_pred = w_rank * rank_pred + w_density * dens_pred

    rank_mae, rank_rmse = mae_rmse(rank_pred, y_test)
    dens_mae, dens_rmse = mae_rmse(dens_pred, y_test)
    final_mae, final_rmse = mae_rmse(final_pred, y_test)

    print("\n========== Results on TEST ==========")
    print(f"[Rank (linear-calibrated)]  MAE={rank_mae:.6f}  RMSE={rank_rmse:.6f}")
    print(f"[Density (count)]          MAE={dens_mae:.6f}  RMSE={dens_rmse:.6f}")
    print(f"[Final fusion]             MAE={final_mae:.6f}  RMSE={final_rmse:.6f}")
    print("------------------------------------")
    print(f"Fusion weights: w_rank={w_rank:.3f}, w_density={w_density:.3f}")


if __name__ == "__main__":
    # Needed for Windows + DataLoader multiprocessing; harmless even with num_workers=0
    import multiprocessing as mp

    mp.freeze_support()
    main()