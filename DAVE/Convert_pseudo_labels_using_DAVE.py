import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.nn import DataParallel
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm

from utils.arg_parser import get_argparser
from models.dave import build_model
from utils.data import pad_image

import math
import types
import numpy as np
import skimage.feature
from models.box_prediction import BoxList, boxlist_nms

try:
    import threadpoolctl
    _orig_threadpool_info = threadpoolctl.threadpool_info

    def _safe_threadpool_info(*args, **kwargs):
        try:
            return _orig_threadpool_info(*args, **kwargs)
        except Exception as e:
            
            return []

    threadpoolctl.threadpool_info = _safe_threadpool_info

    _orig_threadpool_limits = threadpoolctl.threadpool_limits
    import contextlib

    def _safe_threadpool_limits(*args, **kwargs):
        try:
            return _orig_threadpool_limits(*args, **kwargs)
        except Exception:
            return contextlib.nullcontext()

    threadpoolctl.threadpool_limits = _safe_threadpool_limits
    print("[Patch] threadpoolctl patched (ignore errors in threadpool_info/threadpool_limits).")
except Exception:
    pass


#Configuration section

ROOT_DATA_DIR = Path("dataset")

MODEL_PATH = Path(r"pretrained")
K_SHOT = 3                 # 0 / 1 / 3
TWO_PASSES = False
GPU_ID = 0                 # GPU id
EXEMPLAR_STRATEGY = "largest" 
SKIP_EMPTY_IMAGES = True 

OUT_DIR = ROOT_DATA_DIR / "dave_outputs_rsoc_building"


# Helper function for this experiment module
def _img_dir(split: str) -> Path:
    rsoc_root = ROOT_DATA_DIR / "ASPDNet_dataset" / "RSOC_building" / "building"
    if split == "train":
        return rsoc_root / "train_data" / "images"
    if split == "test":
        return rsoc_root / "test_data" / "images"
    raise ValueError(split)


def _pseudo_txt(split: str) -> Path:
    return ROOT_DATA_DIR / "local pseudo" / f"rsoc-building_grid16_{split}.txt"


def load_pseudo_txt(txt_path: Path) -> Dict[str, List[List[float]]]:
    """伪标签 txt -> {image_name: [[x1,y1,x2,y2], ...]}"""
    boxes_by_img: Dict[str, List[List[float]]] = {}
    if not txt_path.is_file():
        raise FileNotFoundError(f"Pseudo label file not found: {txt_path}")

    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) != 5:
                raise ValueError(f"Bad format at {txt_path}:{ln}: {line!r}")
            name, x1, y1, x2, y2 = parts
            box = [float(x1), float(y1), float(x2), float(y2)]
            boxes_by_img.setdefault(name, []).append(box)
    return boxes_by_img


# DAVE preprocessing and inference

def resize(img: Image.Image, img_size: int):
    resize_img = T.Resize((img_size, img_size), antialias=True)
    w, h = img.size
    img_t = T.Compose([
        T.ToTensor(),
        resize_img,
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(img)
    scale = torch.tensor([1.0, 1.0]) / torch.tensor([w, h]) * img_size  # [sx, sy]
    return img_t, scale


def _pick_exemplars_xyxy(boxes: List[List[float]], k: int) -> torch.Tensor:
    """从伪标签框里选 k 个 exemplar（xyxy 原图像素坐标），返回 (1,k,4)。"""
    if k <= 0:
        return torch.zeros((1, 3, 4), dtype=torch.float32)

    if len(boxes) == 0:
        ex = [[0.0, 0.0, 0.0, 0.0] for _ in range(k)]
    else:
        if EXEMPLAR_STRATEGY == "first":
            picked = boxes[:k]
        elif EXEMPLAR_STRATEGY == "largest":
            def area(b):
                return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            picked = sorted(boxes, key=area, reverse=True)[:k]
        else:
            raise ValueError(f"Unknown EXEMPLAR_STRATEGY: {EXEMPLAR_STRATEGY}")

        if len(picked) < k:
            picked = picked + [picked[-1]] * (k - len(picked))
        ex = picked

    return torch.tensor(ex, dtype=torch.float32).unsqueeze(0)


def _weights_name(k: int) -> str:
    if k <= 0:
        return "DAVE_0_shot.pth"
    if k == 1:
        return "DAVE_1_shot.pth"
    return "DAVE_3_shot.pth"


def build_args_from_defaults():
    """
    Build args directly from get_argparser() defaults,
    to avoid argparse SystemExit from required arguments.
    """
    base = get_argparser()
    defaults = {}
    for a in base._actions:
        if getattr(a, "dest", None) and a.dest != "help":
            defaults[a.dest] = a.default
    import argparse
    return argparse.Namespace(**defaults)


def infer_backbone_from_checkpoint(ckpt_model: Dict[str, torch.Tensor]) -> Optional[str]:
    k1 = "backbone.backbone.layer1.0.conv1.weight"
    k2 = "module.backbone.backbone.layer1.0.conv1.weight"
    w = None
    if k1 in ckpt_model:
        w = ckpt_model[k1]
    elif k2 in ckpt_model:
        w = ckpt_model[k2]

    if isinstance(w, torch.Tensor) and w.ndim == 4:
        kh, kw = int(w.shape[2]), int(w.shape[3])
        if kh == 1 and kw == 1:
            return "resnet50"
        if kh == 3 and kw == 3:
            return "resnet18"

    k3 = "input_proj.weight"
    k4 = "module.input_proj.weight"
    ip = ckpt_model.get(k3, ckpt_model.get(k4, None))
    if isinstance(ip, torch.Tensor) and ip.ndim == 4:
        in_ch = int(ip.shape[1])
        if in_ch >= 3000:
            return "resnet50"
        if in_ch <= 1000:
            return "resnet18"

    return None


def set_backbone_args(args, backbone: str):
    """
    Multiple candidate field names
    """
    candidates = ["backbone", "backbone_name", "backbone_type", "arch", "model", "encoder"]
    for name in candidates:
        if hasattr(args, name):
            setattr(args, name, backbone)


def _set_backbone_args_inplace(args, backbone_str: str):
    for key in ["backbone", "backbone_name", "encoder_backbone", "backbone_type"]:
        if hasattr(args, key) and isinstance(getattr(args, key), str):
            setattr(args, key, backbone_str)

    for k, v in vars(args).items():
        lk = k.lower()
        if "backbone" in lk:
            if "lr" in lk or lk.endswith("_lr") or lk.endswith("lr"):
                continue
            if isinstance(v, str):
                try:
                    setattr(args, k, backbone_str)
                except Exception:
                    pass


def _infer_backbone_from_ckpt(ckpt_model: dict) -> str:
    candidates = [
        "module.backbone.backbone.layer1.0.conv1.weight",
        "backbone.backbone.layer1.0.conv1.weight",
    ]
    for k in candidates:
        if k in ckpt_model:
            w = ckpt_model[k]
            if hasattr(w, "shape") and len(w.shape) == 4:
                kh, kw = int(w.shape[-2]), int(w.shape[-1])
                if kh == 1 and kw == 1:
                    return "resnet50"
                if kh == 3 and kw == 3:
                    return "resnet18"
    for k, w in ckpt_model.items():
        if k.endswith("backbone.layer1.0.conv1.weight") and hasattr(w, "shape") and len(w.shape) == 4:
            kh, kw = int(w.shape[-2]), int(w.shape[-1])
            if kh == 1 and kw == 1:
                return "resnet50"
            if kh == 3 and kw == 3:
                return "resnet18"
    # Default to ResNet50 following paper setting
    return "resnet50"


def load_model(device: torch.device, k_shot: int):
    args = build_args_from_defaults()
    args.model_path = str(MODEL_PATH)  # Keep behavior consistent with original demo: load weights from args.model_path

    w_path = MODEL_PATH / _weights_name(k_shot)
    v_path = MODEL_PATH / "verification.pth"
    if not w_path.is_file():
        raise FileNotFoundError(f"Missing weights: {w_path}")
    if not v_path.is_file():
        raise FileNotFoundError(f"Missing weights: {v_path}")

    ckpt = torch.load(str(w_path), map_location="cpu")
    ckpt_model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    backbone_need = _infer_backbone_from_ckpt(ckpt_model)
    _set_backbone_args_inplace(args, backbone_need)
    print(f"[AutoConfig] backbone set to: {backbone_need}  (to match checkpoint)")

    keys = list(ckpt_model.keys()) if isinstance(ckpt_model, dict) else []
    ckpt_has_objectness = any(("objectness" in k) for k in keys)
    ckpt_has_self_attn = any(("decoder" in k and "self_attn" in k) for k in keys)  # attn1=not zero_shot and use_appearance

    if hasattr(args, "use_objectness"):
        args.use_objectness = bool(ckpt_has_objectness)
    if hasattr(args, "use_appearance"):
        if getattr(args, "use_appearance") is False and ckpt_has_self_attn:
            args.use_appearance = True

    if hasattr(args, "use_objectness") and hasattr(args, "use_appearance"):
        if (not args.use_objectness) and (not args.use_appearance) and (not getattr(args, "zero_shot", False)):
            args.use_appearance = True

    print(f"[AutoConfig] use_objectness={getattr(args,'use_objectness',None)} | use_appearance={getattr(args,'use_appearance',None)}")


    if device.type == "cuda":
        torch.cuda.set_device(GPU_ID)
        model = DataParallel(
            build_model(args).to(device),
            device_ids=[GPU_ID],
            output_device=GPU_ID
        )
    else:
        model = build_model(args).to(device)

    model.load_state_dict(ckpt_model, strict=False)

    v_ckpt = torch.load(str(v_path), map_location=device)
    v_model = v_ckpt["model"] if isinstance(v_ckpt, dict) and "model" in v_ckpt else v_ckpt
    pretrained_dict_feat = {
        kk.split("feat_comp.")[1]: vv
        for kk, vv in v_model.items()
        if "feat_comp" in kk
    }
    if hasattr(model, "module"):
        model.module.feat_comp.load_state_dict(pretrained_dict_feat)
    else:
        model.feat_comp.load_state_dict(pretrained_dict_feat)

    patch_model_generate_bbox(model)

    model.eval()
    return model, args



@torch.no_grad()
def infer_one(model, device: torch.device, args, img_path: Path, bboxes_xyxy: torch.Tensor):
    """
    Returns:
      pred_boxes_orig: Tensor[N,4] (original-image pixel coordinates)
      dmap_count: float
      box_count: int
    """
    scale_x = scale_y = 1.0

    image = Image.open(img_path).convert("RGB")
    img, scale = resize(image, args.image_size)
    img = img.unsqueeze(0).to(device)

    # Scale exemplar boxes from original coordinates to resized coordinates
    b = bboxes_xyxy.clone()
    b[:, :, 0] *= scale[0]  # x1
    b[:, :, 2] *= scale[0]  # x2
    b[:, :, 1] *= scale[1]  # y1
    b[:, :, 3] *= scale[1]  # y2
    b = b.to(device)

    denisty_map, _, _, predicted_bboxes = model(img, bboxes=b)

    if TWO_PASSES:
        boxes_predicted = predicted_bboxes.box
        if boxes_predicted is not None and boxes_predicted.numel() > 0:
            scale_y = min(1.0, 50.0 / (boxes_predicted[:, 2] - boxes_predicted[:, 0]).mean().clamp(min=1e-6).item())
            scale_x = min(1.0, 50.0 / (boxes_predicted[:, 3] - boxes_predicted[:, 1]).mean().clamp(min=1e-6).item())

            if scale_x < 1.0 or scale_y < 1.0:
                scale_x = (int(args.image_size * scale_x) // 8 * 8) / args.image_size
                scale_y = (int(args.image_size * scale_y) // 8 * 8) / args.image_size
                resize_ = T.Resize((int(args.image_size * scale_x), int(args.image_size * scale_y)), antialias=True)
                img_resized = resize_(img)
                img_resized = pad_image(img_resized[0]).unsqueeze(0)
            else:
                scale_y = max(1.0, 11.0 / (boxes_predicted[:, 2] - boxes_predicted[:, 0]).mean().clamp(min=1e-6).item())
                scale_x = max(1.0, 11.0 / (boxes_predicted[:, 3] - boxes_predicted[:, 1]).mean().clamp(min=1e-6).item())
                scale_y = min(scale_y, 1.9)
                scale_x = min(scale_x, 1.9)

                scale_x = (int(args.image_size * scale_x) // 8 * 8) / args.image_size
                scale_y = (int(args.image_size * scale_y) // 8 * 8) / args.image_size
                resize_ = T.Resize((int(args.image_size * scale_x), int(args.image_size * scale_y)), antialias=True)
                img_resized = resize_(img)

            if scale_x != 1.0 or scale_y != 1.0:
                b2 = b.clone()
                b2[:, :, 0] *= scale_y
                b2[:, :, 2] *= scale_y
                b2[:, :, 1] *= scale_x
                b2[:, :, 3] *= scale_x
                denisty_map, _, _, predicted_bboxes = model(img_resized, bboxes=b2)

    if predicted_bboxes is None or predicted_bboxes.box is None:
        pred_boxes = torch.zeros((0, 4), dtype=torch.float32)
    else:
        pred_boxes = predicted_bboxes.box.detach().cpu()

    denom = torch.tensor([scale_y * scale[0], scale_x * scale[1], scale_y * scale[0], scale_x * scale[1]])
    pred_boxes_orig = pred_boxes / denom

    dmap_count = float(denisty_map.sum().item()) if denisty_map is not None else 0.0
    return pred_boxes_orig, dmap_count, int(pred_boxes_orig.shape[0])



def _safe_generate_bbox(self, density_map, tlrb, gt_dmap=None):

    if gt_dmap is not None:
        density_map = gt_dmap

    bboxes = []
    for i in range(density_map.shape[0]):
        density = np.array(density_map[i][0].detach().cpu())
        dmap = density.copy()

        if dmap.size == 0:
            empty = BoxList(torch.zeros((0, 4), dtype=torch.float32), (density_map.shape[3], density_map.shape[2]))
            empty.fields["scores"] = torch.zeros((0,), dtype=torch.float32)
            bboxes.append(empty)
            continue

        maxv = float(np.max(dmap)) if dmap.size > 0 else 0.0
        thr = min(maxv / self.d_t if getattr(self, "d_t", 0) not in [0, None] else maxv, self.s_t)
        dmap[dmap < thr] = 0

        a = skimage.feature.peak_local_max(dmap, exclude_border=0)

        boxes = []
        scores = []
        b, l, r, t = tlrb[i]

        for x11, y11 in a:
            box = [
                y11 - b[x11][y11].item(),
                x11 - l[x11][y11].item(),
                y11 + r[x11][y11].item(),
                x11 + t[x11][y11].item()
            ]

            y0 = max(0, int(box[1]))
            y1 = min(int(box[3]), dmap.shape[0])
            x0 = max(0, int(box[0]))
            x1 = min(int(box[2]), dmap.shape[1])

            #Skip empty regions
            if y1 <= y0 or x1 <= x0:
                continue

            region = density[y0:y1, x0:x1]
            if region.size == 0:
                continue

            region_sum = float(region.sum())
            region_max = float(region.max())
            score = (1 - math.fabs(region_sum - 1.0)) * self.d_s + region_max * self.m_s

            boxes.append(box)
            scores.append(score)

        #If no valid boxes remain, return an empty BoxList
        if len(boxes) == 0:
            empty = BoxList(torch.zeros((0, 4), dtype=torch.float32), (density_map.shape[3], density_map.shape[2]))
            empty.fields["scores"] = torch.zeros((0,), dtype=torch.float32)
            bboxes.append(empty)
            continue

        b_list = BoxList(list(boxes), (density_map.shape[3], density_map.shape[2]))
        b_list.fields["scores"] = torch.tensor(scores, dtype=b_list.box.dtype)
        b_list = b_list.clip()

        # Normalize scores
        if getattr(self, "norm_s", False) and len(scores) > 1 and max(scores) != min(scores):
            mn, mx = min(scores), max(scores)
            b_list.fields["scores"] = torch.tensor([(float(s) - mn) / (mx - mn) for s in b_list.fields["scores"]])

        if len(b_list) > 0:
            b_list = boxlist_nms(b_list, b_list.fields["scores"], self.i_thr)

        bboxes.append(b_list)

    return bboxes


def patch_model_generate_bbox(model):
    target = model.module if hasattr(model, "module") else model
    target.generate_bbox = types.MethodType(_safe_generate_bbox, target)
    print("[Patch] generate_bbox patched to safe version (skip empty regions).")


def run_split(split: str, model, device, args) -> List[List[str]]:
    img_dir = _img_dir(split)
    pseudo_file = _pseudo_txt(split)

    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image dir not found: {img_dir}")
    if not pseudo_file.is_file():
        raise FileNotFoundError(f"Pseudo txt not found: {pseudo_file}")

    boxes_by_img = load_pseudo_txt(pseudo_file)

    #Skip images with zero boxes
    img_names = sorted(boxes_by_img.keys()) if SKIP_EMPTY_IMAGES else sorted([p.name for p in img_dir.iterdir() if p.is_file()])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pred = OUT_DIR / f"dave_pred_{split}.txt"

    if out_pred.exists():
        out_pred.unlink()

    rows: List[List[str]] = []
    with out_pred.open("w", encoding="utf-8") as f_pred:
        for name in tqdm(img_names, desc=f"Split={split}"):
            img_path = img_dir / name
            if not img_path.is_file():
                print(f"[Warn] image not found, skipped: {img_path}")
                continue

            pseudo_boxes = boxes_by_img.get(name, [])
            if SKIP_EMPTY_IMAGES and len(pseudo_boxes) == 0:
                continue

            bboxes = _pick_exemplars_xyxy(pseudo_boxes, K_SHOT)
            pred_boxes, dmap_count, pred_n = infer_one(model, device, args, img_path, bboxes)

            for i in range(pred_boxes.shape[0]):
                x1, y1, x2, y2 = pred_boxes[i].tolist()
                f_pred.write(f"{name} {int(round(x1))} {int(round(y1))} {int(round(x2))} {int(round(y2))}\n")

            rows.append([split, name, str(len(pseudo_boxes)), f"{dmap_count:.4f}", str(pred_n)])

    print(f"[Done] {split} pred saved: {out_pred}")
    return rows


def main():
    print("======== DAVE RSOC-Building PyCharm Runner ========")
    print(f"[Config] ROOT_DATA_DIR={ROOT_DATA_DIR}")
    print(f"[Config] MODEL_PATH={MODEL_PATH}")
    print(f"[Config] K_SHOT={K_SHOT} | TWO_PASSES={TWO_PASSES} | SKIP_EMPTY_IMAGES={SKIP_EMPTY_IMAGES}")

    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"[Config] device={device}")

    model, args = load_model(device, K_SHOT)

    all_rows: List[List[str]] = []
    for split in ["train", "test"]:
        all_rows.extend(run_split(split, model, device, args))

    stats_path = OUT_DIR / "dave_stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "image_name", "pseudo_box_count", "dmap_count", "pred_box_count"])
        w.writerows(all_rows)

    print(f"[Done] stats saved: {stats_path}")
    print("===================================================")


if __name__ == "__main__":
    main()
