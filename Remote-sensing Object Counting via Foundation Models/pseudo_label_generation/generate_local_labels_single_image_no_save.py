# -*- coding: utf-8 -*-
"""singleimagetothanpreviewversion(notsaveanyfile, singleimagetothan)

youprint APIlistdisplay, youlocal Sam3Processor onlyhas: 
  add_geometric_prompt / reset_all_prompts / set_image / set_image_batch / set_text_prompt
no image exemplar / image prompt mutualcloseAPI. 

therefore: nomethoduse"take ROI cropbecome patch whenimageprompt"this kindmethod. 
thisscripttake"hasboxpipeline"implementforyoucurrent API support way: 
- youmanualbox  ROI as SAM3  geometryboxprompt(positionmethod), onlyasused for SAM3 part(GD notchange)

showorder: 
1) Baseline: GD + SAM3(pass1: GD boxprompt) + SAM3(pass2: plain text)
2) With ROI: GD + SAM3(pass1: GD box+ROI boxprompt) + SAM3(pass2: ROI boxprompt)

notsaveanyfile: notwritepseudo label, notsave patch, notsavevisualize. 
"""

import os
import sys
import types
import random
from unittest.mock import MagicMock

import numpy as np
import torch
import cv2
from PIL import Image


# ===================== user configuration section(onlyretainsingleimagepreviewneed ) =====================

ROOT_DATA_DIR = "dataset"  # you datasetroot directory

# selectobjectdataset(andoriginalscriptconsistent)
TARGET_DATASET = "RSOC-Building"

# singleimagefromwhich split insideextract(you datasetdifferent split canuse-itydifferent)
TARGET_SPLIT = "train"  # RSOC-Building: train/test; otherusually train/val/test

# youalsocandirectspecifyoneimageimage(notforemptywhenpreferuse)
SINGLE_IMAGE_PATH = r""  # For example: r"dataset/ASPDNet_dataset/train/images/xxx.png"

# isor notrandomselectselectobjectimage; if False and SINGLE_IMAGE_PATH forempty, thentakelistfirstimage
RANDOM_PICK_ONE = True
RANDOM_SEED = 2027

# grid division: keepandoriginalscriptconsistent(default 4×4=16)
GRID_ROWS = 4
GRID_COLS = 4

# ---------- GD (Local GroundingDINO) parameter ----------
GD_BBOX_THRESHOLD = 0.13
GD_TEXT_THRESHOLD = 0.15
GDINO_MODEL_ID = os.getenv("GDINO_MODEL_ID", "grounding_dino_base")

# ---------- SAM3 parameter ----------
LOCAL_SAM3_PATH = r"D:\\sam3_source"  # you local sam3 source codepath
SAM3_CHECKPOINT_PATH = "saved_models/SAM3.pt"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAM3_CONFIDENCE_THRESHOLD = 0.3

# ---------- deduplication parameter ----------
DEDUP_IOU_THR = 0.6
MIN_BOX_AREA = 16.0

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# SAM3 text prompt
SAM3_PROMPT_MAP = {
    "RSOC-Building": "building",
    "RSOC-S-Vehicle": "smallcar.car.vehicle",
    "RSOC-L-Vehicle": "largecar.truck.bus.vehicle",
    "RSOC-Ship": "ship.boat",
    "VD-People": "person",
    "VD-Vehicle": "vehicle",
}

# GD text prompt
GD_PROMPT_MAP = {
    "RSOC-Building": "structure.building.house",
    "RSOC-S-Vehicle": "car",
    "RSOC-L-Vehicle": "truck",
    "RSOC-Ship": "ship.boat",
    "VD-People": "person",
    "VD-Vehicle": "vehicle",
}


# ---------- ROI boxselectparameter ----------
ROI_WINDOW_PREFIX = "ROI"
ROI_MIN_W = 4
ROI_MIN_H = 4


# ===================== utility: adaptive path =====================

def _autofix_root_dir():
    global ROOT_DATA_DIR
    if os.path.isdir(ROOT_DATA_DIR):
        return
    if os.path.isdir("dataset"):
        ROOT_DATA_DIR = "dataset"
        return
    if os.path.isdir("Dataset"):
        ROOT_DATA_DIR = "Dataset"


_autofix_root_dir()


def build_grid_cells(img_w, img_h, rows, cols):
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    xs = np.linspace(0, img_w, cols + 1, dtype=int)
    ys = np.linspace(0, img_h, rows + 1, dtype=int)

    cells = []
    idx = 1
    for r in range(rows):
        for c in range(cols):
            x1, x2 = int(xs[c]), int(xs[c + 1])
            y1, y2 = int(ys[r]), int(ys[r + 1])
            if x2 <= x1 + 1 or y2 <= y1 + 1:
                continue
            cells.append({"idx": idx, "row": r, "col": c, "xyxy": (x1, y1, x2, y2)})
            idx += 1
    return cells


# ===================== GD (Local GroundingDINO) =====================

_GD_PROCESSOR = None
_GD_MODEL = None
_GD_DEVICE_STR = None


def _ensure_gd_prompt(prompt_text: str) -> str:
    if prompt_text is None:
        return ""
    t = str(prompt_text).strip()
    if not t:
        return t
    if not t.endswith("."):
        t = t + " ."
    elif not t.endswith(" ."):
        t = t[:-1].rstrip() + " ."
    return t


def _load_local_gd():
    global _GD_PROCESSOR, _GD_MODEL, _GD_DEVICE_STR
    if _GD_PROCESSOR is not None and _GD_MODEL is not None:
        return _GD_PROCESSOR, _GD_MODEL, _GD_DEVICE_STR

    try:
        from transformers import AutoProcessor, GroundingDinoForObjectDetection
    except Exception as e:
        raise ImportError(
            "failed to import transformers   GroundingDINO related modules. Please install first/Upgrade:pip install -U transformers"
        ) from e

    _GD_DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[GD] Loading local GroundingDINO: {GDINO_MODEL_ID} | device={_GD_DEVICE_STR}")

    _GD_PROCESSOR = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
    _GD_MODEL = GroundingDinoForObjectDetection.from_pretrained(GDINO_MODEL_ID).to(_GD_DEVICE_STR)
    _GD_MODEL.eval()
    return _GD_PROCESSOR, _GD_MODEL, _GD_DEVICE_STR


def call_dino_local_on_rgb(img_rgb, prompt_text):
    processor, model, device = _load_local_gd()
    h, w = img_rgb.shape[:2]
    prompt = _ensure_gd_prompt(prompt_text)

    inputs = processor(images=img_rgb, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[h, w]], device=device)
    try:
        processed = processor.post_process_grounded_object_detection(
            outputs,
            inputs,
            box_threshold=float(GD_BBOX_THRESHOLD),
            text_threshold=float(GD_TEXT_THRESHOLD),
            target_sizes=target_sizes,
        )[0]
    except TypeError:
        input_ids = inputs["input_ids"]
        processed = processor.post_process_grounded_object_detection(
            outputs,
            input_ids,
            target_sizes=target_sizes,
        )[0]
        scores = processed.get("scores", None)
        boxes = processed.get("boxes", None)
        if scores is None or boxes is None:
            return []
        keep = scores > float(GD_BBOX_THRESHOLD)
        boxes = boxes[keep]
        processed = {"boxes": boxes}

    boxes = processed.get("boxes", None)
    if boxes is None:
        return []
    if torch.is_tensor(boxes):
        boxes_np = boxes.detach().cpu().numpy()
    else:
        boxes_np = np.array(boxes)
    return boxes_np.astype(float).tolist()


def call_dino_local_on_pil(pil_img, prompt_text):
    img_rgb = np.array(pil_img.convert("RGB"))
    return call_dino_local_on_rgb(img_rgb, prompt_text)


# ===================== SAM3 import and inference =====================

if sys.platform.startswith("win"):
    def _dummy_decorator(*args, **kwargs):
        if kwargs or (args and not callable(args[0])):
            return lambda f: f
        return args[0]

    mock_triton = types.ModuleType("triton")
    mock_triton.__spec__ = MagicMock()
    mock_triton.__spec__.name = "triton"
    mock_triton.__file__ = "dummy_triton_path"
    mock_triton.__path__ = []
    mock_triton.__version__ = "2.0.0"
    mock_triton.jit = _dummy_decorator
    mock_triton.autotune = _dummy_decorator
    mock_triton.heuristics = _dummy_decorator
    mock_triton.cdiv = lambda x, y: (x + y - 1) // y
    mock_triton.next_power_of_2 = lambda x: 1 << (x - 1).bit_length()

    class DummyConfig:
        def __init__(self, *args, **kwargs):
            pass

    mock_triton.Config = DummyConfig
    mock_triton.language = MagicMock()
    mock_triton.impl = MagicMock()
    sys.modules["triton"] = mock_triton
    sys.modules["triton.language"] = mock_triton.language
    sys.modules["triton.impl"] = mock_triton.impl

    try:
        import flash_attn  # noqa: F401
    except ImportError:
        mock_flash = types.ModuleType("flash_attn")
        mock_flash.__spec__ = MagicMock()
        sys.modules["flash_attn"] = mock_flash
        sys.modules["flash_attn.flash_attn_interface"] = MagicMock()

if os.path.exists(LOCAL_SAM3_PATH) and LOCAL_SAM3_PATH not in sys.path:
    sys.path.insert(0, LOCAL_SAM3_PATH)

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model.box_ops import box_xywh_to_cxcywh
except ImportError as e:
    print(f"[Error] cannot import SAM3 module: {e}")
    print("please confirm LOCAL_SAM3_PATH and sam3 source path is correct.")
    raise


def build_sam3_processor():
    if not os.path.exists(SAM3_CHECKPOINT_PATH):
        raise FileNotFoundError(f"SAM3 weights do not exist: {SAM3_CHECKPOINT_PATH}")

    print("[Model] load SAM3 model ...")
    model = build_sam3_image_model(
        checkpoint_path=SAM3_CHECKPOINT_PATH,
        load_from_HF=False,
        device=str(DEVICE).split(":")[0],
    )
    processor = Sam3Processor(model, confidence_threshold=SAM3_CONFIDENCE_THRESHOLD)
    print("[Model] SAM3 Processor initialization complete")
    return processor


def _apply_sam3_image_exemplars(processor, state, exemplar_pil_list):
    """compatibleoldcall: youlocal Sam3Processor no image exemplar / image prompt API, directreturn state. """
    return state
def sam3_text_plus_multi_boxes(processor, pil_img, boxes_xywh_list, prompt_text, exemplar_pil_list=None):
    width, height = pil_img.size
    state = processor.set_image(pil_img)
    state = _apply_sam3_image_exemplars(processor, state, exemplar_pil_list)
    state = processor.set_text_prompt(state=state, prompt=prompt_text)

    for (x, y, w, h) in boxes_xywh_list:
        if w <= 0 or h <= 0:
            continue
        box_xywh = torch.tensor([[x, y, w, h]], dtype=torch.float32, device=DEVICE)
        box_cxcywh = box_xywh_to_cxcywh(box_xywh)

        norm_box_cxcywh = [
            float(box_cxcywh[0, 0] / width),
            float(box_cxcywh[0, 1] / height),
            float(box_cxcywh[0, 2] / width),
            float(box_cxcywh[0, 3] / height),
        ]
        state = processor.add_geometric_prompt(box=norm_box_cxcywh, label=True, state=state)

    boxes = state.get("boxes", None)
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), dtype=float)

    if torch.is_tensor(boxes):
        boxes_np = boxes.detach().cpu().numpy()
    else:
        boxes_np = np.array(boxes)
    return boxes_np.astype(float)


def sam3_text_only(processor, pil_img, prompt_text, exemplar_pil_list=None):
    state = processor.set_image(pil_img)
    state = _apply_sam3_image_exemplars(processor, state, exemplar_pil_list)
    output = processor.set_text_prompt(state=state, prompt=prompt_text)

    boxes = output.get("boxes", None) if isinstance(output, dict) else None
    if boxes is None:
        boxes = getattr(output, "boxes", None) if output is not None else None
    if boxes is None:
        return np.zeros((0, 4), dtype=float)

    if torch.is_tensor(boxes):
        boxes_np = boxes.detach().cpu().numpy()
    else:
        boxes_np = np.array(boxes)
    return boxes_np.astype(float)


# ===================== ROI boxselect(exemplar) =====================

def _select_roi_one_image(img_bgr, win_name):
    """OpenCV interactivetypeboxselect ROI, return (x,y,w,h) or None"""
    try:
        roi = cv2.selectROI(win_name, img_bgr, fromCenter=False, showCrosshair=True)
    except Exception as e:
        print(f"[Error] cv2.selectROI callfail: {e}")
        return None

    try:
        cv2.destroyWindow(win_name)
    except Exception:
        pass

    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        return None
    if w < int(ROI_MIN_W) or h < int(ROI_MIN_H):
        print(f"[ROI] ROI toosmall({w}x{h}), alreadyskip. ")
        return None
    return (x, y, w, h)


# ===================== counting: box processing / deduplication =====================

def clamp_box_xyxy(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(w - 1), float(x1)))
    y1 = max(0.0, min(float(h - 1), float(y1)))
    x2 = max(0.0, min(float(w - 1), float(x2)))
    y2 = max(0.0, min(float(h - 1), float(y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def box_area_xyxy(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = box_area_xyxy(a) + box_area_xyxy(b) - inter
    return float(inter / union) if union > 0 else 0.0


def nms_dedup_boxes(boxes, iou_thr=0.6, min_area=0.0):
    if boxes is None or len(boxes) == 0:
        return []
    boxes = [list(map(float, b)) for b in boxes]
    boxes = [b for b in boxes if box_area_xyxy(b) >= min_area]
    if not boxes:
        return []
    boxes_sorted = sorted(boxes, key=lambda b: box_area_xyxy(b), reverse=True)
    keep = []
    for b in boxes_sorted:
        ok = True
        for kb in keep:
            if iou_xyxy(b, kb) >= iou_thr:
                ok = False
                break
        if ok:
            keep.append(b)
    return keep


# ===================== dataset: columnimage(reuse v2  directorylogic) =====================

def list_images(img_dir):
    if not img_dir or (not os.path.isdir(img_dir)):
        return []
    return [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(IMAGE_EXTS)]


def _available_splits_for_dataset(dataset_label: str):
    if dataset_label == "RSOC-Building":
        return ["train", "test"]
    if dataset_label in ["RSOC-Ship", "RSOC-S-Vehicle", "RSOC-L-Vehicle"]:
        return ["train", "val", "test"]
    if dataset_label in ["VD-People", "VD-Vehicle"]:
        return ["train", "val", "test"]
    return ["train", "val", "test"]


def _find_dota_list_files(dota_dir: str, split: str, cat_token: str):
    if not os.path.isdir(dota_dir):
        return []
    split_l = split.lower()
    cat_l = cat_token.lower()
    hits = []
    for fn in os.listdir(dota_dir):
        low = fn.lower()
        if not low.endswith(".txt"):
            continue
        if not low.startswith(split_l):
            continue
        if cat_l.replace("-", "") in low.replace("-", "").replace("_", ""):
            hits.append(os.path.join(dota_dir, fn))
    hits.sort(key=lambda p: (len(os.path.basename(p)), p))
    return hits


def _read_dota_name_list(txt_path):
    names = set()
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                s = s.replace("\\", "/")
                base = os.path.basename(s)
                base = os.path.splitext(base)[0]
                if base:
                    names.add(base)
    except Exception:
        return set()
    return names


def _build_allow_set_for_rsoc_mixed(dataset_label: str, split: str):
    cat_token = {
        "RSOC-Ship": "ship",
        "RSOC-S-Vehicle": "small-vehicle",
        "RSOC-L-Vehicle": "large-vehicle",
    }[dataset_label]

    dota_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "DOTA_data")
    allow = None

    list_files = _find_dota_list_files(dota_dir, split, cat_token)
    if list_files:
        allow = set()
        for lp in list_files:
            allow |= _read_dota_name_list(lp)

    if allow is None and split in ["train", "val"]:
        if split == "train":
            label_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "train", "labelTxt-v1.0", "trainset_reclabelTxt")
        else:
            label_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "val", "labelTxt-v1.0", "valset_reclabelTxt")

        if os.path.isdir(label_dir):
            allow = set()
            for fn in os.listdir(label_dir):
                if not fn.endswith(".txt"):
                    continue
                base = os.path.splitext(fn)[0]
                fp = os.path.join(label_dir, fn)
                try:
                    hit = False
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 9:
                                continue
                            cls = parts[-2] if len(parts) >= 10 else parts[-1]
                            if cls == cat_token:
                                hit = True
                                break
                    if hit:
                        allow.add(base)
                except Exception:
                    continue

    return allow


def _list_images_for_dataset_split(dataset_label: str, split: str):
    if dataset_label == "RSOC-Building":
        rsoc_root = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "RSOC_building", "building")
        sub = "train_data" if split == "train" else ("test_data" if split == "test" else None)
        if sub is None:
            return []
        img_dir = os.path.join(rsoc_root, sub, "images")
        return list_images(img_dir)

    if dataset_label in ["RSOC-Ship", "RSOC-S-Vehicle", "RSOC-L-Vehicle"]:
        img_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", split, "images")
        if not os.path.isdir(img_dir):
            return []
        allow = _build_allow_set_for_rsoc_mixed(dataset_label, split)
        if not allow:
            return []
        out = []
        for p in list_images(img_dir):
            base = os.path.splitext(os.path.basename(p))[0]
            if base in allow:
                out.append(p)
        return out

    if dataset_label in ["VD-People", "VD-Vehicle"]:
        vd_root = os.path.join(ROOT_DATA_DIR, "VisDrone-People" if dataset_label == "VD-People" else "VisDrone-Vehicle")
        img_dir = None
        for cand in [os.path.join(vd_root, split, "images"), os.path.join(vd_root, split, "Images")]:
            if os.path.isdir(cand):
                img_dir = cand
                break
        return list_images(img_dir) if img_dir else []

    return []


# ===================== singleimageinference(GD + twice SAM3, support exemplar) =====================

def _roi_intersection_as_cell_xywh(roi_xywh_global, cell_xyxy_global):
    """takeallimage ROI (x,y,w,h) turnbecomesome cell inside  (x,y,w,h), take ROI and cell  intersectionset. 
    ifnointersectionsetreturn None. 
    """
    if roi_xywh_global is None:
        return None
    rx, ry, rw, rh = roi_xywh_global
    if rw <= 0 or rh <= 0:
        return None
    r_x1, r_y1, r_x2, r_y2 = rx, ry, rx + rw, ry + rh
    c_x1, c_y1, c_x2, c_y2 = cell_xyxy_global

    ix1 = max(r_x1, c_x1)
    iy1 = max(r_y1, c_y1)
    ix2 = min(r_x2, c_x2)
    iy2 = min(r_y2, c_y2)
    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw <= 0 or ih <= 0:
        return None

    # turnfor cell coordinate
    return (int(ix1 - c_x1), int(iy1 - c_y1), int(iw), int(ih))


def _process_one_image_grid16(img_path: str, processor, dataset_label: str, roi_xywh_global=None):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return [0] * (GRID_ROWS * GRID_COLS), []

    h_full, w_full = img_bgr.shape[:2]
    pil_full = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    sam_prompt = SAM3_PROMPT_MAP.get(dataset_label, "object")
    gd_prompt = GD_PROMPT_MAP.get(dataset_label, "object")

    cells = build_grid_cells(w_full, h_full, GRID_ROWS, GRID_COLS)
    if len(cells) != GRID_ROWS * GRID_COLS:
        return [0] * (GRID_ROWS * GRID_COLS), []

    counts = [0] * (GRID_ROWS * GRID_COLS)
    boxes_global = []

    for cell in cells:
        idx = cell["idx"]
        cell_x1, cell_y1, cell_x2, cell_y2 = cell["xyxy"]
        cell_w, cell_h = int(cell_x2 - cell_x1), int(cell_y2 - cell_y1)
        pil_cell = pil_full.crop((cell_x1, cell_y1, cell_x2, cell_y2))

        # 1) GD(notchange)
        gd_boxes = call_dino_local_on_pil(pil_cell, gd_prompt)
        gd_boxes = [clamp_box_xyxy(b, cell_w, cell_h) for b in gd_boxes
                    if isinstance(b, (list, tuple)) and len(b) == 4]

        # ROI -> cell prompt(mostmany 1 )
        roi_cell_xywh = _roi_intersection_as_cell_xywh(roi_xywh_global, (cell_x1, cell_y1, cell_x2, cell_y2))
        roi_prompts_xywh = [roi_cell_xywh] if roi_cell_xywh is not None else []

        # 2) SAM3 pass1: text + (GD boxes) + (ROI box optional)
        sam_boxes_pass1 = []
        prompts_xywh = []
        for b in gd_boxes:
            xx1, yy1, xx2, yy2 = [int(round(v)) for v in b]
            ww = max(1, xx2 - xx1)
            hh = max(1, yy2 - yy1)
            prompts_xywh.append((xx1, yy1, ww, hh))
        prompts_xywh.extend(roi_prompts_xywh)

        if len(prompts_xywh) > 0:
            boxes_np = sam3_text_plus_multi_boxes(processor, pil_cell, prompts_xywh, sam_prompt, None)
            for bb in boxes_np:
                sam_boxes_pass1.append(clamp_box_xyxy(list(map(float, bb)), cell_w, cell_h))
        sam_boxes_pass1 = nms_dedup_boxes(sam_boxes_pass1, iou_thr=DEDUP_IOU_THR, min_area=MIN_BOX_AREA)

        # 3) SAM3 pass2: 
        #    - Baseline: plain text
        #    - With ROI: text + ROI geometryboxprompt
        sam_boxes_pass2 = []
        if roi_prompts_xywh:
            boxes_np2 = sam3_text_plus_multi_boxes(processor, pil_cell, roi_prompts_xywh, sam_prompt, None)
        else:
            boxes_np2 = sam3_text_only(processor, pil_cell, sam_prompt, None)

        if getattr(boxes_np2, "shape", None) is not None and boxes_np2.shape[0] > 0:
            for bb in boxes_np2:
                sam_boxes_pass2.append(clamp_box_xyxy(list(map(float, bb)), cell_w, cell_h))
        sam_boxes_pass2 = nms_dedup_boxes(sam_boxes_pass2, iou_thr=DEDUP_IOU_THR, min_area=MIN_BOX_AREA)

        # 4) merge & dedup -> count
        merged = sam_boxes_pass1 + sam_boxes_pass2
        final_boxes = nms_dedup_boxes(merged, iou_thr=DEDUP_IOU_THR, min_area=MIN_BOX_AREA)
        counts[idx - 1] = int(len(final_boxes))

        # 5) cell -> global
        for bb in final_boxes:
            bx1, by1, bx2, by2 = [float(v) for v in bb]
            gx1 = bx1 + float(cell_x1)
            gy1 = by1 + float(cell_y1)
            gx2 = bx2 + float(cell_x1)
            gy2 = by2 + float(cell_y1)
            gx1 = max(0.0, min(float(w_full - 1), gx1))
            gy1 = max(0.0, min(float(h_full - 1), gy1))
            gx2 = max(0.0, min(float(w_full - 1), gx2))
            gy2 = max(0.0, min(float(h_full - 1), gy2))
            boxes_global.append([gx1, gy1, gx2, gy2])

    return counts, boxes_global


# ===================== visualize(notsave) =====================

def _draw_boxes_on_bgr(img_bgr, boxes_xyxy, title_text=None):
    out = img_bgr.copy()
    for b in (boxes_xyxy or []):
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = max(0, x2), max(0, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    if title_text is not None:
        cv2.putText(out, str(title_text), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return out


def _resize_for_display(img_bgr, max_w=1600, max_h=900):
    h, w = img_bgr.shape[:2]
    scale = min(max_w / float(w), max_h / float(h), 1.0)
    if scale >= 1.0:
        return img_bgr
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _show_window(title, img_bgr):
    img_disp = _resize_for_display(img_bgr)
    cv2.imshow(title, img_disp)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow(title)
    return key


# ===================== main: singleimagetothan =====================

def main():
    dataset_label = str(TARGET_DATASET).strip()
    if dataset_label not in SAM3_PROMPT_MAP:
        print(f"[Error] TARGET_DATASET notvalid: {dataset_label}")
        print(f"optional: {list(SAM3_PROMPT_MAP.keys())}")
        return

    avail = _available_splits_for_dataset(dataset_label)
    split = str(TARGET_SPLIT).strip().lower()
    if split not in avail:
        print(f"[Error] TARGET_SPLIT={split} notfitused for {dataset_label}, canuse: {avail}")
        return

    # accurateprepareobjectimage
    img_pool = []
    if SINGLE_IMAGE_PATH and os.path.isfile(SINGLE_IMAGE_PATH):
        target_img_path = SINGLE_IMAGE_PATH
    else:
        img_pool = _list_images_for_dataset_split(dataset_label, split)
        if not img_pool:
            print(f"[Error] notfindtoimage: dataset={dataset_label} split={split}")
            return
        rng = random.Random(int(RANDOM_SEED))
        target_img_path = rng.choice(img_pool) if RANDOM_PICK_ONE else img_pool[0]

    print(f"[Main] ROOT_DATA_DIR={ROOT_DATA_DIR}")
    print(f"[Main] dataset={dataset_label} split={split}")
    print(f"[Main] target image: {target_img_path}")
    print(f"[Main] DEVICE={DEVICE} | GDINO_MODEL_ID={GDINO_MODEL_ID}")

    processor = build_sam3_processor()
    # print processor  canusepromptAPI, theninconfirmrecognizeisor notsupport image exemplar(you outputinsideonlyhas geometric/text)
    methods = [m for m in dir(processor) if any(k in m.lower() for k in ["prompt", "exemplar", "reference", "template", "image"])]
    print(f"[Main] Sam3Processor prompt-related methods: {methods}")

    # readoriginalimage
    img_bgr = cv2.imread(target_img_path)
    if img_bgr is None:
        print("[Error] nomethodreadobjectimage")
        return

    # ----------------- 2) baseline: no ROI -----------------
    print("\n[Run] Baseline: GD + twice SAM3(no ROI)")
    _, boxes_base = _process_one_image_grid16(target_img_path, processor, dataset_label, roi_xywh_global=None)
    base_vis = _draw_boxes_on_bgr(img_bgr, boxes_base, title_text=f"Baseline count={len(boxes_base)}")
    print(f"[Baseline] boxes={len(boxes_base)}")
    key = _show_window("Baseline (no ROI)", base_vis)
    if key in (ord('q'), 27):
        cv2.destroyAllWindows()
        return

    # ----------------- 3) with ROI: manualboxselect ROI(only SAM3 stagegeometryboxprompt) -----------------
    print("\n[ROI] pleaseinsameoneimageobjectimageonboxselect 1  ROI(used for SAM3  geometryboxprompt). ")
    print("      Operation: drag to draw a box -> ENTER/SPACE confirm;ESC/takeeliminatethennotuse ROI. ")
    roi_xywh = _select_roi_one_image(
        img_bgr,
        f"{ROI_WINDOW_PREFIX} - {os.path.basename(target_img_path)}",
    )

    if roi_xywh is None:
        print("[ROI] notselect ROI, thistimes With-ROI pipelinewilldegeneratefor Baseline(no ROI). ")
    else:
        print(f"[ROI] ROI={roi_xywh} willas SAM3  geometryboxprompt(only SAM3 stage). ")

    print("\n[Run] With ROI: GD + twice SAM3(ROI onlyas SAM3 geometryboxprompt)")
    _, boxes_roi = _process_one_image_grid16(target_img_path, processor, dataset_label, roi_xywh_global=roi_xywh)
    roi_vis = _draw_boxes_on_bgr(img_bgr, boxes_roi, title_text=f"With ROI count={len(boxes_roi)}")

    # drawoutyouselect  ROI(green), sidethentoaccording to
    if roi_xywh is not None:
        x, y, w, h = roi_xywh
        cv2.rectangle(roi_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(roi_vis, "ROI", (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    print(f"[WithROI] boxes={len(boxes_roi)}")
    _show_window("With ROI (geometric box prompt for SAM3)", roi_vis)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
