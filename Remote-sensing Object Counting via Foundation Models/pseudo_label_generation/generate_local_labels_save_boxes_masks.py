# -*- coding: utf-8 -*-
"""
GD + SAM3 dual-path object counting + 16 regionlocal ranking(Grid version)
[thisversionchangeobject]
- from"singleimage"extensionto"traversewholedataset"
- forevery split outputonepseudo labeltextfile: 
  dataset/local pseudo/{dataset}_grid16_{split}.txt
- fileinsidetolerateper line: 
  <image_name> x1 y1 x2 y2  (one per lineobject, coordinateforwhole image xyxy)
- notagainoutputimage, notagainoutputleaderboard, notagainread/outputreal label
Description:
- ship / small-vehicle / large-vehicle mixed in ASPDNet_dataset/{train,val,test}/images
  use ASPDNet_dataset/DOTA_data below listfile(if train_ship.txt etc.)comefilter
  if train/val missing list, degenerateforscan labelTxt-v1.0
"""
import os
import sys
import types
from unittest.mock import MagicMock
import numpy as np
from transformers import AutoProcessor, GroundingDinoForObjectDetection
import random
import torch
import cv2
import time
from PIL import Image, ImageDraw, ImageFont
# optional: used forread RSOC_building   .mat realvalue(thisversionnotuse, butretainoriginalcodecompatible)
try:
    import scipy.io as sio  # type: ignore
except Exception:
    sio = None
# ===================== user configuration section =====================
# datasetroot directory(you directoryis dataset)
ROOT_DATA_DIR = "dataset"
# oldconfigurationretain(thisversionnotagainuse SINGLE_IMAGE_PATH pathrunsingleimage)
SINGLE_IMAGE_PATH = r""
# old fallback retain
TARGET_DATASET_FALLBACK = "RSOC-Ship"
# grid division: rows×cols(default 4×4=16)
GRID_ROWS = 4
GRID_COLS = 4
# traversewholedatasetwhennotneedpopup
SHOW_ORIGINAL = False
SHOW_RESULT = False
# ---------- newly added: wholedatasettraverse & pseudo labeloutputconfiguration ----------
# selectneedtraverse objectdataset(each timerunyouchangeherei.e.can)
# optional:  "RSOC-Building", "RSOC-Ship", "RSOC-S-Vehicle", "RSOC-L-Vehicle", "VD-People", "VD-Vehicle"
TARGET_DATASET = "RSOC-S-Vehicle"
# selectneedhandle  split: 
# - "all": automatichandlethisdatasetcanuse all split(recommend)
# - orlist: ["train"] / ["train","val"] / ["test"] etc.
TARGET_SPLITS = "all"
# outputpseudo labeltext directory(willautomaticcreate): dataset/local pseudo
PSEUDO_OUT_DIRNAME = "local pseudo"
# outputfileisor notwith .txt extension(suggest True)
OUTPUT_WITH_TXT_EXT = True
# ---------- GD (Local GroundingDINO) parameter ----------
GD_BBOX_THRESHOLD = 0.13
GD_TEXT_THRESHOLD = 0.15
GDINO_MODEL_ID = os.getenv("GDINO_MODEL_ID", "grounding_dino_base")
# ---------- SAM3 parameter ----------
LOCAL_SAM3_PATH = r"D:\sam3_source"
SAM3_CHECKPOINT_PATH = "saved_models/SAM3.pt"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAM3_CONFIDENCE_THRESHOLD = 0.3
# ---------- count deduplication parameters ----------
DEDUP_IOU_THR = 0.4
MIN_BOX_AREA = 16.0
MIN_MASK_AREA = MIN_BOX_AREA  # mask areathreshold(pixelnumber), defaultand box thresholdconsistent
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
# SAM3 text prompt
SAM3_PROMPT_MAP = {
    "RSOC-Building": "building",
    "RSOC-S-Vehicle": "smallcar",
    "RSOC-L-Vehicle": "largecar.truck",
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
# ---------- optional: manualimageprompt box(wholedatasettraversenotsuggestenable) ----------
ENABLE_MANUAL_IMAGE_PROMPT = False
MANUAL_PROMPT_WINDOW_NAME = "Select SAM3 prompt box (ENTER/SPACE confirm, ESC cancel)"
FALLBACK_TO_CENTER_PROMPT_BOX = True
# ---------- newly added: Exemplar referencebox(10 image, each 1 box; supportjust/negativeprompt) ----------
# idea: firstrandom sample NUM_EXEMPLAR_IMAGES imageimage, letyou per imagemanualboxselect 1  ROI; 
#      thentakethese ROI with"mutualtocoordinate(normalization)+ just/negative label" formsavebelowcome; 
#      inaftercontinuesohasimage  SAM3 inferencestage(pass1 / pass2)takethese ROI map tocurrentimage, 
#      andandcurrent cell requestintersectionsetafterasadditionalgeometric promptboxadd(onlyaffect SAM3, notaffect GD). 
ENABLE_EXEMPLAR_REFERENCE = True
NUM_EXEMPLAR_IMAGES = 10
EXEMPLAR_SELECT_SPLIT = "train"      # fromwhich split samplingdo exemplar; None thenuse splits[0]
EXEMPLAR_RANDOM_SEED = 123
EXEMPLAR_ALLOW_NEGATIVE_LABEL = True # selectboxaftercaninput p/n annotationjust/negative(defaultjust)
EXEMPLAR_WINDOW_NAME = "Select exemplar ROI (ENTER/SPACE confirm, ESC cancel)"
# ---------- Exemplar match(take exemplar  "insidetolerate"ineach imagefind inmutualsimilarregion) ----------
EXEMPLAR_CROP_MAX_SIZE = 256             # exemplar crop maximumsidelong(shrinkcanraiseraisematchspeed)
EXEMPLAR_MATCH_TOPK_PER_EX = 4           # every exemplar inobjectimageinsidetake top-K matchbox
EXEMPLAR_MATCH_THRESHOLD = 0.55          # matchTemplate  threshold(exceedlargeexceedstrictcell)
EXEMPLAR_MATCH_SCALES = [0.7, 0.85, 1.0, 1.2, 1.4]  # manyscalematch
# ---------- newly added: random draw 5 imageoutputvisualize ----------
# Note:thisnotwillaffectpseudo label txt, onlyused foryoufastcheckeffect
SAVE_VIZ_SAMPLES = True
VIZ_NUM_IMAGES = 5
VIZ_OUT_SUBDIR = "viz_samples"
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
    """
    return list[dict]:
      { "idx": int(1..rows*cols), "row": r, "col": c, "xyxy": (x1,y1,x2,y2) }
    """
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
def _to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.array(x)
def _extract_sam3_outputs(state_or_output):
    """
    compatibledifferent SAM3 version outputfield: 
      - boxes: (N,4) xyxy
      - masks: (N,1,H,W) or (N,H,W)
      - scores: (N,)
    return dict: {"boxes": np.ndarray, "masks": np.ndarray[bool], "scores": np.ndarray}
    """
    if state_or_output is None:
        return {"boxes": np.zeros((0, 4), dtype=float),
                "masks": np.zeros((0, 1, 1), dtype=bool),
                "scores": np.zeros((0,), dtype=float)}
    if isinstance(state_or_output, dict):
        boxes = state_or_output.get("boxes", None)
        masks = state_or_output.get("masks", None)
        if masks is None:
            masks = state_or_output.get("mask", None)
        scores = state_or_output.get("scores", None)
    else:
        boxes = getattr(state_or_output, "boxes", None)
        masks = getattr(state_or_output, "masks", None)
        if masks is None:
            masks = getattr(state_or_output, "mask", None)
        scores = getattr(state_or_output, "scores", None)
    boxes_np = _to_numpy(boxes)
    if boxes_np is None:
        boxes_np = np.zeros((0, 4), dtype=float)
    boxes_np = boxes_np.astype(float).reshape(-1, 4) if boxes_np.size else np.zeros((0, 4), dtype=float)
    masks_np = _to_numpy(masks)
    if masks_np is None:
        masks_np = np.zeros((0, 1, 1), dtype=bool)
    # periodwaitshape (N,1,H,W) or (N,H,W)
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0, :, :]
    elif masks_np.ndim == 3:
        pass
    elif masks_np.ndim == 2:
        masks_np = masks_np[None, :, :]
    else:
        masks_np = np.zeros((0, 1, 1), dtype=bool)
    masks_np = (masks_np > 0.5) if masks_np.dtype != np.bool_ else masks_np.astype(bool)
    scores_np = _to_numpy(scores)
    if scores_np is None:
        scores_np = np.zeros((boxes_np.shape[0],), dtype=float)
    scores_np = scores_np.astype(float).reshape(-1) if scores_np.size else np.zeros((boxes_np.shape[0],), dtype=float)
    # alignlength
    n = int(min(boxes_np.shape[0], masks_np.shape[0])) if masks_np.shape[0] > 0 else int(boxes_np.shape[0])
    boxes_np = boxes_np[:n] if boxes_np.shape[0] >= n else np.zeros((0, 4), dtype=float)
    masks_np = masks_np[:n] if masks_np.shape[0] >= n else np.zeros((0, 1, 1), dtype=bool)
    scores_np = scores_np[:n] if scores_np.shape[0] >= n else np.zeros((n,), dtype=float)
    return {"boxes": boxes_np, "masks": masks_np, "scores": scores_np}
def sam3_text_plus_multi_boxes(processor, pil_img, boxes_xywh_list, prompt_text):
    """compatibleoldAPI: allviewforjustsample(label=True)"""
    labels = [True] * (len(boxes_xywh_list) if boxes_xywh_list is not None else 0)
    return sam3_text_plus_multi_boxes_with_labels(processor, pil_img, boxes_xywh_list, labels, prompt_text)
def sam3_text_plus_multi_boxes_with_labels(processor, pil_img, boxes_xywh_list, labels_list, prompt_text):
    """SAM3: text + multiple geometriesboxprompt(supporteveryboxhasjust/negative label). return masks/boxes/scores. """
    if boxes_xywh_list is None:
        boxes_xywh_list = []
    if labels_list is None:
        labels_list = [True] * len(boxes_xywh_list)
    width, height = pil_img.size
    state = processor.set_image(pil_img)
    state = processor.set_text_prompt(state=state, prompt=prompt_text)
    for (xywh, lab) in zip(boxes_xywh_list, labels_list):
        x, y, w, h = xywh
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
        state = processor.add_geometric_prompt(box=norm_box_cxcywh, label=bool(lab), state=state)
    return _extract_sam3_outputs(state)
def sam3_text_only(processor, pil_img, prompt_text):
    """SAM3: puretext prompt. return masks/boxes/scores. """
    state = processor.set_image(pil_img)
    output = processor.set_text_prompt(state=state, prompt=prompt_text)
    return _extract_sam3_outputs(output)
# ===================== manualprompt box(retainbutwholedatasettraversenotsuggestenable) =====================
def _default_center_rel_box():
    return (0.25, 0.25, 0.5, 0.5)
def rel_xywh_to_abs_xywh(rel_xywh, img_w, img_h):
    rx, ry, rw, rh = rel_xywh
    x = int(round(rx * img_w))
    y = int(round(ry * img_h))
    w = int(round(rw * img_w))
    h = int(round(rh * img_h))
    x = max(0, min(int(img_w - 1), x))
    y = max(0, min(int(img_h - 1), y))
    w = max(1, min(int(img_w - x), w))
    h = max(1, min(int(img_h - y), h))
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
# ===================== Exemplar referencebox: manualselect 10 image, each 1 box =====================
# ===================== Mask deduplication(according to mask IoU) =====================
def _bbox_from_mask(mask_bool):
    ys, xs = np.where(mask_bool)
    if xs.size == 0 or ys.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)
    return [x1, y1, x2, y2]
def _mask_area(mask_bool):
    return float(np.count_nonzero(mask_bool))
def mask_iou(a_bool, b_bool):
    inter = np.logical_and(a_bool, b_bool).sum()
    if inter <= 0:
        return 0.0
    union = np.logical_or(a_bool, b_bool).sum()
    return float(inter / union) if union > 0 else 0.0
def nms_dedup_instances(instances, iou_thr=0.8, min_area=10):
    """
    instances: list[{"mask": np.ndarray[bool], "box": [x1,y1,x2,y2], "score": float}]
    firstaccording to score descending(ifno score thenaccording toareadescending), againdo greedy NMS. 
    """
    if not instances:
        return []
    # filtersmallarea
    kept = []
    for ins in instances:
        m = ins.get("mask", None)
        if m is None:
            continue
        area = _mask_area(m)
        if area < float(min_area):
            continue
        ins["_area"] = area
        if ins.get("box", None) is None or len(ins.get("box", [])) != 4:
            ins["box"] = _bbox_from_mask(m)
        kept.append(ins)
    if not kept:
        return []
    def key_fn(ins):
        sc = ins.get("score", None)
        if sc is None:
            return float(ins.get("_area", 0.0))
        return float(sc)
    kept.sort(key=key_fn, reverse=True)
    final = []
    for ins in kept:
        m = ins["mask"]
        bx = ins["box"]
        discard = False
        for win in final:
            # firstuse bbox fastfilter
            if iou_xyxy(bx, win["box"]) < (iou_thr * 0.25):
                continue
            if mask_iou(m, win["mask"]) >= float(iou_thr):
                discard = True
                break
        if not discard:
            final.append(ins)
    # cleanuptemporarywhenfield
    for ins in final:
        if "_area" in ins:
            del ins["_area"]
    return final
def _instances_from_sam3_output(out, img_w, img_h):
    """
    out: {"masks": (N,H,W) bool, "boxes": (N,4) xyxy, "scores": (N,)}
    """
    masks = out.get("masks", None)
    boxes = out.get("boxes", None)
    scores = out.get("scores", None)
    if masks is None or masks.size == 0:
        return []
    n = int(masks.shape[0])
    instances = []
    for i in range(n):
        m = masks[i].astype(bool)
        b = None
        if boxes is not None and boxes.shape[0] > i:
            b = clamp_box_xyxy(list(map(float, boxes[i])), img_w, img_h)
        else:
            b = _bbox_from_mask(m)
        sc = float(scores[i]) if (scores is not None and scores.shape[0] > i) else None
        instances.append({"mask": m, "box": b, "score": sc})
    return instances
def save_instance_label_png(instances, out_png_path, h, w):
    """
    saveone uint16  instancelabelimage: 
      0=background, 1..N=instance id
    """
    label = np.zeros((h, w), dtype=np.uint16)
    for idx, ins in enumerate(instances, start=1):
        m = ins.get("mask", None)
        if m is None:
            continue
        label[m.astype(bool)] = np.uint16(idx)
    img = Image.fromarray(label, mode="I;16")
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    img.save(out_png_path)
def _select_roi_on_image(img_bgr, window_name: str):
    """OpenCV manualboxselect ROI, return (x,y,w,h) or None. """
    try:
        roi = cv2.selectROI(window_name, img_bgr, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(window_name)
    except Exception:
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass
        return None
    x, y, w, h = roi
    if w is None or h is None or int(w) <= 0 or int(h) <= 0:
        return None
    return (int(x), int(y), int(w), int(h))
def _abs_xywh_to_rel_xywh(abs_xywh, img_w, img_h):
    x, y, w, h = abs_xywh
    img_w = max(1, int(img_w))
    img_h = max(1, int(img_h))
    rx = float(x) / float(img_w)
    ry = float(y) / float(img_h)
    rw = float(w) / float(img_w)
    rh = float(h) / float(img_h)
    # clamp
    rx = max(0.0, min(1.0, rx))
    ry = max(0.0, min(1.0, ry))
    rw = max(0.0, min(1.0, rw))
    rh = max(0.0, min(1.0, rh))
    return (rx, ry, rw, rh)
def _rel_xywh_to_abs_xyxy(rel_xywh, img_w, img_h):
    x, y, w, h = rel_xywh_to_abs_xywh(rel_xywh, img_w, img_h)
    return (x, y, x + w, y + h)
def _intersect_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)
def select_exemplar_boxes(dataset_label: str, split_for_exemplar: str, num_images: int):
    """randomselectselect num_images image, image by imagemanualboxselect 1  ROI(optionaljust/negativelabel). 
    [thisversion: take"positionprompt"change to"exampleprompt"]
    - notagaintake ROI recordbecome rel_xywh thenmap toarbitraryimage(thatwillbecome"fixedpositionprompt"). 
    - changeforsave ROI  "insidetolerate(crop)", aftercontinuewillineachwaithandleimageinsideuse template matching findtomutualsimilarregion, 
      againtakefindto candidateboxas SAM3  geometric prompt(just/negative)use. 
    return list[dict]:
      {
        "crop_bgr": np.ndarray,       # exemplar  insidetolerate(BGR)
        "crop_gray": np.ndarray,      # grayscale(used formatch)
        "label": bool,                # True=justsample, False=negativesample
        "src_image": str,             # fromwhichimage
        "roi_abs_xywh": (x,y,w,h),    # onlyused forrecord/debug
      }
    """
    img_list = _list_images_for_dataset_split(dataset_label, split_for_exemplar)
    if not img_list:
        print(f"[Exemplar] split={split_for_exemplar} nofindtoimage, nomethodselect exemplar. ")
        return []
    rnd = random.Random(int(EXEMPLAR_RANDOM_SEED))
    img_list = list(img_list)
    rnd.shuffle(img_list)
    picked = []
    need = max(0, int(num_images))
    print(f"\n[Exemplar] willfrom split={split_for_exemplar} inrandom sample {need} imageimage, letyou per imagemanualboxselect 1  ROI. ")
    print("           Operation: drag to draw a box -> ENTER/SPACE confirm;ESC takeeliminatethisimage. ")
    for img_path in img_list:
        if len(picked) >= need:
            break
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        base = os.path.basename(img_path)
        print(f"\n[Exemplar] ({len(picked)+1}/{need}) select: {base}")
        roi = _select_roi_on_image(img_bgr, EXEMPLAR_WINDOW_NAME)
        if roi is None:
            print("[Exemplar] takeeliminate/notselectin ROI, skipthis image. ")
            continue
        label = True
        if bool(EXEMPLAR_ALLOW_NEGATIVE_LABEL):
            try:
                ans = input("  label? input p(just)/n(negative), default p: ").strip().lower()
                if ans == "n":
                    label = False
            except Exception:
                label = True
        x, y, ww, hh = roi
        x = max(0, min(int(w - 1), int(x)))
        y = max(0, min(int(h - 1), int(y)))
        ww = max(1, min(int(w - x), int(ww)))
        hh = max(1, min(int(h - y), int(hh)))
        crop = img_bgr[y:y+hh, x:x+ww].copy()
        if crop.size == 0:
            print("[Exemplar] ROI crop forempty, skip. ")
            continue
        # in order tomatchspeed, take exemplar crop limitationtomaximumsidelong EXEMPLAR_CROP_MAX_SIZE
        if int(EXEMPLAR_CROP_MAX_SIZE) > 0:
            max_side = max(crop.shape[0], crop.shape[1])
            if max_side > int(EXEMPLAR_CROP_MAX_SIZE):
                scale = float(EXEMPLAR_CROP_MAX_SIZE) / float(max_side)
                new_w = max(1, int(round(crop.shape[1] * scale)))
                new_h = max(1, int(round(crop.shape[0] * scale)))
                crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        picked.append({
            "crop_bgr": crop,
            "crop_gray": crop_gray,
            "label": bool(label),
            "src_image": base,
            "roi_abs_xywh": (int(x), int(y), int(ww), int(hh)),
        })
        print(f"[Exemplar] alreadyrecord crop shape={crop.shape[:2]} | label={'pos' if label else 'neg'}")
    if len(picked) < need:
        print(f"[Exemplar][Warn] onlyselectto  {len(picked)}/{need}  exemplar, willusealreadyselect continuerun. ")
    return picked
def match_exemplars_to_image(img_bgr, exemplar_list):
    """take exemplar  "insidetolerate"incurrentimageinsidedomatch, return prompts_xywh + prompts_labels. """
    if not exemplar_list:
        return [], []
    h, w = img_bgr.shape[:2]
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    prompts_xywh = []
    prompts_labels = []
    scales = list(EXEMPLAR_MATCH_SCALES) if isinstance(EXEMPLAR_MATCH_SCALES, (list, tuple)) else [1.0]
    topk = max(0, int(EXEMPLAR_MATCH_TOPK_PER_EX))
    thr = float(EXEMPLAR_MATCH_THRESHOLD)
    for ex in exemplar_list:
        tmpl = ex.get("crop_gray", None)
        lab = bool(ex.get("label", True))
        if tmpl is None:
            continue
        best = []  # list of (score, x, y, tw, th)
        th0, tw0 = tmpl.shape[:2]
        for s in scales:
            tw = max(8, int(round(tw0 * float(s))))
            th = max(8, int(round(th0 * float(s))))
            if tw >= w or th >= h:
                continue
            tmpl_s = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_AREA) if (tw != tw0 or th != th0) else tmpl
            res = cv2.matchTemplate(img_gray, tmpl_s, cv2.TM_CCOEFF_NORMED)
            # take topk peakvalue
            for _ in range(topk):
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if max_val < thr:
                    break
                x, y = max_loc
                best.append((float(max_val), int(x), int(y), int(tw), int(th)))
                # takethisonepieceregionerasedrop, avoidrepeattaketosameonepeak
                x1 = max(0, x - tw // 2)
                y1 = max(0, y - th // 2)
                x2 = min(res.shape[1], x + tw // 2)
                y2 = min(res.shape[0], y + th // 2)
                res[y1:y2, x1:x2] = -1.0
        # write prompts
        best.sort(key=lambda t: t[0], reverse=True)
        for (score, x, y, tw, th) in best[:topk]:
            # clamp
            x = max(0, min(int(w - 1), x))
            y = max(0, min(int(h - 1), y))
            tw = max(1, min(int(w - x), tw))
            th = max(1, min(int(h - y), th))
            prompts_xywh.append((int(x), int(y), int(tw), int(th)))
            prompts_labels.append(bool(lab))
    return prompts_xywh, prompts_labels
# ===================== datasettraverse: columnimage + DOTA_data filter =====================
def list_images(img_dir):
    if not img_dir or (not os.path.isdir(img_dir)):
        return []
    return [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(IMAGE_EXTS)]
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
def _dataset_label_to_outname(dataset_label: str) -> str:
    m = {
        "RSOC-Building": "rsoc-building",
        "RSOC-Ship": "rsoc-ship",
        "RSOC-S-Vehicle": "rsoc-s-vehicle",
        "RSOC-L-Vehicle": "rsoc-l-vehicle",
        "VD-People": "vd-people",
        "VD-Vehicle": "vd-vehicle",
    }
    return m.get(dataset_label, str(dataset_label).lower())
def _available_splits_for_dataset(dataset_label: str):
    if dataset_label == "RSOC-Building":
        return ["train", "test"]
    if dataset_label in ["RSOC-Ship", "RSOC-S-Vehicle", "RSOC-L-Vehicle"]:
        return ["train", "val", "test"]
    if dataset_label in ["VD-People", "VD-Vehicle"]:
        return ["train", "val", "test"]
    return ["train", "val", "test"]
def _build_allow_set_for_rsoc_mixed(dataset_label: str, split: str):
    cat_token = {
        "RSOC-Ship": "ship",
        "RSOC-S-Vehicle": "small-vehicle",
        "RSOC-L-Vehicle": "large-vehicle",
    }[dataset_label]
    dota_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "DOTA_data")
    allow = None
    # 1) DOTA_data list priority
    list_files = _find_dota_list_files(dota_dir, split, cat_token)
    if list_files:
        allow = set()
        for lp in list_files:
            allow |= _read_dota_name_list(lp)
    # 2) train/val degenerate scan labelTxt(ifmissing list)
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
def _process_one_image_full(img_path, processor, dataset_label, exemplar_list=None):
    """
    [whole imageversion]
    - notagain 4x4 gridcropcut, GD / SAM3 allinoriginalimageonrun
    - SAM3 outputuse mask(instanceset), anddo 3 timesdeduplication: 
        (1) pass1(GD+prompt) internaldeduplication
        (2) pass2(allimage/plain textor exemplar prompt) internaldeduplication
        (3) pass1+pass2 combineandafteragaintimesdeduplication
    Returns:final_instances(list[{"mask","box","score"}])
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return []
    h, w = img_bgr.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    gd_prompt = GD_PROMPT_MAP.get(dataset_label, "object")
    sam_prompt = SAM3_PROMPT_MAP.get(dataset_label, "object")
    # exemplar prompts: use"insidetoleratematch"getthisimageon candidatebox(just/negative)
    exemplar_prompts_xywh, exemplar_prompts_labels = [], []
    if bool(ENABLE_EXEMPLAR_REFERENCE) and exemplar_list:
        exemplar_prompts_xywh, exemplar_prompts_labels = match_exemplars_to_image(img_bgr, exemplar_list)
    # 1) GD(whole image)
    gd_boxes = call_dino_local_on_pil(pil_img, gd_prompt)
    gd_boxes = [clamp_box_xyxy(b, w, h) for b in gd_boxes if isinstance(b, (list, tuple)) and len(b) == 4]
    # 2) SAM3 pass1(GD boxprompt + exemplar prompts)
    prompts_xywh = []
    prompts_labels = []
    # optional: retain"manualprompt box"(defaultdisable)
    if ENABLE_MANUAL_IMAGE_PROMPT:
        manual_prompt_rel = _default_center_rel_box() if FALLBACK_TO_CENTER_PROMPT_BOX else None
        if manual_prompt_rel is not None:
            prompts_xywh.append(rel_xywh_to_abs_xywh(manual_prompt_rel, w, h))
            prompts_labels.append(True)
    for b in gd_boxes:
        xx1, yy1, xx2, yy2 = [int(round(v)) for v in b]
        ww = max(1, xx2 - xx1)
        hh = max(1, yy2 - yy1)
        prompts_xywh.append((xx1, yy1, ww, hh))
        prompts_labels.append(True)
    if exemplar_prompts_xywh:
        prompts_xywh.extend(exemplar_prompts_xywh)
        prompts_labels.extend(exemplar_prompts_labels)
    out1 = {"masks": np.zeros((0, 1, 1), dtype=bool), "boxes": np.zeros((0, 4), dtype=float), "scores": np.zeros((0,), dtype=float)}
    if len(prompts_xywh) > 0:
        out1 = sam3_text_plus_multi_boxes_with_labels(processor, pil_img, prompts_xywh, prompts_labels, sam_prompt)
    inst1 = _instances_from_sam3_output(out1, w, h)
    inst1 = nms_dedup_instances(inst1, iou_thr=DEDUP_IOU_THR, min_area=MIN_MASK_AREA)
    # 3) SAM3 pass2(allimage: exemplar prompts or plain text)
    out2 = None
    if exemplar_prompts_xywh:
        out2 = sam3_text_plus_multi_boxes_with_labels(processor, pil_img, exemplar_prompts_xywh, exemplar_prompts_labels, sam_prompt)
    else:
        out2 = sam3_text_only(processor, pil_img, sam_prompt)
    inst2 = _instances_from_sam3_output(out2, w, h)
    inst2 = nms_dedup_instances(inst2, iou_thr=DEDUP_IOU_THR, min_area=MIN_MASK_AREA)
    # 4) merge & dedup
    merged = inst1 + inst2
    final_instances = nms_dedup_instances(merged, iou_thr=DEDUP_IOU_THR, min_area=MIN_MASK_AREA)
    return final_instances
# =============== visualize(mask overlay) ===============
def _draw_masks_on_image(img_bgr, instances, alpha=0.45):
    if img_bgr is None:
        return None
    if not instances:
        return img_bgr
    h, w = img_bgr.shape[:2]
    overlay = img_bgr.copy()
    # generatesomecolor(fixed seed, avoideach timedifferent)
    rnd = np.random.RandomState(123)
    colors = rnd.randint(0, 255, size=(max(1, len(instances)), 3), dtype=np.uint8)
    for i, ins in enumerate(instances):
        m = ins.get("mask", None)
        if m is None:
            continue
        if m.shape[0] != h or m.shape[1] != w:
            # sizenotconsistentthenskip(logicdiscussonnotthishappen)
            continue
        color = colors[i % colors.shape[0]]
        overlay[m.astype(bool)] = (overlay[m.astype(bool)] * (1 - alpha) + color * alpha).astype(np.uint8)
        # draw bbox theninview(optional)
        bx1, by1, bx2, by2 = [int(round(v)) for v in ins.get("box", [0, 0, 0, 0])]
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (int(color[0]), int(color[1]), int(color[2])), 2)
    return overlay
def _draw_boxes_on_image(img_bgr, boxes_xyxy, color=(0, 0, 255), thickness=2):
    if img_bgr is None:
        return None
    out = img_bgr.copy()
    if boxes_xyxy:
        for b in boxes_xyxy:
            x1, y1, x2, y2 = [int(round(v)) for v in b]
            x1 = max(0, min(out.shape[1]-1, x1))
            x2 = max(0, min(out.shape[1]-1, x2))
            y1 = max(0, min(out.shape[0]-1, y1))
            y2 = max(0, min(out.shape[0]-1, y2))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out

def calc_time(start, end):
    run_time = end - start
    # convertfor when:part:second.millisecond
    m, s = divmod(run_time, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:.3f}"


def main():
    dataset_label = str(TARGET_DATASET).strip()
    if dataset_label not in SAM3_PROMPT_MAP:
        print(f"[Error] TARGET_DATASET notvalid: {dataset_label}")
        print(f"optional: {list(SAM3_PROMPT_MAP.keys())}")
        return
    out_dir = os.path.join(ROOT_DATA_DIR, PSEUDO_OUT_DIRNAME)
    os.makedirs(out_dir, exist_ok=True)
    avail = _available_splits_for_dataset(dataset_label)
    if TARGET_SPLITS == "all":
        splits = avail
    else:
        splits = [s for s in TARGET_SPLITS if s in avail]
    if not splits:
        print(f"[Error] nocanhandle  split. avail={avail} | TARGET_SPLITS={TARGET_SPLITS}")
        return
    print(f"[Main] ROOT_DATA_DIR={ROOT_DATA_DIR}")
    print(f"[Main] TARGET_DATASET={dataset_label} | splits={splits} | mode=full_image")
    print(f"[Main] DEVICE={DEVICE} | GDINO_MODEL_ID={GDINO_MODEL_ID}")
    # onceload, allprocessreuse
    processor = build_sam3_processor()
    # select exemplar referencebox(10 image, each 1 box), onlyasused for SAM3
    exemplar_list = []
    if bool(ENABLE_EXEMPLAR_REFERENCE):
        split_for_exemplar = EXEMPLAR_SELECT_SPLIT
        if (split_for_exemplar is None) or (split_for_exemplar not in avail):
            split_for_exemplar = splits[0]
        exemplar_list = select_exemplar_boxes(dataset_label, split_for_exemplar, int(NUM_EXEMPLAR_IMAGES))
        print(f"[Exemplar] total exemplars = {len(exemplar_list)}")
    out_base = _dataset_label_to_outname(dataset_label)
    grid_tag = "full"

    print("startcountwhen")
    start_time = time.perf_counter()

    for split in splits:
        img_list = _list_images_for_dataset_split(dataset_label, split)
        if not img_list:
            print(f"[Warn] {dataset_label} split={split} nofindtocanhandleimage(maybelack DOTA_data listordirectorynotsavein). ")
            continue
        fname = f"{out_base}_mask_{split}"
        if OUTPUT_WITH_TXT_EXT:
            fname += ".txt"
        out_fp = os.path.join(out_dir, fname)

        # additionalOutput:andoldversionconsistent  bbox label(one per lineobject)
        bbox_fname = f"{out_base}_bbox_{split}"
        if OUTPUT_WITH_TXT_EXT:
            bbox_fname += ".txt"
        bbox_out_fp = os.path.join(out_dir, bbox_fname)
        mask_dir = os.path.join(out_dir, "masks", f"{out_base}_{split}")
        os.makedirs(mask_dir, exist_ok=True)
        print(f"\n[Run] split={split} | images={len(img_list)} | out={out_fp} | bbox_out={bbox_out_fp}")
        # random samplefewquantitysampledovisualizeoutput(notaffectpseudolabel file)
        viz_pick_set = set()
        viz_dir = None
        if bool(SAVE_VIZ_SAMPLES) and int(VIZ_NUM_IMAGES) > 0:
            rnd = random.Random(2026 + len(img_list))
            k = min(int(VIZ_NUM_IMAGES), len(img_list))
            viz_pick_set = set(rnd.sample(img_list, k=k))
            viz_dir = os.path.join(out_dir, VIZ_OUT_SUBDIR, f"{out_base}_{grid_tag}_{split}")
            os.makedirs(viz_dir, exist_ok=True)
            print(f"[Viz] will save {k} samples to: {viz_dir}")
        def _clip_int_box_xyxy(box_xyxy, w, h):
            x1, y1, x2, y2 = [int(round(float(v))) for v in box_xyxy]
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h - 1, y2))
            # guarantee x2>x1, y2>y1(at least 1 pixel)
            if x2 <= x1:
                x2 = min(w - 1, x1 + 1)
            if y2 <= y1:
                y2 = min(h - 1, y1 + 1)
            return x1, y1, x2, y2

        with open(out_fp, "w", encoding="utf-8") as f, open(bbox_out_fp, "w", encoding="utf-8") as fb:
            for i, img_path in enumerate(img_list, start=1):
                img_name = os.path.basename(img_path)
                # handlewhole image: returnfinalinstance masks(alreadydeduplication)
                instances = _process_one_image_full(img_path, processor, dataset_label, exemplar_list=exemplar_list)
                # save mask label png(0=background, 1..N=instance id)
                mask_png_name = os.path.splitext(img_name)[0] + ".png"
                mask_png_path = os.path.join(mask_dir, mask_png_name)
                img_bgr = cv2.imread(img_path)
                if img_bgr is not None:
                    hh, ww = img_bgr.shape[:2]
                else:
                    # OpenCV occasionalreadimagefailwhen, degenerateto PIL getsize(notaffect bbox output)
                    try:
                        with Image.open(img_path) as _im:
                            ww, hh = _im.size
                    except Exception:
                        ww, hh = None, None

                if (hh is not None) and (ww is not None):
                    save_instance_label_png(instances, mask_png_path, hh, ww)
                    # additionaloutput bbox label: everyinstanceonerow(IMG.jpg x1 y1 x2 y2)
                    for ins in instances:
                        box = ins.get("box", None)
                        if box is None and ins.get("mask", None) is not None:
                            box = _bbox_from_mask(ins["mask"])
                        if box is None:
                            continue
                        x1, y1, x2, y2 = _clip_int_box_xyxy(box, ww, hh)
                        fb.write(f"{img_name} {x1} {y1} {x2} {y2}\n")
                # txt per line: image_name  mask_png_relpath  count
                rel_mask = os.path.relpath(mask_png_path, out_dir).replace("\\", "/")
                f.write(f"{img_name} {rel_mask} {len(instances)}\\n")
                # visualizesample: takeinstance mask overlaytooriginalimageandsave(onlysampling)
                if viz_dir is not None and img_path in viz_pick_set:
                    img_bgr2 = cv2.imread(img_path)
                    if img_bgr2 is not None:
                        vis = _draw_masks_on_image(img_bgr2, instances, alpha=0.45)
                        if vis is not None:
                            out_name = os.path.splitext(img_name)[0] + f"_n{len(instances)}.jpg"
                            cv2.imwrite(os.path.join(viz_dir, out_name), vis)
                if i % 20 == 0 or i == len(img_list):
                    print(f"  progress: {i}/{len(img_list)}")

    end_time = time.perf_counter()

    print(f"coderuntotalconsumewhen: {calc_time(start_time, end_time)}")
    print("\n[Done] alreadyfinishwholedatasettraverseandgeneratepseudo labeltext. ")
if __name__ == "__main__":
    main()