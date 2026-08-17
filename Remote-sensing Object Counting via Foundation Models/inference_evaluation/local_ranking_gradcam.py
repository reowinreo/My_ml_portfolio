# -*- coding: utf-8 -*-
"""
GD + SAM3 dual-path object counting + 16 regionlocal ranking(Grid version)

youneed change(mutualtoyouoriginalscript"firstgdagainSAM3_cut_imageprompt box.py"): 
1) onlyhandle"oneimageimage"(caninconfiguration sectiondirectspecify SINGLE_IMAGE_PATH). 
2) notagainrandomcrop 512×512 ROI; andisin"originalimage"onaccording togrid divisionbecome rows×cols(default 4×4=16)region. 
3) toeveryregion(cell)partdo notexecute: GD(local GroundingDINO)+ SAM3 twice(pass1 use GD boxprompt; pass2 plain textnotuseanyprompt box)
   thenaccording toeveryregion combineandcount(final_count)to 16 regionranking, outputleaderboardandvisualize. 

remark: 
- thisscriptas much as possiblereuseyouoriginalprogram : local GroundingDINO(Transformers) + local SAM3 + deduplicationlogic + Windows dummy triton/flash_attn fix. 
- defaultdisable"manualimageprompt box", avoidandyou proposed "pass2 notuseanyprompt box"conflict; ifneedcanselfrowopen(onlyparameterand pass1). 

Usage:
    python gd_sam3_grid_ranking.py

Output:
- controlplatformprint: predictleaderboard + real labelleaderboard(ifhas GT)
- optionalpopup: finalvisualizeresult(SHOW_RESULT=True), notsave
"""

import os
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont

# optional: used forread RSOC_building   .mat realvalue(ifnotinstall scipy, willinrunwhenprompt)
try:
    import scipy.io as sio  # type: ignore
except Exception:
    sio = None
# ===================== user configuration section =====================

# datasetroot directory(youto is dataset; scriptwillautomaticin Dataset/dataset adaptive intervalshould)
ROOT_DATA_DIR = "dataset"

# onlyhandlethisoneimageimage(recommendyoudirectinherefill incompletepath)
# example: r"dataset\ASPDNet_dataset\RSOC_building\building\test_data\images\IMG_0104005.jpg"
SINGLE_IMAGE_PATH = r""

# if SINGLE_IMAGE_PATH forempty, thenfromspecifydatasetinsiderandomselect 1 image(optional)
# "RSOC-Building", "RSOC-S-Vehicle", "RSOC-L-Vehicle", "RSOC-Ship", "VD-People", "VD-Vehicle"
TARGET_DATASET_FALLBACK = "RSOC-L-Vehicle"

# grid division: rows×cols(default 4×4=16, canadjust)
GRID_ROWS = 4
GRID_COLS = 4

# visualize: isor notinrunbeforepopuporiginalimage(PIL defaultviewer)
SHOW_ORIGINAL = True

# Output:according to your needrequest[notsaveanything](notwrite txt, notsaveimage, notfalldiskarrangename). 
# ifneedviewfinalvisualizeresult, canenable SHOW_RESULT(onlypopup, notsave). 
SHOW_RESULT = False
# ---------- GD (Local GroundingDINO) parameter ----------
GD_BBOX_THRESHOLD = 0.13
GD_TEXT_THRESHOLD = 0.15
# HuggingFace model id orlocaldirectory(awaylineenvironmentrecommend)
GDINO_MODEL_ID = os.getenv("GDINO_MODEL_ID", "grounding_dino_base")

# ---------- SAM3 parameter ----------
LOCAL_SAM3_PATH = r"D:\sam3_source"
SAM3_CHECKPOINT_PATH = "saved_models/SAM3.pt"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAM3_CONFIDENCE_THRESHOLD = 0.3

# ---------- count deduplication parameters ----------
DEDUP_IOU_THR = 0.6   # exceedlargeexceed"easycombineandrepeatbox"; 0.5~0.7 oftenuse
MIN_BOX_AREA = 16.0   # filterextremesmallnoisebox(pixel^2)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# SAM3 text prompt (more biased toward"concept")
SAM3_PROMPT_MAP = {
    "RSOC-Building": "building",
    "RSOC-S-Vehicle": "smallcar",
    "RSOC-L-Vehicle": "largecar.truck",
    "RSOC-Ship": "ship.boat",
    "VD-People": "person",
    "VD-Vehicle": "vehicle",
}

# GD text prompt (more biased toward"detectionword")
GD_PROMPT_MAP = {
    "RSOC-Building": "structure.building.house",
    "RSOC-S-Vehicle": "car",
    "RSOC-L-Vehicle": "truck",
    "RSOC-Ship": "ship.boat",
    "VD-People": "person",
    "VD-Vehicle": "vehicle",
}

# ---------- optional: manualimageprompt box(onlyparameterand pass1; pass2 foreverplain text) ----------
ENABLE_MANUAL_IMAGE_PROMPT = False
MANUAL_PROMPT_WINDOW_NAME = "Select SAM3 prompt box (ENTER/SPACE confirm, ESC cancel)"
FALLBACK_TO_CENTER_PROMPT_BOX = True


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


def show_original_with_pil(img_path):
    try:
        Image.open(img_path).show()
    except Exception as e:
        print(f"[Warn] nomethodshoworiginalimage: {e}")





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
            # fallback: ensureat least 2x2
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
    # GroundingDINO commonwriting style: phrasebetweenuse ' . ' partinterval, andwith ' .' end
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
    """
    return boxes: list[[x1,y1,x2,y2], ...] (pixel coordinates, xyxy), coordinate systemand img_rgb consistent. 
    """
    processor, model, device = _load_local_gd()
    h, w = img_rgb.shape[:2]
    prompt = _ensure_gd_prompt(prompt_text)

    inputs = processor(images=img_rgb, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[h, w]], device=device)

    # compatiblenewold post_process_grounded_object_detection API
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

# ---- Windows dummy triton / flash_attn(alonguseyou scriptstyle) ----
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

# import SAM3
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


def sam3_text_plus_box(processor, pil_img, roi_xywh, prompt_text):
    """
    text + geometryboxprompt, inwholeimage pil_img onoutputinstance boxes (xyxy, pixel coordinates)
    """
    width, height = pil_img.size
    state = processor.set_image(pil_img)
    state = processor.set_text_prompt(state=state, prompt=prompt_text)

    x, y, w, h = roi_xywh
    if w <= 0 or h <= 0:
        return np.zeros((0, 4), dtype=float)

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


def sam3_text_plus_multi_boxes(processor, pil_img, boxes_xywh_list, prompt_text, return_state=False):
    """
    text + multiple geometriesboxTip:once set_image + set_text_prompt, againaccordingtimes add_geometric_prompt. 
    return boxes: [M,4] (xyxy)
    - return_state=True when, meanwhilereturninternal state/output so thatraisetake masks/logits constructheatmap
    """
    width, height = pil_img.size
    state = processor.set_image(pil_img)
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
        boxes_np = np.zeros((0, 4), dtype=float)
        return (boxes_np, state) if return_state else boxes_np

    if torch.is_tensor(boxes):
        boxes_np = boxes.detach().cpu().numpy()
    else:
        boxes_np = np.array(boxes)

    boxes_np = boxes_np.astype(float)
    return (boxes_np, state) if return_state else boxes_np

def sam3_text_only(processor, pil_img, prompt_text, return_state=False):
    """
    plain text: notaddanypoint/box/maskprompt. 
    return boxes: [M, 4] (xyxy)
    - return_state=True when, meanwhilereturninternal output so thatraisetake masks/logits constructheatmap
    """
    state = processor.set_image(pil_img)
    output = processor.set_text_prompt(state=state, prompt=prompt_text)

    boxes = output.get("boxes", None) if isinstance(output, dict) else None
    if boxes is None:
        boxes = getattr(output, "boxes", None) if output is not None else None
    if boxes is None:
        boxes_np = np.zeros((0, 4), dtype=float)
        return (boxes_np, output) if return_state else boxes_np

    if torch.is_tensor(boxes):
        boxes_np = boxes.detach().cpu().numpy()
    else:
        boxes_np = np.array(boxes)

    boxes_np = boxes_np.astype(float)
    return (boxes_np, output) if return_state else boxes_np


def sam3_output_to_heatmap(output_or_state, out_h: int, out_w: int, fallback_boxes_xyxy=None):
    """
    sidecaseA: take SAM3 outputinside masks / logits / prob maps turnbecomepixellevelheatmap H (float32, 0~1), andtomanyinstancedo max combineand. 
    - ifoutputinsidetakenottohaseffect mask(orforempty), thendegenerateforuse fallback_boxes_xyxy inboxinsidefill 1.0 generatecoarseheatimage. 
    - key: mustto"emptyinstance (N=0)"dofallback, else arr.min()/arr.max() willreportwrong. 
    """
    H = None

    def _to_numpy(x):
        if x is None:
            return None
        if torch.is_tensor(x):
            x = x.detach().float().cpu().numpy()
        else:
            x = np.array(x)
        return x

    def _sigmoid(z):
        z = np.clip(z, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-z))

    # 1) tryfrom output/state infind masks or logits
    cand = None
    if isinstance(output_or_state, dict):
        for k in ["mask_probs", "masks", "pred_masks", "masks_pred", "mask_logits", "logits", "pred_logits"]:
            if k in output_or_state and output_or_state[k] is not None:
                cand = output_or_state[k]
                break
    else:
        for k in ["mask_probs", "masks", "pred_masks", "mask_logits", "logits"]:
            if hasattr(output_or_state, k):
                v = getattr(output_or_state, k)
                if v is not None:
                    cand = v
                    break

    arr = _to_numpy(cand)

    # ✅ fallback: emptyarray / emptyinstancedirectgo fallback
    if arr is not None:
        try:
            if arr.size == 0:
                arr = None
        except Exception:
            arr = None

    if arr is not None:
        # arr maybe (N,H,W) or (H,W) or (N,1,H,W)
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0, :, :]
        # againtimescheck: N=0
        if arr.ndim >= 3 and arr.shape[0] == 0:
            arr = None

    if arr is not None:
        if arr.ndim == 2:
            arr = arr[None, :, :]
        if arr.ndim == 3:
            # ✅ againtimescheck: N=0
            if arr.shape[0] == 0:
                arr = None

    if arr is not None and arr.ndim == 3:
        # judgeisnotis logits: notin [0,1] maybethenis logits
        if (float(arr.min()) < 0.0) or (float(arr.max()) > 1.0):
            arr = _sigmoid(arr)
        else:
            arr = np.clip(arr, 0.0, 1.0)

        # combineandmanyinstance: max
        H = np.max(arr, axis=0).astype(np.float32)

        # sizealign(somemodelwilloutputlowresolution mask)
        if H.shape[0] != out_h or H.shape[1] != out_w:
            H = cv2.resize(H, (int(out_w), int(out_h)), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    # 2) degenerate: use boxes fillfillheatimage
    if H is None:
        H = np.zeros((int(out_h), int(out_w)), dtype=np.float32)
        if fallback_boxes_xyxy is not None:
            try:
                # numpy (0,4) alsocandirectiteration; foremptythennotwillenterenterloop
                for b in fallback_boxes_xyxy:
                    x1, y1, x2, y2 = [int(round(float(v))) for v in b]
                    x1 = max(0, min(out_w - 1, x1))
                    y1 = max(0, min(out_h - 1, y1))
                    x2 = max(0, min(out_w, x2))
                    y2 = max(0, min(out_h, y2))
                    if x2 > x1 and y2 > y1:
                        H[y1:y2, x1:x2] = 1.0
            except Exception:
                pass

    return H


def _default_center_rel_box():
    # center 50% region: x,y,w,h(mutualtocoordinate)
    return (0.25, 0.25, 0.5, 0.5)


def select_prompt_box_rel_on_image(pil_img, window_name=MANUAL_PROMPT_WINDOW_NAME):
    """
    inwholeimageonusemouseboxselectonerectangle, returnmutualtocoordinate (rx, ry, rw, rh), range [0,1]. 
    - ENTER/SPACE: confirmrecognize
    - ESC: takeeliminate
    """
    img_rgb = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    try:
        r = cv2.selectROI(window_name, img_bgr, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(window_name)
    except Exception as e:
        print(f"[Warn] OpenCV interactiveboxselectfail: {e}")
        return None

    x, y, w, h = [int(v) for v in r]
    if w <= 0 or h <= 0:
        print("[Prompt] notselectinhaseffectbox(maybeaccording to  ESC orboxareafor 0). ")
        return None

    H, W = img_bgr.shape[:2]
    rx = float(x) / float(W)
    ry = float(y) / float(H)
    rw = float(w) / float(W)
    rh = float(h) / float(H)

    rx = max(0.0, min(1.0, rx))
    ry = max(0.0, min(1.0, ry))
    rw = max(0.0, min(1.0 - rx, rw))
    rh = max(0.0, min(1.0 - ry, rh))
    return (rx, ry, rw, rh)


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
    """
    no score  simple NMS: useareawhenrankingaccordingaccording, retainlargeboxprefer. 
    """
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


def shift_boxes_xyxy(boxes, dx, dy, w_full, h_full):
    out = []
    for b in boxes:
        bb = [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]
        out.append(clamp_box_xyxy(bb, w_full, h_full))
    return out


# ===================== dataset fallback: randomselect 1 image(optional) =====================

def list_images(img_dir):
    if not os.path.isdir(img_dir):
        return []
    return [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(IMAGE_EXTS)]



def _find_dota_list_files(dota_dir: str, split: str, cat_token: str):
    """in DOTA_data insidecheckfindsome split + classcorrespond listfile. 

    compatiblecommonname: 
      - train_ship.txt / val_ship.txt / test_ship.txt
      - train_small-vehicle.txt / test_large-vehicle.txt
      - train_small_vehicle.txt etc.
    """
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
        # allow small-vehicle / small_vehicle / smallvehicle etc.form
        if cat_l.replace("-", "") in low.replace("-", "").replace("_", ""):
            hits.append(os.path.join(dota_dir, fn))
    # letmore"fineconfirm" fileprefer(filenameexceedshortusuallyexceedfineconfirm)
    hits.sort(key=lambda p: (len(os.path.basename(p)), p))
    return hits

def _read_dota_name_list(txt_path):
    """
    read dataset/ASPDNet_dataset/DOTA_data below  *_ship.txt / *_small-vehicle.txt / *_large-vehicle.txt
    Returns:allow sample basename set(notcontainextension). 
    compatiblecommonformat: per linemaybe"P1234.png""images/P1234.png""P1234". 
    """
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


def _collect_rsoc_dota_candidates(dataset_label):
    """
    RSOC-Ship / RSOC-S-Vehicle / RSOC-L-Vehicle  imageallmixed in: 
      dataset/ASPDNet_dataset/{train,val,test}/images
    needuse dataset/ASPDNet_dataset/DOTA_data below  split listfile, or(train/val)labelTxt comefilter. 
    """
    cat_token = {
        "RSOC-Ship": "ship",
        "RSOC-S-Vehicle": "small-vehicle",
        "RSOC-L-Vehicle": "large-vehicle",
    }[dataset_label]

    dota_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "DOTA_data")
    all_candidates = []

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", split, "images")
        if not os.path.isdir(img_dir):
            continue

        # 1) prefer DOTA_data  listfilefilter(yousay ship/L/S-vehicle mixed intogether, needitcomeregionpart)
        list_files = _find_dota_list_files(dota_dir, split, cat_token)
        allow = None
        if list_files:
            allow = set()
            for lp in list_files:
                allow |= _read_dota_name_list(lp)

        # 2) ifnotfindtolistfile(commoninsome split), train/val degenerateforread labelTxt insideisor notincludethis class
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
                                # DOTA rec label: 8 coords + class + difficult
                                if len(parts) >= 9 and ((parts[-2] if len(parts) >= 10 else parts[-1]) == cat_token):
                                    hit = True
                                    break
                        if hit:
                            allow.add(base)
                    except Exception:
                        continue

        # 3) collectcandidateimage
        if allow is None:
            # noanyfilterinformation: in order toavoidwrongclass, heredirectskip
            continue

        for img_path in list_images(img_dir):
            base = os.path.splitext(os.path.basename(img_path))[0]
            if base in allow:
                all_candidates.append(img_path)

    return all_candidates


def _resolve_image_path(p):
    """
    trytakeuseuserto path(maybewrite as Dataset/... or dataset/...)parsebecomerealsavein filepath. 
    """
    if not p:
        return p
    pp = p
    if os.path.exists(pp):
        return pp

    # unifypartintervalsymbol
    pp2 = pp.replace("\\", os.sep).replace("/", os.sep)

    # Common:useuserwrite Dataset, butactualis dataset(orreverse come)
    head = pp2.split(os.sep)[0]
    if head.lower() in ["dataset", "dataSet".lower()]:
        tail = os.sep.join(pp2.split(os.sep)[1:])
        cand = os.path.join(ROOT_DATA_DIR, tail)
        if os.path.exists(cand):
            return cand

    # againtryonce: if pp thiscomethenismutualto dataset root directory mutualtopath
    cand = os.path.join(ROOT_DATA_DIR, pp2)
    if os.path.exists(cand):
        return cand

    return p  # originalsamplereturn, letaftercontinuereportwrongmoreclear


def pick_one_image_from_dataset(dataset_label):
    """
    onlyused for SINGLE_IMAGE_PATH foremptywhen: from dataset_label corresponddirectoryrandomselect 1 image. 
    - RSOC-Building: dataset/ASPDNet_dataset/RSOC_building/building/{train_data,test_data}/images
    - RSOC-Ship / RSOC-S-Vehicle / RSOC-L-Vehicle: dataset/ASPDNet_dataset/{train,val,test}/images + DOTA_data list/labelTxt filter
    - VD-People / VD-Vehicle: dataset/VisDrone-*/{train,val,test}/images(or Images)
    """
    candidates = []

    if dataset_label == "RSOC-Building":
        rsoc_root = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "RSOC_building", "building")
        for subdir in ["train_data", "test_data"]:
            img_dir = os.path.join(rsoc_root, subdir, "images")
            candidates += list_images(img_dir)

    elif dataset_label in ["RSOC-S-Vehicle", "RSOC-L-Vehicle", "RSOC-Ship"]:
        candidates += _collect_rsoc_dota_candidates(dataset_label)

    elif dataset_label in ["VD-People", "VD-Vehicle"]:
        vd_root = os.path.join(ROOT_DATA_DIR, "VisDrone-People" if dataset_label == "VD-People" else "VisDrone-Vehicle")
        for split in ["train", "val", "test"]:
            img_dir = None
            for cand in [os.path.join(vd_root, split, "images"), os.path.join(vd_root, split, "Images")]:
                if os.path.isdir(cand):
                    img_dir = cand
                    break
            if img_dir:
                candidates += list_images(img_dir)

    if not candidates:
        return None
    import random
    return random.choice(candidates)




# ===================== real label(GT)read: used for"real labelleaderboard" =====================

def _infer_split_from_path(img_path: str):
    p = img_path.replace("\\", "/").lower()
    for s in ["train", "val", "test"]:
        if f"/{s}/" in p:
            return s
    return None


def _basename_noext(p: str):
    return os.path.splitext(os.path.basename(p))[0]


def _guess_rsoc_category_from_dota_data(base: str, split: str):
    """from DOTA_data  listfilejudgethis imagebelongin ship / small-vehicle / large-vehicle. """
    dota_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "DOTA_data")
    for token in ["ship", "small-vehicle", "large-vehicle"]:
        files = _find_dota_list_files(dota_dir, split, token)
        allow = set()
        for fp in files:
            allow |= _read_dota_name_list(fp)
        if allow and base in allow:
            return token
    return None


def infer_dataset_label_from_image_path(img_path: str):
    """as much as possiblefrompath + DOTA_data listfileautomaticinfer dataset_label, avoidyoumanualchange prompt/realvaluelogic. """
    p = img_path.replace("\\", "/").lower()

    if "rsoc_building" in p:
        return "RSOC-Building"
    if "visdrone-people" in p:
        return "VD-People"
    if "visdrone-vehicle" in p:
        return "VD-Vehicle"

    # ASPDNet_dataset  mixclass: train/val/test/images belowmix  ship / small-vehicle / large-vehicle
    if "aspdnet_dataset" in p and "/images/" in p:
        split = _infer_split_from_path(img_path)
        if split in ["train", "val", "test"]:
            base = _basename_noext(img_path)
            token = _guess_rsoc_category_from_dota_data(base, split)
            if token == "ship":
                return "RSOC-Ship"
            if token == "small-vehicle":
                return "RSOC-S-Vehicle"
            if token == "large-vehicle":
                return "RSOC-L-Vehicle"

    # realinpushnotoutthenfall back
    return TARGET_DATASET_FALLBACK


def _extract_points_from_mat(mat_obj):
    """from .mat inas much as possible"robust"lyfindout (N,2) or (2,N)  pointcoordinatearray. """
    pts = []
    for k, v in mat_obj.items():
        if k.startswith("__"):
            continue
        try:
            arr = np.array(v)
        except Exception:
            continue
        if arr.ndim != 2:
            continue
        # (N,2)
        if arr.shape[1] == 2 and arr.shape[0] >= 1:
            for row in arr:
                try:
                    x, y = float(row[0]), float(row[1])
                    if np.isfinite(x) and np.isfinite(y):
                        pts.append((x, y))
                except Exception:
                    pass
        # (2,N)
        elif arr.shape[0] == 2 and arr.shape[1] >= 1:
            for j in range(arr.shape[1]):
                try:
                    x, y = float(arr[0, j]), float(arr[1, j])
                    if np.isfinite(x) and np.isfinite(y):
                        pts.append((x, y))
                except Exception:
                    pass
    return pts


def _load_rsoc_building_gt_centers(img_path: str):
    if sio is None:
        print("[GT][Warn] notinstall scipy, nomethodread RSOC_building   .mat realvalue. canexecute: pip install scipy")
        return None

    p = img_path.replace("\\", "/")
    base = _basename_noext(img_path)

    # train_data / test_data
    sub = "train_data" if "/train_data/" in p.lower() else ("test_data" if "/test_data/" in p.lower() else None)
    if sub is None:
        # fallback: frompathfind in building directory
        sub = "test_data"

    gt_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "RSOC_building", "building", sub, "ground_truth")
    cand1 = os.path.join(gt_dir, f"GT_{base}.mat")
    cand2 = os.path.join(gt_dir, f"{base}.mat")
    gt_fp = cand1 if os.path.exists(cand1) else (cand2 if os.path.exists(cand2) else None)

    if gt_fp is None:
        print(f"[GT][Warn] RSOC_building notfindto GT mat: try  {cand1} / {cand2}")
        return None

    try:
        mat = sio.loadmat(gt_fp)
    except Exception as e:
        print(f"[GT][Warn] read mat fail: {gt_fp} | {e}")
        return None

    pts = _extract_points_from_mat(mat)
    if not pts:
        print(f"[GT][Warn] mat innotparsetopointcoordinate: {gt_fp}(differentversion mat   key maybenotsame)")
        return None
    return pts


def _load_dota_rec_label_centers(img_path: str, cat_token: str):
    """read DOTA rec label(train/val)andextracttakeobjectcenter point. test usuallyno GT, return None. """
    split = _infer_split_from_path(img_path)
    if split not in ["train", "val"]:
        return None

    base = _basename_noext(img_path)
    if split == "train":
        label_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "train", "labelTxt-v1.0", "trainset_reclabelTxt")
    else:
        label_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "val", "labelTxt-v1.0", "valset_reclabelTxt")
    fp = os.path.join(label_dir, base + ".txt")
    if not os.path.exists(fp):
        return None

    centers = []
    try:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                # 8 coords + class + difficult => at least 10; haswhenwilllack difficult => at least 9
                if len(parts) < 9:
                    continue
                cls = parts[-2] if len(parts) >= 10 else parts[-1]
                if cls != cat_token:
                    continue
                try:
                    nums = list(map(float, parts[:8]))
                except Exception:
                    continue
                xs = nums[0::2]
                ys = nums[1::2]
                cx = sum(xs) / 4.0
                cy = sum(ys) / 4.0
                if np.isfinite(cx) and np.isfinite(cy):
                    centers.append((cx, cy))
    except Exception:
        return None

    return centers


def _find_visdrone_gt_dir(vd_root: str, split: str):
    # you directorynameis "Ground Truth"(withemptycell), meanwhiledosizewrite/belowdividelinecompatible
    cands = ["Ground Truth", "GroundTruth", "ground_truth", "annotations", "Annotations"]
    for name in cands:
        p = os.path.join(vd_root, split, name)
        if os.path.isdir(p):
            return p
    return None


def _load_visdrone_gt_centers(img_path: str, dataset_label: str):
    split = _infer_split_from_path(img_path)
    if split not in ["train", "val", "test"]:
        return None

    vd_root = os.path.join(ROOT_DATA_DIR, "VisDrone-People" if dataset_label == "VD-People" else "VisDrone-Vehicle")
    gt_dir = _find_visdrone_gt_dir(vd_root, split)
    if gt_dir is None:
        return None

    base = _basename_noext(img_path)
    fp = os.path.join(gt_dir, base + ".txt")
    if not os.path.exists(fp):
        return None

    centers = []
    try:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                # VisDrone commonforcommapartinterval: x,y,w,h,score,category,truncation,occlusion
                parts = s.split(",") if "," in s else s.split()
                if len(parts) < 4:
                    continue
                try:
                    x, y, w, h = map(float, parts[:4])
                except Exception:
                    continue

                # score=0 usuallytableshow ignore(ifsavein score field)
                if len(parts) >= 5:
                    try:
                        score = float(parts[4])
                        if score == 0:
                            continue
                    except Exception:
                        pass

                if w <= 0 or h <= 0:
                    continue
                cx = x + w / 2.0
                cy = y + h / 2.0
                if np.isfinite(cx) and np.isfinite(cy):
                    centers.append((cx, cy))
    except Exception:
        return None

    return centers


def get_gt_centers_for_image(img_path: str, dataset_label: str):
    """unifyentry: returnthisimageon"objectcenter point/pointannotation"list; ifno GT thenreturn None. """
    if dataset_label == "RSOC-Building":
        return _load_rsoc_building_gt_centers(img_path)

    if dataset_label in ["RSOC-Ship", "RSOC-S-Vehicle", "RSOC-L-Vehicle"]:
        cat = {"RSOC-Ship": "ship", "RSOC-S-Vehicle": "small-vehicle", "RSOC-L-Vehicle": "large-vehicle"}[dataset_label]
        return _load_dota_rec_label_centers(img_path, cat)

    if dataset_label in ["VD-People", "VD-Vehicle"]:
        return _load_visdrone_gt_centers(img_path, dataset_label)

    return None


def count_centers_in_cell(centers, cell_xyxy):
    if centers is None:
        return None
    x1, y1, x2, y2 = cell_xyxy
    n = 0
    for (cx, cy) in centers:
        if (x1 <= cx < x2) and (y1 <= cy < y2):
            n += 1
    return int(n)


# ===================== visualize =====================

def draw_grid_and_results(
    pil_full,
    cells,
    final_boxes_full,
):
    """
    inoriginalimageon: 
    - drawgridline
    - writeeverycell idx and count(predict)
    - ifsaveinrealvalue, thenmeanwhilewrite true_count
    - drawfinalbox(green)
    Returns:drawmakeafter  PIL.Image(notsave)
    """
    img = pil_full.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    # gridline + text
    for cell in cells:
        x1, y1, x2, y2 = cell["xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline="white", width=2)

        pred = int(cell.get("final_count", 0))
        gt = cell.get("true_count", None)
        if gt is None:
            txt = f"{cell['idx']}({pred})"
        else:
            txt = f"{cell['idx']}(P{pred}/T{int(gt)})"
        draw.text((x1 + 4, y1 + 4), txt, fill="yellow", font=font)

    # finalbox(green)
    for b in final_boxes_full:
        draw.rectangle(b, outline="lime", width=3)

    return img


# ===================== main pipeline =====================

def main():
    # 1) selectimage
    img_path = _resolve_image_path(SINGLE_IMAGE_PATH.strip())
    if not img_path:
        img_path = pick_one_image_from_dataset(TARGET_DATASET_FALLBACK)
        if not img_path:
            print("[Error] SINGLE_IMAGE_PATH foremptyand fallback datasetnotfindtoanyimage. pleaseinconfiguration sectionfillwrite SINGLE_IMAGE_PATH. ")
            return

    if not os.path.exists(img_path):
        # tryautomaticpatch ROOT prefix(compatibleyouoftenuse mutualtowriting style)
        cand = os.path.join(ROOT_DATA_DIR, img_path)
        if os.path.exists(cand):
            img_path = cand
        else:
            print(f"[Error] imagenotsavein: {img_path}")
            return

    # 2) automaticinferdatasetclass(used for prompt + realvalueread)
    dataset_label = infer_dataset_label_from_image_path(img_path)
    sam_prompt = SAM3_PROMPT_MAP.get(dataset_label, "object")
    gd_prompt = GD_PROMPT_MAP.get(dataset_label, "object")
    print(f"[Main] img_path = {img_path}")
    print(f"[Main] grid = {GRID_ROWS}x{GRID_COLS} | GD prompt=\"{gd_prompt}\" | SAM3 prompt=\"{sam_prompt}\"")
    print(f"[Main] device = {DEVICE} | GDINO_MODEL_ID = {GDINO_MODEL_ID}")

    if SHOW_ORIGINAL:
        show_original_with_pil(img_path)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("[Error] OpenCV readimagefail. ")
        return

    h_full, w_full = img_bgr.shape[:2]
    pil_full = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    # 2.5) readrealvalue(used forreal labelleaderboard)
    gt_centers = get_gt_centers_for_image(img_path, dataset_label)
    if gt_centers is None:
        print(f"[GT] No ground-truth found for this image (dataset_label={dataset_label}). True ranking will be skipped.")
    else:
        print(f"[GT] Loaded GT centers: {len(gt_centers)}")

    # 3) grid
    cells = build_grid_cells(w_full, h_full, GRID_ROWS, GRID_COLS)
    if not cells:
        print("[Error] grid divisionfail(image smallor rows/cols notvalid). ")
        return

    # 4) optional: manualprompt box(onlyused for pass1)
    manual_prompt_rel = None
    if ENABLE_MANUAL_IMAGE_PROMPT:
        print("\n[Prompt] pleaseinpopup windowinmanualboxselectonerectangle, as SAM3 pass1  additionalgeometric promptbox. ")
        manual_prompt_rel = select_prompt_box_rel_on_image(pil_full)
        if manual_prompt_rel is None and FALLBACK_TO_CENTER_PROMPT_BOX:
            manual_prompt_rel = _default_center_rel_box()
            print(f"[Prompt] usedefaultcenterprompt box rel_xywh={manual_prompt_rel}")
        elif manual_prompt_rel is not None:
            print(f"[Prompt] alreadyrecordprompt box rel_xywh={manual_prompt_rel}")

    # 5) loadmodel
    processor = build_sam3_processor()

    print("\n[Pipeline] For each grid cell: GD -> SAM3(pass1 with GD boxes) -> SAM3(pass2 text-only) -> merge&dedup -> count\n")

    all_final_boxes_full = []

    for cell in cells:
        idx = cell["idx"]
        x1, y1, x2, y2 = cell["xyxy"]
        cell_w, cell_h = int(x2 - x1), int(y2 - y1)

        pil_cell = pil_full.crop((x1, y1, x2, y2))
        print(f"[Cell {idx:02d}] xyxy=({x1},{y1},{x2},{y2}) | size={cell_w}x{cell_h}")

        # --- 1) GD @ cell ---
        gd_boxes = call_dino_local_on_pil(pil_cell, gd_prompt)
        gd_boxes = [clamp_box_xyxy(b, cell_w, cell_h) for b in gd_boxes
                    if isinstance(b, (list, tuple)) and len(b) == 4]
        print(f"  [GD@cell] boxes = {len(gd_boxes)}")

        # --- 2) SAM3 pass1(use GD boxasprompt; once-ity multi-box morehigheffect) ---
        sam_boxes_pass1 = []
        prompts_xywh = []
        if manual_prompt_rel is not None:
            prompts_xywh.append(rel_xywh_to_abs_xywh(manual_prompt_rel, cell_w, cell_h))
        for b in gd_boxes:
            xx1, yy1, xx2, yy2 = [int(round(v)) for v in b]
            ww = max(1, xx2 - xx1)
            hh = max(1, yy2 - yy1)
            prompts_xywh.append((xx1, yy1, ww, hh))

        if len(prompts_xywh) > 0:
            boxes_np, sam_state1 = sam3_text_plus_multi_boxes(processor, pil_cell, prompts_xywh, sam_prompt, return_state=True)
            # sidecaseA: preferfrom SAM3 outputintake masks/logits shapebecomeheatmap; iftakenottothenuse boxes degeneratefillfill
            heat1 = sam3_output_to_heatmap(sam_state1, cell_h, cell_w, fallback_boxes_xyxy=boxes_np)
            for bb in boxes_np:
                sam_boxes_pass1.append(clamp_box_xyxy(list(map(float, bb)), cell_w, cell_h))
        else:
            heat1 = np.zeros((cell_h, cell_w), dtype=np.float32)

        sam_boxes_pass1 = nms_dedup_boxes(sam_boxes_pass1, iou_thr=DEDUP_IOU_THR, min_area=MIN_BOX_AREA)
        print(f"  [SAM3 pass1@cell] merged = {len(sam_boxes_pass1)}")

        # --- 3) SAM3 pass2(plain text: notuseanyprompt box) ---
        sam_boxes_pass2 = []
        boxes_np2, sam_out2 = sam3_text_only(processor, pil_cell, sam_prompt, return_state=True)
        heat2 = sam3_output_to_heatmap(sam_out2, cell_h, cell_w, fallback_boxes_xyxy=boxes_np2)
        if boxes_np2.shape[0] > 0:
            for bb in boxes_np2:
                sam_boxes_pass2.append(clamp_box_xyxy(list(map(float, bb)), cell_w, cell_h))

        sam_boxes_pass2 = nms_dedup_boxes(sam_boxes_pass2, iou_thr=DEDUP_IOU_THR, min_area=MIN_BOX_AREA)
        print(f"  [SAM3 pass2@cell] merged = {len(sam_boxes_pass2)}")

        # --- 4) merge & dedup ---
        merged = sam_boxes_pass1 + sam_boxes_pass2
        final_boxes = nms_dedup_boxes(merged, iou_thr=DEDUP_IOU_THR, min_area=MIN_BOX_AREA)

        # ===== sidecaseA: useheatmap partsubstitute"instancecount" =====
        # combineandtwopathheatmap: use max, avoidrepeatregionoverlaychange"moreheat"
        if 'heat1' in locals() and 'heat2' in locals():
            if heat1 is None and heat2 is None:
                heat = sam3_output_to_heatmap(None, cell_h, cell_w, fallback_boxes_xyxy=final_boxes)
            elif heat1 is None:
                heat = heat2
            elif heat2 is None:
                heat = heat1
            else:
                heat = np.maximum(heat1, heat2)
        else:
            heat = sam3_output_to_heatmap(None, cell_h, cell_w, fallback_boxes_xyxy=final_boxes)

        # averagevaluepartawaymethod: threshold=allimagemean, beforescenepixelratioasthis cell  "numbergenerationlogic"
        thr = float(np.mean(heat)) if heat is not None else 0.0
        fg = (heat > thr) if heat is not None else np.zeros((cell_h, cell_w), dtype=bool)
        fg_ratio = float(np.sum(fg)) / float(max(1, cell_w * cell_h))

        # in order tonotlargechangeyouoriginalcome print/rankingformat, heretake 0~1   fg_ratio linearityscaleto 0~1000  integerscore
        final_count = int(round(fg_ratio * 1000.0))

        # additionalrecordsomeoptionaldebuginformation(notaffectotherlogic)
        cell["heat_fg_ratio"] = float(fg_ratio)
        cell["heat_thr"] = float(thr)

        cell["gd_n"] = int(len(gd_boxes))
        cell["sam1_n"] = int(len(sam_boxes_pass1))
        cell["sam2_n"] = int(len(sam_boxes_pass2))
        cell["final_count"] = final_count

        # realvaluecount: usecenter point/pointannotationfallincurrent cell  number(ifno GT thenfor None)
        cell["true_count"] = count_centers_in_cell(gt_centers, cell["xyxy"]) if 'gt_centers' in locals() else None

        # shift to full image coords for visualization
        final_full = shift_boxes_xyxy(final_boxes, x1, y1, w_full, h_full)
        all_final_boxes_full.extend(final_full)

        print(f"  [Count@cell] final={final_count} (GD={cell['gd_n']}, SAM1={cell['sam1_n']}, SAM2={cell['sam2_n']})")

    # 6) 16 cellranking(according to final_count)
    ranking = sorted(cells, key=lambda d: int(d.get("final_count", 0)), reverse=True)

    print("\n[Ranking] Grid cells by merged count (desc):")
    lines = []
    for rank, cell in enumerate(ranking, start=1):
        x1, y1, x2, y2 = cell["xyxy"]
        line = (
            f"Rank {rank:2d}: blocknumber={cell['idx']:02d}  | "
            f"recognitionresult={cell.get('final_count', 0):3d} | "
        )
        print("  " + line)
        lines.append(line)

    # 6.5) real labelleaderboard(according to true_count)
    if gt_centers is None:
        print("\n[True Ranking] Skipped (no ground-truth for this image / split).")
    else:
        true_ranking = sorted(cells, key=lambda d: int(d.get("true_count", 0) or 0), reverse=True)
        print("\n[True Ranking] Grid cells by TRUE count (desc):")
        for rank, cell in enumerate(true_ranking, start=1):
            x1, y1, x2, y2 = cell["xyxy"]
            line = (
                f"Rank {rank:2d}: blocknumber={cell['idx']:02d} | "
                f"real label={int(cell.get('true_count', 0) or 0):3d} | "
                f"predict={cell.get('final_count', 0):3d} | "
            )
            print("  " + line)

    # ======= according to your needrequest: notsaveanything(notwrite txt, notsaveimage, notfalldiskarrangename) =======

    # 7) visualize(optional): onlypopup, notsave
    if SHOW_RESULT:
        img_vis = draw_grid_and_results(pil_full, cells, all_final_boxes_full)
        try:
            img_vis.show()
        except Exception as e:
            print(f"[Warn] nomethodpopupresultimage: {e}")

    print("\n[Done] finish: singleimagegrid(cantune)  GD + SAM3 dual-pathcount + 16 regionranking. ")
# =======



if __name__ == "__main__":
    main()
