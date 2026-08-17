# -*- coding: utf-8 -*-
"""
GD + SAM3(twice)samplingcheckscript(Mask output + deduplication + 'example/content hint'ROI)

you proposed newneedrequest, I logicsolution: 
1) accordingoldeach timerunfirstrandom draw 10 imageimage, letyoumanualboxselect ROI(thisonestep  ROI useas'example/content hint', notistakecoordinatedirectmap tootherimage). 
2) aftercontinuenotrunalldataset, andisagainrandom draw 10 imageimagedo once"samplingevaluation". 
3) tothis 10 imageexecuteoriginalcome  GD + SAM3 twicepipeline(logicnotchange): 
   - pass1: first GD detectionget box, againtake box feed as geometric prompt to SAM3 get masks
   - pass2: SAM3 inwhole imagedo once(text/conceptsegmentation), alsooutput masks
   - deduplication: pass1 internaldeduplication, pass2 internaldeduplication, then pass1+pass2 againdeduplication
   - additional: take'example/content hint'throughtemplatematchincurrentimagefind intocandidate box, asadditionalgeometric promptadd SAM3(thisthenis'content hint' falllypractice)
4) tothis 10 imageautomaticOutput:overlay  mask + mask outsideconnectbox  visualizeimage(automaticpopupdisplay + savetolocal), andprint: 
   - each image number, filename, real labelcount(from ground_truth/GT_*.mat read center), pseudo labelcount(final mask number)
   - real labelleaderboard, pseudo labelleaderboard(allforthis 10 imageinternalranking)
5) additional: outputonepseudo label txt(andyoupreviousformatconsistent): one per lineobject
   IMG.jpg x1 y1 x2 y2
   here boxfrom mask  outsideconnectrectangle(xyxy, whole imagecoordinate)

Note:
- thisscriptdefaultuseyou"Meta official sam3 source code + Sam3Processor" callway; 
- real labelreadaccording to RSOC-Building   GT_*.mat commonformatdo robustparse; 
- 'example/content hint'inofficial repo insidemorestandard call exemplar prompt, butcrossimagepropagatepassinofficialimplementinsideandnottotalisdirectprovideunify API; 
  hereuse"youmanualboxout  ROI insidetolerate -> inobjectimageinsidedotemplatematch -> get box -> feed as geometric prompt to SAM3" wayimplement, 
  etc.valueinlet ROI  "insidetolerate"affectaftercontinuesegmentation, andnotiscoordinatehard migrationmove. 

Usage:
- directin PyCharm/VSCode runthis .py i.e.can(notneedcommand lineparameter)
"""

import os
import sys
import types
import random
from unittest.mock import MagicMock
from transformers import AutoProcessor, GroundingDinoForObjectDetection
import numpy as np
import cv2
import torch
from PIL import Image

try:
    import scipy.io as sio  # used forread GT_*.mat
except Exception:
    sio = None


# ===================== youneedchange configuration section =====================

ROOT_DATA_DIR = "dataset"

# selectdataset(herefirstaccording to RSOC-Building; ifyouwantsupportotherdataset, alsocanaccording to yourpreviousscript maptableextension)
TARGET_DATASET = "RSOC-Building"
SPLIT_FOR_EXEMPLAR = "train"    # select ROI   10 which image it comes from split
SPLIT_FOR_SAMPLE   = "test"     # samplingevaluation  10 which image it comes from split(alsocanchange to train)

NUM_EXEMPLAR = 10               # each timerunselect ROI  imagenumber
NUM_SAMPLE   = 10               # each timerunsamplingevaluation imagenumber

# ---------- GroundingDINO ----------
GD_BBOX_THRESHOLD = 0.13
GD_TEXT_THRESHOLD = 0.15
GDINO_MODEL_ID = os.getenv("GDINO_MODEL_ID", "grounding_dino_base")  # youlocalcanuse  id

# ---------- SAM3 ----------
LOCAL_SAM3_PATH = r"D:\sam3_source"         # you  sam3 source codepath
SAM3_CHECKPOINT_PATH = "saved_models/SAM3.pt"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAM3_CONFIDENCE_THRESHOLD = 0.3

# ---------- prompt ----------
SAM3_PROMPT_MAP = {
    "RSOC-Building": "building with rectangular or complex roof from top view . house rooftop in urban or rural area .",
}
GD_PROMPT_MAP = {
    "RSOC-Building": "building with rectangular or complex roof from top view . house rooftop in urban or rural area .",
}

# ---------- deduplication ----------
DEDUP_BOX_IOU_THR = 0.6          # used for box deduplication(fallback/auxiliary)
DEDUP_MASK_IOU_THR = 0.4        # used for mask deduplication(main)
MIN_BOX_AREA = 16.0

# ---------- 'example/content hint'templatematch ----------
TM_SCALES = [0.75, 1.0, 1.25]    # templatematchscale
TM_TOPK_PER_EX = 3               # every exemplar inobjectimagemostmanytakeseveralmatchbox
TM_SCORE_THR = 0.55              # matchthreshold(notenoughthentunelow/tunehigh)

# ---------- visualize ----------
OUT_DIR = os.path.join(ROOT_DATA_DIR, "viz_samples_rankcheck")
os.makedirs(OUT_DIR, exist_ok=True)
SHOW_WINDOW = True
DISPLAY_MS = 1200  # each imageautomaticshowhow many milliseconds, 0=alwaysetc.according tokey

# ---------- outputpseudo label txt(each timerunwriteone) ----------
WRITE_PSEUDO_TXT = True


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ===================== pathanddataset: RSOC-Building =====================

def _autofix_root_dir():
    global ROOT_DATA_DIR
    if os.path.isdir(ROOT_DATA_DIR):
        return
    if os.path.isdir("dataset"):
        ROOT_DATA_DIR = "dataset"

_autofix_root_dir()

def list_images(img_dir: str):
    if not img_dir or (not os.path.isdir(img_dir)):
        return []
    out = []
    for fn in os.listdir(img_dir):
        if fn.lower().endswith(IMAGE_EXTS):
            out.append(os.path.join(img_dir, fn))
    out.sort()
    return out

def _rsoc_building_dirs(split: str):
    rsoc_root = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "RSOC_building", "building")
    sub = "train_data" if split == "train" else ("test_data" if split == "test" else None)
    if sub is None:
        return None, None
    img_dir = os.path.join(rsoc_root, sub, "images")
    gt_dir = os.path.join(rsoc_root, sub, "ground_truth")
    return img_dir, gt_dir

def get_split_images(dataset_label: str, split: str):
    if dataset_label != "RSOC-Building":
        raise NotImplementedError("thisscriptfirstaccording to RSOC-Building writedead; ifneedotherdatasetcanaccording to youroldscriptlogicextension. ")
    img_dir, _ = _rsoc_building_dirs(split)
    return list_images(img_dir)

def get_gt_mat_path_for_image(dataset_label: str, split: str, image_basename: str):
    if dataset_label != "RSOC-Building":
        return None
    _, gt_dir = _rsoc_building_dirs(split)
    if gt_dir is None or (not os.path.isdir(gt_dir)):
        return None
    stem = os.path.splitext(image_basename)[0]
    mat_name = f"GT_{stem}.mat"
    p = os.path.join(gt_dir, mat_name)
    return p if os.path.exists(p) else None


# ===================== GroundingDINO(transformers) =====================

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

def call_dino_local_on_pil(pil_img: Image.Image, prompt_text: str):
    processor, model, device = _load_local_gd()
    img_rgb = np.array(pil_img.convert("RGB"))
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


# ===================== SAM3 import and inference(Meta official repo) =====================

# Windows below somecompatible(alonguseyouoldscript  patch logic)
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
    processor = Sam3Processor(model, confidence_threshold=float(SAM3_CONFIDENCE_THRESHOLD))
    print("[Model] SAM3 Processor initialization complete")
    return processor

def _sam3_state_to_output_dict(state_or_output):
    """
    take state / output unifyextractbecome dict: {"masks":..., "boxes":..., "scores":...}
    compatibleseveral kindscommonreturnform. 
    """
    if state_or_output is None:
        return {"masks": None, "boxes": None, "scores": None}
    if isinstance(state_or_output, dict):
        return {
            "masks": state_or_output.get("masks", None),
            "boxes": state_or_output.get("boxes", None),
            "scores": state_or_output.get("scores", None),
        }
    # has implementmaybeobjectattribute
    masks = getattr(state_or_output, "masks", None)
    boxes = getattr(state_or_output, "boxes", None)
    scores = getattr(state_or_output, "scores", None)
    return {"masks": masks, "boxes": boxes, "scores": scores}

def _to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.array(x)

def sam3_text_only_masks(processor: Sam3Processor, pil_img: Image.Image, prompt_text: str):
    state = processor.set_image(pil_img)
    output = processor.set_text_prompt(state=state, prompt=prompt_text)
    od = _sam3_state_to_output_dict(output)
    masks = _to_numpy(od["masks"])
    boxes  = _to_numpy(od["boxes"])
    scores = _to_numpy(od["scores"])
    return masks, boxes, scores

def sam3_text_plus_boxes_masks(processor: Sam3Processor, pil_img: Image.Image, boxes_xywh_list, labels_list, prompt_text: str):
    """
    text + multiple geometriesbox(just/negative) -> output masks/boxes/scores
    Note:different sam3 versionto"addgeometric promptafterifwhattriggerinference"implementomithasdifference; 
    hereas much as possibleuseandoldscriptconsistent style: first set_image + set_text_prompt getconcept, then add_geometric_prompt update state, 
    finalfrom state insidetake masks/boxes/scores(ifnothendegenerateto output). 
    """
    if boxes_xywh_list is None:
        boxes_xywh_list = []
    if labels_list is None:
        labels_list = [True] * len(boxes_xywh_list)

    width, height = pil_img.size
    state = processor.set_image(pil_img)

    # firstsetsetconcept(text)
    state_or_out = processor.set_text_prompt(state=state, prompt=prompt_text)
    # has implementreturn output dict, has directreturn state
    # in order tokeepandoldscriptconsistent, wecontinuetake prompts addto state on
    # prefer state(set_image  ), againtryuse set_text_prompt  returnas state
    state_use = state_or_out if isinstance(state_or_out, dict) else state_or_out
    if isinstance(state_or_out, dict):
        # dict alsomaybethenis state
        state_use = state_or_out

    # try reset prompts(ifsavein)
    try:
        processor.reset_all_prompts(state_use)
    except Exception:
        pass

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
        state_use = processor.add_geometric_prompt(box=norm_box_cxcywh, label=bool(lab), state=state_use)

    # tryagaintimestriggerinference(ifimplementinsideneed)
    out2 = None
    try:
        out2 = processor.set_text_prompt(state=state_use, prompt=prompt_text)
    except Exception:
        out2 = None

    # finalOutput:prefer out2, itstimes state_use
    od = _sam3_state_to_output_dict(out2 if out2 is not None else state_use)
    masks = _to_numpy(od["masks"])
    boxes  = _to_numpy(od["boxes"])
    scores = _to_numpy(od["scores"])
    return masks, boxes, scores


# ===================== basistool: box / mask / deduplication =====================

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

def _ensure_hw_mask(mask):
    """take (1,H,W)/(H,W,1) etc.unifybecome (H,W)"""
    if mask is None:
        return None
    m = np.asarray(mask)

    # Common:torch->numpy afteris float/bool allmaybe
    # first squeeze dropsohas size=1  dimension
    m = np.squeeze(m)

    # squeeze afterstillmaybe 3D(very fewnumberimplement/return), fallbacktakefirstimage/firstchannel
    if m.ndim == 3:
        # maybe (C,H,W) or (H,W,C), allas much as possibletakeonechannel
        if m.shape[0] in (1, 2, 3):   # (C,H,W)
            m = m[0]
        elif m.shape[-1] in (1, 2, 3):  # (H,W,C)
            m = m[..., 0]
        else:
            m = m[0]  # fallback

    if m.ndim != 2:
        raise ValueError(f"Mask must be 2D (H,W) after squeeze, got shape={m.shape}")

    return m.astype(bool)


def mask_to_xyxy(mask_bool):
    m = _ensure_hw_mask(mask_bool)
    ys, xs = np.where(m)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    # Note:x2/y2 use +1 only theniscloseopeninterval [x1,x2)
    return [float(x1), float(y1), float(x2 + 1), float(y2 + 1)]


def dedup_masks(masks, iou_thr=0.75):
    """
    masks: (N,H,W) / (N,1,H,W) / list[(H,W)] / list[(1,H,W)] ...
    return: list[mask_bool(H,W)]
    """
    if masks is None:
        return []

    ms = []
    if isinstance(masks, list):
        for m in masks:
            if m is None:
                continue
            ms.append(_ensure_hw_mask(m))
    else:
        arr = np.asarray(masks)
        # maybe (N,1,H,W)
        if arr.ndim == 4 and arr.shape[1] == 1:
            for i in range(arr.shape[0]):
                ms.append(_ensure_hw_mask(arr[i, 0]))
        # (N,H,W)
        elif arr.ndim == 3:
            for i in range(arr.shape[0]):
                ms.append(_ensure_hw_mask(arr[i]))
        # (H,W) single
        elif arr.ndim == 2:
            ms.append(_ensure_hw_mask(arr))
        else:
            # fallback: one by one squeeze
            for i in range(arr.shape[0]):
                ms.append(_ensure_hw_mask(arr[i]))

    ms = [m for m in ms if m is not None and m.sum() > 0]
    ms.sort(key=lambda m: int(m.sum()), reverse=True)

    keep = []
    for m in ms:
        ok = True
        for km in keep:
            inter = np.logical_and(m, km).sum()
            if inter <= 0:
                continue
            union = np.logical_or(m, km).sum()
            iou = float(inter / union) if union > 0 else 0.0
            if iou >= iou_thr:
                ok = False
                break
        if ok:
            keep.append(m)
    return keep



# ===================== real labelread: GT_*.mat(RSOC-Building) =====================

def load_gt_count_from_mat(mat_path: str):
    if mat_path is None or (not os.path.exists(mat_path)) or sio is None:
        return None
    try:
        d = sio.loadmat(mat_path)
    except Exception:
        return None

    if "center" not in d:
        # fallback: hassomeversioncall "annPoints" etc., hereonlyexample
        for k in d.keys():
            if k.lower().startswith("center"):
                d["center"] = d[k]
                break
    if "center" not in d:
        return None

    c = d["center"]
    # Common:object array, (2,1): 
    # center[0,0] = Nx2 pointcoordinate
    # center[1,0] = N(or 1x1)
    try:
        if c.dtype == object and c.size >= 1:
            coords = c[0, 0]
            if hasattr(coords, "shape") and len(coords.shape) == 2:
                n = int(coords.shape[0])
                return n
            # ifseconditemisnumber
            if c.size >= 2:
                v = c[1, 0]
                try:
                    return int(np.array(v).reshape(-1)[0])
                except Exception:
                    pass
        # if center itselfthenis Nx2
        arr = np.array(c)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return int(arr.shape[0])
        if arr.size == 1:
            return int(arr.reshape(-1)[0])
    except Exception:
        return None
    return None


# ===================== 'example/content hint': ROI select + templatematchget boxes =====================

def select_roi_on_image(img_bgr, window_name: str):
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

def pick_exemplar_crops(image_paths, num_ex=10, seed=123):
    rnd = random.Random(int(seed))
    paths = list(image_paths)
    rnd.shuffle(paths)
    picked = []
    need = int(num_ex)
    print(f"\n[Exemplar] willrandom sample {need} image, letyou per imageboxselect ROI(as 'example/content hint'). ")
    print("          Operation: drag to draw a box -> ENTER/SPACE confirm;ESC takeeliminatethis image. ")
    for p in paths:
        if len(picked) >= need:
            break
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        base = os.path.basename(p)
        print(f"[Exemplar] ({len(picked)+1}/{need}) select: {base}")
        roi = select_roi_on_image(bgr, "Select exemplar ROI (content-based)")
        if roi is None:
            print("  - notselect ROI, skip")
            continue
        x, y, w, h = roi
        crop = bgr[y:y+h, x:x+w].copy()
        if crop.size == 0:
            print("  - ROI crop forempty, skip")
            continue
        picked.append({"src": base, "crop": crop})
        print(f"  - OK ROI={roi}")
    if len(picked) < need:
        print(f"[Exemplar][Warn] onlyselectto {len(picked)}/{need}  exemplar, willusealreadyselect continuerun. ")
    return picked

def template_match_boxes(img_bgr, exemplar_crops):
    """
    Returns:list[xyxy] as'content hint'inobjectimageinsidecandidatebox
    """
    if img_bgr is None or len(exemplar_crops) == 0:
        return []

    H, W = img_bgr.shape[:2]
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    boxes = []

    for ex in exemplar_crops:
        tmpl_bgr = ex["crop"]
        if tmpl_bgr is None or tmpl_bgr.size == 0:
            continue
        tmpl_gray0 = cv2.cvtColor(tmpl_bgr, cv2.COLOR_BGR2GRAY)
        th0, tw0 = tmpl_gray0.shape[:2]

        for sc in TM_SCALES:
            tw = int(max(8, round(tw0 * sc)))
            th = int(max(8, round(th0 * sc)))
            if tw >= W or th >= H:
                continue
            tmpl_gray = cv2.resize(tmpl_gray0, (tw, th), interpolation=cv2.INTER_AREA)

            res = cv2.matchTemplate(img_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
            # take topk
            flat = res.reshape(-1)
            if flat.size == 0:
                continue
            # takecandidatepointindex(manytakeonepointagainfilter)
            k = min(int(TM_TOPK_PER_EX * 5), flat.size)
            idxs = np.argpartition(-flat, k-1)[:k]
            # ranking
            idxs = idxs[np.argsort(-flat[idxs])]

            taken = 0
            for idx in idxs:
                score = float(flat[idx])
                if score < float(TM_SCORE_THR):
                    break
                y = int(idx // res.shape[1])
                x = int(idx %  res.shape[1])
                x1, y1 = x, y
                x2, y2 = x + tw, y + th
                x1 = max(0, min(W-1, x1))
                y1 = max(0, min(H-1, y1))
                x2 = max(0, min(W,   x2))
                y2 = max(0, min(H,   y2))
                if (x2-x1) * (y2-y1) < MIN_BOX_AREA:
                    continue
                boxes.append([float(x1), float(y1), float(x2), float(y2)])
                taken += 1
                if taken >= int(TM_TOPK_PER_EX):
                    break

    # firstdo once box deduplication(avoidtemplatematchgeneratelargequantityoverlap)
    boxes = nms_dedup_boxes(boxes, iou_thr=0.5, min_area=MIN_BOX_AREA)
    return boxes


# ===================== visualize: overlay mask + box =====================

def overlay_masks_and_boxes(img_bgr, masks_bool_list, idx_text="", gt_cnt=None, pseudo_cnt=None):
    out = img_bgr.copy()
    H, W = out.shape[:2]
    if masks_bool_list:
        # overlay
        overlay = out.copy()
        for m in masks_bool_list:
            if m is None:
                continue
            m = m.astype(bool)
            if m.shape[0] != H or m.shape[1] != W:
                # ifsizenotconsistent, try resize(mostneighbor)
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

            color = (0, 255, 0)  # green
            overlay[m] = (overlay[m] * 0.3 + np.array(color) * 0.7).astype(np.uint8)

            bb = mask_to_xyxy(m)
            if bb is not None:
                x1, y1, x2, y2 = [int(round(v)) for v in bb]
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 1)

        out = cv2.addWeighted(overlay, 0.6, out, 0.4, 0)

    # title text
    lines = [str(idx_text)]
    if gt_cnt is not None:
        lines.append(f"GT={gt_cnt}")
    if pseudo_cnt is not None:
        lines.append(f"Pseudo={pseudo_cnt}")
    y = 25
    for t in lines:
        cv2.putText(out, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28
    return out


# ===================== mainPipeline:sampling 10 imageandoutputleaderboard =====================

def run_one_image(processor: Sam3Processor, img_path: str, exemplar_crops):
    bgr = cv2.imread(img_path)
    if bgr is None:
        return None
    H, W = bgr.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    sam_prompt = SAM3_PROMPT_MAP.get(TARGET_DATASET, "object")
    gd_prompt  = GD_PROMPT_MAP.get(TARGET_DATASET, "object")

    # 1) exemplar content -> boxes
    ex_boxes_xyxy = template_match_boxes(bgr, exemplar_crops)

    # 2) GD
    gd_boxes_xyxy = call_dino_local_on_pil(pil, gd_prompt)
    gd_boxes_xyxy = [clamp_box_xyxy(bb, W, H) for bb in gd_boxes_xyxy if isinstance(bb, (list, tuple)) and len(bb) == 4]
    gd_boxes_xyxy = nms_dedup_boxes(gd_boxes_xyxy, iou_thr=0.6, min_area=MIN_BOX_AREA)

    # 3) pass1: boxes = GD boxes + exemplar boxes
    prompts_xywh = []
    prompts_lab  = []
    # gd boxes -> xywh
    for bb in gd_boxes_xyxy:
        x1, y1, x2, y2 = [int(round(v)) for v in bb]
        prompts_xywh.append((x1, y1, max(1, x2-x1), max(1, y2-y1)))
        prompts_lab.append(True)
    # exemplar boxes -> xywh
    for bb in ex_boxes_xyxy:
        x1, y1, x2, y2 = [int(round(v)) for v in bb]
        prompts_xywh.append((x1, y1, max(1, x2-x1), max(1, y2-y1)))
        prompts_lab.append(True)

    masks1, boxes1, scores1 = (None, None, None)
    if len(prompts_xywh) > 0:
        masks1, boxes1, scores1 = sam3_text_plus_boxes_masks(processor, pil, prompts_xywh, prompts_lab, sam_prompt)
    masks1_list = dedup_masks(masks1, iou_thr=DEDUP_MASK_IOU_THR)

    # 4) pass2: text-only(stillcanadd exemplar boxes asadditionalgeometric prompt, help'content hint'morefocus)
    prompts_xywh2 = []
    prompts_lab2  = []
    for bb in ex_boxes_xyxy:
        x1, y1, x2, y2 = [int(round(v)) for v in bb]
        prompts_xywh2.append((x1, y1, max(1, x2-x1), max(1, y2-y1)))
        prompts_lab2.append(True)

    if len(prompts_xywh2) > 0:
        masks2, boxes2, scores2 = sam3_text_plus_boxes_masks(processor, pil, prompts_xywh2, prompts_lab2, sam_prompt)
    else:
        masks2, boxes2, scores2 = sam3_text_only_masks(processor, pil, sam_prompt)
    masks2_list = dedup_masks(masks2, iou_thr=DEDUP_MASK_IOU_THR)

    # 5) merge & dedup again
    merged = masks1_list + masks2_list
    merged = dedup_masks(merged, iou_thr=DEDUP_MASK_IOU_THR)

    # 6) generatefinal box(used foroutputpseudo label)
    final_boxes_xyxy = []
    for m in merged:
        bb = mask_to_xyxy(m)
        if bb is None:
            continue
        bb = clamp_box_xyxy(bb, W, H)
        if box_area_xyxy(bb) >= MIN_BOX_AREA:
            final_boxes_xyxy.append(bb)
    final_boxes_xyxy = nms_dedup_boxes(final_boxes_xyxy, iou_thr=DEDUP_BOX_IOU_THR, min_area=MIN_BOX_AREA)

    return {
        "bgr": bgr,
        "H": H,
        "W": W,
        "masks": merged,
        "boxes": final_boxes_xyxy,
        "gd_boxes": gd_boxes_xyxy,
        "ex_boxes": ex_boxes_xyxy,
    }

def main():
    print(f"[Device] {DEVICE}")
    if TARGET_DATASET not in SAM3_PROMPT_MAP:
        raise ValueError(f"TARGET_DATASET={TARGET_DATASET} noin SAM3_PROMPT_MAP inconfiguration. ")

    # 1) columnimage
    ex_imgs = get_split_images(TARGET_DATASET, SPLIT_FOR_EXEMPLAR)
    sample_imgs = get_split_images(TARGET_DATASET, SPLIT_FOR_SAMPLE)

    if len(ex_imgs) == 0:
        raise RuntimeError(f"split={SPLIT_FOR_EXEMPLAR} findnottoimage, pleasecheck ROOT_DATA_DIR anddirectorystructure. ")
    if len(sample_imgs) == 0:
        raise RuntimeError(f"split={SPLIT_FOR_SAMPLE} findnottoimage, pleasecheck ROOT_DATA_DIR anddirectorystructure. ")

    # 2) select exemplar ROI(content hint)
    exemplar_crops = pick_exemplar_crops(ex_imgs, num_ex=NUM_EXEMPLAR, seed=123)
    if len(exemplar_crops) == 0:
        print("[Warn] noselecttoany exemplar ROI. aftercontinuewilldegeneratefornotuse'content hint'. ")

    # 3) sampling 10 imageimage
    rnd = random.Random(2026)
    sample_imgs = list(sample_imgs)
    rnd.shuffle(sample_imgs)
    picked = sample_imgs[:int(NUM_SAMPLE)]

    print(f"\n[Sample] willrandom sample {len(picked)} imageimageperform GD+SAM3(twice)samplingcheck: ")
    for i, p in enumerate(picked, 1):
        print(f"  {i:02d}. {os.path.basename(p)}")

    # 4) load SAM3 processor(oncei.e.can)
    processor = build_sam3_processor()

    # 5) run 10 image
    results = []
    pseudo_txt_lines = []
    for idx, img_path in enumerate(picked, 1):
        base = os.path.basename(img_path)
        print(f"\n[Run] ({idx}/{len(picked)}) {base}")

        out = run_one_image(processor, img_path, exemplar_crops)
        if out is None:
            print("  - readfail, skip")
            continue

        pseudo_cnt = int(len(out["masks"]))
        # real labelcount
        gt_mat = get_gt_mat_path_for_image(TARGET_DATASET, SPLIT_FOR_SAMPLE, base)
        gt_cnt = load_gt_count_from_mat(gt_mat) if gt_mat else None

        # writepseudo labelrow(one per lineobject)
        if WRITE_PSEUDO_TXT:
            for bb in out["boxes"]:
                x1, y1, x2, y2 = [int(round(v)) for v in bb]
                pseudo_txt_lines.append(f"{base} {x1} {y1} {x2} {y2}")

        # visualize
        viz = overlay_masks_and_boxes(
            out["bgr"],
            out["masks"],
            idx_text=f"[{idx:02d}] {base}",
            gt_cnt=gt_cnt,
            pseudo_cnt=pseudo_cnt
        )
        save_path = os.path.join(OUT_DIR, f"{idx:02d}_{os.path.splitext(base)[0]}_viz.jpg")
        cv2.imwrite(save_path, viz)
        print(f"  - saved: {save_path}")
        print(f"  - GT={gt_cnt} | Pseudo={pseudo_cnt}")

        if SHOW_WINDOW:
            cv2.imshow("GD+SAM3 sample check (mask+boxes)", viz)
            if DISPLAY_MS and int(DISPLAY_MS) > 0:
                key = cv2.waitKey(int(DISPLAY_MS)) & 0xFF
            else:
                key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            if key == 27:  # ESC
                print("[User] ESC pressed, stop early.")
                break

        results.append({
            "idx": idx,
            "name": base,
            "gt": gt_cnt if gt_cnt is not None else -1,
            "pseudo": pseudo_cnt,
        })

    # 6) writepseudo label txt
    if WRITE_PSEUDO_TXT and len(pseudo_txt_lines) > 0:
        txt_path = os.path.join(OUT_DIR, f"pseudo_{TARGET_DATASET}_{SPLIT_FOR_SAMPLE}_sample{len(results)}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for line in pseudo_txt_lines:
                f.write(line + "\n")
        print(f"\n[Output] pseudo label txt alreadywriteout: {txt_path}")

    # 7) printleaderboard(inthis 10 imageinternal)
    if len(results) == 0:
        print("\n[Done] nohaseffectresult. ")
        return

    # GT arrangerow(GT=-1  arrangetoaftersurface)
    gt_sorted = sorted(results, key=lambda r: (r["gt"] if r["gt"] >= 0 else -1), reverse=True)
    pseudo_sorted = sorted(results, key=lambda r: r["pseudo"], reverse=True)

    print("\n================== real label(GT)leaderboard(thistimessampling) ==================")
    for rank, r in enumerate(gt_sorted, 1):
        print(f"#{rank:02d}  [{r['idx']:02d}] {r['name']}  GT={r['gt']}  Pseudo={r['pseudo']}  (err={r['pseudo']-r['gt'] if r['gt']>=0 else 'NA'})")

    print("\n================== pseudo label(Pseudo)leaderboard(thistimessampling) ================")
    for rank, r in enumerate(pseudo_sorted, 1):
        print(f"#{rank:02d}  [{r['idx']:02d}] {r['name']}  Pseudo={r['pseudo']}  GT={r['gt']}  (err={r['pseudo']-r['gt'] if r['gt']>=0 else 'NA'})")

    print(f"\n[Done] visualizeoutputdirectory: {OUT_DIR}")

if __name__ == "__main__":
    main()
