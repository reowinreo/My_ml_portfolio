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
import math
from unittest.mock import MagicMock

import numpy as np
import cv2
import torch
from PIL import Image

try:
    import scipy.io as sio  # used forread GT_*.mat
except Exception:
    sio = None


# ===================== youneedchange configuration section =====================
TARGET_IMAGE_NAME = "P0703.png"   # youwantrun thatimagename(withextension)

ROOT_DATA_DIR = "dataset"

# selectdataset(herefirstaccording to RSOC-Building; ifyouwantsupportotherdataset, alsocanaccording to yourpreviousscript maptableextension)
TARGET_DATASET = "RSOC-Ship"
SPLIT_FOR_EXEMPLAR = "train"    # select ROI   10 which image it comes from split
SPLIT_FOR_SAMPLE   = "train"     # samplingevaluation  10 which image it comes from split(alsocanchange to train)

NUM_EXEMPLAR = 10               # each timerunselect ROI  imagenumber
NUM_SAMPLE   = 1               # each timerunsamplingevaluation imagenumber

# ---------- GroundingDINO ----------
GD_BBOX_THRESHOLD = 0.13
GD_TEXT_THRESHOLD = 0.15
GDINO_MODEL_ID = os.getenv("GDINO_MODEL_ID", "grounding_dino_base")  # youlocalcanuse  id

# ---------- SAM3 ----------
LOCAL_SAM3_PATH = r"D:\sam3_source"         # you  sam3 source codepath
SAM3_CHECKPOINT_PATH = "saved_models/SAM3.pt"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAM3_CONFIDENCE_THRESHOLD = 0.3


# ===================== SAHI patchconfiguration(newly added) =====================
# Objective:takeoriginalcut image into 4x4=16 block(overlap 10%), everyblock resize to 512x512 againfeed GD / SAM3
SLICE_ROWS = 4
SLICE_COLS = 2
SLICE_OVERLAP_RATIO = 0.10   # 10%
MODEL_INPUT_SIZE = 512       # GD / SAM3 inputsize(sideshape)

# ---------- prompt ----------
SAM3_PROMPT_MAP = {
    "RSOC-Building": "building",
    "RSOC-Ship": "ship",
}
GD_PROMPT_MAP = {
    "RSOC-Building": "structure.building.house",
    "RSOC-Ship": "ship.boat",
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
OUT_DIR = os.path.join(ROOT_DATA_DIR, "viz_ship_method_B")
os.makedirs(OUT_DIR, exist_ok=True)
SHOW_WINDOW = True
DISPLAY_MS = 1200  # each imageautomaticshowhow many milliseconds, 0=alwaysetc.according tokey


# ===================== random perturbation(newly added): deleteobject / turnmoveobject =====================
# Description:
# - delete_ratio: with"object"forunitrandomdelete ratio(for example 0.10=delete 10%)
# - move_ratio: with"object"forunitrandomtranslate ratio(for example 0.02=translate 2%)
# - translateamplitude: dx,dy randomtake [-MOVE_SHIFT_MAX, -MOVE_SHIFT_MIN] U [MOVE_SHIFT_MIN, MOVE_SHIFT_MAX]
PERTURB_ENABLE = True
RANDOM_SEED = 12345  # fixedrandom seed; notwantfixedthenset to None
GD_BOX_DELETE_RATIO_FOR_PROTECT = 0.7  # used for"protect mask"  GD deleteboxratio(pleaseandAprogram  PERTURB_DELETE_RATIO keepconsistent)
MASK_PROTECT_IOU_THR = 0.45  # mask outsideconnectbox and(deleteboxafterremaining )GD box IoU>=thisthreshold, thenthis mask notallowbyrandomdelete

PERTURB_DELETE_RATIO = 0.55
PERTURB_MOVE_RATIO   = 0.15
MOVE_SHIFT_MIN = 10
MOVE_SHIFT_MAX = 50

# ---------- outputpseudo label txt(each timerunwriteone) ----------
WRITE_PSEUDO_TXT = False  # thisversiononlysavevisualizeimage


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def _norm_name_token(s: str):
    """Normalize a filename/path token from DOTA_data list or filesystem for robust matching."""
    if s is None:
        return ""
    # remove BOM and whitespace
    s = str(s).replace("\ufeff", "").strip()
    if not s:
        return ""
    s = s.replace("\\", "/")
    # strip quotes
    if (len(s) >= 2) and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        s = s[1:-1].strip()
    base = os.path.basename(s)
    stem = os.path.splitext(base)[0]
    return stem.lower().strip()


# ===================== SAHI patchtool(newly added) =====================

def _compute_slice_starts(length: int, n_slices: int, overlap: float):
    """generate n_slices startpoint, makepatchbetweenapproximately overlap overlap(classsimilar SAHI   overlap_ratio). """
    length = int(length)
    n_slices = int(n_slices)
    overlap = float(overlap)
    if n_slices <= 1:
        return [0], length

    # SAHI oftenuseidea: step = slice * (1 - overlap)
    # andtotaloverwrite: slice*(n - (n-1)*overlap) ~= length
    denom = max(1e-6, (n_slices - (n_slices - 1) * overlap))
    slice_len = int(math.ceil(length / denom))
    slice_len = max(1, min(slice_len, length))
    step = int(round(slice_len * (1.0 - overlap)))
    step = max(1, step)

    starts = []
    for i in range(n_slices):
        s = i * step
        if s + slice_len > length:
            s = max(0, length - slice_len)
        starts.append(int(s))

    # deduplicationandpad(extremeendsmallimagewhenmaybeoutnowrepeat)
    starts = list(dict.fromkeys(starts))
    while len(starts) < n_slices:
        # uniformly insert
        cand = int(round((len(starts) / max(1, n_slices - 1)) * max(0, length - slice_len)))
        if cand not in starts:
            starts.append(cand)
        else:
            starts.append(max(0, min(length - slice_len, cand + 1)))
        starts = list(dict.fromkeys(starts))
    starts = starts[:n_slices]
    starts.sort()
    return starts, slice_len


def sahi_slice_image_16(bgr_full, rows=4, cols=4, overlap_ratio=0.10):
    """takeoriginalcut image into rows x cols block(default 4x4=16), return tile list. 

    every tile: dict(
        x1,y1,x2,y2,   # inoriginalimageon coordinate(rightbelowforopeninterval)
        tile_bgr       # originalresolutioncropout  tile(not resize)
    )
    """
    H, W = bgr_full.shape[:2]
    ys, slice_h = _compute_slice_starts(H, rows, overlap_ratio)
    xs, slice_w = _compute_slice_starts(W, cols, overlap_ratio)

    tiles = []
    for y1 in ys:
        y2 = min(H, y1 + slice_h)
        for x1 in xs:
            x2 = min(W, x1 + slice_w)
            tile = bgr_full[y1:y2, x1:x2].copy()
            tiles.append({
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                "tile_bgr": tile
            })
    return tiles


def _resize_to_512(tile_bgr, out_size=512):
    """take tile resize become out_size x out_size, used for GD / SAM3 input. """
    return cv2.resize(tile_bgr, (int(out_size), int(out_size)), interpolation=cv2.INTER_LINEAR)


def _map_box_512_to_full(bb_512, x1, y1, tile_w, tile_h, out_size=512):
    """take 512 inputspace  xyxy box mapbackoriginalimageglobalcoordinate. """
    sx = float(tile_w) / float(out_size)
    sy = float(tile_h) / float(out_size)
    bx1, by1, bx2, by2 = bb_512
    fx1 = x1 + bx1 * sx
    fy1 = y1 + by1 * sy
    fx2 = x1 + bx2 * sx
    fy2 = y1 + by2 * sy
    return [float(fx1), float(fy1), float(fx2), float(fy2)]


def _map_box_tile_to_512(bb_tile, tile_w, tile_h, out_size=512):
    """take tile originalresolutionspace  xyxy box map to 512 inputspace. """
    sx = float(out_size) / float(max(1, tile_w))
    sy = float(out_size) / float(max(1, tile_h))
    x1, y1, x2, y2 = bb_tile
    return [float(x1 * sx), float(y1 * sy), float(x2 * sx), float(y2 * sy)]


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


def _rsoc_mixed_dirs(split: str):
    """RSOC ship/L-vehicle/S-vehicle mixdatasetdirectory(onlyuse images). """
    base = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", split)
    img_dir = os.path.join(base, "images")
    # train/val has labelTxt-v1.0(canas fallback)
    label_dir = os.path.join(base, "labelTxt-v1.0")
    return img_dir, label_dir

def _find_dota_list_files(dota_dir: str, split: str, cat_token: str):
    """in DOTA_data belowfindlistfile(morerobust). 

    compatiblecommonname: 
      - train_ship.txt / trainship.txt / train-ship.txt / train_ship_list.txt ...
      - val_ship*.txt
      - test_ship*.txt

    ruleas much as possibleclose toyouoldversionscript: 
      - mustwith split beginning(train/val/test)
      - filenameneedincludeclass token(ignore - and _)
    """
    if not dota_dir or (not os.path.isdir(dota_dir)):
        return []
    split_l = str(split).lower().strip()
    cat_l = str(cat_token).lower().strip()
    cat_norm = cat_l.replace("-", "").replace("_", "")
    hits = []
    for fn in os.listdir(dota_dir):
        low = fn.lower()
        if not low.endswith(".txt"):
            continue
        if not low.startswith(split_l):
            continue
        low_norm = low.replace("-", "").replace("_", "")
        if cat_norm in low_norm:
            hits.append(os.path.join(dota_dir, fn))
    hits.sort(key=lambda p: (len(os.path.basename(p)), p))
    return hits

def _read_dota_name_list(txt_path: str):
    """read DOTA_data  list: one per lineimagename/path, return *normalized stem*   set(allsmallwrite). """
    if not txt_path or (not os.path.exists(txt_path)):
        return set()
    names = set()
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if (not s) or s.startswith("#"):
                    continue
                stem = _norm_name_token(s)
                if stem:
                    names.add(stem)
    except Exception:
        return set()
    return names



def _iter_label_txt_files(label_root: str):
    """traverse labelTxt-v1.0 below sohasannotation txt(compatiblehasno trainset_reclabelTxt/valset_reclabelTxt). """
    if not label_root or (not os.path.isdir(label_root)):
        return []
    subdirs = []
    for subcand in ("trainset_reclabelTxt", "valset_reclabelTxt", "testset_reclabelTxt"):
        p = os.path.join(label_root, subcand)
        if os.path.isdir(p):
            subdirs.append(p)
    if not subdirs:
        subdirs = [label_root]

    out = []
    for d in subdirs:
        for fn in os.listdir(d):
            if fn.lower().endswith(".txt"):
                out.append(os.path.join(d, fn))
    out.sort()
    return out

def _label_file_contains_class(label_fp: str, want_cls: str):
    """parse DOTA/RSOC rotateboxannotation txt, judgeisor notincludespecifyclass. """
    want = str(want_cls).lower().strip()
    try:
        with open(label_fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                # typicalformat: x1 y1 x2 y2 x3 y3 x4 y4 cls difficulty
                # or: ... cls
                if len(parts) < 9:
                    continue
                cls = parts[-2] if len(parts) >= 10 else parts[-1]
                if str(cls).lower() == want:
                    return True
    except Exception:
        return False
    return False

def _build_allow_set_for_ship(split: str):
    """build ship image allow-set(normalized stems, lowercase). 

    ship/S-vehicle/L-vehicle mixed in ASPDNet_dataset/{train,val,test}/images: 
    - prefer ASPDNet_dataset/DOTA_data below listfileregionclassificationdo not; 
    - iflistmissing, thento train/val degenerate scan labelTxt-v1.0. 
    """
    # 1) DOTA_data list priority
    dota_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "DOTA_data")
    allow = set()
    for lp in _find_dota_list_files(dota_dir, split=split, cat_token="ship"):
        allow |= _read_dota_name_list(lp)
    if allow:
        return allow

    # 2) fallback: only train/val use labelTxt-v1.0 scanclass
    if split not in ("train", "val"):
        return None
    _, label_dir = _rsoc_mixed_dirs(split)
    if not os.path.isdir(label_dir):
        return None

    for fp in _iter_label_txt_files(label_dir):
        if _label_file_contains_class(fp, "ship"):
            allow.add(_norm_name_token(fp))

    return allow if allow else None



def get_split_images(dataset_label: str, split: str):
    if dataset_label == "RSOC-Building":
        img_dir, _ = _rsoc_building_dirs(split)
        return list_images(img_dir)

    if dataset_label == "RSOC-Ship":
        img_dir, _ = _rsoc_mixed_dirs(split)
        imgs = list_images(img_dir)

        if len(imgs) == 0:
            raise RuntimeError(
                f"[RSOC-Ship] split={split}  imagedirectoryforemptyornotsavein. \n"
                f"img_dir={img_dir}\n"
                f"please confirmyou imagein dataset/ASPDNet_dataset/{split}/images below. "
            )

        # ship/S-vehicle/L-vehicle mixed intogether: mustdofilter
        allow = _build_allow_set_for_ship(split)
        if allow is None or len(allow) == 0:
            raise RuntimeError(
                f"[RSOC-Ship] nomethodconfirmset split={split} inwhichsomeimagebelongin ship. \n"
                f"img_dir={img_dir} (found {len(imgs)} images)\n"
                f"please confirm: dataset/ASPDNet_dataset/DOTA_data belowsaveinclasssimilar {split}_ship.txt  list(one per linefilename), \n"
                f"or(only train/val)dataset/ASPDNet_dataset/{split}/labelTxt-v1.0 belowsaveincorrespondannotation txt. "
            )

        # normalize matching (case-insensitive, extension-agnostic)
        out = []
        allow_norm = set([a.lower().strip() for a in allow if a])
        for p in imgs:
            stem = _norm_name_token(p)
            if stem in allow_norm:
                out.append(p)

        if len(out) == 0:
            # providemoretoolvolume diagnose, avoid"looks likehaslistbutfilterfor 0"
            some_imgs = [os.path.basename(x) for x in imgs[:10]]
            some_allow = list(sorted(list(allow_norm)))[:10]
            raise RuntimeError(
                f"[RSOC-Ship] split={split} filterafterimagenumberfor 0. \n"
                f"img_dir={img_dir} (total images={len(imgs)})\n"
                f"DOTA/label allow-set size={len(allow_norm)}\n"
                f"example images before10: {some_imgs}\n"
                f"example allow before10(stem): {some_allow}\n"
                f"usuallyreason: listinsidefilenameand images insideactualfilenamenotconsistent(sizewrite/prefix/isor notcropblockextensionetc.). "
            )

        return out

    raise NotImplementedError(f"notsupport dataset: {dataset_label}")



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
    print(f"\n[Exemplar] willrandom sample {need} image, letyou per imageboxselect ROI(as'example/content hint'). ")
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




# ===================== random perturbation: masks(newly added) =====================
def _seed_to_int(seed):
    if seed is None:
        return None
    try:
        return int(seed)
    except Exception:
        return int(abs(hash(str(seed))) % (2**31 - 1))

def _select_random_indices(n: int, k: int, seed=None):
    """returnlengthfor k  randombelowlabellist(based on seed, canreproduce). """
    n = int(n); k = int(k)
    if n <= 0 or k <= 0:
        return []
    k = min(k, n)
    if seed is None:
        idxs = list(range(n))
        random.shuffle(idxs)
        return idxs[:k]
    rng = random.Random(_seed_to_int(seed))
    idxs = list(range(n))
    rng.shuffle(idxs)
    return idxs[:k]

def _rand_signed_step(minv: int, maxv: int, rng=None) -> int:
    minv = int(minv); maxv = int(maxv)
    if maxv < minv:
        maxv = minv
    r = rng if rng is not None else random
    v = r.randint(minv, maxv)
    return v if r.random() < 0.5 else -v

def translate_mask_bool(mask_bool: np.ndarray, dx: int, dy: int):
    """will bool mask translate(dx,dy), superoutpartdropout(notloopbackvolume). """
    if mask_bool is None:
        return None
    m = mask_bool.astype(bool)
    H, W = m.shape[:2]
    dx = int(dx); dy = int(dy)
    out = np.zeros((H, W), dtype=bool)

    dst_x1 = max(0, dx)
    dst_y1 = max(0, dy)
    dst_x2 = min(W, W + dx)
    dst_y2 = min(H, H + dy)

    src_x1 = max(0, -dx)
    src_y1 = max(0, -dy)
    src_x2 = min(W, W - dx)
    src_y2 = min(H, H - dy)

    if (dst_x2 <= dst_x1) or (dst_y2 <= dst_y1) or (src_x2 <= src_x1) or (src_y2 <= src_y1):
        return out

    out[dst_y1:dst_y2, dst_x1:dst_x2] = m[src_y1:src_y2, src_x1:src_x2]
    return out

def _gd_boxes_after_A_deletion(gd_boxes_full, delete_ratio, seed):
    """according to A program "deletebox"logic(onlydelete, nottranslate), usesameone seed get"remaining  GD box". """
    if gd_boxes_full is None:
        return []
    boxes = [list(map(float, b)) for b in gd_boxes_full if b is not None]
    n = len(boxes)
    if n == 0:
        return []
    del_k = int(round(n * float(delete_ratio)))
    if del_k <= 0:
        return boxes
    del_seed = None if seed is None else (_seed_to_int(seed) + 1000)
    del_idxs = set(_select_random_indices(n, del_k, seed=del_seed))
    return [b for i, b in enumerate(boxes) if i not in del_idxs]

def perturb_masks_list(masks_list, delete_ratio=0.0, move_ratio=0.0, shift_min=10, shift_max=50,
                      seed=None, protect_boxes=None, protect_iou_thr=0.0):
    """to mask list dorandomdeletesubtract/translateperturb. 

    modifypoint(according to your needrequest): 
    - deletestage: firstaccording to seed reproduce A programwilldeletewhichsome GD box, therebyget"remaining  GD box". 
    - compute mask and"remaining GD box"  IoU(use mask outsideconnectboxand box   IoU): if >= protect_iou_thr, thenthis mask notallowbyrandomdelete. 
    - rest mask againaccording to delete_ratio randomdelete. 
    """
    if masks_list is None:
        return []
    masks = [m.astype(bool) for m in masks_list if m is not None]
    n = len(masks)
    if n == 0:
        return []

    # 1) randomdelete(withprotectlogic, canreproduce)
    del_k = int(round(n * float(delete_ratio)))
    protect_set = set()
    if protect_boxes and float(protect_iou_thr) > 0:
        pb = [list(map(float, b)) for b in protect_boxes if b is not None]
        for i, m in enumerate(masks):
            bb = mask_to_xyxy(m)
            if bb is None:
                continue
            for b in pb:
                try:
                    if iou_xyxy(bb, b) >= float(protect_iou_thr):
                        protect_set.add(i)
                        break
                except Exception:
                    continue

    if del_k > 0:
        cand = [i for i in range(n) if i not in protect_set]
        if cand:
            del_k = min(del_k, len(cand))
            del_seed = None if seed is None else (_seed_to_int(seed) + 4000)
            del_idxs = set(_select_random_indices(len(cand), del_k, seed=del_seed))
            # del_idxs rolein cand  positionon
            del_set = set([cand[j] for j in del_idxs])
            masks = [m for i, m in enumerate(masks) if i not in del_set]
    if not masks:
        return []

    # 2) randomtranslate(keeporiginallogic; canreproduce)
    move_k = int(round(len(masks) * float(move_ratio)))
    if move_k > 0:
        mv_seed = None if seed is None else (_seed_to_int(seed) + 5000)
        step_seed = None if seed is None else (_seed_to_int(seed) + 6000)
        mv_idxs = set(_select_random_indices(len(masks), move_k, seed=mv_seed))
        step_rng = None if step_seed is None else random.Random(_seed_to_int(step_seed))
        new_masks = []
        for i, m in enumerate(masks):
            if i in mv_idxs:
                dx = _rand_signed_step(shift_min, shift_max, rng=step_rng)
                dy = _rand_signed_step(shift_min, shift_max, rng=step_rng)
                m2 = translate_mask_bool(m, dx=dx, dy=dy)
                if m2 is not None and m2.sum() > 0:
                    new_masks.append(m2)
            else:
                new_masks.append(m)
        masks = new_masks

    return masks

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
        y += 28
    return out



def overlay_boxes_only(img_bgr, boxes_xyxy, title_text=""):
    out = img_bgr.copy()
    H, W = out.shape[:2]
    if boxes_xyxy:
        for bb in boxes_xyxy:
            x1, y1, x2, y2 = [int(round(v)) for v in bb]
            x1 = max(0, min(W - 1, x1))
            y1 = max(0, min(H - 1, y1))
            x2 = max(0, min(W - 1, x2))
            y2 = max(0, min(H - 1, y2))
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return out


def run_gd_only(img_path: str):
    """methodA: only GD(newly added: firstdo SAHI 4x4=16 patch, everyblock resize to 512x512 againrun GD), finaltakeboxmapbackoriginalimageanddeduplication. """
    bgr_full = cv2.imread(img_path)
    if bgr_full is None:
        return None
    H, W = bgr_full.shape[:2]
    gd_prompt  = GD_PROMPT_MAP.get(TARGET_DATASET, "object")

    tiles = sahi_slice_image_16(bgr_full, rows=SLICE_ROWS, cols=SLICE_COLS, overlap_ratio=SLICE_OVERLAP_RATIO)

    all_boxes_full = []
    for t in tiles:
        x1, y1, x2, y2 = t["x1"], t["y1"], t["x2"], t["y2"]
        tile = t["tile_bgr"]
        if tile is None or tile.size == 0:
            continue
        tile_h, tile_w = tile.shape[:2]

        tile_512 = _resize_to_512(tile, out_size=MODEL_INPUT_SIZE)
        pil_512 = Image.fromarray(cv2.cvtColor(tile_512, cv2.COLOR_BGR2RGB))

        gd_boxes_512 = call_dino_local_on_pil(pil_512, gd_prompt)
        # mapbackallimage
        for bb in gd_boxes_512:
            if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                continue
            full_bb = _map_box_512_to_full(bb, x1, y1, tile_w, tile_h, out_size=MODEL_INPUT_SIZE)
            full_bb = clamp_box_xyxy(full_bb, W, H)
            if box_area_xyxy(full_bb) >= MIN_BOX_AREA:
                all_boxes_full.append(full_bb)

    all_boxes_full = nms_dedup_boxes(all_boxes_full, iou_thr=0.6, min_area=MIN_BOX_AREA)
    return {"bgr": bgr_full, "H": H, "W": W, "gd_boxes": all_boxes_full}


# ===================== mainPipeline:sampling 10 imageandoutputleaderboard =====================


def run_one_image(processor: Sam3Processor, img_path: str, exemplar_crops):
    """methodB/C/D: GD + SAM3(twice). (newly added: SAHI 4x4=16 patch, everyblock resize to 512x512 againrun GD / SAM3)

    Output:
      - masks: allimagecoordinate systembelow  bool mask list(alreadyglobaldeduplication)
      - boxes: masks outsideconnectbox(allimagecoordinate system)
      - gd_boxes: GD-only  box(allimagecoordinate system, fromsohas tile combineanddeduplication)
      - ex_boxes: exemplar templatematchinallimagecoordinate systembelow candidatebox(fromsohas tile combineanddeduplication)
    """
    bgr_full = cv2.imread(img_path)
    if bgr_full is None:
        return None
    H, W = bgr_full.shape[:2]

    sam_prompt = SAM3_PROMPT_MAP.get(TARGET_DATASET, "object")
    gd_prompt  = GD_PROMPT_MAP.get(TARGET_DATASET, "object")

    tiles = sahi_slice_image_16(bgr_full, rows=SLICE_ROWS, cols=SLICE_COLS, overlap_ratio=SLICE_OVERLAP_RATIO)

    all_masks_full = []
    all_gd_boxes_full = []
    all_ex_boxes_full = []

    for t in tiles:
        x0, y0, x2, y2 = t["x1"], t["y1"], t["x2"], t["y2"]
        tile = t["tile_bgr"]
        if tile is None or tile.size == 0:
            continue
        tile_h, tile_w = tile.shape[:2]

        # 1) exemplar content -> boxes(in tile originalresolutionondotemplatematch)
        ex_boxes_tile = template_match_boxes(tile, exemplar_crops)  # tile coordinate system xyxy
        # map to 512 inputspace(used for SAM3 prompts)
        ex_boxes_512 = [_map_box_tile_to_512(bb, tile_w, tile_h, out_size=MODEL_INPUT_SIZE) for bb in ex_boxes_tile]

        # meanwhilesavetoallimagecoordinate system(used for debug/visualize)
        for bb in ex_boxes_tile:
            full_bb = [bb[0] + x0, bb[1] + y0, bb[2] + x0, bb[3] + y0]
            full_bb = clamp_box_xyxy(full_bb, W, H)
            if box_area_xyxy(full_bb) >= MIN_BOX_AREA:
                all_ex_boxes_full.append(full_bb)

        # 2) GD(in tile_512 onrun)
        tile_512 = _resize_to_512(tile, out_size=MODEL_INPUT_SIZE)
        pil_512 = Image.fromarray(cv2.cvtColor(tile_512, cv2.COLOR_BGR2RGB))

        gd_boxes_512 = call_dino_local_on_pil(pil_512, gd_prompt)
        # mapback tile coordinate(originalresolution)
        gd_boxes_tile = []
        for bb in gd_boxes_512:
            if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                continue
            # firstmap to tile originalresolutioncoordinate(mutualto tile)
            sx = float(tile_w) / float(MODEL_INPUT_SIZE)
            sy = float(tile_h) / float(MODEL_INPUT_SIZE)
            bx1, by1, bx2, by2 = bb
            tb = [bx1 * sx, by1 * sy, bx2 * sx, by2 * sy]
            tb = clamp_box_xyxy(tb, tile_w, tile_h)
            if box_area_xyxy(tb) >= MIN_BOX_AREA:
                gd_boxes_tile.append(tb)

        gd_boxes_tile = nms_dedup_boxes(gd_boxes_tile, iou_thr=0.6, min_area=MIN_BOX_AREA)

        # saveallimage gd boxes
        for bb in gd_boxes_tile:
            full_bb = [bb[0] + x0, bb[1] + y0, bb[2] + x0, bb[3] + y0]
            full_bb = clamp_box_xyxy(full_bb, W, H)
            if box_area_xyxy(full_bb) >= MIN_BOX_AREA:
                all_gd_boxes_full.append(full_bb)

        # 3) pass1: prompts = GD boxes + exemplar boxes(alluse 512 inputspacecoordinate)
        prompts_xywh = []
        prompts_lab  = []

        # GD boxes(512)
        for bb in gd_boxes_512:
            if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                continue
            x1, y1, x2b, y2b = [int(round(v)) for v in bb]
            w = max(1, x2b - x1)
            h = max(1, y2b - y1)
            prompts_xywh.append((x1, y1, w, h))
            prompts_lab.append(True)

        # exemplar boxes(512)
        for bb in ex_boxes_512:
            x1, y1, x2b, y2b = [int(round(v)) for v in bb]
            w = max(1, x2b - x1)
            h = max(1, y2b - y1)
            prompts_xywh.append((x1, y1, w, h))
            prompts_lab.append(True)

        masks1 = None
        if len(prompts_xywh) > 0:
            masks1, _, _ = sam3_text_plus_boxes_masks(processor, pil_512, prompts_xywh, prompts_lab, sam_prompt)
        masks1_list = dedup_masks(masks1, iou_thr=DEDUP_MASK_IOU_THR)

        # 4) pass2: text-only(oruse exemplar boxes doadditionalgeometric prompt)
        prompts_xywh2 = []
        prompts_lab2  = []
        for bb in ex_boxes_512:
            x1, y1, x2b, y2b = [int(round(v)) for v in bb]
            w = max(1, x2b - x1)
            h = max(1, y2b - y1)
            prompts_xywh2.append((x1, y1, w, h))
            prompts_lab2.append(True)

        if len(prompts_xywh2) > 0:
            masks2, _, _ = sam3_text_plus_boxes_masks(processor, pil_512, prompts_xywh2, prompts_lab2, sam_prompt)
        else:
            masks2, _, _ = sam3_text_only_masks(processor, pil_512, sam_prompt)

        masks2_list = dedup_masks(masks2, iou_thr=DEDUP_MASK_IOU_THR)

        # 5) merge & tile-level dedup (in 512 space)
        merged_512 = masks1_list + masks2_list
        merged_512 = dedup_masks(merged_512, iou_thr=DEDUP_MASK_IOU_THR)

        # 6) take tile   mask(512x512)mapbackoriginalimageallresolutioncoordinate: resize back tile size, again paste toallimage
        for m512 in merged_512:
            if m512 is None:
                continue
            m512 = m512.astype(np.uint8)
            mtile = cv2.resize(m512, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST).astype(bool)

            full_m = np.zeros((H, W), dtype=bool)
            full_m[y0:y2, x0:x2] = mtile
            if full_m.sum() > 0:
                all_masks_full.append(full_m)

    # 7) allimageleveldo notagaindeduplication(cross tile overlap deduplication)
    merged_full = dedup_masks(all_masks_full, iou_thr=DEDUP_MASK_IOU_THR)

    # --- newly added: according to A programdeleteboxresult, get"remaining  GD box", used forprotect mask notbydelete ---
    gd_boxes_remaining_for_protect = _gd_boxes_after_A_deletion(
        all_gd_boxes_full,
        delete_ratio=GD_BOX_DELETE_RATIO_FOR_PROTECT,
        seed=RANDOM_SEED,
    )

    if PERTURB_ENABLE:
        merged_full = perturb_masks_list(
            merged_full,
            delete_ratio=PERTURB_DELETE_RATIO,
            move_ratio=PERTURB_MOVE_RATIO,
            shift_min=MOVE_SHIFT_MIN,
            shift_max=MOVE_SHIFT_MAX,
            seed=RANDOM_SEED,
            protect_boxes=gd_boxes_remaining_for_protect,
            protect_iou_thr=MASK_PROTECT_IOU_THR,
        )
        # perturbafteragaindo onceallimagededuplication, avoidtranslateafteroverlap
        merged_full = dedup_masks(merged_full, iou_thr=DEDUP_MASK_IOU_THR)

        # allimage boxes(from mask outsideconnectbox)
        final_boxes_xyxy = []
        for m in merged_full:
            bb = mask_to_xyxy(m)
            if bb is None:
                continue
            bb = clamp_box_xyxy(bb, W, H)
            if box_area_xyxy(bb) >= MIN_BOX_AREA:
                final_boxes_xyxy.append(bb)
        final_boxes_xyxy = nms_dedup_boxes(final_boxes_xyxy, iou_thr=DEDUP_BOX_IOU_THR, min_area=MIN_BOX_AREA)

        all_gd_boxes_full = nms_dedup_boxes(all_gd_boxes_full, iou_thr=0.6, min_area=MIN_BOX_AREA)
        all_ex_boxes_full = nms_dedup_boxes(all_ex_boxes_full, iou_thr=0.5, min_area=MIN_BOX_AREA)

        return {
            "bgr": bgr_full,
            "H": H,
            "W": W,
            "masks": merged_full,
            "boxes": final_boxes_xyxy,
            "gd_boxes": all_gd_boxes_full,
            "ex_boxes": all_ex_boxes_full,
        }

def main():
    print(f"[Device] {DEVICE}")
    if TARGET_DATASET not in SAM3_PROMPT_MAP:
        raise ValueError(f"TARGET_DATASET={TARGET_DATASET} noin SAM3_PROMPT_MAP inconfiguration. ")

    # 1) listcanuseimage
    sample_imgs = get_split_images(TARGET_DATASET, SPLIT_FOR_SAMPLE)
    if len(sample_imgs) == 0:
        raise RuntimeError(f"split={SPLIT_FOR_SAMPLE} findnottoimage, pleasecheck ROOT_DATA_DIR anddirectorystructure. ")

    # 2) fixedselectobjectimage(usetop TARGET_IMAGE_NAME control)
    target_lower = TARGET_IMAGE_NAME.strip().lower()
    if not target_lower:
        raise RuntimeError("TARGET_IMAGE_NAME forempty. pleaseinfiletoptake TARGET_IMAGE_NAME change toyouneedrun imagename(if P0053.png). ")

    img_path = None
    for p in sample_imgs:
        if os.path.basename(p).lower() == target_lower:
            img_path = p
            break

    if img_path is None:
        show_n = min(30, len(sample_imgs))
        examples = [os.path.basename(p) for p in sample_imgs[:show_n]]
        raise RuntimeError(
            f"in split={SPLIT_FOR_SAMPLE}   ship imagelistfind innottoobjectimage: {TARGET_IMAGE_NAME}\n"
            f"pleasecheck: 1) filenameisor notcompleteallconsistent(containextension) 2) this imageisor notbelongin ship(DOTA_data list) 3) split isor notjustconfirm\n"
            f"examplebefore {show_n} canusefilename: {examples}"
        )

    base = os.path.basename(img_path)
    stem = os.path.splitext(base)[0]
    print(f"\n[Pick] objectimage(fixedspecify): {base}")

    # 3) exemplar(optional)
    exemplar_crops = []


    # 4) load SAM3 processor(oncei.e.can)
    processor = build_sam3_processor()

    # 5) runthismethod
    out = run_one_image(processor, img_path, exemplar_crops=exemplar_crops)

    if out is None:
        raise RuntimeError("outputforempty(maybeimagereadfail, ormodelinferencereturnempty). ")

    title = "B) GD+SAM3(text) | " + base
    viz = overlay_masks_and_boxes(out["bgr"], out["masks"], idx_text=title, gt_cnt=None, pseudo_cnt=None)

    os.makedirs(OUT_DIR, exist_ok=True)
    save_path = os.path.join(OUT_DIR, f"{stem}_B.jpg")
    cv2.imwrite(save_path, viz)
    print(f"[Saved] {save_path}")

    if SHOW_WINDOW:
        cv2.imshow("B) GD+SAM3(text)", viz)
        if DISPLAY_MS and int(DISPLAY_MS) > 0:
            cv2.waitKey(int(DISPLAY_MS))
        else:
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"\n[Done] resultimagealreadysaveto: {OUT_DIR}")


if __name__ == "__main__":
    main()
