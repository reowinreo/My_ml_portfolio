# -*- coding: utf-8 -*-
"""
GD + SAM3 dual-path object counting + arrangeleaderboard(ROI 512×512 version)

youneeddo task(restoredescribe): 
- originalpipelineis: whole imageonfirstuse GroundingDINO getbox; againuse SAM3 dotwicerecognition(pass1: one by one GD boxcrop; pass2: allimageplain text), finallycombineanddeduplicationcountandoutputleaderboard. 
- nowinin order toavoid SAM3 forcescaleto 1024×1024 withcome "largeimageinformationlose", wechangefor: 
  1) toeachoriginalimagerandomcropone 512×512   ROI(guaranteenotexceedboundary); 
  2) onlyin ROI onexecute: GD once + SAM3 twice + combineanddeduplication; 
  3) realcountalsochangefor"ROI inside" realvalue(RSOC-Building usepointfallin ROI inside; RSOC-*/Ship useannotationboxcenterfallin ROI inside; VisDrone use density map in ROI insidesum); 
  4) leaderboard/visualizelogickeep: stillprint"real labelleaderboard/pseudo labelleaderboard", andoutputvisualize(inoriginalimageondraw ROI box + resultbox). 

datasetrealvaluefileposition(andyou  dataset directorystructureconsistent): 
- RSOC-Building: 
  dataset/ASPDNet_dataset/RSOC_building/building/{train_data,test_data}/ground_truth/GT_*.mat
- RSOC-S-Vehicle / RSOC-L-Vehicle / RSOC-Ship(DOTA stylerotatebox txt): 
  dataset/ASPDNet_dataset/train/labelTxt-v1.0/trainset_reclabelTxt/*.txt
  dataset/ASPDNet_dataset/val/labelTxt-v1.0/valset_reclabelTxt/*.txt
- VD-People / VD-Vehicle(density map .npy): 
  dataset/VisDrone-People/{train,val,test}/Ground Truth/*.npy
  dataset/VisDrone-Vehicle/{train,val,test}/Ground Truth/*.npy

remark: 
- youprevious scriptalreadythroughtake GD change tolocal GroundingDINO(transformers), herecontinuealonguse; 
- GD  modelpath/ID stillin"user configuration section"insidecan be changed(GDINO_MODEL_ID). 
"""

import os
import sys
import random
import types
from unittest.mock import MagicMock

import numpy as np
import scipy.io as sio
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont


# ===================== user configuration section =====================

# datasetroot directory(youto is dataset; scriptwillautomaticin Dataset/dataset adaptive intervalshould)
ROOT_DATA_DIR = "Dataset"

# selectonedatasetclass: 
# "RSOC-Building", "RSOC-S-Vehicle", "RSOC-L-Vehicle", "RSOC-Ship",
# "VD-People", "VD-Vehicle"
TARGET_DATASET = "RSOC-Ship"

# randomselectmanyfewimage
NUM_SAMPLES = 10

# isor notpopuporiginalimage(PIL defaultviewer)
SHOW_ORIGINAL = True

# ROI setting: in ROI onrun GD+SAM3
USE_RANDOM_ROI = True
ROI_SIZE = 512
ROI_SEED = None  # setbecomeintegercanreproducerandom ROI, for example 123

# --------- newly added: samplefilterand ROI constraint ---------
# functionality1: inselectselect 10 imageprevious, firstcompute"whole imagerealvalue", onlyfromrealvalue > 80  imageinsiderandomselect
FULL_TRUE_THRESHOLD = 80

# functionality2: foreach imageselect 512×512 ROI when, as much as possibleensure ROI insiderealvalue >= 50
ROI_TRUE_THRESHOLD = 50

# ROI searchtimesnumberonlimit
ROI_MAX_TRIES = 200

# --------- newly added: SAM3 "imageprompt box" (manualboxselectonce, reusetothisbatch 10 image ROI) ---------
# isor notenable: inenterenter SAM3 previous, willpopupwindowletyouin"firstimage ROI image"onmanualdrawonebox; 
# thisboxwillaccording tomutualtocoordinatereusetothisbatch sohas ROI on, as SAM3  geometric promptbox(image prompt). 
ENABLE_MANUAL_IMAGE_PROMPT = True

# OpenCV interactivewindownamecall
MANUAL_PROMPT_WINDOW_NAME = "Select SAM3 prompt box on ROI (ENTER/SPACE confirm, ESC cancel)"

# ifyoutakeeliminateselect/boxselectnoeffect, isor notusedefaultcenterboxfallback
FALLBACK_TO_CENTER_PROMPT_BOX = True


# outputvisualizedirectory
OUT_DIR = "output_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- GD (Local GroundingDINO) parameter ----------
GD_BBOX_THRESHOLD = 0.13
GD_TEXT_THRESHOLD = 0.15

# youcantake GDINO_MODEL_ID change to: 
# 1) HuggingFace model id(if: "grounding_dino_base")
# 2) youlocalbelowloadgood modeldirectorypath(awaylineenvironmentrecommend)
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
    "RSOC-L-Vehicle": "vehicle",
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

# RSOC in DOTA labelTxt  classnamemap(realcount)
RSOC_DOTA_CATEGORY_MAP = {
    "RSOC-S-Vehicle": "small-vehicle",
    "RSOC-L-Vehicle": "large-vehicle",
    "RSOC-Ship": "ship",
}


# ===================== utility: adaptive path =====================

def _autofix_root_dir():
    """
    youto directorycall dataset(smallwrite), oldscriptdefault Dataset(largewrite); 
    hereautomaticselectsavein that. 
    """
    global ROOT_DATA_DIR
    if os.path.isdir(ROOT_DATA_DIR):
        return
    if os.path.isdir("dataset"):
        ROOT_DATA_DIR = "dataset"
        return
    if os.path.isdir("Dataset"):
        ROOT_DATA_DIR = "Dataset"

_autofix_root_dir()


# ===================== auxiliary: showoriginalimage =====================

def show_original_with_pil(img_path, img_id):
    try:
        print(f"  [Display] currentlyshoworiginalimage (ID={img_id})...")
        Image.open(img_path).show()
    except Exception as e:
        print(f"  [Warn] nomethodshoworiginalimage: {e}")


# ===================== newly added: manualselect"imageprompt box"(used for SAM3) =====================

def _default_center_rel_box():
    # center 50% region: x,y,w,h(mutualtocoordinate)
    return (0.25, 0.25, 0.5, 0.5)


def select_prompt_box_rel_on_roi(pil_roi, window_name=MANUAL_PROMPT_WINDOW_NAME):
    """
    infirstimage ROI imageonusemouseboxselectonerectangle, returnmutualtocoordinate (rx, ry, rw, rh), range [0,1]. 
    - ENTER/SPACE: confirmrecognize
    - ESC: takeeliminate
    """
    roi_rgb = np.array(pil_roi.convert("RGB"))
    roi_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)

    try:
        r = cv2.selectROI(window_name, roi_bgr, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(window_name)
    except Exception as e:
        print(f"  [Warn] OpenCV interactiveboxselectfail: {e}")
        return None

    x, y, w, h = [int(v) for v in r]
    if w <= 0 or h <= 0:
        print("  [Prompt] notselectinhaseffectbox(maybeaccording to  ESC orboxareafor 0). ")
        return None

    roi_h, roi_w = roi_bgr.shape[:2]
    rx = float(x) / float(roi_w)
    ry = float(y) / float(roi_h)
    rw = float(w) / float(roi_w)
    rh = float(h) / float(roi_h)

    rx = max(0.0, min(1.0, rx))
    ry = max(0.0, min(1.0, ry))
    rw = max(0.0, min(1.0 - rx, rw))
    rh = max(0.0, min(1.0 - ry, rh))
    return (rx, ry, rw, rh)


def rel_xywh_to_abs_xywh(rel_xywh, roi_w, roi_h):
    """
    (rx,ry,rw,rh) -> (x,y,w,h) pixel coordinates, anddoboundaryandminimumsizeprotect. 
    """
    rx, ry, rw, rh = rel_xywh
    x = int(round(rx * roi_w))
    y = int(round(ry * roi_h))
    w = int(round(rw * roi_w))
    h = int(round(rh * roi_h))

    x = max(0, min(int(roi_w - 1), x))
    y = max(0, min(int(roi_h - 1), y))
    w = max(1, min(int(roi_w - x), w))
    h = max(1, min(int(roi_h - y), h))
    return (x, y, w, h)



# ===================== datasetindexandrealvalueread =====================

def list_images(img_dir):
    if not os.path.isdir(img_dir):
        return []
    return [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(IMAGE_EXTS)]


def load_rsoc_building_points(mat_path):
    data = sio.loadmat(mat_path)
    if "center" not in data:
        return np.zeros((0, 2), dtype=float)
    try:
        pts = np.asarray(data["center"][0, 0], dtype=float)
        # pts: [N,2] with (x,y)
        return pts
    except Exception:
        return np.zeros((0, 2), dtype=float)


def build_samples_for_dataset(dataset_label):
    """
    return list[dict]: 
      {
        "img_path": str,
        "true_count_full": float,   # whole imagerealvalue(onlyused forreference)
        "gt_type": "points"/"boxes"/"density",
        "gt_path": str | None,      # realvaluefilepath(boxes/density mustneed; points used forbacktrace)
        "gt_data": object | None,   # points: np.ndarray [N,2]; otherfor None
      }
    """
    samples = []

    # 1) RSOC-Building: .mat inpointnumber
    if dataset_label == "RSOC-Building":
        rsoc_root = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "RSOC_building", "building")
        for subdir in ["train_data", "test_data"]:
            img_dir = os.path.join(rsoc_root, subdir, "images")
            gt_dir = os.path.join(rsoc_root, subdir, "ground_truth")
            imgs = list_images(img_dir)
            for img_path in imgs:
                base = os.path.splitext(os.path.basename(img_path))[0]
                gt_path = os.path.join(gt_dir, f"GT_{base}.mat")
                if not os.path.exists(gt_path):
                    continue
                pts = load_rsoc_building_points(gt_path)
                samples.append({
                    "img_path": img_path,
                    "true_count_full": float(pts.shape[0]),
                    "gt_type": "points",
                    "gt_path": gt_path,
                    "gt_data": pts,
                })

    # 2) RSOC-S/L-Vehicle, RSOC-Ship: labelTxt rotatebox
    elif dataset_label in ["RSOC-S-Vehicle", "RSOC-L-Vehicle", "RSOC-Ship"]:
        target_cat = RSOC_DOTA_CATEGORY_MAP[dataset_label]

        def collect_split(split):
            if split == "train":
                img_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "train", "images")
                label_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "train",
                                         "labelTxt-v1.0", "trainset_reclabelTxt")
            elif split == "val":
                img_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "val", "images")
                label_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "val",
                                         "labelTxt-v1.0", "valset_reclabelTxt")
            else:
                return

            if not os.path.isdir(label_dir) or not os.path.isdir(img_dir):
                return

            base_to_img = {os.path.splitext(os.path.basename(p))[0]: p for p in list_images(img_dir)}

            for txt_name in os.listdir(label_dir):
                if not txt_name.endswith(".txt"):
                    continue
                base = os.path.splitext(txt_name)[0]
                if base not in base_to_img:
                    continue

                gt_path = os.path.join(label_dir, txt_name)

                count = 0
                with open(gt_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        parts = line.strip().split()
                        # typicalformat: 8 coords + class + difficult
                        if len(parts) >= 9 and parts[-2] == target_cat:
                            count += 1

                if count > 0:
                    samples.append({
                        "img_path": base_to_img[base],
                        "true_count_full": float(count),
                        "gt_type": "boxes",
                        "gt_path": gt_path,
                        "gt_data": None,
                        "target_cat": target_cat,
                    })

        collect_split("train")
        collect_split("val")

    # 3) VD-People / VD-Vehicle: density map sum
    elif dataset_label in ["VD-People", "VD-Vehicle"]:
        vd_root = os.path.join(ROOT_DATA_DIR, "VisDrone-People" if dataset_label == "VD-People" else "VisDrone-Vehicle")
        for split in ["train", "val", "test"]:
            img_dir = None
            for cand in [os.path.join(vd_root, split, "images"), os.path.join(vd_root, split, "Images")]:
                if os.path.isdir(cand):
                    img_dir = cand
                    break
            gt_dir = os.path.join(vd_root, split, "Ground Truth")
            if not img_dir or not os.path.isdir(gt_dir):
                continue

            for fname in os.listdir(gt_dir):
                if not fname.lower().endswith(".npy"):
                    continue
                base = os.path.splitext(fname)[0]
                img_path = None
                for ext in IMAGE_EXTS:
                    cand = os.path.join(img_dir, base + ext)
                    if os.path.exists(cand):
                        img_path = cand
                        break
                if not img_path:
                    continue

                gt_path = os.path.join(gt_dir, fname)
                try:
                    density = np.load(gt_path)
                    samples.append({
                        "img_path": img_path,
                        "true_count_full": float(density.sum()),
                        "gt_type": "density",
                        "gt_path": gt_path,
                        "gt_data": None,
                    })
                except Exception:
                    pass
    else:
        raise ValueError(f"notknowdatasetlabel: {dataset_label}")

    print(f"[Index] {dataset_label}: total {len(samples)} one withrealcount sample")
    return samples


def choose_samples_with_constraints(samples, num_samples=10):
    """
    newly addedtwoitemconstraint: 
    1) firstaccording towhole imagerealvalue true_count_full > FULL_TRUE_THRESHOLD filter; 
    2) foreach imageselect ROI when, as much as possibleguarantee ROI insiderealvalue >= ROI_TRUE_THRESHOLD. 

    return chosen list(everyelementwillinclude: id / roi_xyxy / true_count). 
    """
    if not samples:
        return []

    candidates = [s for s in samples if float(s.get("true_count_full", 0.0)) > float(FULL_TRUE_THRESHOLD)]
    print(f"[Filter] whole imagerealvalue > {FULL_TRUE_THRESHOLD}  number of samples: {len(candidates)} / {len(samples)}")
    if not candidates:
        print("[Warn] noanysamplefullenoughwhole imagerealvaluethreshold, nomethodaccording toneedrequestextracttake. ")
        return []

    random.shuffle(candidates)

    chosen = []
    backups = []  # notreachto ROI_TRUE_THRESHOLD  prepareselect(used forfallback)

    for s in candidates:
        if len(chosen) >= num_samples:
            break

        img_bgr = cv2.imread(s["img_path"])
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]

        roi_xyxy, roi_true = choose_best_roi_for_sample(s, w, h, ROI_SIZE, ROI_MAX_TRIES)
        s["roi_xyxy"] = roi_xyxy
        s["true_count"] = roi_true

        if roi_true >= float(ROI_TRUE_THRESHOLD):
            chosen.append(s)
        else:
            backups.append(s)

    # fallback: ifstrictcellfullenough ROI_TRUE_THRESHOLD  samplenotenough num_samples, thenfromprepareselectinsideaccording to ROI realvaluefromlargetosmallpad
    if len(chosen) < num_samples and backups:
        backups_sorted = sorted(backups, key=lambda x: float(x.get("true_count", 0.0)), reverse=True)
        need = num_samples - len(chosen)
        chosen.extend(backups_sorted[:need])
        print(
            f"[Warn] onlyfindto {len(chosen) - min(need, len(backups_sorted))} image ROI realvalue >= {ROI_TRUE_THRESHOLD}  image; "
            f"alreadyuseprepareselectpad {min(need, len(backups_sorted))} image(maybelowinthreshold). "
        )

    # ifstillnotenough, thendirectreturnnowhas(tooutprompt)
    if len(chosen) < num_samples:
        print(f"[Warn] finalonlyselectto {len(chosen)} / {num_samples} imagesample. youcandecreaselowthresholdoradddatasetscale. ")

    for i, s in enumerate(chosen):
        s["id"] = i + 1
    return chosen


def print_real_ranking(chosen):
    ranking_real = sorted(chosen, key=lambda x: x.get("true_count", 0.0), reverse=True)
    print("\n[Ranking] real labelleaderboard(ROI insiderealvalue): ")
    for rank, s in enumerate(ranking_real, start=1):
        roi = s.get("roi_xyxy", None)
        roi_str = f"ROI=({roi[0]},{roi[1]},{roi[2]},{roi[3]})" if roi else "ROI=None"
        print(f"  Rank {rank:2d}: ID={s['id']:2d} | TrueROI={s.get('true_count', 0.0):.2f} | {roi_str} | {s['img_path']}")


# ===================== ROI realvaluecompute =====================

_DOTA_CENTER_CACHE = {}
_DENSITY_CACHE = {}

def _load_dota_centers(gt_txt_path, target_cat):
    """
    parse DOTA stylerotatebox txt, returnthis classobject center pointarray [N,2]. 
    hereuse"center pointfallin ROI inside"comecount(simple, robust, noneedmanysideshapemutualintersection). 
    """
    key = (gt_txt_path, target_cat)
    if key in _DOTA_CENTER_CACHE:
        return _DOTA_CENTER_CACHE[key]

    centers = []
    try:
        with open(gt_txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                if parts[-2] != target_cat:
                    continue
                # before 8 iscoordinate
                try:
                    coords = list(map(float, parts[:8]))
                    xs = coords[0::2]
                    ys = coords[1::2]
                    cx = float(sum(xs) / 4.0)
                    cy = float(sum(ys) / 4.0)
                    centers.append((cx, cy))
                except Exception:
                    continue
    except Exception:
        centers = []

    arr = np.asarray(centers, dtype=float) if centers else np.zeros((0, 2), dtype=float)
    _DOTA_CENTER_CACHE[key] = arr
    return arr


def _load_density(gt_npy_path):
    if gt_npy_path in _DENSITY_CACHE:
        return _DENSITY_CACHE[gt_npy_path]
    try:
        d = np.load(gt_npy_path)
    except Exception:
        d = None
    _DENSITY_CACHE[gt_npy_path] = d
    return d


def compute_true_count_in_roi(sample, roi_xyxy):
    """
    roi_xyxy: (x1,y1,x2,y2) in full-image coordinates.
    return float(ROI insiderealvalue)
    """
    x1, y1, x2, y2 = roi_xyxy
    gt_type = sample.get("gt_type")

    if gt_type == "points":
        pts = sample.get("gt_data", None)
        if pts is None or len(pts) == 0:
            return 0.0
        xs = pts[:, 0]
        ys = pts[:, 1]
        m = (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)
        return float(np.sum(m))

    if gt_type == "boxes":
        gt_path = sample.get("gt_path")
        target_cat = sample.get("target_cat")
        if not gt_path or not target_cat:
            return 0.0
        centers = _load_dota_centers(gt_path, target_cat)
        if centers.shape[0] == 0:
            return 0.0
        xs = centers[:, 0]
        ys = centers[:, 1]
        m = (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)
        return float(np.sum(m))

    if gt_type == "density":
        gt_path = sample.get("gt_path")
        if not gt_path:
            return 0.0
        density = _load_density(gt_path)
        if density is None:
            return 0.0
        # density usuallyandimagesamesize; hereaccording topixelwindowsum
        # Note:numpy indexis [y, x]
        yy1 = max(0, int(round(y1)))
        yy2 = min(int(density.shape[0]), int(round(y2)))
        xx1 = max(0, int(round(x1)))
        xx2 = min(int(density.shape[1]), int(round(x2)))
        if yy2 <= yy1 or xx2 <= xx1:
            return 0.0
        return float(np.sum(density[yy1:yy2, xx1:xx2]))

    return 0.0


def choose_random_roi_xyxy(img_w, img_h, roi_size=512):
    """
    in [0,w]x[0,h] insiderandomselectone roi_size×roi_size  window(guaranteenotexceedboundary). 
    return (x1,y1,x2,y2)
    """
    if img_w < roi_size or img_h < roi_size:
        # logicdiscussonyouguarantee >=512, butheredofallback: pastesidetakeminimumcanrowwindow
        roi_size = int(min(img_w, img_h))
    x1 = random.randint(0, max(0, img_w - roi_size))
    y1 = random.randint(0, max(0, img_h - roi_size))
    return (x1, y1, x1 + roi_size, y1 + roi_size)


def _clamp_roi_top_left(x1, y1, img_w, img_h, roi_size):
    x1 = int(max(0, min(int(img_w - roi_size), int(x1))))
    y1 = int(max(0, min(int(img_h - roi_size), int(y1))))
    return x1, y1


def _best_roi_on_density_map(density, roi_size):
    """in density map onfind sum maximum  roi_size×roi_size window, return (x1,y1,best_sum). """
    if density is None:
        return 0, 0, 0.0
    H, W = density.shape[:2]
    if H < roi_size or W < roi_size:
        roi_size = int(min(H, W))
    # productpartimage: pad 1, theninusehalfopenintervalsum
    integ = np.pad(density.astype(np.float64), ((1, 0), (1, 0)), mode="constant")
    integ = integ.cumsum(axis=0).cumsum(axis=1)
    # windowandmatrix S: shape (H-roi+1, W-roi+1)
    S = (
        integ[roi_size:, roi_size:]
        - integ[:-roi_size, roi_size:]
        - integ[roi_size:, :-roi_size]
        + integ[:-roi_size, :-roi_size]
    )
    if S.size == 0:
        return 0, 0, 0.0
    y1, x1 = np.unravel_index(np.argmax(S), S.shape)
    best = float(S[y1, x1])
    return int(x1), int(y1), best


def choose_best_roi_for_sample(sample, img_w, img_h, roi_size, max_tries):
    """
    forsingleimageselectone ROI: 
    - density: directselect density sum maximum window(usuallymoststable); 
    - points/boxes: use"withobjectpoint/center pointforintroducelead" randomsampling, take ROI insiderealvaluemaximum that; 
    - fallback: purerandomsamplingtakemaximum. 

    Returns:roi_xyxy, best_true
    """
    gt_type = sample.get("gt_type")

    # fallback: if roi_size super imageboundary
    roi_size_eff = int(min(int(roi_size), int(img_w), int(img_h)))
    if roi_size_eff <= 0:
        return (0, 0, 0, 0), 0.0

    # --- density: directrequestmaximumwindow ---
    if gt_type == "density":
        density = _load_density(sample.get("gt_path"))
        x1, y1, best_sum = _best_roi_on_density_map(density, roi_size_eff)
        x1, y1 = _clamp_roi_top_left(x1, y1, img_w, img_h, roi_size_eff)
        roi = (x1, y1, x1 + roi_size_eff, y1 + roi_size_eff)
        # forstableproperstartsee, useunifyfunctionagaincomputeonetimes(avoid density shape notconsistentwhenoutwrong)
        best_true = float(compute_true_count_in_roi(sample, roi))
        return roi, best_true

    # --- points/boxes: withobjectpoint/center pointintroduceleadsampling ---
    anchors = None
    if gt_type == "points":
        anchors = sample.get("gt_data")
    elif gt_type == "boxes":
        gt_path = sample.get("gt_path")
        target_cat = sample.get("target_cat")
        if gt_path and target_cat:
            anchors = _load_dota_centers(gt_path, target_cat)

    best_roi = choose_random_roi_xyxy(img_w, img_h, roi_size_eff)
    best_true = float(compute_true_count_in_roi(sample, best_roi))

    if anchors is None or len(anchors) == 0:
        # no anchors: purerandom best-of-N
        for _ in range(int(max_tries)):
            roi = choose_random_roi_xyxy(img_w, img_h, roi_size_eff)
            t = float(compute_true_count_in_roi(sample, roi))
            if t > best_true:
                best_true = t
                best_roi = roi
        return best_roi, best_true

    anchors = np.asarray(anchors, dtype=float)
    # limitation tries
    max_tries = int(max(10, max_tries))
    for _ in range(max_tries):
        # takeonerandom anchor, initsjitter aroundmove, make ROI moremaybeincludemoremanyobject
        idx = random.randint(0, anchors.shape[0] - 1)
        cx, cy = float(anchors[idx, 0]), float(anchors[idx, 1])

        jitter = roi_size_eff * 0.25
        x1 = cx - roi_size_eff / 2.0 + random.uniform(-jitter, jitter)
        y1 = cy - roi_size_eff / 2.0 + random.uniform(-jitter, jitter)
        x1, y1 = _clamp_roi_top_left(x1, y1, img_w, img_h, roi_size_eff)
        roi = (int(x1), int(y1), int(x1 + roi_size_eff), int(y1 + roi_size_eff))

        t = float(compute_true_count_in_roi(sample, roi))
        if t > best_true:
            best_true = t
            best_roi = roi

        # earlystop: ifalreadythroughreachtothreshold, directreturn(reducenomeaningsampling)
        if best_true >= float(ROI_TRUE_THRESHOLD):
            break

    return best_roi, best_true


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
        # shapeif "car." -> "car ."
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
            "failed to import transformers   GroundingDINO related modules. Please install first/Upgrade:"
            "pip install -U transformers"
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


def sam3_text_plus_multi_boxes(processor, pil_img, boxes_xywh_list, prompt_text):
    """
    text + multiple geometriesboxTip:
    - first set_image + set_text_prompt
    - againaccordingtimes add_geometric_prompt(box=..., label=True)
    return boxes: [M,4] (xyxy)

    purpose: take"manualprompt box + GD box"togetherasprompt, avoidtoevery GD boxreverserestorecropinference. 
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
        return np.zeros((0, 4), dtype=float)

    if torch.is_tensor(boxes):
        boxes_np = boxes.detach().cpu().numpy()
    else:
        boxes_np = np.array(boxes)
    return boxes_np.astype(float)


def sam3_text_only(processor, pil_img, prompt_text):
    """
    plain text PCS(notaddanypoint/box/maskprompt): 
    - onlyuse prompt_text inwhole imagefindsohasmatchinstance
    return boxes: [M, 4] (xyxy)
    """
    state = processor.set_image(pil_img)
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


# ===================== visualize(inoriginalimageondraw ROI + box) =====================

def draw_boxes(img_path, roi_xyxy, prompt_box_full, gd_boxes, sam_boxes_pass1, sam_boxes_pass2, final_boxes, img_id, text):
    try:
        image = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except Exception:
            font = ImageFont.load_default()

        # ROI: yellow
        if roi_xyxy is not None:
            draw.rectangle(list(map(float, roi_xyxy)), outline="yellow", width=4)

        # manualprompt box: orange
        if prompt_box_full is not None:
            draw.rectangle(list(map(float, prompt_box_full)), outline="orange", width=4)

        # GD: red
        for b in gd_boxes:
            draw.rectangle(b, outline="red", width=3)

        # SAM pass1: blue
        for b in sam_boxes_pass1:
            draw.rectangle(b, outline="blue", width=2)

        # SAM pass2: purple
        for b in sam_boxes_pass2:
            draw.rectangle(b, outline="magenta", width=2)

        # Final: green
        for b in final_boxes:
            draw.rectangle(b, outline="lime", width=3)

        draw.text((10, 10), text, fill="yellow", font=font)

        save_path = os.path.join(OUT_DIR, f"roi512_merged_id_{img_id}.jpg")
        image.save(save_path)
        print(f"  [Saved] visualize: {save_path}")
    except Exception as e:
        print(f"  [Warn] visualizefail: {e}")


# ===================== main pipeline =====================

def main():
    if ROI_SEED is not None:
        random.seed(int(ROI_SEED))
        np.random.seed(int(ROI_SEED))

    print(f"[Main] Dataset={TARGET_DATASET} | sample_n={NUM_SAMPLES} | ROOT={ROOT_DATA_DIR}")
    if TARGET_DATASET not in SAM3_PROMPT_MAP or TARGET_DATASET not in GD_PROMPT_MAP:
        print("[Error] TARGET_DATASET notinpromptwordmaptableinside, pleasecheck. ")
        return

    sam_prompt = SAM3_PROMPT_MAP[TARGET_DATASET]
    gd_prompt = GD_PROMPT_MAP[TARGET_DATASET]
    print(
        f"[Main] GD prompt=\"{gd_prompt}\" | SAM3 prompt=\"{sam_prompt}\" | "
        f"ROI={ROI_SIZE}x{ROI_SIZE} | full_true>{FULL_TRUE_THRESHOLD} | roi_true>={ROI_TRUE_THRESHOLD} | roi_max_tries={ROI_MAX_TRIES}"
    )

    samples = build_samples_for_dataset(TARGET_DATASET)
    if not samples:
        print("[Main] notfindtosample, end. ")
        return

    # newly added: firstfilterwhole imagerealvalue, againforeach imageselectselect"as much as possiblehighrealvalue"  ROI
    chosen = choose_samples_with_constraints(samples, NUM_SAMPLES)
    if not chosen:
        return

    print_real_ranking(chosen)

    # ---------- newly added: inenterenter SAM3 previous, firstin"firstimage ROI image"onmanualboxselectoneprompt box, andrestoreused forthisbatch ----------
    manual_prompt_rel = None
    if ENABLE_MANUAL_IMAGE_PROMPT:
        s0 = chosen[0]
        img0_bgr = cv2.imread(s0["img_path"])
        if img0_bgr is None or s0.get("roi_xyxy", None) is None:
            print("[Prompt] firstimagereadfailor ROI missing, nomethodperformmanualboxselect. ")
        else:
            pil0_full = Image.fromarray(cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2RGB))
            rx1, ry1, rx2, ry2 = s0["roi_xyxy"]
            pil0_roi = pil0_full.crop((rx1, ry1, rx2, ry2))
            print("\n[Prompt] pleaseinpopup windowinto'firstimage ROI image'manualboxselectonerectangle, as SAM3  imageprompt box. ")
            manual_prompt_rel = select_prompt_box_rel_on_roi(pil0_roi)

        if manual_prompt_rel is None and FALLBACK_TO_CENTER_PROMPT_BOX:
            manual_prompt_rel = _default_center_rel_box()
            print(f"[Prompt] usedefaultcenterprompt box rel_xywh={manual_prompt_rel}")
        elif manual_prompt_rel is not None:
            print(f"[Prompt] alreadyrecordprompt box rel_xywh={manual_prompt_rel}")
    else:
        print("[Prompt] alreadydisablemanualimageprompt boxfunctionality. ")

    processor = build_sam3_processor()

    print("\n[Pipeline] (ROI) GD -> SAM3(pass1 in-box crops) -> SAM3(pass2 ROI text-only) -> merge&dedup -> ranking\n")

    for s in chosen:
        img_id = s["id"]
        img_path = s["img_path"]
        roi_xyxy = s.get("roi_xyxy", None)
        print(f"\n[{img_id}/{len(chosen)}] Processing: {img_path}")

        if SHOW_ORIGINAL:
            show_original_with_pil(img_path, img_id)

        img_bgr = cv2.imread(img_path)
        if img_bgr is None or roi_xyxy is None:
            print("  [Warn] readfailor ROI missing, recordfor 0. ")
            s["pseudo_count"] = 0.0
            continue

        h_full, w_full = img_bgr.shape[:2]
        pil_full = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        rx1, ry1, rx2, ry2 = roi_xyxy
        pil_roi = pil_full.crop((rx1, ry1, rx2, ry2))
        roi_w, roi_h = pil_roi.size  # shouldfor 512x512

        print(f"  [ROI] xyxy={roi_xyxy} | size={roi_w}x{roi_h} | trueROI={s.get('true_count', 0.0):.2f}")

        # ---------- 1) GD(ROI on)takebox ----------
        gd_boxes_roi = call_dino_local_on_pil(pil_roi, gd_prompt)
        gd_boxes_roi = [clamp_box_xyxy(b, roi_w, roi_h) for b in gd_boxes_roi
                        if isinstance(b, (list, tuple)) and len(b) == 4]
        print(f"  [GD@ROI] boxes = {len(gd_boxes_roi)}")

                # ---------- 2) SAM3 pass1 ----------
        # ifenable "manualimageprompt box", thenadopt: in ROI ononce-ityaddaddmanygeometric promptbox(manualprompt box + GD box), avoidtoevery GD boxreverserestorecropinference. 
        # elsekeeporiginallogic: one by one GD boxcropafterin crop onprompt. 
        sam_boxes_pass1_roi = []

        if manual_prompt_rel is not None:
            mp_xywh = rel_xywh_to_abs_xywh(manual_prompt_rel, roi_w, roi_h)

            gd_xywh_list = []
            for b in gd_boxes_roi:
                x1, y1, x2, y2 = [int(round(v)) for v in b]
                w = max(1, x2 - x1)
                h = max(1, y2 - y1)
                gd_xywh_list.append((x1, y1, w, h))

            multi_prompts = [mp_xywh] + gd_xywh_list
            boxes_np = sam3_text_plus_multi_boxes(processor, pil_roi, multi_prompts, sam_prompt)
            for bb in boxes_np:
                sam_boxes_pass1_roi.append(clamp_box_xyxy(list(map(float, bb)), roi_w, roi_h))
        else:
            for b in gd_boxes_roi:
                x1, y1, x2, y2 = [int(round(v)) for v in b]
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = pil_roi.crop((x1, y1, x2, y2))
                cw, ch = crop.size
                if cw <= 2 or ch <= 2:
                    continue

                crop_boxes = sam3_text_plus_box(processor, crop, (0, 0, cw, ch), sam_prompt)

                for cb in crop_boxes:
                    cb = list(map(float, cb))
                    cb[0] += x1
                    cb[1] += y1
                    cb[2] += x1
                    cb[3] += y1
                    sam_boxes_pass1_roi.append(clamp_box_xyxy(cb, roi_w, roi_h))

        sam_boxes_pass1_roi = nms_dedup_boxes(sam_boxes_pass1_roi, iou_thr=DEDUP_IOU_THR, min_area=MIN_BOX_AREA)
        print(f"  [SAM3 pass1@ROI] merged = {len(sam_boxes_pass1_roi)}")

        # ---------- 3) SAM3 pass2 ----------
        # according to your needrequest: pass2 notuse GD boxprompt; ifenable manualprompt box, thenuse"manualprompt box"asonlyonegeometric prompt, elsekeepplain text. 
        sam_boxes_pass2_roi = []

        if manual_prompt_rel is not None:
            mp_xywh = rel_xywh_to_abs_xywh(manual_prompt_rel, roi_w, roi_h)
            boxes_np = sam3_text_plus_box(processor, pil_roi, mp_xywh, sam_prompt)
            for bb in boxes_np:
                sam_boxes_pass2_roi.append(clamp_box_xyxy(list(map(float, bb)), roi_w, roi_h))
        else:
            full_boxes = sam3_text_only(processor, pil_roi, sam_prompt)
            if full_boxes.shape[0] > 0:
                for fb in full_boxes:
                    sam_boxes_pass2_roi.append(clamp_box_xyxy(list(map(float, fb)), roi_w, roi_h))

        sam_boxes_pass2_roi = nms_dedup_boxes(sam_boxes_pass2_roi, iou_thr=DEDUP_IOU_THR, min_area=MIN_BOX_AREA)
        print(f"  [SAM3 pass2@ROI] merged = {len(sam_boxes_pass2_roi)}")

        # ---------- 4) combineandtwiceresultdeduplication(ROI inside) ----------
        merged_roi = sam_boxes_pass1_roi + sam_boxes_pass2_roi
        final_boxes_roi = nms_dedup_boxes(merged_roi, iou_thr=DEDUP_IOU_THR, min_area=MIN_BOX_AREA)
        final_count = float(len(final_boxes_roi))
        s["pseudo_count"] = final_count

        print(f"  [Count@ROI] pseudoROI={final_count:.2f} | trueROI={s.get('true_count', 0.0):.2f}")        # visualize: take ROI coordinate system boxtranslatebackoriginalimagecoordinate system
        prompt_box_full = None
        if manual_prompt_rel is not None:
            mp_xywh = rel_xywh_to_abs_xywh(manual_prompt_rel, roi_w, roi_h)
            mx, my, mw, mh = mp_xywh
            prompt_box_full = [mx + rx1, my + ry1, mx + mw + rx1, my + mh + ry1]

        gd_boxes_full = shift_boxes_xyxy(gd_boxes_roi, rx1, ry1, w_full, h_full)
        sam1_full = shift_boxes_xyxy(sam_boxes_pass1_roi, rx1, ry1, w_full, h_full)
        sam2_full = shift_boxes_xyxy(sam_boxes_pass2_roi, rx1, ry1, w_full, h_full)
        final_full = shift_boxes_xyxy(final_boxes_roi, rx1, ry1, w_full, h_full)

        text = (f"ID:{img_id} | ROI:{roi_w}x{roi_h} | "
                f"GD:{len(gd_boxes_roi)} | SAM1:{len(sam_boxes_pass1_roi)} | SAM2:{len(sam_boxes_pass2_roi)} | "
                f"Final:{int(final_count)} | TrueROI:{s.get('true_count', 0.0):.0f}")
        draw_boxes(img_path, roi_xyxy, prompt_box_full, gd_boxes_full, sam1_full, sam2_full, final_full, img_id, text)

    # ---------- 5) pseudo labelleaderboard ----------
    ranking_pseudo = sorted(chosen, key=lambda x: x.get("pseudo_count", 0.0), reverse=True)
    print("\npseudo labelleaderboard: ")
    for rank, s in enumerate(ranking_pseudo, start=1):
        roi = s.get("roi_xyxy", None)
        roi_str = f"ROI=({roi[0]},{roi[1]},{roi[2]},{roi[3]})" if roi else "ROI=None"
        print(f"  Rank {rank:2d}: ID={s['id']:2d} | detection={s.get('pseudo_count', 0.0):.2f} | real label={s.get('true_count', 0.0):.2f}")

    print("\n[Done] finish ROI(512x512) on  GD + SAM3 dual-pathcount + leaderboard. ")


if __name__ == "__main__":
    main()
