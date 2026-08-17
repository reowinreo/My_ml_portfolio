# -*- coding: utf-8 -*-
"""
GD + SAM3 dual-path object counting + leaderboard(combineandversion)

needrequestimplement: 
1) firstuse GroundingDINO (DeepDataSpace API) getdetectionbox; 
2) No.once SAM3: in"sohas GD boxinside"doinference(toeveryboxcropafteruse SAM3 count), get pass1; 
3) secondtimes SAM3: in"allimage"doinference(useone exemplar box trigger SAM3 textconceptsegmentation), get pass2; 
4) will pass1 + pass2  resultdodeduplicationcombineand, getfinalcount pseudo_count; 
5) retain"real labelleaderboard"and"pseudo labelleaderboard" printformat/logic. 

Description:
- youneed: 
  - DDS API Token: suggestputtoenvironmentvariable DDS_API_TOKEN; ordirectchangebelowsurface API_TOKEN. 
  - local SAM3 source codedirectory LOCAL_SAM3_PATH andweight SAM3_CHECKPOINT_PATH. 

referencecombineandselfyou provide twoscript: 
- callgd-apiviewonegroupimage effect.py(GD API + real/pseudoleaderboardframework)
- leaderboard.py(SAM3 load + manydatasetrealcount + leaderboard)
"""

import os
import sys
import time
import random
import json
import base64
import io
import types
from unittest.mock import MagicMock

import requests
import urllib3
import numpy as np
import scipy.io as sio
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont

# disable SSL warning(DDS API)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ===================== user configuration section =====================

# DeepDataSpace Token: preferfromenvironmentvariableread, avoidtake Token writeentercode
API_TOKEN = os.getenv("DDS_API_TOKEN", "89572a070f19b4a331925dcbebe05c09").strip()  # orone who: API_TOKEN = "xxx"

# datasetroot directory
ROOT_DATA_DIR = "Dataset"

# selectonedatasetclass: 
# "RSOC-Building", "RSOC-S-Vehicle", "RSOC-L-Vehicle", "RSOC-Ship",
# "VD-People", "VD-Vehicle"
TARGET_DATASET = "RSOC-Building"

# randomselectmanyfewimage
NUM_SAMPLES = 10

# isor notpopuporiginalimage(PIL defaultviewer)
SHOW_ORIGINAL = True

# outputvisualizedirectory
OUT_DIR = "output_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- GD (DDS API) parameter ----------
GD_MODEL = "GroundingDino-1.6-Pro"
GD_BBOX_THRESHOLD = 0.15
GD_IOU_THRESHOLD = 0.3
GD_POLL_MAX_ITERS = 20
GD_POLL_SLEEP_SEC = 2

# ---------- SAM3 parameter ----------
LOCAL_SAM3_PATH = r"D:\sam3_source"
SAM3_CHECKPOINT_PATH = "saved_models/SAM3.pt"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAM3_CONFIDENCE_THRESHOLD = 0.25

# ---------- count deduplication parameters ----------
DEDUP_IOU_THR = 0.7   # exceedlargeexceed"easycombineandrepeatbox"; 0.5~0.7 oftenuse
MIN_BOX_AREA = 16.0   # filterextremesmallnoisebox(pixel^2)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# SAM3 text prompt (more biased toward"concept")
SAM3_PROMPT_MAP = {
    "RSOC-Building": "rooftop.structure",
    "RSOC-S-Vehicle": "smallcar",
    "RSOC-L-Vehicle": "vehicle",
    "RSOC-Ship": "ship",
    "VD-People": "person",
    "VD-Vehicle": "vehicle",
}

# GD text prompt (more biased toward"detectionword")
GD_PROMPT_MAP = {
    "RSOC-Building": "rooftop.structure",
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

# ===================== auxiliary: showoriginalimage =====================

def show_original_with_pil(img_path, img_id):
    try:
        print(f"  [Display] currentlyshoworiginalimage (ID={img_id})...")
        Image.open(img_path).show()
    except Exception as e:
        print(f"  [Warn] nomethodshoworiginalimage: {e}")

# ===================== datasetandrealcount =====================

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
        return pts
    except Exception:
        return np.zeros((0, 2), dtype=float)

def build_samples_for_dataset(dataset_label):
    """
    return list[dict]: 
      {
        "img_path": str,
        "true_count": float,
        "gt_type": "points"/"boxes"/"density",
        "gt_data": correspondformatdata
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
                    "true_count": float(pts.shape[0]),
                    "gt_type": "points",
                    "gt_data": pts,
                })

    # 2) RSOC-S/L-Vehicle, RSOC-Ship: labelTxt boxcount
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

                count = 0
                with open(os.path.join(label_dir, txt_name), "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 9 and parts[-2] == target_cat:
                            count += 1

                if count > 0:
                    samples.append({
                        "img_path": base_to_img[base],
                        "true_count": float(count),
                        "gt_type": "boxes",
                        "gt_data": None,
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

                try:
                    density = np.load(os.path.join(gt_dir, fname))
                    samples.append({
                        "img_path": img_path,
                        "true_count": float(density.sum()),
                        "gt_type": "density",
                        "gt_data": None,
                    })
                except Exception:
                    pass
    else:
        raise ValueError(f"notknowdatasetlabel: {dataset_label}")

    print(f"[Index] {dataset_label}: total {len(samples)} one withrealcount sample")
    return samples

def build_real_ranking(samples, num_samples=10):
    if len(samples) == 0:
        return []

    chosen = samples.copy() if len(samples) <= num_samples else random.sample(samples, num_samples)
    for i, s in enumerate(chosen):
        s["id"] = i + 1

    ranking_real = sorted(chosen, key=lambda x: x["true_count"], reverse=True)
    print("\n[Ranking] real labelleaderboard: ")
    for rank, s in enumerate(ranking_real, start=1):
        print(f"  Rank {rank:2d}: ID={s['id']:2d} | True={s['true_count']:.2f} | {s['img_path']}")
    return chosen

# ===================== GD (DDS API) =====================

def compress_image_to_base64(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=75)  # notscale, onlycompress
        raw_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{raw_b64}"
    except Exception as e:
        print(f"  [Image Error] {e}")
        return None

def create_session():
    s = requests.Session()
    s.trust_env = False
    return s

def call_dinox_api(img_path, prompt_text):
    """
    return boxes: list[[x1,y1,x2,y2], ...] (pixel coordinates)
    """
    if not API_TOKEN:
        print("  [Error] notsetting DDS_API_TOKEN(environmentvariable)or API_TOKEN, skip GD. ")
        return []

    CREATE_TASK_URL = "https://api.deepdataspace.com/v2/task/grounding_dino/detection"
    QUERY_TASK_URL = "https://api.deepdataspace.com/v2/task_status/{}"
    headers = {"Content-Type": "application/json", "Token": API_TOKEN}

    img_b64 = compress_image_to_base64(img_path)
    if not img_b64:
        return []

    payload = {
        "model": GD_MODEL,
        "image": img_b64,
        "prompt": {"type": "text", "text": prompt_text},
        "targets": ["bbox"],
        "bbox_threshold": GD_BBOX_THRESHOLD,
        "iou_threshold": GD_IOU_THRESHOLD,
    }

    session = create_session()
    task_uuid = None
    for _ in range(3):
        try:
            resp = session.post(CREATE_TASK_URL, headers=headers, json=payload, verify=False, timeout=30)
            if resp.status_code == 200 and resp.json().get("code") == 0:
                task_uuid = resp.json()["data"]["task_uuid"]
                break
        except Exception:
            pass
        time.sleep(1)

    if not task_uuid:
        print("  [Fail] taskraiseintersectionfail")
        return []

    for _ in range(GD_POLL_MAX_ITERS):
        time.sleep(GD_POLL_SLEEP_SEC)
        try:
            q_resp = session.get(QUERY_TASK_URL.format(task_uuid), headers=headers, verify=False, timeout=10)
            if q_resp.status_code != 200:
                continue
            q_data = q_resp.json()
            if q_data.get("code") != 0:
                continue
            status = q_data["data"]["status"]
            if status == "success":
                result = q_data["data"].get("result", {})
                obj_list = result.get("objects", [])
                if not obj_list and isinstance(result, list):
                    obj_list = result
                boxes = [obj["bbox"] for obj in obj_list if isinstance(obj, dict) and "bbox" in obj]
                return boxes
            if status == "failed":
                return []
        except Exception:
            pass

    return []

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
        # compatible: hassomeimplementtakeresultwriteback state
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

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]

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

# ===================== visualize =====================

def draw_boxes(img_path, gd_boxes, sam_boxes_pass1, sam_boxes_pass2, final_boxes, img_id, text):
    try:
        image = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except Exception:
            font = ImageFont.load_default()

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

        save_path = os.path.join(OUT_DIR, f"merged_id_{img_id}.jpg")
        image.save(save_path)
        print(f"  [Saved] visualize: {save_path}")
    except Exception as e:
        print(f"  [Warn] visualizefail: {e}")

# ===================== main pipeline =====================

def main():
    print(f"[Main] Dataset={TARGET_DATASET} | sample_n={NUM_SAMPLES}")
    if TARGET_DATASET not in SAM3_PROMPT_MAP or TARGET_DATASET not in GD_PROMPT_MAP:
        print("[Error] TARGET_DATASET notinpromptwordmaptableinside, pleasecheck. ")
        return

    sam_prompt = SAM3_PROMPT_MAP[TARGET_DATASET]
    gd_prompt = GD_PROMPT_MAP[TARGET_DATASET]
    print(f"[Main] GD prompt=\"{gd_prompt}\" | SAM3 prompt=\"{sam_prompt}\"")

    samples = build_samples_for_dataset(TARGET_DATASET)
    if not samples:
        print("[Main] notfindtosample, end. ")
        return

    chosen = build_real_ranking(samples, NUM_SAMPLES)
    if not chosen:
        return

    processor = build_sam3_processor()

    print("\n[Pipeline] GD -> SAM3(pass1 in-box crops) -> SAM3(pass2 full-image) -> merge&dedup -> ranking\n")

    for s in chosen:
        img_id = s["id"]
        img_path = s["img_path"]
        print(f"\n[{img_id}/{len(chosen)}] Processing: {img_path}")

        if SHOW_ORIGINAL:
            show_original_with_pil(img_path, img_id)

        # readimage
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print("  [Warn] readfail, recordfor 0. ")
            s["pseudo_count"] = 0.0
            continue

        h, w = img_bgr.shape[:2]
        pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        # ---------- 1) GD takebox ----------
        gd_boxes = call_dinox_api(img_path, gd_prompt)
        gd_boxes = [clamp_box_xyxy(b, w, h) for b in gd_boxes if isinstance(b, (list, tuple)) and len(b) == 4]
        print(f"  [GD] boxes = {len(gd_boxes)}")

        # ---------- 2) SAM3 pass1: insohas GD boxinside ----------
        sam_boxes_pass1 = []
        for b in gd_boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in b]
            if x2 <= x1 or y2 <= y1:
                continue

            crop = pil_img.crop((x1, y1, x2, y2))
            cw, ch = crop.size
            if cw <= 2 or ch <= 2:
                continue

            # in crop on, use"wholeblock crop as exemplar box"trigger SAM3
            crop_boxes = sam3_text_plus_box(processor, crop, (0, 0, cw, ch), sam_prompt)

            # mapbackwhole imagecoordinate
            for cb in crop_boxes:
                cb = list(map(float, cb))
                cb[0] += x1
                cb[1] += y1
                cb[2] += x1
                cb[3] += y1
                sam_boxes_pass1.append(clamp_box_xyxy(cb, w, h))

        sam_boxes_pass1 = nms_dedup_boxes(sam_boxes_pass1, iou_thr=DEDUP_IOU_THR, min_area=MIN_BOX_AREA)
        print(f"  [SAM3 pass1] in-box merged = {len(sam_boxes_pass1)}")

        # ---------- 3) SAM3 pass2: allimage ----------
        sam_boxes_pass2 = []

        # plain text PCS: notuseanyprompt box(notdependency GD), inwhole imageonfindsohasmatchinstance
        full_boxes = sam3_text_only(processor, pil_img, sam_prompt)
        sam_boxes_pass2 = []
        if full_boxes.shape[0] > 0:
            for fb in full_boxes:
                sam_boxes_pass2.append(clamp_box_xyxy(list(map(float, fb)), w, h))

        sam_boxes_pass2 = nms_dedup_boxes(sam_boxes_pass2, iou_thr=DEDUP_IOU_THR, min_area=MIN_BOX_AREA)
        print(f"  [SAM3 pass2] full-image merged = {len(sam_boxes_pass2)}")

        # ---------- 4) combineandtwiceresultdeduplication ----------
        merged = sam_boxes_pass1 + sam_boxes_pass2
        final_boxes = nms_dedup_boxes(merged, iou_thr=DEDUP_IOU_THR, min_area=MIN_BOX_AREA)
        final_count = float(len(final_boxes))
        s["pseudo_count"] = final_count

        print(f"  [Count] pseudo={final_count:.2f} | true={s['true_count']:.2f}")

        # visualize(GD/twice SAM3/finalcombineand)
        text = (f"ID:{img_id} | GD:{len(gd_boxes)} | "
                f"SAM1:{len(sam_boxes_pass1)} | SAM2:{len(sam_boxes_pass2)} | "
                f"Final:{int(final_count)} | True:{s['true_count']:.0f}")
        draw_boxes(img_path, gd_boxes, sam_boxes_pass1, sam_boxes_pass2, final_boxes, img_id, text)

    # ---------- 5) leaderboard ----------
    ranking_pseudo = sorted(chosen, key=lambda x: x.get("pseudo_count", 0.0), reverse=True)
    print("\n[Ranking] combineandcount pseudo labelleaderboard: ")
    for rank, s in enumerate(ranking_pseudo, start=1):
        print(f"  Rank {rank:2d}: ID={s['id']:2d} | Pseudo={s.get('pseudo_count', 0.0):.2f} | True={s['true_count']:.2f}")

    print("\n[Done] finish GD + SAM3 dual-pathcount + leaderboard. ")

if __name__ == "__main__":
    main()
