# -*- coding: utf-8 -*-
"""
10 imageimagereal vs Grounding DINO 1.6 Pro (V6 showoriginalimageversion)
functionalityUpgrade:
1. [newly added] incall API before, willfirstpopuporiginalimageprovideview. 
2. [inherit] include V5  sohaspathadaptand V4   API fix. 
"""

import os
import sys
import time
import random
import json
import base64
import requests
import urllib3
import io
from PIL import Image, ImageDraw, ImageFont
import scipy.io as sio
import numpy as np
# disable SSL warning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ===================== user configuration section =====================

# [important]pleaseinherefill inyou  DeepDataSpace API Token
API_TOKEN = "89572a070f19b4a331925dcbebe05c09"

# datasetroot directory
ROOT_DATA_DIR = "Dataset"

# currenttest set
TARGET_DATASET = "RSOC-S-Vehicle"
NUM_SAMPLES = 10

# text promptmap
DATASET_PROMPT_MAP = {
    "RSOC-Building": "structure",
    "RSOC-S-Vehicle": "car",
    "RSOC-L-Vehicle": "truck",
    "RSOC-Ship": "boat",
    "VD-People": "person",
    "VD-Vehicle": "vehicle",
}

RSOC_DOTA_CATEGORY_MAP = {
    "RSOC-S-Vehicle": "small-vehicle",
    "RSOC-L-Vehicle": "large-vehicle",
    "RSOC-Ship": "ship",
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ===================== auxiliaryfunctionality: showoriginalimage =====================

def show_original_with_pil(img_path, img_id):
    """
    callsystemdefaultviewershoworiginalimage
    """
    try:
        print(f"  [Display] currentlyshoworiginalimage (ID={img_id})...")
        img = Image.open(img_path)
        img.show()  # thiswillpopupsystemimageviewer
        # Note:codenotwilltemporarilystopetc.waityouclose image, itwillcontinuetowardbelowrun. 
        # ifyouhopecodetemporarilystopetc.wait, caninhereadd time.sleep(2) orone whouse input()
        # time.sleep(1)
    except Exception as e:
        print(f"  [Warn] nomethodshoworiginalimage: {e}")


# ===================== datasetpathhandle =====================

def list_images(img_dir):
    if not os.path.isdir(img_dir): return []
    return [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(IMAGE_EXTS)]


def load_rsoc_building_points(mat_path):
    data = sio.loadmat(mat_path)
    if "center" not in data: return 0.0
    try:
        return float(data["center"][0, 0].shape[0])
    except:
        return 0.0


def build_samples_for_dataset(dataset_label):
    samples = []
    print(f"[Loader] currentlyload {dataset_label} data...")

    if dataset_label == "RSOC-Building":
        rsoc_root = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "RSOC_building", "building")
        for subdir in ["train_data", "test_data"]:
            img_dir = os.path.join(rsoc_root, subdir, "images")
            gt_dir = os.path.join(rsoc_root, subdir, "ground_truth")
            if not os.path.isdir(img_dir): continue
            imgs = list_images(img_dir)
            for img_path in imgs:
                base = os.path.splitext(os.path.basename(img_path))[0]
                gt_path = os.path.join(gt_dir, f"GT_{base}.mat")
                if os.path.exists(gt_path):
                    count = load_rsoc_building_points(gt_path)
                    samples.append({"img_path": img_path, "true_count": count})

    elif dataset_label in ["RSOC-S-Vehicle", "RSOC-L-Vehicle", "RSOC-Ship"]:
        target_cat = RSOC_DOTA_CATEGORY_MAP[dataset_label]
        for split in ["train", "val"]:
            img_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", split, "images")
            label_folder_name = f"{split}set_reclabelTxt"
            label_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", split, "labelTxt-v1.0", label_folder_name)
            if not os.path.isdir(label_dir) or not os.path.isdir(img_dir): continue

            img_list = list_images(img_dir)
            base_to_img = {os.path.splitext(os.path.basename(p))[0]: p for p in img_list}

            for txt_name in os.listdir(label_dir):
                if not txt_name.endswith(".txt"): continue
                base = os.path.splitext(txt_name)[0]
                if base not in base_to_img: continue
                count = 0
                with open(os.path.join(label_dir, txt_name), "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 9 and parts[-2] == target_cat:
                            count += 1
                if count > 0:
                    samples.append({"img_path": base_to_img[base], "true_count": float(count)})

    elif dataset_label in ["VD-People", "VD-Vehicle"]:
        sub_root = "VisDrone-People" if dataset_label == "VD-People" else "VisDrone-Vehicle"
        vd_root = os.path.join(ROOT_DATA_DIR, sub_root)
        for split in ["train", "val", "test"]:
            path_candidates = [
                os.path.join(vd_root, split, "images"),
                os.path.join(vd_root, split, "Images")
            ]
            img_dir = None
            for p in path_candidates:
                if os.path.isdir(p):
                    img_dir = p
                    break
            gt_dir = os.path.join(vd_root, split, "Ground Truth")
            if not img_dir or not os.path.isdir(gt_dir): continue
            for fname in os.listdir(gt_dir):
                if not fname.endswith(".npy"): continue
                base = os.path.splitext(fname)[0]
                img_path = None
                for ext in IMAGE_EXTS:
                    cand = os.path.join(img_dir, base + ext)
                    if os.path.exists(cand):
                        img_path = cand
                        break
                if img_path:
                    try:
                        density = np.load(os.path.join(gt_dir, fname))
                        samples.append({"img_path": img_path, "true_count": float(density.sum())})
                    except:
                        pass

    print(f"[Loader] {dataset_label}: totalload {len(samples)} sample")
    return samples


def build_real_ranking(samples, num_samples=10):
    if len(samples) <= num_samples:
        chosen = samples.copy()
    else:
        chosen = random.sample(samples, num_samples)
    for i, s in enumerate(chosen): s["id"] = i + 1
    ranking_real = sorted(chosen, key=lambda x: x["true_count"], reverse=True)
    print("\n[Ranking] real labelleaderboard: ")
    for rank, s in enumerate(ranking_real, start=1):
        print(f"  Rank {rank}: ID={s['id']} | True={s['true_count']:.2f} | {s['img_path']}")
    return chosen


# ===================== API calllogic =====================

def compress_image_to_base64_v4(img_path, max_size=1024):
    try:
        img = Image.open(img_path).convert("RGB")

        # ===== removescale: notagainaccording to max_size resize =====
        # w, h = img.size
        # scale = min(1.0, max_size / max(w, h))
        # if scale < 1.0:
        #     img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        # ============================================

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=75)  # stillthenkeeporiginalcome  JPEG compress
        raw_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{raw_base64}"
    except Exception as e:
        print(f"  [Image Error] {e}")
        return None



def create_session():
    s = requests.Session()
    s.trust_env = False
    return s


def call_dinox_api_v4(img_path, prompt_text):
    CREATE_TASK_URL = "https://api.deepdataspace.com/v2/task/grounding_dino/detection"
    QUERY_TASK_URL = "https://api.deepdataspace.com/v2/task_status/{}"
    headers = {"Content-Type": "application/json", "Token": API_TOKEN}

    img_b64 = compress_image_to_base64_v4(img_path)
    if not img_b64: return []

    payload = {
        "model": "GroundingDino-1.6-Pro",
        "image": img_b64,
        "prompt": {"type": "text", "text": prompt_text},
        "targets": ["bbox"],
        "bbox_threshold": 0.13,
        "iou_threshold": 0.3
    }

    session = create_session()
    task_uuid = None
    for i in range(3):
        try:
            resp = session.post(CREATE_TASK_URL, headers=headers, json=payload, verify=False, timeout=30)
            if resp.status_code == 200 and resp.json().get("code") == 0:
                task_uuid = resp.json()["data"]["task_uuid"]
                break
            else:
                time.sleep(1)
        except Exception:
            time.sleep(1)

    if not task_uuid:
        print("  [Fail] taskraiseintersectionfail")
        return []

    for _ in range(20):
        time.sleep(2)
        try:
            q_resp = session.get(QUERY_TASK_URL.format(task_uuid), headers=headers, verify=False, timeout=10)
            if q_resp.status_code == 200:
                q_data = q_resp.json()
                if q_data["code"] == 0:
                    status = q_data["data"]["status"]
                    if status == "success":
                        result = q_data["data"].get("result", {})
                        obj_list = result.get("objects", [])
                        if not obj_list and isinstance(result, list): obj_list = result
                        boxes = [obj["bbox"] for obj in obj_list if "bbox" in obj]
                        return boxes
                    elif status == "failed":
                        return []
        except:
            pass
    return []


def draw_boxes_and_show(img_path, boxes, img_id, prompt):
    try:
        image = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        for box in boxes:
            if len(box) == 4:
                draw.rectangle(box, outline="red", width=3)

        text = f"ID: {img_id} | Prompt: {prompt} | Count: {len(boxes)}"
        draw.text((10, 10), text, fill="yellow", font=font)

        out_dir = "output_results"
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"result_id_{img_id}.jpg")
        image.save(save_path)
        print(f"  [Saved] resultimage: {save_path}")
    except Exception:
        pass


# ===================== mainprogram =====================

def main():
    if "YOUR_" in API_TOKEN:
        print("[Error] pleasefirstinscripttopfill injustconfirm  API_TOKEN")
        return

    print(f"[Main] dataset: {TARGET_DATASET}")
    prompt_text = DATASET_PROMPT_MAP.get(TARGET_DATASET, "object")

    samples = build_samples_for_dataset(TARGET_DATASET)
    if not samples: return
    chosen = build_real_ranking(samples, NUM_SAMPLES)

    print("\n[API] startcall DeepDataSpace V6 (showoriginalimageversion) ...")

    for s in chosen:
        img_id = s["id"]
        print(f"\n[Image ID={img_id}] handlein...")

        # --- newly addedfunctionality: firstshoworiginalimage ---
        show_original_with_pil(s['img_path'], img_id)
        # ------------------------

        detected_boxes = call_dinox_api_v4(s['img_path'], prompt_text)

        s["pseudo_count"] = float(len(detected_boxes))
        print(f"  API recognition: {int(s['pseudo_count'])} | real: {s['true_count']:.2f}")

        if s["pseudo_count"] > 0:
            draw_boxes_and_show(s['img_path'], detected_boxes, img_id, prompt_text)

    ranking_pseudo = sorted(chosen, key=lambda x: x.get("pseudo_count", 0.0), reverse=True)
    print("\n[Ranking] API pseudo labelleaderboard: ")
    for rank, s in enumerate(ranking_pseudo, start=1):
        print(f"  Rank {rank:2d}: ID={s['id']:2d} | Pseudo={s.get('pseudo_count', 0):.2f} | True={s['true_count']:.2f}")


if __name__ == "__main__":
    main()