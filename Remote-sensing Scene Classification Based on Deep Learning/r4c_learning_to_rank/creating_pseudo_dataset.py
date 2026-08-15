# -*- coding: utf-8 -*-
import os
import cv2
import random
import numpy as np
from config import *

# Helper function for this experiment module
def list_images(img_dir):
    if not img_dir or not os.path.isdir(img_dir): return []
    return [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(IMAGE_EXTS)]

def _find_dota_list_files(dota_dir: str, split: str, cat_token: str):
    if not os.path.isdir(dota_dir): return []
    return sorted([os.path.join(dota_dir, fn) for fn in os.listdir(dota_dir) 
                   if fn.lower().endswith(".txt") and fn.lower().startswith(split.lower()) 
                   and cat_token.lower().replace("-", "") in fn.lower().replace("-", "").replace("_", "")],
                  key=lambda p: (len(os.path.basename(p)), p))

def _read_dota_name_list(txt_path):
    names = set()
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"): names.add(os.path.splitext(os.path.basename(s.replace("\\", "/")))[0])
    except Exception: pass
    return names

def _dataset_label_to_outname(dataset_label: str) -> str:
    return {"RSOC-Building": "rsoc-building", "RSOC-Ship": "rsoc-ship", "RSOC-S-Vehicle": "rsoc-s-vehicle",
            "RSOC-L-Vehicle": "rsoc-l-vehicle", "VD-People": "vd-people", "VD-Vehicle": "vd-vehicle"}.get(dataset_label, str(dataset_label).lower())

def _available_splits_for_dataset(dataset_label: str):
    if dataset_label == "RSOC-Building": return ["train", "test"]
    return ["train", "val", "test"]

def _build_allow_set_for_rsoc_mixed(dataset_label: str, split: str):
    cat_token = {"RSOC-Ship": "ship", "RSOC-S-Vehicle": "small-vehicle", "RSOC-L-Vehicle": "large-vehicle"}[dataset_label]
    dota_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "DOTA_data")
    allow = None
    list_files = _find_dota_list_files(dota_dir, split, cat_token)
    if list_files:
        allow = set()
        for lp in list_files: allow |= _read_dota_name_list(lp)
    if allow is None and split in ["train", "val"]:
        label_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", split, "labelTxt-v1.0", f"{split}set_reclabelTxt")
        if os.path.isdir(label_dir):
            allow = set()
            for fn in os.listdir(label_dir):
                if not fn.endswith(".txt"): continue
                base = os.path.splitext(fn)[0]
                try:
                    with open(os.path.join(label_dir, fn), "r", encoding="utf-8", errors="ignore") as f:
                        if any(len(p:=line.strip().split()) >= 9 and (p[-2] if len(p)>=10 else p[-1]) == cat_token for line in f):
                            allow.add(base)
                except Exception: continue
    return allow

def _list_images_for_dataset_split(dataset_label: str, split: str):
    if dataset_label == "RSOC-Building":
        sub = "train_data" if split == "train" else ("test_data" if split == "test" else None)
        return list_images(os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "RSOC_building", "building", sub, "images")) if sub else []
    if dataset_label in ["RSOC-Ship", "RSOC-S-Vehicle", "RSOC-L-Vehicle"]:
        img_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", split, "images")
        allow = _build_allow_set_for_rsoc_mixed(dataset_label, split)
        return [p for p in list_images(img_dir) if os.path.splitext(os.path.basename(p))[0] in allow] if allow else []
    if dataset_label in ["VD-People", "VD-Vehicle"]:
        vd_root = os.path.join(ROOT_DATA_DIR, "VisDrone-People" if dataset_label == "VD-People" else "VisDrone-Vehicle")
        for cand in [os.path.join(vd_root, split, "images"), os.path.join(vd_root, split, "Images")]:
            if os.path.isdir(cand): return list_images(cand)
    return []

def select_exemplar_boxes(dataset_label: str, split_for_exemplar: str, num_images: int):
    img_list = _list_images_for_dataset_split(dataset_label, split_for_exemplar)
    if not img_list: return []
    random.Random(int(EXEMPLAR_RANDOM_SEED)).shuffle(img_list)
    picked, need = [], max(0, int(num_images))
    print(f"\n[Exemplar] Will randomly sample {need} images from split={split_for_exemplar}; you will manually draw one ROI for each image.")
    for img_path in img_list:
        if len(picked) >= need: break
        if (img_bgr := cv2.imread(img_path)) is None: continue
        h, w = img_bgr.shape[:2]
        base = os.path.basename(img_path)
        print(f"\n[Exemplar] ({len(picked)+1}/{need}) Selected: {base}")
        try:
            roi = cv2.selectROI(EXEMPLAR_WINDOW_NAME, img_bgr, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow(EXEMPLAR_WINDOW_NAME)
        except Exception:
            try: cv2.destroyWindow(EXEMPLAR_WINDOW_NAME)
            except Exception: pass
            continue
        if roi is None or roi[2] <= 0 or roi[3] <= 0: continue
        
        x, y, ww, hh = max(0, min(w-1, roi[0])), max(0, min(h-1, roi[1])), max(1, min(w-roi[0], roi[2])), max(1, min(h-roi[1], roi[3]))
        label = True
        if EXEMPLAR_ALLOW_NEGATIVE_LABEL:
            if input("  label? enter p(pos)/n(neg), default p: ").strip().lower() == "n": label = False
        
        crop = img_bgr[y:y+hh, x:x+ww].copy()
        if crop.size == 0: continue
        if EXEMPLAR_CROP_MAX_SIZE > 0 and max(crop.shape[:2]) > EXEMPLAR_CROP_MAX_SIZE:
            scale = EXEMPLAR_CROP_MAX_SIZE / max(crop.shape[:2])
            crop = cv2.resize(crop, (max(1, int(round(crop.shape[1]*scale))), max(1, int(round(crop.shape[0]*scale)))), interpolation=cv2.INTER_AREA)
        
        picked.append({"crop_bgr": crop, "crop_gray": cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), "label": label, "src_image": base, "roi_abs_xywh": (x, y, ww, hh)})
        print(f"[Exemplar] Recorded crop shape={crop.shape[:2]} | label={'pos' if label else 'neg'}")
    return picked

def match_exemplars_to_image(img_bgr, exemplar_list):
    if not exemplar_list: return [], []
    h, w = img_bgr.shape[:2]
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    prompts_xywh, prompts_labels = [], []
    scales, topk, thr = EXEMPLAR_MATCH_SCALES, max(0, int(EXEMPLAR_MATCH_TOPK_PER_EX)), float(EXEMPLAR_MATCH_THRESHOLD)
    for ex in exemplar_list:
        if (tmpl := ex.get("crop_gray", None)) is None: continue
        best, th0, tw0 = [], tmpl.shape[0], tmpl.shape[1]
        for s in scales:
            tw, th = max(8, int(round(tw0 * s))), max(8, int(round(th0 * s)))
            if tw >= w or th >= h: continue
            tmpl_s = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_AREA) if (tw != tw0 or th != th0) else tmpl
            res = cv2.matchTemplate(img_gray, tmpl_s, cv2.TM_CCOEFF_NORMED)
            for _ in range(topk):
                _, max_val, _, (x, y) = cv2.minMaxLoc(res)
                if max_val < thr: break
                best.append((float(max_val), int(x), int(y), tw, th))
                res[max(0, y - th//2):min(res.shape[0], y + th//2), max(0, x - tw//2):min(res.shape[1], x + tw//2)] = -1.0
        for (_, x, y, tw, th) in sorted(best, key=lambda t: t[0], reverse=True)[:topk]:
            prompts_xywh.append((max(0, min(w-1, x)), max(0, min(h-1, y)), max(1, min(w-x, tw)), max(1, min(h-y, th))))
            prompts_labels.append(ex.get("label", True))
    return prompts_xywh, prompts_labels
