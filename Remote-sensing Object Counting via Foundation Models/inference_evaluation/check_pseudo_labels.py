# -*- coding: utf-8 -*-
"""
check_pseudo_labels.py

functionality: 
1. readscript1generate pseudo label .npy file(pseudo_labels/pseudo_xxx.npy)
2. randomselectoutifdosample(default3), print: 
   - imagepath
   - datasetlabel dataset_label
   - split (train/val/test)
   - globalpseudocount global_count
   - localpseudocount patch_counts(4x4)
3. fromrealannotationincompute"realobjectnumber"(onlystatisticscurrentsamplesobelongclass)andoutput. 

usemethodExample:
    python check_pseudo_labels.py --npy pseudo_labels/pseudo_RSOC_Building.npy
    python check_pseudo_labels.py --npy pseudo_labels/pseudo_VD_People.npy --num 5
"""

import os
import argparse
import random

import numpy as np

# ------------- optionaldependency: used forread RSOC-Building   .mat -------------
try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[Warn] notinstall scipy, nomethodread RSOC-Building   .mat realannotation. "
          "ifneedsupport, pleasefirstinstall: pip install scipy")

# dataset root(andpreviousscriptkeepconsistent)
ROOT_DATA_DIR = "Dataset"

# RSOC in DOTA classnamemap(script1inalsoisthissampleuse )
RSOC_DOTA_CATEGORY_MAP = {
    "RSOC-S-Vehicle": "small-vehicle",
    "RSOC-L-Vehicle": "large-vehicle",
    "RSOC-Ship": "ship",
}

# VisDrone in class ID map(youcanaccording toneedadjust)
# officialdefinelargecauseis:  1 pedestrian, 2 people,
# 3 bicycle, 4 car, 5 van, 6 truck, 7 tricycle, 8 awning-tricycle,
# 9 bus, 10 motor, 11 others
VD_PEOPLE_CAT_IDS = {1, 2}                      # rowperson / people
VD_VEHICLE_CAT_IDS = {3, 4, 5, 6, 7, 8, 9, 10}  # eachkindvehicle(youcanaccording toneeddeletesubtract)


# ============================================================
# someutility functions
# ============================================================

def format_patch_counts(patch_counts, grid=4):
    """takelength16 onedimensionlistformat-izebecome4x4 string, so thatview. """
    arr = list(patch_counts)
    if len(arr) != grid * grid:
        return str(arr)
    lines = []
    for r in range(grid):
        row = arr[r * grid:(r + 1) * grid]
        lines.append(" ".join(f"{v:3d}" for v in row))
    return "\n" + "\n".join(lines)


# ============================================================
# readrealannotation: RSOC-Building(.mat)
# ============================================================

def count_gt_rsoc_building(img_path, split):
    """
    according to RSOC-Building   ground_truth statisticsrealobjectnumber. 
    directorystructureExample:
        RSOC_building/building/train_data/ground_truth/GT_IMG_000053.mat
        RSOC_building/building/train_data/images/IMG_000053.jpg
    """
    if not HAS_SCIPY:
        return None

    rsoc_root = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "RSOC_building", "building")
    if split == "train":
        split_dir = "train_data"
    else:
        # hereunifytakenon- train allwhenbecome test_data, youalsocanaccording toneedfinepart
        split_dir = "test_data"

    img_name = os.path.basename(img_path)  # e.g. "IMG_000053.jpg"
    base = os.path.splitext(img_name)[0]   # "IMG_000053"
    gt_name = "GT_" + base + ".mat"
    gt_path = os.path.join(rsoc_root, split_dir, "ground_truth", gt_name)

    if not os.path.isfile(gt_path):
        print(f"    [GT-RSOC-Building] findnotto .mat annotationfile: {gt_path}")
        return None

    mat = sio.loadmat(gt_path)
    # commonsituation: mat inhas 'location' or 'annPoints' thissample variable, shapeis (N,2)
    for key in ["location", "annPoints", "gt", "points"]:
        if key in mat and isinstance(mat[key], np.ndarray):
            arr = mat[key]
            # usuallyis (N,2) or (2,N)
            if arr.ndim == 2:
                # heresimpleuseNo.onedimensionwhenasnumber
                return int(max(arr.shape[0], arr.shape[1]))
    # ifonsurfacenotmatchto, thentryfindonemostimage (N,2)  array
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray) and value.ndim == 2:
            return int(max(value.shape[0], value.shape[1]))

    print(f"    [GT-RSOC-Building] nomethodfrom {gt_path} infercount, pleasemanualview .mat structure. ")
    return None


# ============================================================
# readrealannotation: RSOC otherthreeclass(DOTA labelTxt)
# ============================================================

def count_gt_rsoc_dota(img_path, split, dataset_label):
    """
    to RSOC-S-Vehicle / RSOC-L-Vehicle / RSOC-Ship:
    statistics DOTA annotationinbelonginthisclass objectnumber. 
    DOTA annotationformat: x1 y1 x2 y2 x3 y3 x4 y4 category difficult
    """
    if dataset_label not in RSOC_DOTA_CATEGORY_MAP:
        return None
    target_cat = RSOC_DOTA_CATEGORY_MAP[dataset_label]

    if split == "train":
        label_dir = os.path.join(
            ROOT_DATA_DIR, "ASPDNet_dataset", "train",
            "labelTxt-v1.0", "trainset_reclabelTxt"
        )
    elif split == "val":
        label_dir = os.path.join(
            ROOT_DATA_DIR, "ASPDNet_dataset", "val",
            "labelTxt-v1.0", "valset_reclabelTxt"
        )
    else:
        # test notmusthaslabel, ifhasyoucanselfrowpatchfillpath
        label_dir = None

    if label_dir is None or not os.path.isdir(label_dir):
        print("    [GT-RSOC] current split no labelTxt directory, nomethodstatisticsrealnumber. ")
        return None

    base = os.path.splitext(os.path.basename(img_path))[0]
    txt_path = os.path.join(label_dir, base + ".txt")
    if not os.path.isfile(txt_path):
        print(f"    [GT-RSOC] findnotto labelTxt file: {txt_path}")
        return None

    count = 0
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            cat = parts[-2]
            if cat == target_cat:
                count += 1
    return count


# ============================================================
# readrealannotation: VisDrone (VD-People & VD-Vehicle)
# ============================================================

def count_gt_visdrone(img_path, split, dataset_label):
    """
    VisDrone   Ground Truth txt structurelargecausefor: 
        bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, truncation, occlusion
    weaccording to object_category filter: 
        - VD-People: onlynumberclassIDin VD_PEOPLE_CAT_IDS in object
        - VD-Vehicle: onlynumberclassIDin VD_VEHICLE_CAT_IDS in object
    """
    if dataset_label == "VD-People":
        root_name = "VisDrone-People"
        valid_ids = VD_PEOPLE_CAT_IDS
    elif dataset_label == "VD-Vehicle":
        root_name = "VisDrone-Vehicle"
        valid_ids = VD_VEHICLE_CAT_IDS
    else:
        return None

    vd_root = os.path.join(ROOT_DATA_DIR, root_name)

    # Ground Truth directory: onegeneralis "Ground Truth"
    gt_dir_candidates = [
        os.path.join(vd_root, split, "Ground Truth"),
        os.path.join(vd_root, split, "ground_truth"),
        os.path.join(vd_root, split, "annotations"),
    ]

    gt_dir = None
    for d in gt_dir_candidates:
        if os.path.isdir(d):
            gt_dir = d
            break

    if gt_dir is None:
        print(f"    [GT-VD] findnotto Ground Truth directory(try : {gt_dir_candidates})")
        return None

    base = os.path.splitext(os.path.basename(img_path))[0]
    txt_path = os.path.join(gt_dir, base + ".txt")

    if not os.path.isfile(txt_path):
        print(f"    [GT-VD] findnottoannotationfile: {txt_path}")
        return None

    count = 0
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # VisDrone usuallyiscommapartinterval
            parts = line.replace(",", " ").split()
            if len(parts) < 6:
                continue
            try:
                cat_id = int(parts[5])
            except ValueError:
                continue
            if cat_id in valid_ids:
                count += 1
    return count


# ============================================================
# mainlogic: readpseudo label, randomselectsampleandprint
# ============================================================

def check_pseudo_labels(npy_path, num_samples=3):
    if not os.path.isfile(npy_path):
        print(f"[Error] findnottopseudolabel file: {npy_path}")
        return

    arr = np.load(npy_path, allow_pickle=True)
    entries = arr.tolist()
    n = len(entries)
    if n == 0:
        print("[Error] pseudolabel fileforempty. ")
        return

    k = min(num_samples, n)
    samples = random.sample(entries, k)

    print(f"[Info] from {n} sampleinrandomselecttake {k} performcheck. ")
    print("=" * 80)

    for idx, e in enumerate(samples):
        img_path = e.get("img_path")
        split = e.get("split", "train")
        dataset_label = e.get("dataset_label", "Unknown")
        global_count = e.get("global_count", None)
        patch_counts = e.get("patch_counts", [])

        print(f"[Sample {idx+1}/{k}]")
        print(f"  imagepath       : {img_path}")
        print(f"  datasetlabel     : {dataset_label}")
        print(f"  split          : {split}")
        print(f"  pseudo label-globalnumber: {global_count}")
        print(f"  pseudo label-localnumber(length={len(patch_counts)}): {patch_counts}")
        print("  pseudo label-localnumber(4x4grid):")
        print(format_patch_counts(patch_counts, grid=4))

        # ---- realnumber ----
        gt_count = None
        if dataset_label == "RSOC-Building":
            gt_count = count_gt_rsoc_building(img_path, split)
        elif dataset_label in RSOC_DOTA_CATEGORY_MAP:
            gt_count = count_gt_rsoc_dota(img_path, split, dataset_label)
        elif dataset_label in ["VD-People", "VD-Vehicle"]:
            gt_count = count_gt_visdrone(img_path, split, dataset_label)
        else:
            print("  [Warn] notknowdatasetlabel, temporarilynotsupportrealnumberstatistics. ")

        if gt_count is not None:
            print(f"  realobjectnumber     : {gt_count}")
        else:
            print("  realobjectnumber     : (notcansuccessread, detailedseeonsideprompt)")

        print("-" * 80)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy", type=str, required=True,
                        help="pseudo label .npy filepath, for example pseudo_labels/pseudo_RSOC_Building.npy")
    parser.add_argument("--num", type=int, default=3,
                        help="randomcheck number of samplesquantity(default3)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    check_pseudo_labels(args.npy, num_samples=args.num)
