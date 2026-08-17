# -*- coding: utf-8 -*-
"""
10 imageimagereal vs SAM3 pseudocountleaderboardscript

functionality: 
1)fromonedatasetclass(TARGET_DATASET)inrandomselectout 10 imageimage(notenough 10 imagethenallselect). 
2)tothis 10 imageimage: 
    - according toselectinordernumberfor ID=1..N(rankingbeforefirstnumber). 
    - according toreal label(GT)computeeach realnumber true_count. 
    - according to true_count ranking, output"real labelleaderboard", use ID number. 
3)thenuse SAM3: 
    - toeveryoneimageaccordingtimespopup, letyoudrawone ROI(rectanglebox). 
    - use text prompt + ROI box(as exemplar)inwhole imageoninference, getpseudocount pseudo_count. 
    - finallyaccording to pseudo_count ranking, output"SAM3 pseudo labelleaderboard", sameuse ID number. 

Description:
- TARGET_DATASET infiletopConfiguration:
    "RSOC-Building", "RSOC-S-Vehicle", "RSOC-L-Vehicle", "RSOC-Ship",
    "VD-People", "VD-Vehicle"
- text promptwordalsoaccording todatasetfixedmap. 
"""

import os
import sys
import types
import random
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

import cv2
import torch
import scipy.io as sio

# ===================== configuration section =====================

ROOT_DATA_DIR = "Dataset"  # you  Dataset root directory

LOCAL_SAM3_PATH = r"D:\sam3_source"
SAM3_CHECKPOINT_PATH = "saved_models/SAM3.pt"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# selectonedatasetclass(according toneedmodify): 
# "RSOC-Building", "RSOC-S-Vehicle", "RSOC-L-Vehicle", "RSOC-Ship",
# "VD-People", "VD-Vehicle"
TARGET_DATASET = "VD-People"

NUM_SAMPLES = 10  # randomselectmanyfewimage

# SAM3 text prompt: externallabel -> promptword
DATASET_PROMPT_MAP = {
    "RSOC-Building": "building",
    "RSOC-S-Vehicle": "small vehicle",
    "RSOC-L-Vehicle": "large vehicle",
    "RSOC-Ship": "boat",
    "VD-People": "person",
    "VD-Vehicle": "vehicle",
}

# RSOC in DOTA labelTxt  classnamemap
RSOC_DOTA_CATEGORY_MAP = {
    "RSOC-S-Vehicle": "small-vehicle",
    "RSOC-L-Vehicle": "large-vehicle",
    "RSOC-Ship": "ship",
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ===================== Windows Fix(according toyouoriginalscript) =====================
if sys.platform.startswith("win"):
    print("[Windows Fix] mount dummy triton / flash_attn ...")

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
        import flash_attn  # noqa:F401
    except ImportError:
        mock_flash = types.ModuleType("flash_attn")
        mock_flash.__spec__ = MagicMock()
        sys.modules["flash_attn"] = mock_flash
        sys.modules["flash_attn.flash_attn_interface"] = MagicMock()

    print("[Windows Fix] dummy triton / flash_attn mountfinish")

# ===================== import SAM3 =====================
if os.path.exists(LOCAL_SAM3_PATH) and LOCAL_SAM3_PATH not in sys.path:
    sys.path.insert(0, LOCAL_SAM3_PATH)

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model.box_ops import box_xywh_to_cxcywh
    print("[Info] successimport SAM3 API")
except ImportError as e:
    print(f"[Error] cannot import SAM3 module: {e}")
    sys.exit(1)


def build_sam3_processor():
    if not os.path.exists(SAM3_CHECKPOINT_PATH):
        print(f"[Error] SAM3 weight filenotsavein: {SAM3_CHECKPOINT_PATH}")
        sys.exit(1)

    print("[Model] load SAM3 model ...")
    model = build_sam3_image_model(
        checkpoint_path=SAM3_CHECKPOINT_PATH,
        load_from_HF=False,
        device=str(DEVICE).split(":")[0],
    )
    processor = Sam3Processor(model, confidence_threshold=0.3)
    print("[Model] SAM3 Processor initialization complete")
    return processor


# ===================== utility functions =====================
def list_images(img_dir):
    if not os.path.isdir(img_dir):
        return []
    return [
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith(IMAGE_EXTS)
    ]


# ---------- parse RSOC-Building   .mat ----------
def load_rsoc_building_points(mat_path):
    data = sio.loadmat(mat_path)
    if "center" not in data:
        return np.zeros((0, 2), dtype=float)
    c = data["center"]
    try:
        pts = c[0, 0]
        pts = np.asarray(pts, dtype=float)
        return pts
    except Exception:
        return np.zeros((0, 2), dtype=float)


# ---------- build"withrealcount" samplelist ----------
def build_samples_for_dataset(dataset_label):
    """
    return list[dict]: 
      {
        "img_path": str,
        "true_count": float,
        "gt_type": "points"/"boxes"/"density",
        "gt_data":  correspondformat data(pointarray / box / npypath)
      }
    """
    samples = []

    # 1) RSOC-Building: .mat inpoint number
    if dataset_label == "RSOC-Building":
        rsoc_root = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "RSOC_building", "building")
        for split_name, subdir in [("train", "train_data"), ("test", "test_data")]:
            img_dir = os.path.join(rsoc_root, subdir, "images")
            gt_dir = os.path.join(rsoc_root, subdir, "ground_truth")
            imgs = list_images(img_dir)
            for img_path in imgs:
                base = os.path.splitext(os.path.basename(img_path))[0]
                gt_name = f"GT_{base}.mat"
                gt_path = os.path.join(gt_dir, gt_name)
                if not os.path.exists(gt_path):
                    continue
                pts = load_rsoc_building_points(gt_path)
                count = pts.shape[0]
                samples.append({
                    "img_path": img_path,
                    "true_count": float(count),
                    "gt_type": "points",
                    "gt_data": pts,
                })

    # 2) RSOC otherthreeclass: DOTA labelTxt box(onlystatisticscorrespondclass)
    elif dataset_label in ["RSOC-S-Vehicle", "RSOC-L-Vehicle", "RSOC-Ship"]:
        target_cat = RSOC_DOTA_CATEGORY_MAP[dataset_label]

        def collect_split(split):
            if split == "train":
                img_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "train", "images")
                label_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset",
                                         "train", "labelTxt-v1.0", "trainset_reclabelTxt")
            elif split == "val":
                img_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "val", "images")
                label_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset",
                                         "val", "labelTxt-v1.0", "valset_reclabelTxt")
            else:
                return

            if not os.path.isdir(label_dir):
                return

            img_list = list_images(img_dir)
            base_to_img = {
                os.path.splitext(os.path.basename(p))[0]: p
                for p in img_list
            }

            for txt_name in os.listdir(label_dir):
                if not txt_name.lower().endswith(".txt"):
                    continue
                txt_path = os.path.join(label_dir, txt_name)
                base = os.path.splitext(txt_name)[0]

                if base not in base_to_img:
                    continue
                img_path = base_to_img[base]

                boxes = []
                with open(txt_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 9:
                            continue
                        x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])
                        cat = parts[-2]
                        if cat != target_cat:
                            continue
                        boxes.append([x1, y1, x2, y2, x3, y3, x4, y4])

                if len(boxes) == 0:
                    continue

                samples.append({
                    "img_path": img_path,
                    "true_count": float(len(boxes)),
                    "gt_type": "boxes",
                    "gt_data": np.array(boxes, dtype=float),
                })

        collect_split("train")
        collect_split("val")

    # 3) VD-People / VD-Vehicle: density map   sum whencount
    elif dataset_label in ["VD-People", "VD-Vehicle"]:
        vd_root = os.path.join(ROOT_DATA_DIR,
                               "VisDrone-People" if dataset_label == "VD-People" else "VisDrone-Vehicle")

        for split in ["train", "val", "test"]:
            img_dir1 = os.path.join(vd_root, split, "images")
            img_dir2 = os.path.join(vd_root, split, "Images")
            if os.path.isdir(img_dir1):
                img_dir = img_dir1
            elif os.path.isdir(img_dir2):
                img_dir = img_dir2
            else:
                continue

            gt_dir = os.path.join(vd_root, split, "Ground Truth")
            if not os.path.isdir(gt_dir):
                continue

            for fname in os.listdir(gt_dir):
                if not fname.lower().endswith(".npy"):
                    continue
                npy_path = os.path.join(gt_dir, fname)
                base = os.path.splitext(fname)[0]

                img_path = None
                for ext in IMAGE_EXTS:
                    cand = os.path.join(img_dir, base + ext)
                    if os.path.exists(cand):
                        img_path = cand
                        break
                if img_path is None:
                    continue

                density = np.load(npy_path)
                count = float(density.sum())

                samples.append({
                    "img_path": img_path,
                    "true_count": count,
                    "gt_type": "density",
                    "gt_data": npy_path,
                })

    else:
        raise ValueError(f"notknowdatasetlabel: {dataset_label}")

    print(f"[Index] {dataset_label}: total {len(samples)} one withrealcount sample")
    return samples


# ===================== realcountleaderboard =====================
def build_real_ranking(samples, num_samples=10):
    """
    fromsohassampleinrandomselectout num_samples , andnumberfor ID 1..N. 
    Returns:
      chosen: list[dict], every dict addfield "id"
      ranking_real: according to true_count descendingranking  list[dict](sameinclude id)
    """
    if len(samples) == 0:
        return [], []

    if len(samples) <= num_samples:
        chosen = samples.copy()
    else:
        chosen = random.sample(samples, num_samples)

    # number: 1..N, orderthenis"randomselectout order"
    for i, s in enumerate(chosen):
        s["id"] = i + 1

    print("\n[Select] randomselectout sample(originalorder & number): ")
    for s in chosen:
        print(f"  ID={s['id']:2d} | true_count={s['true_count']:.2f} | {s['img_path']}")

    # according torealcountranking(descending)
    ranking_real = sorted(chosen, key=lambda x: x["true_count"], reverse=True)

    print("\n[Ranking] real labelleaderboard(according to true_count descending): ")
    for rank, s in enumerate(ranking_real, start=1):
        print(f"  Rank {rank:2d}: ID={s['id']:2d} | true={s['true_count']:.2f} | {s['img_path']}")

    return chosen, ranking_real


# ===================== SAM3: text + singletimes ROI boxpseudocount =====================
def sam3_text_plus_box(processor, pil_img, roi_xywh, prompt_text):
    """
    text + onegeometryboxTip:
    - textspecify"iswhat"; 
    - youdraw  ROI (xywh) as exemplar; 
    - inwhole imageonoutputinstance. 
    return boxes: [M, 4] (xyxy)
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

    state = processor.add_geometric_prompt(
        box=norm_box_cxcywh, label=True, state=state
    )

    boxes = state.get("boxes", None)
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), dtype=float)

    if torch.is_tensor(boxes):
        boxes_np = boxes.detach().cpu().numpy()
    else:
        boxes_np = np.array(boxes)
    return boxes_np.astype(float)


# ===================== main pipeline =====================
def main():
    print(f"[Main] dataset: {TARGET_DATASET}")
    prompt_text = DATASET_PROMPT_MAP[TARGET_DATASET]
    print(f"[Main] SAM3 text prompt: \"{prompt_text}\"")

    # 1) buildsohassample(withrealcount)
    samples = build_samples_for_dataset(TARGET_DATASET)
    if len(samples) == 0:
        print("[Main] nosample, end. ")
        return

    # 2) randomselect 10 image, number & printreal labelleaderboard
    chosen, ranking_real = build_real_ranking(samples, NUM_SAMPLES)
    if len(chosen) == 0:
        print("[Main] noselectoutsample, end. ")
        return

    # 3) initialize SAM3
    processor = build_sam3_processor()

    # 4) tothis 10 imageimage by imagepopup, letyoubox ROI, thenuse text+box getpseudocount
    print("\n[SAM3] nowintoeachimageperform: drawonce ROI, thenusetext+boxgeneratepseudocount. ")
    for s in chosen:
        img_id = s["id"]
        img_path = s["img_path"]
        print(f"\n[Image ID={img_id}] {img_path}")
        print(f"  realcount true_count = {s['true_count']:.2f}")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print("  [Warn] readimagefail, pseudocountrecordfor 0. ")
            s["pseudo_count"] = 0.0
            continue

        h, w = img_bgr.shape[:2]
        pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        # letyoudraw ROI
        disp = img_bgr.copy()
        msg = f"ID={img_id}: Draw ROI then press ENTER (ESC to skip)"
        cv2.putText(disp, msg, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        roi = cv2.selectROI("Select ROI", disp, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")

        x, y, rw, rh = roi
        if rw <= 0 or rh <= 0:
            print("  [Info] notselect ROI, pseudocountrecordfor 0. ")
            s["pseudo_count"] = 0.0
            continue

        print(f"  ROI: x={x}, y={y}, w={rw}, h={rh}")

        boxes = sam3_text_plus_box(processor, pil_img, (x, y, rw, rh), prompt_text)
        pseudo_count = float(boxes.shape[0])
        s["pseudo_count"] = pseudo_count

        print(f"  SAM3 pseudocount pseudo_count = {pseudo_count:.2f}")

    # 5) according topseudocountranking, print SAM3 pseudo labelleaderboard(usesame ID)
    ranking_pseudo = sorted(chosen, key=lambda x: x.get("pseudo_count", 0.0), reverse=True)

    print("\n[Ranking] SAM3 pseudo labelleaderboard(according to pseudo_count descending): ")
    for rank, s in enumerate(ranking_pseudo, start=1):
        print(
            f"  Rank {rank:2d}: ID={s['id']:2d} | pseudo={s.get('pseudo_count', 0.0):.2f} "
            f"| true={s['true_count']:.2f} | {s['img_path']}"
        )

    print("\n[Done] finish 10 imageimage real vs SAM3 pseudocountleaderboard. ")


if __name__ == "__main__":
    main()
