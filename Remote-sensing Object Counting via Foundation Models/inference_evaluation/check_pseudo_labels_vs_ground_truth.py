import os
import random

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

ROOT_DATA_DIR = "Dataset"       # datasetroot directory
PSEUDO_ROOT = "pseudo_labels"   # pseudo labelroot directory

# selectneedcheck datasetlabel(andpseudo labelin  dataset_label correspond): 
# optional:  "RSOC-Building", "RSOC-S-Vehicle", "RSOC-L-Vehicle", "RSOC-Ship",
#       "VD-People", "VD-Vehicle"
DATASET_LABEL = "RSOC-L-Vehicle"

# random drawmanyfewsamplecomecheck
NUM_SAMPLES = 1

# VisDrone density mappathstrategy: 
#   if DENSITY_SAME_DIR = True: recognizefordensity mapandimagesamedirectory, samename, onlyisextensionchangefor .npy
#   elseuse VD_DENSITY_ROOT, fakesetstructurefor VD_DENSITY_ROOT/<split>/<same_name>.npy
DENSITY_SAME_DIR = True
VD_DENSITY_ROOT = ""  # if DENSITY_SAME_DIR=False, needinherefilldensity maproot directory
# ============================================================

# RSOC otherthreeclass  DOTA classnamemap
RSOC_DOTA_CATEGORY_MAP = {
    "RSOC-S-Vehicle": "small-vehicle",
    "RSOC-L-Vehicle": "large-vehicle",
    "RSOC-Ship": "ship",
}

# VisDrone class ID set(ifafterneeduseboxandnotisdensity mapcanuse, nowinonlyusedensity map)
VD_PEOPLE_CAT_IDS = {1, 2}
VD_VEHICLE_CAT_IDS = {4, 5, 6, 7, 8, 9}

# RSOC-Building   .mat need scipy
try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[Warn] notinstall scipy, RSOC-Building nomethodread .mat realvalue, pleasefirst pip install scipy")


# ----------------- utility functions -----------------
def format_patch_counts(patch_counts, grid=4):
    """takelength16 onedimensionlistformat-izebecome4x4 string. """
    arr = list(patch_counts)
    if len(arr) != grid * grid:
        return str(arr)
    lines = []
    for r in range(grid):
        row = arr[r * grid:(r + 1) * grid]
        lines.append(" ".join(f"{v:6.2f}" for v in row))
    return "\n" + "\n".join(lines)


def get_grid_index(x, y, W, H, grid=4):
    """tosetcoordinate (x,y) andimage size (W,H), return 4x4 gridin index(0-15). """
    if W == 0 or H == 0:
        return None
    gx = int(x * grid / W)
    gy = int(y * grid / H)
    gx = min(max(gx, 0), grid - 1)
    gy = min(max(gy, 0), grid - 1)
    return gy * grid + gx


def draw_grid_on_pil(img, grid=4, color=(255, 0, 0), width=1):
    """in PIL Image ondraw 4x4 gridline. """
    draw = ImageDraw.Draw(img)
    W, H = img.size
    for i in range(1, grid):
        x = int(W * i / grid)
        draw.line([(x, 0), (x, H)], fill=color, width=width)
        y = int(H * i / grid)
        draw.line([(0, y), (W, y)], fill=color, width=width)
    return img


# ============================================================
# 1. RSOC-Building: from .mat   center readrealvaluepoint
# ============================================================
def load_rsoc_building_centers(img_path, split):
    if not HAS_SCIPY:
        return None

    rsoc_root = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "RSOC_building", "building")
    # pseudo labelinside split onegeneralis "train" / "test", heresimplemap
    if "train" in split:
        split_dir = "train_data"
    else:
        split_dir = "test_data"

    img_name = os.path.basename(img_path)   # IMG_000053.jpg
    base = os.path.splitext(img_name)[0]    # IMG_000053
    mat_name = "GT_" + base + ".mat"
    mat_path = os.path.join(rsoc_root, split_dir, "ground_truth", mat_name)

    if not os.path.isfile(mat_path):
        print(f"    [RSOC-Building] findnotto .mat annotationfile: {mat_path}")
        return None

    mat = sio.loadmat(mat_path)
    if "center" not in mat:
        print(f"    [RSOC-Building] .mat inno 'center' field: {mat_path}")
        return None

    center = mat["center"]
    try:
        pts = np.array(center[0, 0], dtype=np.float32)  # (N,2)
    except Exception as e:
        print(f"    [RSOC-Building] parse center fail: {e}")
        return None

    if pts.ndim != 2 or pts.shape[1] != 2:
        print(f"    [RSOC-Building] center shapeexception: {pts.shape}")
        return None

    return pts  # (N,2)


def compute_rsoc_building_gt_counts(img_path, split, grid=4):
    pts = load_rsoc_building_centers(img_path, split)
    if pts is None:
        return None, None

    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    global_count = pts.shape[0]
    patch_counts = np.zeros(grid * grid, dtype=np.float32)

    for (x, y) in pts:
        idx = get_grid_index(x, y, W, H, grid=grid)
        if idx is not None:
            patch_counts[idx] += 1

    return global_count, patch_counts


def show_rsoc_building(img_path, split, pts, grid=4):
    """display: originalimage + point + grid"""
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)

    r = max(2, min(W, H) // 200)
    for (x, y) in pts:
        x1, y1 = x - r, y - r
        x2, y2 = x + r, y + r
        draw.ellipse([x1, y1, x2, y2], outline=(0, 255, 0), width=2)

    img_draw = draw_grid_on_pil(img_draw, grid=grid, color=(255, 0, 0), width=2)
    img_draw.show(title=os.path.basename(img_path))


# ============================================================
# 2. RSOC otherthreeclass: from DOTA labelTxt readbox
# ============================================================
def load_rsoc_dota_boxes(img_path, split, dataset_label):
    if dataset_label not in RSOC_DOTA_CATEGORY_MAP:
        return []

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
        # test notmusthaslabel
        label_dir = None

    if label_dir is None or not os.path.isdir(label_dir):
        print("    [RSOC-DOTA] current split no labelTxt directory. ")
        return []

    base = os.path.splitext(os.path.basename(img_path))[0]
    txt_path = os.path.join(label_dir, base + ".txt")
    if not os.path.isfile(txt_path):
        print(f"    [RSOC-DOTA] findnotto labelTxt file: {txt_path}")
        return []

    boxes = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("imagesource:") or line.startswith("gsd:"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            cat = parts[-2]
            if cat != target_cat:
                continue
            try:
                coords = list(map(float, parts[:8]))
            except ValueError:
                continue
            if len(coords) == 8:
                boxes.append(coords)
    return boxes


def compute_rsoc_dota_gt_counts(img_path, split, dataset_label, grid=4):
    boxes = load_rsoc_dota_boxes(img_path, split, dataset_label)
    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    global_count = len(boxes)
    patch_counts = np.zeros(grid * grid, dtype=np.float32)

    for box in boxes:
        x1, y1, x2, y2, x3, y3, x4, y4 = box
        cx = (x1 + x2 + x3 + x4) / 4.0
        cy = (y1 + y2 + y3 + y4) / 4.0
        idx = get_grid_index(cx, cy, W, H, grid=grid)
        if idx is not None:
            patch_counts[idx] += 1

    return global_count, patch_counts


def show_rsoc_dota(img_path, boxes, grid=4):
    """display: originalimage + greenbox + grid"""
    img = Image.open(img_path).convert("RGB")
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)

    for box in boxes:
        x1, y1, x2, y2, x3, y3, x4, y4 = box
        poly = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        draw.line(poly + [poly[0]], fill=(0, 255, 0), width=2)

    img_draw = draw_grid_on_pil(img_draw, grid=grid, color=(255, 0, 0), width=2)
    img_draw.show(title=os.path.basename(img_path))


# ============================================================
# 3. VisDrone: fromdensity map .npy readrealvalue
# ============================================================
def infer_vd_density_path(img_path, split):
    """
    you  VisDrone datastructureis: 
        Dataset/VisDrone-People/val/images/*.jpg
        Dataset/VisDrone-People/val/Ground Truth/*.npy

    thereforeonlyneedwill images replacefor Ground Truth i.e.can. 
    """
    base = os.path.splitext(os.path.basename(img_path))[0]  # 0000013_...

    # imagedirectory
    img_dir = os.path.dirname(img_path)

    # take "images" replacebecome "Ground Truth"
    gt_dir = img_dir.replace("images", "Ground Truth")

    density_path = os.path.join(gt_dir, base + ".npy")
    return density_path



def compute_vd_density_gt_counts(img_path, split, dataset_label, grid=4):
    density_path = infer_vd_density_path(img_path, split)
    if not os.path.isfile(density_path):
        print(f"    [VD] findnottodensity mapfile: {density_path}")
        return None, None, None

    density = np.load(density_path)

    # -------- compatibleeachkinddimension  density --------
    if density.ndim == 2:
        # standardsituation (H, W)
        pass
    elif density.ndim == 3:
        # maybe (1, H, W), (H, W, 1) or (C, H, W)/(H, W, C)
        if 1 in density.shape:
            # hassinglechannel, squeezedropfor (H, W)
            density = np.squeeze(density)
        else:
            # manychannel: simplelyinchanneldimensiononsum, get (H, W)
            # fakesetchannelinbefore (C,H,W) orinafter (H,W,C), according towhichdimensionsmallonepointguessonebelow
            if density.shape[0] <= 4:  # viewfor (C, H, W)
                density = density.sum(axis=0)
            elif density.shape[-1] <= 4:  # viewfor (H, W, C)
                density = density.sum(axis=-1)
            else:
                # realinlooknotout, thenfirst squeeze onebelow, looklookcannotcanbecome 2D
                density = np.squeeze(density)
                if density.ndim != 2:
                    print(f"    [VD] nomethodhandle  density shape: {density.shape}")
                    return None, None, None
    else:
        # dimensiontoohigh situation, trycompressto 2D
        density = np.squeeze(density)
        if density.ndim != 2:
            print(f"    [VD] nomethodhandle  density shape: {density.shape}")
            return None, None, None

    H, W = density.shape  # herethennotwillagainreport too many values  

    global_count = float(density.sum())
    patch_counts = np.zeros(grid * grid, dtype=np.float32)
    ph = H // grid
    pw = W // grid

    for gy in range(grid):
        for gx in range(grid):
            y0 = gy * ph
            y1 = (gy + 1) * ph if gy < grid - 1 else H
            x0 = gx * pw
            x1 = (gx + 1) * pw if gx < grid - 1 else W
            patch = density[y0:y1, x0:x1]
            patch_counts[gy * grid + gx] = float(patch.sum())

    return global_count, patch_counts, density



def show_vd_density(img_path, density, grid=4):
    """displaydensity map + 4x4 grid"""
    H, W = density.shape
    plt.figure(figsize=(6, 4))
    plt.imshow(density, cmap="jet")
    plt.title(os.path.basename(img_path))
    plt.colorbar()

    for i in range(1, grid):
        y = H * i / grid
        x = W * i / grid
        plt.axhline(y, color="white", linewidth=0.5)
        plt.axvline(x, color="white", linewidth=0.5)

    plt.tight_layout()
    plt.show()


# ============================================================
# mainlogic
# ============================================================
def main():
    pseudo_path = os.path.join(PSEUDO_ROOT, f"pseudo_{DATASET_LABEL.replace('-', '_')}.npy")
    if not os.path.isfile(pseudo_path):
        print(f"[Error] pseudolabel filenotsavein: {pseudo_path}")
        return

    arr = np.load(pseudo_path, allow_pickle=True)
    entries = arr.tolist()
    n = len(entries)
    if n == 0:
        print("[Error] pseudolabel fileforempty. ")
        return

    k = min(NUM_SAMPLES, n)
    samples = random.sample(entries, k)
    print(f"[Info] dataset={DATASET_LABEL}, from {n} sampleinextracttake {k} performcheck. ")
    print("=" * 100)

    for idx, e in enumerate(samples):
        img_path = e.get("img_path")
        split = e.get("split", "train")
        dataset_label = e.get("dataset_label", "Unknown")
        pseudo_global = e.get("global_count", None)
        pseudo_patch = np.array(e.get("patch_counts", []), dtype=np.float32)

        print(f"[Sample {idx+1}/{k}]")
        print(f"  imagepath       : {img_path}")
        print(f"  datasetlabel     : {dataset_label}")
        print(f"  split          : {split}")
        print(f"  pseudo label-globalnumber: {pseudo_global}")
        print(f"  pseudo label-localnumber (len={len(pseudo_patch)}):")
        print(format_patch_counts(pseudo_patch, grid=4))

        if not os.path.isfile(img_path):
            print("  [Warn] imagefilenotsavein, nomethodreadrealvalue/visualize. ")
            print("-" * 100)
            continue

        # -------- according todatasetclasstypereadrealvalueanddisplay --------
        gt_global = None
        gt_patch = None

        # 1) RSOC-Building
        if dataset_label == "RSOC-Building":
            gt_global, gt_patch = compute_rsoc_building_gt_counts(img_path, split, grid=4)
            if gt_global is not None:
                print(f"  realvalue-globalnumber : {gt_global}")
                print(f"  realvalue-localnumber :")
                print(format_patch_counts(gt_patch, grid=4))
            else:
                print("  realvalue-global/localnumber : (nomethodread)")

            pts = load_rsoc_building_centers(img_path, split)
            if pts is not None:
                show_rsoc_building(img_path, split, pts, grid=4)

        # 2) RSOC S/L/Ship
        elif dataset_label in RSOC_DOTA_CATEGORY_MAP:
            gt_global, gt_patch = compute_rsoc_dota_gt_counts(img_path, split, dataset_label, grid=4)
            if gt_global is not None:
                print(f"  realvalue-globalnumber : {gt_global}")
                print(f"  realvalue-localnumber :")
                print(format_patch_counts(gt_patch, grid=4))
            else:
                print("  realvalue-global/localnumber : (nomethodread)")

            boxes = load_rsoc_dota_boxes(img_path, split, dataset_label)
            if boxes:
                show_rsoc_dota(img_path, boxes, grid=4)

        # 3) VisDrone
        elif dataset_label in ["VD-People", "VD-Vehicle"]:
            gt_global, gt_patch, density = compute_vd_density_gt_counts(
                img_path, split, dataset_label, grid=4
            )
            if gt_global is not None:
                print(f"  realvalue-globalnumber (density.sum) : {gt_global:.3f}")
                print(f"  realvalue-localnumber (density per patch):")
                print(format_patch_counts(gt_patch, grid=4))
                show_vd_density(img_path, density, grid=4)
            else:
                print("  realvalue-global/localnumber : (nomethodread density .npy)")

        else:
            print("  [Warn] notknow dataset_label, temporarilynotsupportrealvalueread. ")

        print("-" * 100)


if __name__ == "__main__":
    main()
