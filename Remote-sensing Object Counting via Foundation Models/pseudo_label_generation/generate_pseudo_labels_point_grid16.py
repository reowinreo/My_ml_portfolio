import os
import sys
import types
from unittest.mock import MagicMock

import numpy as np
from PIL import Image, ImageDraw

import torch
import scipy.io as sio  # parse .mat

# ========================================================
# 0. configurationparameter(only modify here)
# ========================================================

# dataset root
ROOT_DATA_DIR = "Dataset"  # you workprocessinside Dataset folder

# SAM3 source code / weightpath
LOCAL_SAM3_PATH = r"D:\sam3_source"
SAM3_CHECKPOINT_PATH = "saved_models/SAM3.pt"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# objectclass(frompaperinside 6 insideselectone)
#   "RSOC-Building"
#   "RSOC-S-Vehicle"
#   "RSOC-L-Vehicle"
#   "RSOC-Ship"
#   "VD-People"
#   "VD-Vehicle"
TARGET_DATASET = "RSOC-Building"

# selectoutrealobjectnumbermostmany before TOP_K image
TOP_K = 10

# SAM automaticpointsampling  point grid(hope SAM3 internaladoptmoredense point)
POINT_GRID_SIZE = 16

# pseudo label / visualizeoutputdirectory
PSEUDO_DEBUG_DIR = "pseudo_labels_debug"
VIS_DEBUG_DIR = "debug_vis_pg16"

os.makedirs(PSEUDO_DEBUG_DIR, exist_ok=True)
os.makedirs(VIS_DEBUG_DIR, exist_ok=True)

# datasetlabel -> SAM text promptword
DATASET_PROMPT_MAP = {
    "RSOC-Building": "building",
    "RSOC-S-Vehicle": "small vehicle",
    "RSOC-L-Vehicle": "large vehicle",
    "RSOC-Ship": "ship",
    "VD-People": "person",
    "VD-Vehicle": "vehicle",
}

# RSOC 3 non- building class, correspond DOTA txt insideclassname
RSOC_DOTA_CATEGORY_MAP = {
    "RSOC-S-Vehicle": "small-vehicle",
    "RSOC-L-Vehicle": "large-vehicle",
    "RSOC-Ship": "ship",
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# ========================================================
# 1. Windows Fix(according toyouoriginalcomescript writing style)
# ========================================================
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

# ========================================================
# 2. import SAM3
# ========================================================
if os.path.exists(LOCAL_SAM3_PATH) and LOCAL_SAM3_PATH not in sys.path:
    sys.path.insert(0, LOCAL_SAM3_PATH)

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    print("[Info] successimport SAM3 API")
except ImportError as e:
    print(f"[Error] cannot import SAM3 module: {e}")
    sys.exit(1)


def list_images(img_dir):
    if not os.path.isdir(img_dir):
        return []
    return [
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith(IMAGE_EXTS)
    ]


# ========================================================
# 3. real labelparse & selectout TOP_K sample
# ========================================================

def collect_rsoc_building_samples():
    """
    RSOC-Building:
        Dataset/ASPDNet_dataset/RSOC_building/building/{train_data,test_data}/
            - images/IMG_****.jpg
            - ground_truth/GT_IMG_****.mat  (center[0,0] insideis N×2 coordinate)
    """
    root = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "RSOC_building", "building")
    samples = []

    for split_name, split_dir in [("train", "train_data"), ("test", "test_data")]:
        img_dir = os.path.join(root, split_dir, "images")
        gt_dir = os.path.join(root, split_dir, "ground_truth")
        if not os.path.isdir(gt_dir):
            continue

        for fname in os.listdir(gt_dir):
            if not fname.lower().endswith(".mat"):
                continue
            gt_path = os.path.join(gt_dir, fname)
            base = os.path.splitext(fname)[0]  # e.g. GT_IMG_000053
            # removeprefix "GT_"
            if base.startswith("GT_"):
                img_base = base[3:]    # IMG_000053
            else:
                img_base = base

            # trymanykindextension
            img_path = None
            for ext in IMAGE_EXTS:
                cand = os.path.join(img_dir, img_base + ext)
                if os.path.isfile(cand):
                    img_path = cand
                    break
            if img_path is None:
                print(f"[Warn] findnotto RSOC-Building correspondimage: {img_base}.*")
                continue

            # read .mat, statisticspointnumber
            try:
                mat = sio.loadmat(gt_path)
                center = mat["center"]
                pts = center[0, 0]  # (N,2)
                gt_count = int(pts.shape[0])
            except Exception as e:
                print(f"[Warn] parse {gt_path} fail: {e}")
                continue

            samples.append({
                "img_path": img_path,
                "gt_path": gt_path,
                "split": split_name,
                "dataset_label": "RSOC-Building",
                "gt_count": gt_count,
                "gt_type": "center_points",
            })

    return samples


def collect_rsoc_dota_samples(dataset_label):
    """
    RSOC-S/L-Vehicle, RSOC-Ship:
        Dataset/ASPDNet_dataset/{train,val}/
            - images/*.*
            - labelTxt-v1.0/*set_reclabelTxt/*.txt
        txt per line: x1 y1 x2 y2 x3 y3 x4 y4 category difficult
    """
    target_cat = RSOC_DOTA_CATEGORY_MAP[dataset_label]
    base_root = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset")
    samples = []

    def collect_split(split):
        if split not in ("train", "val"):
            return
        label_dir = os.path.join(base_root, split, "labelTxt-v1.0",
                                 f"{split}set_reclabelTxt")
        img_dir = os.path.join(base_root, split, "images")
        if not os.path.isdir(label_dir):
            return
        img_list = list_images(img_dir)
        base_to_img = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in img_list
        }

        for fname in os.listdir(label_dir):
            if not fname.lower().endswith(".txt"):
                continue
            gt_path = os.path.join(label_dir, fname)
            base = os.path.splitext(fname)[0]

            # realcount: onlystatisticscurrentclass
            count = 0
            polys = []  # forvisualizeretainmanysideshape
            with open(gt_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # skipheadpart: imagesource, gsd etc.
                    if not line[0].isdigit():
                        continue
                    parts = line.split()
                    if len(parts) < 10:
                        continue
                    cat = parts[-2]
                    if cat != target_cat:
                        continue
                    try:
                        nums = list(map(float, parts[:8]))
                        poly = [(nums[0], nums[1]),
                                (nums[2], nums[3]),
                                (nums[4], nums[5]),
                                (nums[6], nums[7])]
                        polys.append(poly)
                        count += 1
                    except Exception:
                        continue

            if base not in base_to_img:
                # findnottocorrespondimage
                continue
            img_path = base_to_img[base]

            samples.append({
                "img_path": img_path,
                "gt_path": gt_path,
                "split": split,
                "dataset_label": dataset_label,
                "gt_count": count,
                "gt_type": "dota_boxes",
                "gt_polygons": polys,  # used forvisualize
            })

    collect_split("train")
    collect_split("val")

    return samples


def collect_vd_density_samples(dataset_label):
    """
    VD-People / VD-Vehicle:
        Dataset/VisDrone-XXX/{train,val,test}/
            - Ground Truth/*.npy (density map)
            - images/*.*
        realcount ~ density.sum()
    """
    if dataset_label == "VD-People":
        vd_root = os.path.join(ROOT_DATA_DIR, "VisDrone-People")
    else:
        vd_root = os.path.join(ROOT_DATA_DIR, "VisDrone-Vehicle")

    samples = []

    for split in ["train", "val", "test"]:
        gt_dir = os.path.join(vd_root, split, "Ground Truth")
        img_dir1 = os.path.join(vd_root, split, "images")
        img_dir2 = os.path.join(vd_root, split, "Images")
        if os.path.isdir(img_dir1):
            img_dir = img_dir1
        elif os.path.isdir(img_dir2):
            img_dir = img_dir2
        else:
            continue
        if not os.path.isdir(gt_dir):
            continue

        img_list = list_images(img_dir)
        base_to_img = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in img_list
        }

        for fname in os.listdir(gt_dir):
            if not fname.lower().endswith(".npy"):
                continue
            gt_path = os.path.join(gt_dir, fname)
            base = os.path.splitext(fname)[0]
            if base not in base_to_img:
                continue
            img_path = base_to_img[base]

            try:
                density = np.load(gt_path)
                gt_count = float(density.sum())
            except Exception as e:
                print(f"[Warn] parse VD density {gt_path} fail: {e}")
                continue

            samples.append({
                "img_path": img_path,
                "gt_path": gt_path,
                "split": split,
                "dataset_label": dataset_label,
                "gt_count": gt_count,
                "gt_type": "density_map",
            })

    return samples


def collect_samples_for_target_dataset(dataset_label):
    if dataset_label == "RSOC-Building":
        samples = collect_rsoc_building_samples()
    elif dataset_label in ("RSOC-S-Vehicle", "RSOC-L-Vehicle", "RSOC-Ship"):
        samples = collect_rsoc_dota_samples(dataset_label)
    elif dataset_label in ("VD-People", "VD-Vehicle"):
        samples = collect_vd_density_samples(dataset_label)
    else:
        raise ValueError(f"notknow TARGET_DATASET: {dataset_label}")

    print(f"[GT] {dataset_label} totalcollectto {len(samples)} sample(hasreal label)")
    return samples


def select_top_k_samples(samples, k):
    if len(samples) == 0:
        return []
    samples_sorted = sorted(samples, key=lambda x: x["gt_count"], reverse=True)
    top = samples_sorted[:k]
    print(f"[GT] realcountmosthigh before {len(top)} sample: ")
    for i, s in enumerate(top):
        print(f"  #{i+1:02d}: count={s['gt_count']} | {s['img_path']}")
    return top


# ========================================================
# 4. SAM3 model & point grid setting
# ========================================================

def build_sam3_model_and_processor(point_grid_size=16):
    if not os.path.exists(SAM3_CHECKPOINT_PATH):
        print(f"[Error] SAM3 weight filenotsavein: {SAM3_CHECKPOINT_PATH}")
        sys.exit(1)

    print("[Model] load SAM3 modelin ...")
    model = build_sam3_image_model(
        checkpoint_path=SAM3_CHECKPOINT_PATH,
        load_from_HF=False,
        device=str(DEVICE).split(":")[0],
    )

    # tryusedifferentwaytake point grid propagateto Sam3Processor
    processor = None
    last_err = None
    tried = []

    trial_kwargs_list = [
        {"confidence_threshold": 0.3, "points_per_side": point_grid_size},
        {"confidence_threshold": 0.3, "point_grids": point_grid_size},
        {"confidence_threshold": 0.3},  # fall backmostsafeall way(followyouoriginalcome scriptsame)
    ]

    for kwargs in trial_kwargs_list:
        try:
            processor = Sam3Processor(model, **kwargs)
            tried.append(kwargs)
            print(f"[Model] use Sam3Processor Arguments: {kwargs}")
            break
        except TypeError as e:
            last_err = e
            tried.append(kwargs)
            continue

    if processor is None:
        print("[Error] nomethoduseanyparameterconstruct Sam3Processor, pleaseaccording toyoulocal sam3 implementmanualmodify. ")
        print("  alreadytry parametergroupcombine:")
        for kw in tried:
            print("   ", kw)
        print("  finallyError:", last_err)
        sys.exit(1)

    # ifconstructwhennotto point  parameter, againtrythroughattributesetting
    set_ok = False
    for attr in ["points_per_side", "point_grid_size", "point_grids"]:
        if hasattr(processor, attr):
            try:
                setattr(processor, attr, point_grid_size)
                print(f"[Model] will processor.{attr} set to {point_grid_size}")
                set_ok = True
                break
            except Exception:
                continue
    if not set_ok:
        print("[Warn] nofindtoobvious  point grid attribute(points_per_side / point_grid_size / point_grids), ")
        print("       ifyouconfirmrecognize Sam3Processor hasmutualcloseAPI, pleasein build_sam3_model_and_processor insidemanualchange. ")

    print("[Model] SAM3 modeland Processor initialization complete")
    return model, processor


# ========================================================
# 5. SAM3 count(global + 4x4 patch)
# ========================================================

def sam3_count_instances_and_patches(processor, img_path, prompt_text, grid=(4, 4)):
    """
    tooneimage: 
        - SAM3 segmentation + text prompt(singleclass)
        - Returns:
            global_count: instancenumber
            patch_counts: (grid_h * grid_w,) every patch  instancenumber
    """
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[Warn] openimagefail: {img_path} ({e})")
        return 0, np.zeros(grid[0] * grid[1], dtype=np.int32)

    W, H = img.size
    grid_h, grid_w = grid

    state = processor.set_image(img)
    output = processor.set_text_prompt(state=state, prompt=prompt_text)
    masks = output.get("masks", None)

    if masks is None or len(masks) == 0:
        return 0, np.zeros(grid_h * grid_w, dtype=np.int32)

    if torch.is_tensor(masks):
        masks_np = masks.detach().cpu().numpy()
    else:
        masks_np = np.array(masks)

    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0, :, :]  # (N,H,W)

    N, Mh, Mw = masks_np.shape
    # resize tooriginalimagesize
    if (Mw, Mh) != (W, H):
        masks_resized = []
        for i in range(N):
            m = (masks_np[i] > 0.5).astype(np.uint8) * 255
            m_img = Image.fromarray(m).resize((W, H), resample=Image.NEAREST)
            m_bin = (np.array(m_img) > 127).astype(np.uint8)
            masks_resized.append(m_bin)
        masks_np = np.stack(masks_resized, axis=0)
    else:
        masks_np = (masks_np > 0.5).astype(np.uint8)

    global_count = N

    patch_counts = np.zeros(grid_h * grid_w, dtype=np.int32)
    ph = H // grid_h
    pw = W // grid_w

    for idx in range(N):
        m = masks_np[idx]
        for gy in range(grid_h):
            for gx in range(grid_w):
                y0 = gy * ph
                y1 = (gy + 1) * ph if gy < grid_h - 1 else H
                x0 = gx * pw
                x1 = (gx + 1) * pw if gx < grid_w - 1 else W
                patch = m[y0:y1, x0:x1]
                if patch.sum() > 0:
                    patch_index = gy * grid_w + gx
                    patch_counts[patch_index] += 1

    return global_count, patch_counts


# ========================================================
# 6. visualize: real label + 4x4 grid
# ========================================================

def draw_grid(draw: ImageDraw.Draw, W, H, grid=(4, 4), line_color=(0, 255, 0), width=1):
    gw, gh = grid
    # verticalline
    for i in range(1, gw):
        x = int(W * i / gw)
        draw.line([(x, 0), (x, H)], fill=line_color, width=width)
    # horizontalline
    for j in range(1, gh):
        y = int(H * j / gh)
        draw.line([(0, y), (W, y)], fill=line_color, width=width)


def visualize_sample(sample, out_path):
    img = Image.open(sample["img_path"]).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    # draw 4x4 grid
    draw_grid(draw, W, H, grid=(4, 4), line_color=(0, 255, 0), width=2)

    gt_type = sample["gt_type"]

    if gt_type == "center_points":
        # RSOC-Building: drawpoint
        mat = sio.loadmat(sample["gt_path"])
        center = mat["center"]
        pts = center[0, 0]  # (N,2)
        for (x, y) in pts:
            r = 4
            draw.ellipse([(x - r, y - r), (x + r, y + r)],
                         outline=(255, 0, 0), width=2)
    elif gt_type == "dota_boxes":
        # RSOC-S/L-Vehicle/Ship: drawthis class manysideshape
        polys = sample.get("gt_polygons", [])
        for poly in polys:
            if len(poly) < 4:
                continue
            # closemanysideshape
            pts = poly + [poly[0]]
            draw.line(pts, fill=(255, 0, 0), width=2)
    elif gt_type == "density_map":
        # VD: temporarilywhenonlydrawgrid, notadditionaldrawpoint(becauseisdensity map, noawayscatterobjectcoordinate)
        pass

    img.save(out_path)
    print(f"[Vis] savevisualizeto: {out_path}")


# ========================================================
# 7. mainPipeline:collectsample -> select TOP_K -> SAM3 pseudo label -> visualize
# ========================================================

def main():
    print(f"[Main] TARGET_DATASET = {TARGET_DATASET}")
    print(f"[Main] POINT_GRID_SIZE = {POINT_GRID_SIZE}, TOP_K = {TOP_K}")

    # ---------- 1) firstaccording toreal labelcollectsohassample ----------
    samples = collect_samples_for_target_dataset(TARGET_DATASET)
    if len(samples) == 0:
        print("[Main] nosample, backout. ")
        return

    # ---------- 2) selectoutrealcountmostmany before TOP_K ----------
    top_samples = select_top_k_samples(samples, TOP_K)
    if len(top_samples) == 0:
        print("[Main] TOP_K sampleforempty, backout. ")
        return

    # ---------- 3) build SAM3 + Processor (point grid ~ 16) ----------
    _, processor = build_sam3_model_and_processor(point_grid_size=POINT_GRID_SIZE)
    prompt_text = DATASET_PROMPT_MAP[TARGET_DATASET]
    print(f"[Main] SAM3 text prompt: \"{prompt_text}\"")

    # ---------- 4) totheseimagegenerate SAM3 pseudo label ----------
    pseudo_entries = []
    for idx, s in enumerate(top_samples):
        img_path = s["img_path"]
        print(f"[SAM3] ({idx+1}/{len(top_samples)}) {img_path}")
        global_count_sam3, patch_counts_sam3 = sam3_count_instances_and_patches(
            processor, img_path, prompt_text, grid=(4, 4)
        )
        entry = {
            "img_path": img_path,
            "gt_path": s["gt_path"],
            "split": s["split"],
            "dataset_label": s["dataset_label"],
            "gt_count": float(s["gt_count"]),
            "gt_type": s["gt_type"],
            "global_count_sam3": int(global_count_sam3),
            "patch_counts_sam3": patch_counts_sam3.astype(np.int32).tolist(),
        }
        pseudo_entries.append(entry)

        # meanwhileoutputoneimagevisualizeimage(realannotation + 4x4 grid)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        vis_name = f"{TARGET_DATASET.replace('-', '_')}_{idx+1:02d}_{base_name}_gt_grid.jpg"
        vis_path = os.path.join(VIS_DEBUG_DIR, vis_name)
        visualize_sample(s, vis_path)

    # ---------- 5) savepseudo label ----------
    pseudo_arr = np.array(pseudo_entries, dtype=object)
    save_path = os.path.join(
        PSEUDO_DEBUG_DIR,
        f"pseudo_debug_{TARGET_DATASET.replace('-', '_')}_pg{POINT_GRID_SIZE}_top{len(top_samples)}.npy"
    )
    np.save(save_path, pseudo_arr)
    print(f"[Main] totalsave {len(pseudo_entries)} itempseudo labelto: {save_path}")
    print("[Main] finish. ")


if __name__ == "__main__":
    main()
