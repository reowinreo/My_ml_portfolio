# -*- coding: utf-8 -*-
"""
script1: use SAM3 forsohasdatasetgeneratepseudo label(once-ityruncomplete)
Output:pseudo_labels/xxxx.npy, everydatasetonefile
"""

import os
import sys
import types
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

import torch

# ================= globalconfiguration =================
ROOT_DATA_DIR = "Dataset"  # youworkprocessinsidedataset root
LOCAL_SAM3_PATH = r"D:\sam3_source"
SAM3_CHECKPOINT_PATH = "saved_models/SAM3.pt"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.makedirs("pseudo_labels", exist_ok=True)

# sohasneedhandle datasetlabel
ALL_DATASET_LABELS = [
    "RSOC-Building",
    "RSOC-S-Vehicle",
    "RSOC-L-Vehicle",
    "RSOC-Ship",
    "VD-People",
    "VD-Vehicle",
]

# used forshow/statistics label -> SAM3 text promptword
DATASET_PROMPT_MAP = {
    "RSOC-Building": "building",
    "RSOC-S-Vehicle": "small vehicle",
    "RSOC-L-Vehicle": "large vehicle",
    "RSOC-Ship": "ship",
    "VD-People": "person",
    "VD-Vehicle": "vehicle",
}

# RSOC inside DOTA labelTxt in classname
RSOC_DOTA_CATEGORY_MAP = {
    "RSOC-S-Vehicle": "small-vehicle",
    "RSOC-L-Vehicle": "large-vehicle",
    "RSOC-Ship": "ship",
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ============== Windows Fix(according toyouoriginalscript)=================
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


# ============== import SAM3 =================
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


# ============== builddataindex =================
def build_image_list_for_dataset(dataset_label):
    """
    return list[dict], everyelement: 
        {
          "img_path": "...",
          "split": "train"/"val"/"test",
          "dataset_label": dataset_label
        }
    """
    results = []

    # ---------- RSOC-Building ----------
    if dataset_label == "RSOC-Building":
        rsoc_build_root = os.path.join(
            ROOT_DATA_DIR, "ASPDNet_dataset", "RSOC_building", "building"
        )
        for split_name, split_dir in [("train", "train_data"), ("test", "test_data")]:
            img_dir = os.path.join(rsoc_build_root, split_dir, "images")
            imgs = list_images(img_dir)
            for p in imgs:
                results.append({
                    "img_path": p,
                    "split": split_name,
                    "dataset_label": dataset_label
                })

    # ---------- RSOC otherthreeclass, rely DOTA labelfilter ----------
    elif dataset_label in ["RSOC-S-Vehicle", "RSOC-L-Vehicle", "RSOC-Ship"]:
        target_cat = RSOC_DOTA_CATEGORY_MAP[dataset_label]

        def collect_from_split(split):
            if split == "train":
                label_dir = os.path.join(
                    ROOT_DATA_DIR, "ASPDNet_dataset", "train",
                    "labelTxt-v1.0", "trainset_reclabelTxt"
                )
                img_dir = os.path.join(
                    ROOT_DATA_DIR, "ASPDNet_dataset", "train", "images"
                )
            elif split == "val":
                label_dir = os.path.join(
                    ROOT_DATA_DIR, "ASPDNet_dataset", "val",
                    "labelTxt-v1.0", "valset_reclabelTxt"
                )
                img_dir = os.path.join(
                    ROOT_DATA_DIR, "ASPDNet_dataset", "val", "images"
                )
            else:  # test notlabel, thendirectalluse
                label_dir = None
                img_dir = os.path.join(
                    ROOT_DATA_DIR, "ASPDNet_dataset", "test", "images"
                )

            if label_dir is None or not os.path.isdir(label_dir):
                imgs = list_images(img_dir)
                for p in imgs:
                    results.append({
                        "img_path": p,
                        "split": split,
                        "dataset_label": dataset_label
                    })
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

                cat_found = False
                with open(txt_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 9:
                            continue
                        cat = parts[-2]
                        if cat == target_cat:
                            cat_found = True
                            break
                if not cat_found:
                    continue

                if base in base_to_img:
                    img_path = base_to_img[base]
                    results.append({
                        "img_path": img_path,
                        "split": split,
                        "dataset_label": dataset_label
                    })

        collect_from_split("train")
        collect_from_split("val")
        # ifhasneedalsocanadd test: 
        # collect_from_split("test")

    # ---------- VisDrone ----------
    elif dataset_label in ["VD-People", "VD-Vehicle"]:
        vd_root = os.path.join(
            ROOT_DATA_DIR, "VisDrone-People" if dataset_label == "VD-People" else "VisDrone-Vehicle"
        )
        for split in ["train", "val", "test"]:
            img_dir1 = os.path.join(vd_root, split, "images")
            img_dir2 = os.path.join(vd_root, split, "Images")  # preventsizewrite
            if os.path.isdir(img_dir1):
                img_dir = img_dir1
            elif os.path.isdir(img_dir2):
                img_dir = img_dir2
            else:
                continue
            imgs = list_images(img_dir)
            for p in imgs:
                results.append({
                    "img_path": p,
                    "split": split,
                    "dataset_label": dataset_label
                })

    else:
        raise ValueError(f"notknowdatasetlabel: {dataset_label}")

    print(f"[Index] {dataset_label}: findto {len(results)} imageimage")
    return results


# ============== SAM3 inference + pseudo label =================
def build_sam3_model_and_processor():
    if not os.path.exists(SAM3_CHECKPOINT_PATH):
        print(f"[Error] SAM3 weight filenotsavein: {SAM3_CHECKPOINT_PATH}")
        sys.exit(1)

    print("[Model] load SAM3 modelin ...")
    model = build_sam3_image_model(
        checkpoint_path=SAM3_CHECKPOINT_PATH,
        load_from_HF=False,
        device=str(DEVICE).split(":")[0],
    )
    processor = Sam3Processor(model, confidence_threshold=0.3)
    print("[Model] SAM3 modeland Processor initialization complete")
    return processor


def sam3_count_instances_and_patches(processor, img_path, prompt_text, grid=(4, 4)):
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
        masks_np = masks_np[:, 0, :, :]

    N, Mh, Mw = masks_np.shape
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


def build_pseudo_for_one_dataset(dataset_label):
    prompt_text = DATASET_PROMPT_MAP[dataset_label]
    print(f"\n[Pseudo] dataset: {dataset_label}, promptword: \"{prompt_text}\"")

    img_list = build_image_list_for_dataset(dataset_label)
    if len(img_list) == 0:
        print("[Pseudo] noimage, skip")
        return

    processor = build_sam3_model_and_processor()  # everydatasetrepeatuseone processor

    pseudo_entries = []
    for idx, info in enumerate(img_list):
        img_path = info["img_path"]
        split = info["split"]
        if not os.path.isfile(img_path):
            print(f"[Warn] imagenotsavein, skip: {img_path}")
            continue

        print(f"[Pseudo] ({idx+1}/{len(img_list)}) {split} | {img_path}")
        g_count, p_counts = sam3_count_instances_and_patches(
            processor, img_path, prompt_text, grid=(4, 4)
        )
        entry = {
            "img_path": img_path,
            "split": split,
            "dataset_label": dataset_label,
            "global_count": int(g_count),
            "patch_counts": p_counts.astype(np.int32).tolist(),
        }
        pseudo_entries.append(entry)

    arr = np.array(pseudo_entries, dtype=object)
    save_path = os.path.join("pseudo_labels",
                             f"pseudo_{dataset_label.replace('-', '_')}.npy")
    np.save(save_path, arr)
    print(f"[Pseudo] {dataset_label} total {len(pseudo_entries)} itempseudo label, alreadysaveto {save_path}")


def main():
    print("[Main] startforsohasdatasetgeneratepseudo label ...")
    for ds in ALL_DATASET_LABELS:
        build_pseudo_for_one_dataset(ds)
    print("[Main] allfinish. ")


if __name__ == "__main__":
    main()
