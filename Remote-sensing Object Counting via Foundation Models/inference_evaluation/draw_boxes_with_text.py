# -*- coding: utf-8 -*-
"""
interactivetype ROI + SAM3 pseudo labelsmallrealverifyscript(singleimagemanytimesbox, support"manytimesiterationoverlayboxprompt")

functionality: 
1)selectonedatasetclass(RSOC-Building / RSOC-S-Vehicle / RSOC-L-Vehicle / RSOC-Ship / VD-People / VD-Vehicle)
2)readreal label, computeeach image objectnumber
3)fromobjectnumber Top 10% inrandomselect[oneimage]sample
4)tothisimagesample, youcanmanytimesdraw ROI: 
    - SAM3 executetwokindmode: 
        a) puretext prompt(allimage) -> count_text_only(onlycomputeonce, globaltotalshare)
        b) text + sohasalreadydraw ROI boxprompt(allimage)-> each timeallinbeforesurfacebasisonaddonebox, getnew  count_text_box
    - each timedrawcompleteonebox: 
        * generateoneitempseudo labelrecord
        * outputtwoimagevisualizeimage: 
            - SAM image(grid + current ROI + "text+sohasbox"result)
            - GT image(grid + current ROI + real label)
    - according to ESC / drawout 0 size ROI endinteractive

configurationallintopconstantmodifyi.e.can, notusecommand lineparameter. 
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

ROOT_DATA_DIR = "Dataset"  # youprojectinside Dataset root directory

LOCAL_SAM3_PATH = r"D:\sam3_source"
SAM3_CHECKPOINT_PATH = "saved_models/SAM3.pt"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# selectonedatasetclass(according toneedmodify): 
# "RSOC-Building", "RSOC-S-Vehicle", "RSOC-L-Vehicle", "RSOC-Ship",
# "VD-People", "VD-Vehicle"
TARGET_DATASET = "RSOC-L-Vehicle"

# selectimagerule: take Top 10% highdensityinsiderandomselect 1 image
TOP_PERCENT = 0.1

# pseudo labelandvisualizeoutputdirectory
PSEUDO_OUT_DIR = "pseudo_labels_roi_experiment"
VIS_OUT_DIR = "vis_roi_experiment"

os.makedirs(PSEUDO_OUT_DIR, exist_ok=True)
os.makedirs(VIS_OUT_DIR, exist_ok=True)

# SAM3 text prompt: externallabel -> promptword
DATASET_PROMPT_MAP = {
    "RSOC-Building": "building",
    "RSOC-S-Vehicle": "small vehicle",
    "RSOC-L-Vehicle": "large vehicle",
    "RSOC-Ship": "ship",
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
    samples = []

    # 1) RSOC-Building: use .mat in pointnumber
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
                    "count": float(count),
                    "gt_type": "points",
                    "gt_data": pts,
                })

    # 2) RSOC   small-vehicle / large-vehicle / ship, use DOTA labelTxt box
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
                    "count": float(len(boxes)),
                    "gt_type": "boxes",
                    "gt_data": np.array(boxes, dtype=float),
                })

        collect_split("train")
        collect_split("val")

    # 3) VD-People / VD-Vehicle: use density map .npy   sum whencount
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
                    "count": count,
                    "gt_type": "density",
                    "gt_data": npy_path,
                })

    else:
        raise ValueError(f"notknowdatasetlabel: {dataset_label}")

    print(f"[Index] {dataset_label}: total {len(samples)} one withrealcount sample")
    return samples


# ===================== selectoneimage Top10%  highdensitysample =====================
def select_one_dense_sample(samples, top_percent=0.1):
    if len(samples) == 0:
        return None

    samples_sorted = sorted(samples, key=lambda x: x["count"], reverse=True)
    n_top = max(1, int(len(samples_sorted) * top_percent))
    top_list = samples_sorted[:n_top]

    # notselectcountmaximum thatimage(ifhasmanyimage)
    if len(top_list) > 1:
        candidates = top_list[1:]
    else:
        candidates = top_list

    chosen = random.choice(candidates)
    print(f"[Select] from {len(samples)} imageintake Top{top_percent*100:.0f}%   {n_top} image, "
          f"removemaximumcountafter, from {len(candidates)} imageinrandomselect 1 image. ")
    return chosen


# ===================== SAM3 call: text-only =====================
def sam3_text_only(processor, pil_img, prompt_text):
    """
    puretext prompt: inwholeimageonfindoutsohasmatch prompt  instance. 
    return boxes_text: [N, 4] (xyxy)
    """
    state = processor.set_image(pil_img)
    out = processor.set_text_prompt(state=state, prompt=prompt_text)
    boxes = out.get("boxes", None)
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), dtype=float)
    if torch.is_tensor(boxes):
        boxes_np = boxes.detach().cpu().numpy()
    else:
        boxes_np = np.array(boxes)
    return boxes_np.astype(float)


def compute_patch_counts_from_boxes(img_w, img_h, boxes, grid=(4, 4)):
    grid_h, grid_w = grid
    ph = img_h / grid_h
    pw = img_w / grid_w
    counts = np.zeros(grid_h * grid_w, dtype=np.int32)

    for b in boxes:
        x1, y1, x2, y2 = b
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        gx = int(cx // pw)
        gy = int(cy // ph)
        if gx < 0 or gx >= grid_w or gy < 0 or gy >= grid_h:
            continue
        idx = gy * grid_w + gx
        counts[idx] += 1
    return counts


# ===================== visualize: grid + ROI + SAM / GT =====================
def draw_grid(img_bgr, color=(0, 255, 0), thickness=1):
    h, w = img_bgr.shape[:2]
    # verticalline
    for i in range(1, 4):
        x = int(w * i / 4)
        cv2.line(img_bgr, (x, 0), (x, h), color, thickness)
    # horizontalline
    for j in range(1, 4):
        y = int(h * j / 4)
        cv2.line(img_bgr, (0, y), (w, y), color, thickness)


def draw_roi(img_bgr, roi, color=(255, 255, 0), thickness=2):
    x, y, w, h = roi
    cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, thickness)


def draw_sam_boxes(img_bgr, boxes, color=(0, 0, 255), thickness=2):
    for b in boxes:
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)


def draw_gt_points(img_bgr, pts, color=(0, 0, 255), radius=2, thickness=-1):
    for (x, y) in pts:
        cv2.circle(img_bgr, (int(x), int(y)), radius, color, thickness)


def draw_gt_boxes_quads(img_bgr, boxes, color=(0, 0, 255), thickness=2):
    for quad in boxes:
        x1, y1, x2, y2, x3, y3, x4, y4 = quad
        pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)
        cv2.polylines(img_bgr, [pts], isClosed=True, color=color, thickness=thickness)


def overlay_density_heatmap(img_bgr, density, alpha=0.4):
    h, w = img_bgr.shape[:2]
    dh, dw = density.shape
    if (dh, dw) != (h, w):
        density_resized = cv2.resize(density, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        density_resized = density

    if density_resized.max() > 0:
        dens_norm = density_resized / density_resized.max()
    else:
        dens_norm = density_resized
    dens_uint8 = (dens_norm * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(dens_uint8, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)
    return blended


# ===================== main pipeline =====================
def main():
    print(f"[Main] dataset: {TARGET_DATASET}")
    prompt_text = DATASET_PROMPT_MAP[TARGET_DATASET]
    print(f"[Main] SAM3 text prompt: \"{prompt_text}\"")

    # 1) buildwithrealcount samplelist
    samples = build_samples_for_dataset(TARGET_DATASET)
    if len(samples) == 0:
        print("[Main] nosample, end. ")
        return

    # 2) from Top10% inselect[oneimage]sample
    chosen = select_one_dense_sample(samples, TOP_PERCENT)
    if chosen is None:
        print("[Main] notselectoutsample, end. ")
        return

    img_path = chosen["img_path"]
    gt_type = chosen["gt_type"]
    gt_data = chosen["gt_data"]
    true_count = chosen["count"]

    print(f"\n[Chosen Sample]")
    print(f"  image: {img_path}")
    print(f"  realcount: {true_count:.2f}")

    # 3) readimage
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("[Main] readimagefail, end. ")
        return
    h, w = img_bgr.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    # 3.1 initialize SAM3
    processor = build_sam3_processor()

    # 3.2 firstcomputeoncepuretext prompt(allimage)result
    boxes_text_only = sam3_text_only(processor, pil_img, prompt_text)
    count_text_only = int(boxes_text_only.shape[0])
    print(f"[SAM] puretext prompt(allimage)detectionto {count_text_only} instance")

    # 3.3 buildstandone"interactive state": text + image, onlydo once
    interactive_state = processor.set_image(pil_img)
    interactive_state = processor.set_text_prompt(state=interactive_state, prompt=prompt_text)

    pseudo_entries = []
    roi_index = 0

    # 4) manytimesbox ROI interactive
    while True:
        roi_index += 1
        disp = img_bgr.copy()
        msg = f"Draw ROI #{roi_index} then press ENTER (ESC to finish)"
        cv2.putText(disp, msg, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        roi = cv2.selectROI("Select ROI", disp, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")

        x, y, rw, rh = roi
        if rw <= 0 or rh <= 0:
            print(f"[ROI] notselecthaseffect ROI, endinteractive. ")
            break

        print(f"\n[ROI #{roi_index}] x={x}, y={y}, w={rw}, h={rh}")

        # ====== keychange: insameone interactive_state onchaseaddgeometric prompt ======
        width, height = pil_img.size
        box_xywh = torch.tensor([[x, y, rw, rh]], dtype=torch.float32, device=DEVICE)
        box_cxcywh = box_xywh_to_cxcywh(box_xywh)
        norm_box_cxcywh = [
            float(box_cxcywh[0, 0] / width),
            float(box_cxcywh[0, 1] / height),
            float(box_cxcywh[0, 2] / width),
            float(box_cxcywh[0, 3] / height),
        ]

        interactive_state = processor.add_geometric_prompt(
            box=norm_box_cxcywh, label=True, state=interactive_state
        )

        boxes = interactive_state.get("boxes", None)
        if boxes is None or len(boxes) == 0:
            boxes_text_box = np.zeros((0, 4), dtype=float)
        else:
            if torch.is_tensor(boxes):
                boxes_text_box = boxes.detach().cpu().numpy().astype(float)
            else:
                boxes_text_box = np.array(boxes, dtype=float)

        count_text_box = int(boxes_text_box.shape[0])
        print(f"  text + sohas ROI boxprompt(allimage)currentdetectionto {count_text_box} instance")

        # based on"text+sohasbox"resultdo 4×4 patch count
        sam_patch_counts = compute_patch_counts_from_boxes(w, h, boxes_text_box, grid=(4, 4))

        # pseudo labelrecord
        pseudo_entry = {
            "img_path": img_path,
            "dataset_label": TARGET_DATASET,
            "roi": [int(x), int(y), int(rw), int(rh)],
            "roi_index": int(roi_index),
            "sam_text_only_count": int(count_text_only),
            "sam_text_box_count": int(count_text_box),
            "sam_patch_counts_text_box": sam_patch_counts.astype(int).tolist(),
            "true_count_full": float(true_count),
        }
        pseudo_entries.append(pseudo_entry)

        # SAM visualize(use"text+sohasbox" currentresult)
        sam_vis = img_bgr.copy()
        draw_grid(sam_vis)
        draw_roi(sam_vis, (x, y, rw, rh))
        draw_sam_boxes(sam_vis, boxes_text_box)

        sam_vis_name = os.path.join(
            VIS_OUT_DIR,
            f"{TARGET_DATASET.replace('-', '_')}_sam_roi{roi_index}.jpg"
        )
        cv2.imwrite(sam_vis_name, sam_vis)
        print(f"  [Save] SAM visualizesaveto: {sam_vis_name}")

        # GT visualize
        gt_vis = img_bgr.copy()
        draw_grid(gt_vis)
        draw_roi(gt_vis, (x, y, rw, rh))

        if gt_type == "points":
            pts = gt_data
            draw_gt_points(gt_vis, pts)
        elif gt_type == "boxes":
            boxes_gt = gt_data
            draw_gt_boxes_quads(gt_vis, boxes_gt)
        elif gt_type == "density":
            density = np.load(gt_data)
            gt_vis = overlay_density_heatmap(gt_vis, density)

        gt_vis_name = os.path.join(
            VIS_OUT_DIR,
            f"{TARGET_DATASET.replace('-', '_')}_gt_roi{roi_index}.jpg"
        )
        cv2.imwrite(gt_vis_name, gt_vis)
        print(f"  [Save] GT visualizesaveto: {gt_vis_name}")

    # 5) savepseudo label
    if len(pseudo_entries) > 0:
        arr = np.array(pseudo_entries, dtype=object)
        save_path = os.path.join(
            PSEUDO_OUT_DIR,
            f"pseudo_roi_{TARGET_DATASET.replace('-', '_')}.npy"
        )
        np.save(save_path, arr)
        print(f"\n[Pseudo] totalsave {len(pseudo_entries)} item ROI pseudo labelto: {save_path}")
    else:
        print("\n[Pseudo] nogenerateanypseudo label, maybeyoudirect ESC backout . ")


if __name__ == "__main__":
    main()
