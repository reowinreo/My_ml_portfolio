# -*- coding: utf-8 -*-
"""
SAM3 confidencevisualize(singlerandomsample)- fixversion

youprevious reportwrongreason: 
- codefrom state insidetaketo  `masks` notis (N,H,W)  twovaluemask/masklogits, 
  andisshapeif (N,H,W,512)  manychanneltensor(orothermanychannelfeature). 
- cv2.findContours onlysupportsinglechannel CV_8UC1, thereforereport CV_8UC512. 

thisscript: 
1) alonguseyou<leaderboard.py> datasetreadand SAM3 initialize(contain Windows Fix)
2) random draw 1 image, use SAM3 doinstancedetection/segmentation
3) onlydisplay score >= VIS_SCORE_THRESHOLD  instance: 
   - ifcanraisetaketojustconfirm instance masks: draw mask boundary + score
   - ifnomethodraisetake masks: degeneratefordraw box + score(stillcanlook score isor notreliable)
4) mustsetsaveoutput png, andoptionalpopupdisplay

changeArguments:
- TARGET_DATASET
- VIS_SCORE_THRESHOLD(default 0.1)
- PROCESSOR_CONFIDENCE_THRESHOLD(wantlooktomorelowpartcandidatethenset 0.0)
- SHOW_WINDOW(isor notpopup)
"""

import os
import sys
import types
import random
from unittest.mock import MagicMock

import numpy as np
import cv2
import torch
import scipy.io as sio
from PIL import Image

# ===================== configuration section(andleaderboardscriptkeepconsistent) =====================

ROOT_DATA_DIR = "Dataset"
LOCAL_SAM3_PATH = r"D:\sam3_source"
SAM3_CHECKPOINT_PATH = "saved_models/SAM3.pt"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TARGET_DATASET = "RSOC-Building"

# youneed threshold: onlydraw score >= thisthreshold instance(canselfrowchange)
VIS_SCORE_THRESHOLD = 0.1

# Processor insidesetfilterthreshold: wantobservelowpartfragment, takeitsetlow(if 0.0)
PROCESSOR_CONFIDENCE_THRESHOLD = 0.00

# isor notpopupdisplay(saveonesetwilldo)
SHOW_WINDOW = True

DATASET_PROMPT_MAP = {
    "RSOC-Building": "building",
    "RSOC-S-Vehicle": "small vehicle",
    "RSOC-L-Vehicle": "large vehicle",
    "RSOC-Ship": "ship",
    "VD-People": "person",
    "VD-Vehicle": "vehicle",
}

RSOC_DOTA_CATEGORY_MAP = {
    "RSOC-S-Vehicle": "small-vehicle",
    "RSOC-L-Vehicle": "large-vehicle",
    "RSOC-Ship": "ship",
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# ===================== Windows Fix(andyouoriginalscriptconsistent) =====================
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

# ===================== import SAM3(andyouoriginalscriptconsistent) =====================
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
    processor = Sam3Processor(model, confidence_threshold=float(PROCESSOR_CONFIDENCE_THRESHOLD))
    print(f"[Model] Processor initialization complete (confidence_threshold={PROCESSOR_CONFIDENCE_THRESHOLD})")
    return processor


# ===================== datasetread(andyouoriginalscriptkeepconsistent) =====================

def list_images(img_dir):
    if not os.path.isdir(img_dir):
        return []
    return [
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith(IMAGE_EXTS)
    ]


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


def build_samples_for_dataset(dataset_label):
    samples = []

    if dataset_label == "RSOC-Building":
        rsoc_root = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "RSOC_building", "building")
        for _, subdir in [("train", "train_data"), ("test", "test_data")]:
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

    elif dataset_label in ["RSOC-S-Vehicle", "RSOC-L-Vehicle", "RSOC-Ship"]:
        target_cat = RSOC_DOTA_CATEGORY_MAP[dataset_label]

        def collect_split(split):
            if split == "train":
                img_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "train", "images")
                label_dir = os.path.join(
                    ROOT_DATA_DIR,
                    "ASPDNet_dataset",
                    "train",
                    "labelTxt-v1.0",
                    "trainset_reclabelTxt",
                )
            elif split == "val":
                img_dir = os.path.join(ROOT_DATA_DIR, "ASPDNet_dataset", "val", "images")
                label_dir = os.path.join(
                    ROOT_DATA_DIR,
                    "ASPDNet_dataset",
                    "val",
                    "labelTxt-v1.0",
                    "valset_reclabelTxt",
                )
            else:
                return

            if not os.path.isdir(label_dir):
                return

            img_list = list_images(img_dir)
            base_to_img = {os.path.splitext(os.path.basename(p))[0]: p for p in img_list}

            for txt_name in os.listdir(label_dir):
                if not txt_name.lower().endswith(".txt"):
                    continue
                txt_path = os.path.join(label_dir, txt_name)
                base = os.path.splitext(txt_name)[0]
                if base not in base_to_img:
                    continue
                img_path = base_to_img[base]

                n = 0
                with open(txt_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 9:
                            continue
                        cat = parts[-2]
                        if cat != target_cat:
                            continue
                        n += 1

                if n == 0:
                    continue

                samples.append({
                    "img_path": img_path,
                    "true_count": float(n),
                    "gt_type": "boxes",
                    "gt_data": None,
                })

        collect_split("train")
        collect_split("val")

    elif dataset_label in ["VD-People", "VD-Vehicle"]:
        vd_root = os.path.join(
            ROOT_DATA_DIR,
            "VisDrone-People" if dataset_label == "VD-People" else "VisDrone-Vehicle",
        )

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


# ===================== SAM3 inferenceand masks raisetake(fixcore) =====================

def _to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize_masks_candidate(m, n_expected, h, w):
    """trytakecandidate m wholelogicbecome (N,H,W)   uint8; notsymbolcombinethenreturn None."""
    if m is None:
        return None
    m = _to_numpy(m)
    if m is None:
        return None

    m = np.asarray(m)

    # Common:float logits -> bin
    if m.dtype != np.uint8 and m.dtype != np.bool_:
        # notneedtoobviousisfeature  (..,512) randombinarize: firstdoshapejudgedo not
        pass

    # allowshape: 
    # (N,H,W), (N,1,H,W), (N,H,W,1), (H,W,N)
    if m.ndim == 4:
        # (N,1,H,W)
        if m.shape[0] == n_expected and m.shape[1] == 1 and m.shape[2] == h and m.shape[3] == w:
            m = m[:, 0, :, :]
        # (N,H,W,1)
        elif m.shape[0] == n_expected and m.shape[1] == h and m.shape[2] == w and m.shape[3] == 1:
            m = m[:, :, :, 0]
        else:
            # shapeif (N,H,W,512) or (H,W,512,N) etc.--directjudgesetfornon-instance mask
            return None

    if m.ndim == 3:
        # (H,W,N) -> (N,H,W)
        if m.shape[0] == h and m.shape[1] == w and m.shape[2] == n_expected:
            m = np.transpose(m, (2, 0, 1))
        # (N,H,W)
        elif m.shape[0] == n_expected and m.shape[1] == h and m.shape[2] == w:
            pass
        else:
            # (H,W,512) or (N,H,512) ofclass--notisinstance mask
            return None

    if m.ndim == 2:
        # onlyhasoneimage mask, nomethodcorrespondmanyinstance, unless n_expected==1
        if n_expected == 1 and m.shape[0] == h and m.shape[1] == w:
            m = m[None, :, :]
        else:
            return None

    if m.ndim != 3:
        return None

    # toheremustis (N,H,W)
    if m.shape[0] != n_expected or m.shape[1] != h or m.shape[2] != w:
        return None

    # binarize
    if m.dtype == np.bool_:
        m = m.astype(np.uint8)
    elif m.dtype != np.uint8:
        # recognizeforis logits / prob
        m = (m > 0).astype(np.uint8)

    return m


def extract_instance_masks_from_state(state, n_expected, h, w, debug=False):
    """from state insideselectselectjustconfirm instance masks. """
    # preferorder: displaytype logits / bin > masks
    candidate_keys = [
        "masks_bin",
        "pred_masks",
        "masks_logits",
        "mask_logits",
        "masks_logit",
        "mask_logit",
        "masks",
        "mask",
    ]

    for k in candidate_keys:
        if k not in state:
            continue
        cand = state.get(k)
        norm = _normalize_masks_candidate(cand, n_expected=n_expected, h=h, w=w)
        if norm is not None:
            if debug:
                arr = _to_numpy(cand)
                print(f"[Mask] use key='{k}', raw_shape={None if arr is None else arr.shape}, raw_dtype={None if arr is None else arr.dtype}")
            return norm
        else:
            if debug:
                arr = _to_numpy(cand)
                if arr is not None:
                    print(f"[Mask] skip key='{k}', raw_shape={arr.shape}, raw_dtype={arr.dtype}")

    return None


def sam3_segment_with_scores(processor, pil_img, prompt_text, use_full_image_box=True, debug_masks=False):
    width, height = pil_img.size

    state = processor.set_image(pil_img)
    state = processor.set_text_prompt(state=state, prompt=prompt_text)

    if use_full_image_box:
        x, y, w, h = 0, 0, width, height
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
    scores = state.get("scores", None)

    boxes = _to_numpy(boxes)
    scores = _to_numpy(scores)

    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=float), None, state

    boxes = np.asarray(boxes, dtype=float)
    if scores is None:
        scores = np.ones((boxes.shape[0],), dtype=float)
    else:
        scores = np.asarray(scores, dtype=float).reshape(-1)

    n = int(min(len(scores), boxes.shape[0]))
    boxes = boxes[:n]
    scores = scores[:n]

    # raisetake masks
    masks = extract_instance_masks_from_state(state, n_expected=n, h=height, w=width, debug=debug_masks)

    return boxes, scores, masks, state


# ===================== visualize: draw mask boundary + score(no mask thendraw box) =====================

def _ensure_single_channel_mask(m, h, w):
    """takesingleinstance mask compressbecome (H,W) uint8; failreturn None."""
    if m is None:
        return None
    m = np.asarray(m)

    # allow (H,W), (1,H,W), (H,W,1)
    if m.ndim == 3:
        if m.shape[0] == 1 and m.shape[1] == h and m.shape[2] == w:
            m = m[0]
        elif m.shape[2] == 1 and m.shape[0] == h and m.shape[1] == w:
            m = m[:, :, 0]
        else:
            # manychannel: notdrawcontour
            return None

    if m.ndim != 2:
        return None

    if m.shape[0] != h or m.shape[1] != w:
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    if m.dtype != np.uint8:
        m = m.astype(np.uint8)

    # guaranteeis 0/255
    if m.max() <= 1:
        m = (m * 255).astype(np.uint8)
    else:
        m = (m > 0).astype(np.uint8) * 255

    return m


def draw_instances_with_scores(img_bgr, boxes, scores, masks, score_thr=0.1):
    vis = img_bgr.copy()
    h, w = vis.shape[:2]

    idxs = np.where(scores >= score_thr)[0]
    if len(idxs) == 0:
        return vis, 0

    # lowpartfirstdraw, highpartafterdraw
    idxs = idxs[np.argsort(scores[idxs])]

    for i in idxs:
        s = float(scores[i])
        cx, cy = None, None

        drew_mask = False
        if masks is not None and i < masks.shape[0]:
            m2 = _ensure_single_channel_mask(masks[i], h=h, w=w)
            if m2 is not None:
                cnts, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(cnts) > 0:
                    cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
                    c = max(cnts, key=cv2.contourArea)
                    M = cv2.moments(c)
                    if M["m00"] > 1e-6:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    drew_mask = True

        # degeneratefordraw box
        if not drew_mask:
            x1, y1, x2, y2 = boxes[i].tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cx, cy = x1, max(0, y1 - 5)

        cv2.putText(
            vis,
            f"{s:.2f}",
            (int(cx), int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    return vis, int(len(idxs))


def main():
    print(f"[Main] dataset: {TARGET_DATASET}")
    prompt_text = DATASET_PROMPT_MAP[TARGET_DATASET]
    print(f"[Main] text prompt: \"{prompt_text}\"")
    print(f"[Main] visualizethreshold: VIS_SCORE_THRESHOLD={VIS_SCORE_THRESHOLD}")

    samples = build_samples_for_dataset(TARGET_DATASET)
    if len(samples) == 0:
        print("[Main] nosample, end. ")
        return

    sample = random.choice(samples)
    img_path = sample["img_path"]
    true_count = sample["true_count"]
    print(f"[Pick] randomsample: {img_path}")
    print(f"[Pick] true_count = {true_count:.2f}")

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("[Error] readimagefail. ")
        return

    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    processor = build_sam3_processor()

    boxes, scores, masks, state = sam3_segment_with_scores(
        processor,
        pil_img,
        prompt_text,
        use_full_image_box=True,
        debug_masks=True,  # print masks key/shape, sidethenyouconfirmrecognizetobottomtaketo iswhat
    )

    print(f"[SAM3] returninstancenumber: {len(scores)} (notthreshold)")

    vis, kept = draw_instances_with_scores(img_bgr, boxes, scores, masks, score_thr=float(VIS_SCORE_THRESHOLD))
    print(f"[Vis] score >= {VIS_SCORE_THRESHOLD}  instancenumber: {kept}")

    base = os.path.splitext(os.path.basename(img_path))[0]
    out_name = f"sam3_confidence_vis_{TARGET_DATASET}_{base}_thr{VIS_SCORE_THRESHOLD:.2f}.png"
    cv2.imwrite(out_name, vis)
    print(f"[Save] alreadysave: {out_name}")

    if SHOW_WINDOW:
        cv2.imshow("SAM3 confidence", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
