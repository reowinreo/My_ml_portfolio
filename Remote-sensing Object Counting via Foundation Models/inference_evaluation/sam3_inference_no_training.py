
import os
import sys
import types
import random
from unittest.mock import MagicMock

import torch
from PIL import Image, ImageDraw
import numpy as np

# ========================================================
# 0. Windows compatibilitypatchsmall (Triton / FlashAttn simulate)
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
# 1. pathconfiguration
# ========================================================
LOCAL_SAM3_PATH = r"D:\sam3_source"
SAM3_CHECKPOINT_PATH = "saved_models/SAM3.pt"

# yousaythisthreedatasetallinproject  Dataset folderbelow
ROOT_DATA_DIR = "Dataset"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if os.path.exists(LOCAL_SAM3_PATH) and LOCAL_SAM3_PATH not in sys.path:
    sys.path.insert(0, LOCAL_SAM3_PATH)

# ========================================================
# 2. import SAM3 buildfunctionand Processor
# ========================================================
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    print("[Info] successimport SAM3 API")
except ImportError as e:
    print(f"[Error] cannot import SAM3 module: {e}")
    sys.exit(1)

# ========================================================
# 3. datasetconfiguration & labelmap
# ========================================================
def build_dataset_configs(root_dir: str):
    configs = []

    # ---------- RSOC-Building ----------
    # ASPDNet_dataset/RSOC_building/building/train_data/images/*.jpg
    # ASPDNet_dataset/RSOC_building/building/test_data/images/*.jpg
    rsoc_build_root = os.path.join(root_dir, "ASPDNet_dataset", "RSOC_building", "building")
    for split in ["train_data", "test_data"]:
        img_dir = os.path.join(rsoc_build_root, split, "images")
        configs.append({
            "dataset_label": "RSOC-Building",  # originallabel
            "prompt": "building",             # SAM3  text prompt
            "image_dir": img_dir
        })

    # ---------- VisDrone-People ----------
    # VisDrone-People/{train,val,test}/images/*.jpg
    vd_people_root = os.path.join(root_dir, "VisDrone-People")
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(vd_people_root, split, "images")
        configs.append({
            "dataset_label": "VD-People",
            "prompt": "people",   # alsocanchange to "person" / "crowd"
            "image_dir": img_dir
        })

    # ---------- VisDrone-Vehicle ----------
    # VisDrone-Vehicle/{train,val,test}/images/*.jpg
    vd_vehicle_root = os.path.join(root_dir, "VisDrone-Vehicle")
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(vd_vehicle_root, split, "images")
        configs.append({
            "dataset_label": "VD-Vehicle",
            "prompt": "vehicle",  # alsocanchange to "car and truck"
            "image_dir": img_dir
        })

    # afterneedadd RSOC-S-Vehicle / RSOC-L-Vehicle / RSOC-Ship, according tothisformatagain append i.e.can
    return configs


def list_images(img_dir: str):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    if not os.path.isdir(img_dir):
        return []
    return [
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith(exts)
    ]


def choose_random_sample(root_dir: str):
    configs = build_dataset_configs(root_dir)
    candidates = []

    for cfg in configs:
        imgs = list_images(cfg["image_dir"])
        if imgs:
            candidates.append((cfg, imgs))

    if not candidates:
        print(f"[Error] in {root_dir} belownofindtoanyhaseffectimage, pleasecheckpath. ")
        sys.exit(1)

    cfg, imgs = random.choice(candidates)
    img_path = random.choice(imgs)
    return cfg, img_path

# ========================================================
# 4. overlaysohasinstance  mask and bbox
# ========================================================
def overlay_masks_and_boxes(image: Image.Image, masks, boxes, scores, prompt_text: str):
    """
    inoriginalimageonoverlay **sohasinstance**   mask and bbox, andwriteonscoreandprompttext. 
    masks: Tensor/ndarray, shape [N, 1, H, W] or [N, H, W]
    boxes: Tensor/ndarray, shape [N, 4] (x1, y1, x2, y2)
    scores: Tensor/ndarray, shape [N]
    """
    if masks is None or len(masks) == 0:
        return image

    # turn numpy
    if torch.is_tensor(masks):
        masks_np = masks.detach().cpu().numpy()
    else:
        masks_np = np.array(masks)

    if torch.is_tensor(boxes):
        boxes_np = boxes.detach().cpu().numpy()
    else:
        boxes_np = np.array(boxes)

    if torch.is_tensor(scores):
        scores_np = scores.detach().cpu().numpy()
    else:
        scores_np = np.array(scores)

    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    H_img, W_img = img.size[1], img.size[0]  # PIL size = (W, H)

    num_inst = masks_np.shape[0]

    # traversesohasinstance
    for i in range(num_inst):
        mask = masks_np[i]
        box = boxes_np[i]
        score = float(scores_np[i])

        # compatible [1, H, W]
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]

        mask = (mask > 0.5).astype(np.uint8)
        h, w = mask.shape

        # if mask sizeandoriginalimagenotconsistent, then resize
        if (w, h) != img.size:
            mask_img = Image.fromarray(mask * 255).resize(img.size, resample=Image.NEAREST)
            mask = np.array(mask_img) // 255
            h, w = mask.shape

        # overlaygreensemi-transparent mask
        for y in range(h):
            for x in range(w):
                if mask[y, x]:
                    draw.point((x, y), fill=(0, 255, 0, 60))  # sohasinstancesameonekindcolor

        # drawred bbox
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=2)

        # writeonerowsmallcharacter: No. i instance + score
        text = f"{prompt_text} #{i+1} | {score:.2f}"
        # avoidtextdrawtoimageoutside
        tx = max(0, int(x1))
        ty = max(0, int(y1) - 12)
        draw.text((tx, ty), text, fill=(255, 255, 0, 255))

    out = Image.alpha_composite(img, overlay).convert("RGB")
    return out

# ========================================================
# 5. main pipeline
# ========================================================
def main():
    # ---------- 5.1 randomselectdataset & image ----------
    cfg, img_path = choose_random_sample(ROOT_DATA_DIR)

    dataset_label = cfg["dataset_label"]  # originallabel (RSOC-Building / VD-People / VD-Vehicle)
    prompt_text = cfg["prompt"]          # SAM3 text prompt ("building" / "people" / "vehicle")

    img = Image.open(img_path).convert("RGB")

    print("\n[Sample]")
    print(f"  datasetoriginallabel: {dataset_label}")
    print(f"  SAM3 prompttext : \"{prompt_text}\"")
    print(f"  imagepath      : {img_path}")

    # ---------- 5.2 load SAM3 model ----------
    if not os.path.exists(SAM3_CHECKPOINT_PATH):
        print(f"[Error] localweight filenotsavein: {SAM3_CHECKPOINT_PATH}")
        sys.exit(1)

    print("\n[Model] load SAM3 modelin ...")
    try:
        model = build_sam3_image_model(
            checkpoint_path=SAM3_CHECKPOINT_PATH,
            load_from_HF=False,
            device=str(DEVICE).split(":")[0],
        )
    except Exception as e:
        print(f"[Fatal] build SAM3 modelfail: {e}")
        sys.exit(1)

    processor = Sam3Processor(model, confidence_threshold=0.3)
    print("[Model] modeland Processor initialization complete")

    # ---------- 5.3 SAM3 inference ----------
    print("\n[Infer] settingimage ...")
    state = processor.set_image(img)

    print(f"[Infer] text prompt: \"{prompt_text}\"")
    output = processor.set_text_prompt(
        state=state,
        prompt=prompt_text
    )

    masks = output.get("masks", None)
    boxes = output.get("boxes", None)
    scores = output.get("scores", None)

    if masks is None or len(masks) == 0:
        print("[Result] SAM3 inthisimageonnotfindtofullenoughtext prompt instance. ")
        print("[Show] displayoriginalimage")
        img.show(title=f"Original ({dataset_label})")
    else:
        num_inst = len(masks)
        print(f"[Result] SAM3 findto  {num_inst} instance. ")
        if scores is not None:
            scores_np = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.array(scores)
            print("  Top 5 scores:", np.round(-np.sort(-scores_np)[:5], 3))

        vis_img = overlay_masks_and_boxes(img, masks, boxes, scores, prompt_text)

        print("[Show] displayoriginalimage")
        img.show(title=f"Original ({dataset_label})")

        print("[Show] display SAM3 maskresult(allinstance)")
        vis_img.show(title=f"SAM3 Result ({dataset_label})")

    print("\n[Done] inferenceend. ")


if __name__ == "__main__":
    main()
