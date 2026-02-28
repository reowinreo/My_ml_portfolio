# -*- coding: utf-8 -*-
import torch
import numpy as np
from config import GDINO_MODEL_ID, GD_BBOX_THRESHOLD, GD_TEXT_THRESHOLD

_GD_PROCESSOR = None
_GD_MODEL = None
_GD_DEVICE_STR = None

# Helper function for this experiment module
def _ensure_gd_prompt(prompt_text: str) -> str:
    if prompt_text is None: return ""
    t = str(prompt_text).strip()
    if not t: return t
    if not t.endswith("."): t = t + " ."
    elif not t.endswith(" ."): t = t[:-1].rstrip() + " ."
    return t

def _load_local_gd():
    global _GD_PROCESSOR, _GD_MODEL, _GD_DEVICE_STR
    if _GD_PROCESSOR is not None and _GD_MODEL is not None:
        return _GD_PROCESSOR, _GD_MODEL, _GD_DEVICE_STR
    try:
        from transformers import AutoProcessor, GroundingDinoForObjectDetection
    except ImportError as e:
        raise ImportError("未能导入 transformers。请先安装：pip install -U transformers") from e
    _GD_DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[GD] Loading local GroundingDINO: {GDINO_MODEL_ID} | device={_GD_DEVICE_STR}")
    _GD_PROCESSOR = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
    _GD_MODEL = GroundingDinoForObjectDetection.from_pretrained(GDINO_MODEL_ID).to(_GD_DEVICE_STR)
    _GD_MODEL.eval()
    return _GD_PROCESSOR, _GD_MODEL, _GD_DEVICE_STR

def call_dino_local_on_rgb(img_rgb, prompt_text):
    processor, model, device = _load_local_gd()
    h, w = img_rgb.shape[:2]
    prompt = _ensure_gd_prompt(prompt_text)
    inputs = processor(images=img_rgb, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad(): outputs = model(**inputs)
    target_sizes = torch.tensor([[h, w]], device=device)
    try:
        processed = processor.post_process_grounded_object_detection(
            outputs, inputs, box_threshold=float(GD_BBOX_THRESHOLD),
            text_threshold=float(GD_TEXT_THRESHOLD), target_sizes=target_sizes
        )[0]
    except TypeError:
        processed = processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], target_sizes=target_sizes
        )[0]
        scores, boxes = processed.get("scores", None), processed.get("boxes", None)
        if scores is None or boxes is None: return []
        keep = scores > float(GD_BBOX_THRESHOLD)
        processed = {"boxes": boxes[keep]}
        
    boxes = processed.get("boxes", None)
    if boxes is None: return []
    return boxes.detach().cpu().numpy().astype(float).tolist() if torch.is_tensor(boxes) else np.array(boxes).astype(float).tolist()

def call_dino_local_on_pil(pil_img, prompt_text):
    return call_dino_local_on_rgb(np.array(pil_img.convert("RGB")), prompt_text)
