# -*- coding: utf-8 -*-
import os
import sys
import types
from unittest.mock import MagicMock
import torch
import numpy as np
from config import LOCAL_SAM3_PATH, SAM3_CHECKPOINT_PATH, SAM3_CONFIDENCE_THRESHOLD, DEVICE

# SAM3 Windows Triton compatibility shim
if sys.platform.startswith("win"):
    def _dummy_decorator(*args, **kwargs):
        if kwargs or (args and not callable(args[0])): return lambda f: f
        return args[0]
    mock_triton = types.ModuleType("triton")
    mock_triton.__spec__ = MagicMock()
    mock_triton.__spec__.name = "triton"
    mock_triton.jit = _dummy_decorator
    mock_triton.autotune = _dummy_decorator
    mock_triton.heuristics = _dummy_decorator
    mock_triton.cdiv = lambda x, y: (x + y - 1) // y
    mock_triton.next_power_of_2 = lambda x: 1 << (x - 1).bit_length()
    mock_triton.Config = type("DummyConfig", (), {"__init__": lambda *a, **kw: None})
    mock_triton.language = mock_triton.impl = MagicMock()
    sys.modules["triton"] = mock_triton
    sys.modules["triton.language"] = mock_triton.language
    sys.modules["triton.impl"] = mock_triton.impl
    try: import flash_attn  # noqa: F401
    except ImportError:
        mock_flash = types.ModuleType("flash_attn")
        mock_flash.__spec__ = MagicMock()
        sys.modules["flash_attn"] = mock_flash
        sys.modules["flash_attn.flash_attn_interface"] = MagicMock()

if os.path.exists(LOCAL_SAM3_PATH) and LOCAL_SAM3_PATH not in sys.path:
    sys.path.insert(0, LOCAL_SAM3_PATH)

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.box_ops import box_xywh_to_cxcywh

# Helper function for this experiment module
def build_sam3_processor():
    if not os.path.exists(SAM3_CHECKPOINT_PATH):
        raise FileNotFoundError(f"SAM3 权重不存在: {SAM3_CHECKPOINT_PATH}")
    print("[Model] 加载 SAM3 模型 ...")
    model = build_sam3_image_model(checkpoint_path=SAM3_CHECKPOINT_PATH, load_from_HF=False, device=str(DEVICE).split(":")[0])
    processor = Sam3Processor(model, confidence_threshold=SAM3_CONFIDENCE_THRESHOLD)
    print("[Model] SAM3 Processor 初始化完成")
    return processor

def _to_numpy(x):
    if x is None: return None
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.array(x)

def _extract_sam3_outputs(state_or_output):
    if state_or_output is None: return {"boxes": np.zeros((0, 4)), "masks": np.zeros((0, 1, 1), dtype=bool), "scores": np.zeros((0,))}
    is_dict = isinstance(state_or_output, dict)
    boxes = state_or_output.get("boxes", None) if is_dict else getattr(state_or_output, "boxes", None)
    masks = (state_or_output.get("masks", state_or_output.get("mask", None)) if is_dict 
             else getattr(state_or_output, "masks", getattr(state_or_output, "mask", None)))
    scores = state_or_output.get("scores", None) if is_dict else getattr(state_or_output, "scores", None)

    boxes_np = _to_numpy(boxes)
    boxes_np = boxes_np.astype(float).reshape(-1, 4) if boxes_np is not None and boxes_np.size else np.zeros((0, 4), dtype=float)
    masks_np = _to_numpy(masks)
    masks_np = masks_np if masks_np is not None else np.zeros((0, 1, 1), dtype=bool)
    if masks_np.ndim == 4 and masks_np.shape[1] == 1: masks_np = masks_np[:, 0, :, :]
    elif masks_np.ndim == 2: masks_np = masks_np[None, :, :]
    masks_np = (masks_np > 0.5) if masks_np.dtype != np.bool_ else masks_np.astype(bool)
    scores_np = _to_numpy(scores)
    scores_np = scores_np.astype(float).reshape(-1) if scores_np is not None and scores_np.size else np.zeros((boxes_np.shape[0],))
    
    n = int(min(boxes_np.shape[0], masks_np.shape[0])) if masks_np.shape[0] > 0 else int(boxes_np.shape[0])
    return {"boxes": boxes_np[:n], "masks": masks_np[:n], "scores": scores_np[:n]}

def sam3_text_plus_multi_boxes_with_labels(processor, pil_img, boxes_xywh_list, labels_list, prompt_text):
    if boxes_xywh_list is None: boxes_xywh_list = []
    if labels_list is None: labels_list = [True] * len(boxes_xywh_list)
    width, height = pil_img.size
    state = processor.set_text_prompt(state=processor.set_image(pil_img), prompt=prompt_text)
    for (x, y, w, h), lab in zip(boxes_xywh_list, labels_list):
        if w <= 0 or h <= 0: continue
        box_cxcywh = box_xywh_to_cxcywh(torch.tensor([[x, y, w, h]], dtype=torch.float32, device=DEVICE))
        norm_box = [float(box_cxcywh[0, i] / (width if i%2==0 else height)) for i in range(4)]
        state = processor.add_geometric_prompt(box=norm_box, label=bool(lab), state=state)
    return _extract_sam3_outputs(state)

def sam3_text_only(processor, pil_img, prompt_text):
    return _extract_sam3_outputs(processor.set_text_prompt(state=processor.set_image(pil_img), prompt=prompt_text))
