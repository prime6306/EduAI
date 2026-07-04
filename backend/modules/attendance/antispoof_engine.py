"""
modules/attendance/antispoof_engine.py
Loads the MobileNetV2 + ResNet18 ensemble pkl and runs inference.
Migrated from with_antiSpoofing.py — Firebase removed, MongoDB only.
"""
import pickle
import time

import cv2
import numpy as np
from PIL import Image

from backend.config import get_settings

_models_dict = None
_device = None
_transform = None


def _get_transform():
    global _transform
    if _transform is None:
        from torchvision import transforms
        _transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    return _transform


def _get_device():
    import torch
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def load_models() -> dict:
    import torch
    global _models_dict
    if _models_dict is not None:
        return _models_dict
    path = get_settings().antispoof_model_path
    device = _get_device()
    try:
        raw = torch.load(path, map_location=device, weights_only=False)
        # Handle both formats: dict of full models or dict with 'models' key
        if isinstance(raw, dict) and "models" in raw:
            models = raw["models"]
            loaded = {}
            for name, obj in models.items():
                if isinstance(obj, torch.nn.Module):
                    obj.to(device)
                    obj.eval()
                    loaded[name] = obj
            _models_dict = loaded
        elif isinstance(raw, dict):
            loaded = {}
            for name, obj in raw.items():
                if isinstance(obj, torch.nn.Module):
                    obj.to(device)
                    obj.eval()
                    loaded[name] = obj
            _models_dict = loaded
        else:
            raise ValueError("Unexpected model format")
        print(f"[AntiSpoof] Loaded {len(_models_dict)} model(s) on {device}")
    except Exception as e:
        print(f"[AntiSpoof] WARNING: Could not load model: {e}. Running in bypass mode.")
        _models_dict = {}
    return _models_dict


def is_real_face(face_img: np.ndarray) -> tuple[bool, float]:
    """
    Returns (is_real: bool, inference_time_sec: float).
    If models not loaded, returns (True, 0.0) — bypass mode.
    """
    import torch
    models = load_models()
    if not models:
        return True, 0.0  # bypass if model unavailable

    t0 = time.perf_counter()
    device = _get_device()
    pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    tensor = _get_transform()(pil_img).unsqueeze(0).to(device)

    probs_sum = None
    with torch.no_grad():
        for model in models.values():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)
            probs_sum = probs if probs_sum is None else probs_sum + probs

    avg = probs_sum / len(models)
    pred = torch.argmax(avg, dim=1).item()
    return pred == 1, time.perf_counter() - t0  # 1 = real
