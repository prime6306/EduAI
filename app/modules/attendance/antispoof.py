"""
Anti-spoof ensemble (MobileNetV2 + ResNet18, PyTorch). This is a PRETRAINED
artifact supplied by the operator — this module only ever loads and runs
inference on it, it never trains one. Drop the .pkl at
Config.ANTISPOOF_MODEL_PATH (default ./models/antispoof_fullmodels.pkl).
"""
import os
import threading

_lock = threading.Lock()
_model_cache = {"model": None, "device": "cpu", "loaded": False}


def load_antispoof_model(app=None):
    """Load the pretrained ensemble once into a module-level singleton.
    Runs on CPU automatically if no GPU is available."""
    path = app.config["ANTISPOOF_MODEL_PATH"] if app else None
    if not path or not os.path.exists(path):
        with _lock:
            _model_cache.update(model=None, loaded=False)
        raise FileNotFoundError(
            f"Anti-spoof model not found at '{path}'. "
            "This is a pretrained artifact — copy your .pkl there; it is never trained by this app."
        )

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(path, map_location=device)
    except Exception:
        import pickle
        with open(path, "rb") as f:
            model = pickle.load(f)

    if isinstance(model, dict):
        for sub_model in model.values():
            if hasattr(sub_model, "eval"):
                sub_model.eval()
            if hasattr(sub_model, "to"):
                sub_model.to(device)
    else:
        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "to"):
            model = model.to(device)

    with _lock:
        _model_cache["model"] = model
        _model_cache["device"] = device
        _model_cache["loaded"] = True
    return _model_cache


def get_model_cache():
    return _model_cache


def _preprocess(face_crop_rgb_160x160):
    """160x160 RGB uint8 array -> normalised (1,3,160,160) float tensor."""
    import numpy as np
    import torch
    from torchvision import transforms

    if face_crop_rgb_160x160.shape[:2] != (160, 160):
        from PIL import Image
        face_crop_rgb_160x160 = np.array(Image.fromarray(face_crop_rgb_160x160).resize((160, 160)))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(face_crop_rgb_160x160)
    return tensor.unsqueeze(0)


def predict_is_real(face_crop_rgb_160x160) -> float:
    """
    Returns the ensemble's probability, in [0, 1], that the given 160x160
    RGB face crop is a REAL (not spoofed) face.

    Contract for the supplied .pkl (matching the spec's "MobileNetV2 +
    ResNet18 ensemble"): it should unpickle to either
      - a dict of named torch.nn.Module sub-models (e.g.
        {"mobilenet": ..., "resnet": ...}), each outputting 2-class logits
        where index 1 = real, index 0 = spoof — probabilities are averaged
        across sub-models, or
      - a single torch.nn.Module with the same 2-class output convention.
    If your actual pickle's structure or class-index convention differs,
    this is the only function that needs adjusting.
    """
    if not _model_cache["loaded"]:
        raise RuntimeError("Anti-spoof model is not loaded. Check ANTISPOOF_MODEL_PATH.")

    import torch

    model = _model_cache["model"]
    device = _model_cache["device"]
    tensor = _preprocess(face_crop_rgb_160x160).to(device)

    with torch.no_grad():
        if isinstance(model, dict):
            probs = []
            for sub_model in model.values():
                logits = sub_model(tensor)
                prob_real = torch.softmax(logits, dim=1)[0, 1].item()
                probs.append(prob_real)
            return sum(probs) / len(probs) if probs else 0.0
        else:
            logits = model(tensor)
            return torch.softmax(logits, dim=1)[0, 1].item()
