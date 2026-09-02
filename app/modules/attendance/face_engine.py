"""
Face encoding cache + matching. `face_recognition` (dlib) is a heavy native
dependency — this module isolates it so the rest of the app can import
safely even where dlib isn't installed yet, and so the encodings file is
loaded once into a module-level cache per the spec's non-functional
requirements (loaded at startup, invalidated on enroll/re-encode).
"""
import os
import pickle
import threading

_lock = threading.Lock()
_cache = {"encodings": [], "student_ids": [], "loaded": False}


def load_face_encodings(app=None):
    """Load ENcodedFile.p into the module-level cache. Safe to call multiple
    times; call again after enroll/re-encode to invalidate the cache."""
    path = app.config["FACE_ENCODINGS_PATH"] if app else None
    if not path or not os.path.exists(path):
        with _lock:
            _cache.update(encodings=[], student_ids=[], loaded=False)
        raise FileNotFoundError(
            f"Face encodings file not found at '{path}'. "
            "Enrol students via /students/add to create it."
        )
    with open(path, "rb") as f:
        data = pickle.load(f)
    with _lock:
        _cache["encodings"] = data.get("encodings", [])
        _cache["student_ids"] = data.get("student_ids", [])
        _cache["loaded"] = True
    return _cache


def get_cache():
    return _cache


def invalidate_cache(app):
    """Call after /api/students/enroll-face or /api/students/reencode."""
    return load_face_encodings(app)


def detect_faces(image_array):
    """
    image_array: RGB numpy array (e.g. from face_recognition.load_image_file).
    Returns a list of {location, encoding, crop_160} for every face found —
    location is (top, right, bottom, left); crop_160 is a 160x160 RGB crop
    ready for the anti-spoof model.
    """
    import face_recognition
    import numpy as np
    from PIL import Image

    locations = face_recognition.face_locations(image_array)
    encodings = face_recognition.face_encodings(image_array, locations)

    faces = []
    for loc, enc in zip(locations, encodings):
        top, right, bottom, left = loc
        crop = image_array[max(top, 0):bottom, max(left, 0):right]
        if crop.size == 0:
            continue
        crop_160 = np.array(Image.fromarray(crop).resize((160, 160)))
        faces.append({"location": loc, "encoding": enc, "crop_160": crop_160})
    return faces


def match_encoding(encoding, tolerance: float = 0.6):
    """Compares a face encoding against the cached roster. Returns
    (student_id, distance) on a match within tolerance, else (None, None)."""
    import face_recognition
    import numpy as np

    cache = get_cache()
    if not cache["loaded"] or not cache["encodings"]:
        return None, None

    distances = face_recognition.face_distance(cache["encodings"], encoding)
    best_idx = int(np.argmin(distances))
    best_distance = float(distances[best_idx])
    if best_distance <= tolerance:
        return cache["student_ids"][best_idx], best_distance
    return None, None
