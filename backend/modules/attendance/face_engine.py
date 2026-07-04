"""
modules/attendance/face_engine.py
Face recognition + attendance marking.
Firebase REMOVED — all data goes to MongoDB.
Migrated from groupPhoto.py / with_antiSpoofing.py.
"""
import os
import pickle
import time
from datetime import datetime

import cv2
import face_recognition
import numpy as np

from backend.config import get_settings
from backend.db.mongodb import get_sync_db
from backend.modules.attendance.antispoof_engine import is_real_face

_encodings: list | None = None
_student_ids: list | None = None


def _load_encodings():
    global _encodings, _student_ids
    if _encodings is not None:
        return _encodings, _student_ids
    path = get_settings().face_encodings_path
    if not os.path.exists(path):
        print(f"[FaceEngine] WARNING: Encoding file not found at {path}")
        _encodings, _student_ids = [], []
        return [], []
    with open(path, "rb") as f:
        data = pickle.load(f)
    _encodings, _student_ids = data[0], data[1]
    print(f"[FaceEngine] Loaded {len(_student_ids)} face encodings")
    return _encodings, _student_ids


def _get_student(student_id: str) -> dict | None:
    db = get_sync_db()
    return db.students.find_one({"student_id": student_id})


def _update_attendance(student_id: str, student_info: dict) -> dict:
    """Update attendance in MongoDB. Returns updated doc."""
    db = get_sync_db()
    now = datetime.now()
    last_str = student_info.get("last_attendance_date", "2000-01-01 00:00:00")
    try:
        last_dt = datetime.strptime(last_str, "%Y-%m-%d %H:%M:%S")
    except Exception:
        last_dt = datetime(2000, 1, 1)

    time_diff = (now - last_dt).total_seconds()
    if time_diff < 10:
        return {"status": "duplicate", "message": "Already marked recently"}

    new_count = student_info.get("total_attendance", 0) + 1
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    db.students.update_one(
        {"student_id": student_id},
        {"$set": {"total_attendance": new_count, "last_attendance_date": now_str}},
        upsert=False,
    )
    # Log to attendance_logs collection
    db.attendance_logs.insert_one({
        "student_id": student_id,
        "name": student_info.get("name", student_id),
        "timestamp": now,
        "total_attendance": new_count,
    })
    return {"status": "marked", "total_attendance": new_count, "timestamp": now_str}


def process_image(image_path: str) -> dict:
    """
    Full pipeline:
    1. Read image
    2. Detect faces
    3. Match encodings
    4. Anti-spoofing check
    5. Update MongoDB attendance
    """
    t_total = time.perf_counter()
    encodings, student_ids = _load_encodings()

    if not os.path.exists(image_path):
        return {"status": "error", "message": "Image not found"}

    img = cv2.imread(image_path)
    if img is None:
        return {"status": "error", "message": "Failed to read image"}

    img_small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    face_locs = face_recognition.face_locations(img_rgb)
    face_encs = face_recognition.face_encodings(img_rgb, face_locs)

    if not face_encs:
        return {"status": "error", "message": "No faces detected", "timings": {}}

    results = []
    per_face = []

    for enc, loc in zip(face_encs, face_locs):
        t_face = time.perf_counter()

        if not encodings:
            results.append({"status": "no_encodings", "message": "No face encodings loaded"})
            continue

        t_recog = time.perf_counter()
        matches = face_recognition.compare_faces(encodings, enc)
        distances = face_recognition.face_distance(encodings, enc)
        best_idx = int(np.argmin(distances))
        t_recog_end = time.perf_counter()

        if not matches[best_idx]:
            results.append({"status": "no_match"})
            continue

        student_id = student_ids[best_idx]

        # Crop for anti-spoof
        top, right, bottom, left = [v * 4 for v in loc]
        face_crop = img[top:bottom, left:right]
        real, spoof_time = is_real_face(face_crop)

        if not real:
            results.append({"student_id": student_id, "status": "spoof"})
            per_face.append({"student_id": student_id, "antispoof_time": spoof_time, "result": "spoof"})
            continue

        # DB lookup + update
        student_info = _get_student(student_id)
        if not student_info:
            # Auto-create minimal record
            db = get_sync_db()
            db.students.insert_one({
                "student_id": student_id,
                "name": student_id,
                "branch": "Unknown",
                "year": "Unknown",
                "total_attendance": 0,
                "last_attendance_date": "2000-01-01 00:00:00",
            })
            student_info = _get_student(student_id)

        attendance_result = _update_attendance(student_id, student_info)
        results.append({
            "student_id": student_id,
            "name": student_info.get("name", student_id),
            **attendance_result,
        })
        per_face.append({
            "student_id": student_id,
            "recognition_time": round(t_recog_end - t_recog, 4),
            "antispoof_time": round(spoof_time, 4),
            "face_time": round(time.perf_counter() - t_face, 4),
        })

    return {
        "status": "success",
        "faces_detected": len(face_encs),
        "students": results,
        "timings": {
            "per_face": per_face,
            "total_time": round(time.perf_counter() - t_total, 4),
        },
    }
