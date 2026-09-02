"""
Registering students and managing their face encodings. `ENcodedFile.p`
holds two parallel lists — encodings and student_ids — kept in sync by
every function here; the in-memory cache (face_engine) is always
invalidated after a write so matching immediately reflects the change.
"""
import csv
import io
import os
import pickle
from datetime import datetime

from app.extensions import db
from . import face_engine

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def add_student(student_id: str, name: str, branch: str, year: str, email: str = "") -> dict:
    if db.students.find_one({"student_id": student_id}):
        raise ValueError(f"Student ID '{student_id}' is already registered.")
    doc = {
        "student_id": student_id, "name": name, "branch": branch, "year": year,
        "email": email, "total_attendance": 0, "last_attendance_date": None,
        "face_enrolled": False, "created_at": datetime.utcnow(),
    }
    db.students.insert_one(doc)
    return doc


def _load_encodings_file(app) -> dict:
    path = app.config["FACE_ENCODINGS_PATH"]
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {"encodings": [], "student_ids": []}


def _save_encodings_file(app, data: dict) -> None:
    path = app.config["FACE_ENCODINGS_PATH"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def enroll_face(app, student_id: str, image_filepath: str) -> None:
    """Detects a face in the given photo, saves it to models/Images/, and
    (re)writes the student's encoding into ENcodedFile.p."""
    try:
        import face_recognition
    except ImportError as exc:
        raise ValueError(
            "Face recognition is not available on this system. "
            "Ensure dlib and its Visual C++ dependencies are installed."
        ) from exc

    image = face_recognition.load_image_file(image_filepath)
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        raise ValueError("No face detected in that photo. Try a clearer, front-facing shot.")
    encoding = encodings[0]

    os.makedirs(app.config["STUDENT_IMAGES_DIR"], exist_ok=True)
    dest_path = os.path.join(app.config["STUDENT_IMAGES_DIR"], f"{student_id}.jpg")
    from PIL import Image
    Image.open(image_filepath).convert("RGB").save(dest_path, "JPEG")

    data = _load_encodings_file(app)
    if student_id in data["student_ids"]:
        idx = data["student_ids"].index(student_id)
        data["encodings"][idx] = encoding
    else:
        data["encodings"].append(encoding)
        data["student_ids"].append(student_id)
    _save_encodings_file(app, data)

    db.students.update_one({"student_id": student_id}, {"$set": {"face_enrolled": True}})
    face_engine.invalidate_cache(app)


def reencode_all(app) -> dict:
    """Rebuilds ENcodedFile.p from every photo in models/Images/."""
    try:
        import face_recognition
    except ImportError as exc:
        raise ValueError(
            "Face recognition is not available on this system."
        ) from exc

    images_dir = app.config["STUDENT_IMAGES_DIR"]
    os.makedirs(images_dir, exist_ok=True)

    encodings, student_ids = [], []
    succeeded, failed = [], []

    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith(IMAGE_EXTENSIONS):
            continue
        student_id = os.path.splitext(fname)[0]
        try:
            image = face_recognition.load_image_file(os.path.join(images_dir, fname))
            found = face_recognition.face_encodings(image)
            if not found:
                failed.append(student_id)
                continue
            encodings.append(found[0])
            student_ids.append(student_id)
            succeeded.append(student_id)
        except Exception:  # noqa: BLE001
            failed.append(student_id)

    _save_encodings_file(app, {"encodings": encodings, "student_ids": student_ids})
    db.students.update_many({"student_id": {"$in": succeeded}}, {"$set": {"face_enrolled": True}})
    db.students.update_many({"student_id": {"$in": failed}}, {"$set": {"face_enrolled": False}})
    face_engine.invalidate_cache(app)

    return {"succeeded": succeeded, "failed": failed}


def delete_student(app, student_id: str) -> bool:
    result = db.students.delete_one({"student_id": student_id})
    if result.deleted_count == 0:
        return False

    data = _load_encodings_file(app)
    if student_id in data["student_ids"]:
        idx = data["student_ids"].index(student_id)
        data["student_ids"].pop(idx)
        data["encodings"].pop(idx)
        _save_encodings_file(app, data)

    image_path = os.path.join(app.config["STUDENT_IMAGES_DIR"], f"{student_id}.jpg")
    if os.path.exists(image_path):
        os.remove(image_path)

    db.attendance_logs.delete_many({"student_id": student_id})
    face_engine.invalidate_cache(app)
    return True


def list_students(search: str = "") -> list[dict]:
    query = {}
    if search:
        query["$or"] = [
            {"student_id": {"$regex": search, "$options": "i"}},
            {"name": {"$regex": search, "$options": "i"}},
        ]
    return list(db.students.find(query).sort("student_id", 1))


def export_attendance_csv() -> str:
    students = list_students()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Student ID", "Name", "Branch", "Year", "Total Attendance", "Last Attendance", "Face Enrolled"])
    for s in students:
        writer.writerow([
            s.get("student_id"), s.get("name"), s.get("branch"), s.get("year"),
            s.get("total_attendance", 0),
            s.get("last_attendance_date").strftime("%Y-%m-%d %H:%M") if s.get("last_attendance_date") else "",
            "Yes" if s.get("face_enrolled") else "No",
        ])
    return buf.getvalue()
