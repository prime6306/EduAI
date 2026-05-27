"""
modules/attendance/router.py – Attendance endpoints (image upload + logs).
"""
import os
import tempfile

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.auth.router import get_current_user
from backend.db.mongodb import get_async_db
from backend.modules.attendance.face_engine import process_image
from backend.modules.mlflow_tracker import log_run

router = APIRouter(prefix="/attendance", tags=["Attendance"])


@router.post("/mark")
async def mark_attendance(
    file: UploadFile = File(...),
    user=Depends(get_current_user),
):
    """Upload group/individual photo to mark attendance."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(400, "Only JPG/PNG images allowed")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = process_image(tmp_path)
        marked = sum(1 for s in result.get("students", []) if s.get("status") == "marked")
        log_run("attendance",
                params={"uploaded_by": user.get("email")},
                metrics={"faces_detected": result.get("faces_detected", 0), "marked": marked})
        return result
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        os.unlink(tmp_path)


@router.get("/logs")
async def get_attendance_logs(limit: int = 50, user=Depends(get_current_user)):
    db = get_async_db()
    query = {} if user.get("role") == "teacher" else {"student_id": user.get("student_id")}
    logs = await db.attendance_logs.find(query).sort("timestamp", -1).limit(limit).to_list(limit)
    for log in logs:
        log["_id"] = str(log["_id"])
        if "timestamp" in log:
            log["timestamp"] = log["timestamp"].isoformat()
    return logs


@router.get("/students")
async def list_students(user=Depends(get_current_user)):
    if user.get("role") not in ("teacher", "admin"):
        raise HTTPException(403, "Teachers only")
    db = get_async_db()
    students = await db.students.find({}, {"_id": 0, "student_id": 1, "name": 1,
                                           "branch": 1, "year": 1, "total_attendance": 1}).to_list(200)
    return students
