"""
A structured, in-platform replacement for WhatsApp-based attendance fix
requests. All server-side guards from the spec are enforced here, not
just in the UI: 30-day window, 2-per-week cap, and no correction for a
date already marked present.
"""
from datetime import datetime, timedelta

from bson import ObjectId
from bson.errors import InvalidId

from app.extensions import db
from . import attendance_service

MAX_PER_WEEK = 2
MAX_DAYS_AGO = 30
REASON_CATEGORIES = ["Medical", "Technical Issue", "Proxy Error", "Other"]


def _day_bounds(d):
    start = d.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)


def submit_correction(student_id, requested_date, reason_category, explanation, proof_filename=None):
    if len(explanation.strip()) < 20 or len(explanation.strip()) > 500:
        raise ValueError("Explanation must be between 20 and 500 characters.")

    days_ago = (datetime.utcnow().date() - requested_date.date()).days
    if days_ago > MAX_DAYS_AGO:
        raise ValueError(f"Cannot request a correction for a date more than {MAX_DAYS_AGO} days ago.")
    if days_ago < 0:
        raise ValueError("Cannot request a correction for a future date.")

    day_start, day_end = _day_bounds(requested_date)
    already_present = db.attendance_logs.find_one({
        "student_id": student_id, "timestamp": {"$gte": day_start, "$lt": day_end},
    })
    if already_present:
        raise ValueError("This date is already marked present - no correction needed.")

    week_ago = datetime.utcnow() - timedelta(days=7)
    recent_count = db.attendance_corrections.count_documents({
        "student_id": student_id, "submitted_at": {"$gte": week_ago},
    })
    if recent_count >= MAX_PER_WEEK:
        raise ValueError(f"You've reached the limit of {MAX_PER_WEEK} correction requests per week.")

    doc = {
        "student_id": student_id,
        "teacher_id": None,
        "requested_date": day_start,
        "reason_category": reason_category,
        "explanation": explanation.strip(),
        "proof_filename": proof_filename,
        "status": "Pending",
        "teacher_comment": "",
        "action_taken_at": None,
        "messages": [],
        "seen_by_student": True,
        "submitted_at": datetime.utcnow(),
    }
    result = db.attendance_corrections.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


def get(correction_id):
    try:
        oid = ObjectId(correction_id)
    except (InvalidId, TypeError):
        return None
    return db.attendance_corrections.find_one({"_id": oid})


def list_for_student(student_id):
    return list(db.attendance_corrections.find({"student_id": student_id}).sort("submitted_at", -1))


def list_queue(status=None, student_id=None, reason_category=None, date_from=None, date_to=None):
    query = {}
    if status:
        query["status"] = status
    if student_id:
        query["student_id"] = student_id
    if reason_category:
        query["reason_category"] = reason_category
    if date_from or date_to:
        date_query = {}
        if date_from:
            date_query["$gte"] = date_from
        if date_to:
            date_query["$lte"] = date_to
        query["requested_date"] = date_query
    return list(db.attendance_corrections.find(query).sort("submitted_at", -1))


def count_pending():
    return db.attendance_corrections.count_documents({"status": "Pending"})


def approve(correction_id, teacher_id):
    doc = get(correction_id)
    if not doc or doc["status"] != "Pending":
        return False

    student = db.students.find_one({"student_id": doc["student_id"]})
    name = student["name"] if student else doc["student_id"]
    attendance_service.mark_attendance(doc["student_id"], name, session_name="Manual correction")

    db.attendance_corrections.update_one(
        {"_id": doc["_id"]},
        {"$set": {
            "status": "Approved", "teacher_id": teacher_id,
            "action_taken_at": datetime.utcnow(), "seen_by_student": False,
        }},
    )
    return True


def reject(correction_id, teacher_id, reason):
    if len(reason.strip()) < 10:
        raise ValueError("Rejection reason must be at least 10 characters.")
    doc = get(correction_id)
    if not doc or doc["status"] != "Pending":
        return False
    db.attendance_corrections.update_one(
        {"_id": doc["_id"]},
        {"$set": {
            "status": "Rejected", "teacher_id": teacher_id, "teacher_comment": reason.strip(),
            "action_taken_at": datetime.utcnow(), "seen_by_student": False,
        }},
    )
    return True


def bulk_approve(correction_ids, teacher_id):
    return sum(1 for cid in correction_ids if approve(cid, teacher_id))


def bulk_reject(correction_ids, teacher_id, reason):
    return sum(1 for cid in correction_ids if reject(cid, teacher_id, reason))


def add_message(correction_id, sender, content):
    doc = get(correction_id)
    if not doc:
        return False
    db.attendance_corrections.update_one(
        {"_id": doc["_id"]},
        {"$push": {"messages": {"from": sender, "content": content, "timestamp": datetime.utcnow()}},
         "$set": {"seen_by_student": sender != "student"}},
    )
    return True


def get_unseen_for_student(student_id):
    return list(db.attendance_corrections.find({
        "student_id": student_id, "status": {"$in": ["Approved", "Rejected"]}, "seen_by_student": False,
    }))


def mark_seen(correction_id, student_id):
    db.attendance_corrections.update_one(
        {"_id": ObjectId(correction_id), "student_id": student_id}, {"$set": {"seen_by_student": True}}
    )
