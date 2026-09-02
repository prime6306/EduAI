"""
Scheduling/expiry status is computed on read (comparing scheduled_for /
expires_at against "now") rather than via a background job - this keeps
the behaviour correct (an announcement becomes visible the moment its
scheduled time passes) without adding APScheduler infrastructure.
"""
from datetime import datetime

from bson import ObjectId
from bson.errors import InvalidId

from app.extensions import db

CATEGORIES = ["Exam", "Assignment", "Holiday", "General", "Urgent"]


def compute_status(doc: dict) -> str:
    now = datetime.utcnow()
    if doc.get("expires_at") and doc["expires_at"] <= now:
        return "Archived"
    if doc.get("scheduled_for") and doc["scheduled_for"] > now:
        return "Scheduled"
    return "Active"


def create_announcement(
    teacher_id, title, body_html, category,
    visibility_type, visible_to_students,
    scheduled_for=None, expires_at=None, attachment_filename=None,
):
    doc = {
        "teacher_id": teacher_id,
        "title": title[:100],
        "body_html": body_html,
        "category": category,
        "attachment_filename": attachment_filename,
        "visibility_type": visibility_type,
        "visible_to_students": visible_to_students,
        "scheduled_for": scheduled_for,
        "expires_at": expires_at,
        "read_by": [],
        "edited": False,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    result = db.announcements.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


def get(announcement_id):
    try:
        oid = ObjectId(announcement_id)
    except (InvalidId, TypeError):
        return None
    return db.announcements.find_one({"_id": oid})


def list_for_teacher(teacher_id):
    docs = list(db.announcements.find({"teacher_id": teacher_id}).sort("created_at", -1))
    for d in docs:
        d["status"] = compute_status(d)
    return docs


def _visible_to(doc, student_id):
    if doc.get("visibility_type") == "all":
        return True
    return student_id in (doc.get("visible_to_students") or [])


def list_for_student(student_id, category=None):
    query = {}
    if category and category != "All":
        query["category"] = category
    docs = list(db.announcements.find(query).sort("created_at", -1))
    out = []
    for d in docs:
        status = compute_status(d)
        if status in ("Archived", "Scheduled"):
            continue
        if not _visible_to(d, student_id):
            continue
        d["status"] = status
        d["is_read"] = any(r["student_id"] == student_id for r in d.get("read_by", []))
        out.append(d)
    return out


def get_urgent_unread(student_id):
    return [d for d in list_for_student(student_id, category="Urgent") if not d["is_read"]]


def unread_count(student_id):
    return len(get_urgent_unread(student_id)) + sum(
        1 for d in list_for_student(student_id) if not d["is_read"] and d["category"] != "Urgent"
    )


def update_announcement(announcement_id, teacher_id, **fields):
    doc = get(announcement_id)
    if not doc or doc["teacher_id"] != teacher_id:
        return False
    fields["updated_at"] = datetime.utcnow()
    if compute_status(doc) == "Active":
        fields["edited"] = True
    db.announcements.update_one({"_id": doc["_id"]}, {"$set": fields})
    return True


def delete_announcement(announcement_id, teacher_id):
    doc = get(announcement_id)
    if not doc or doc["teacher_id"] != teacher_id:
        return False
    db.announcements.delete_one({"_id": doc["_id"]})
    return True


def mark_read(announcement_id, student_id):
    doc = get(announcement_id)
    if not doc:
        return False
    if any(r["student_id"] == student_id for r in doc.get("read_by", [])):
        return True
    db.announcements.update_one(
        {"_id": doc["_id"]},
        {"$push": {"read_by": {"student_id": student_id, "read_at": datetime.utcnow()}}},
    )
    return True
