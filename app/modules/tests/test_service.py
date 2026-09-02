"""
Test builder backend: drafts start empty and accumulate questions from any
mix of AI generation, the question bank, or manual entry, each question
tagged with a stable uuid so the builder/preview/attempt/grading code can
all reference questions by id rather than fragile list position.
"""
import uuid
from datetime import datetime

from bson import ObjectId
from bson.errors import InvalidId

from app.extensions import db
from . import prompts
from app.modules.nlp.llm_client import chat_json

QUESTION_TYPES = ["mcq", "tf", "short", "long"]


def create_draft(teacher_id, title, subject, instructions, time_limit_minutes):
    doc = {
        "teacher_id": teacher_id,
        "title": title,
        "subject": subject,
        "instructions": instructions,
        "time_limit_minutes": time_limit_minutes,
        "available_from": None,
        "available_until": None,
        "assigned_to": [],
        "questions": [],
        "shuffle_questions": False,
        "shuffle_options": False,
        "status": "draft",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    result = db.tests.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


def get(test_id, teacher_id=None):
    try:
        oid = ObjectId(test_id)
    except (InvalidId, TypeError):
        return None
    query = {"_id": oid}
    if teacher_id:
        query["teacher_id"] = teacher_id
    return db.tests.find_one(query)


def list_for_teacher(teacher_id):
    return list(db.tests.find({"teacher_id": teacher_id}).sort("created_at", -1))


def _is_available_now(doc):
    now = datetime.utcnow()
    if doc.get("available_from") and doc["available_from"] > now:
        return False
    if doc.get("available_until") and doc["available_until"] < now:
        return False
    return True


def list_for_student(student_id):
    docs = list(db.tests.find({"status": "published"}).sort("created_at", -1))
    out = []
    for d in docs:
        assigned = d.get("assigned_to") or []
        if assigned and student_id not in assigned:
            continue
        d["is_available"] = _is_available_now(d)
        d["is_upcoming"] = bool(d.get("available_from") and d["available_from"] > datetime.utcnow())
        d["is_expired"] = bool(d.get("available_until") and d["available_until"] < datetime.utcnow())
        out.append(d)
    return out


def update_metadata(test_id, teacher_id, **fields):
    doc = get(test_id, teacher_id)
    if not doc:
        return False
    fields["updated_at"] = datetime.utcnow()
    db.tests.update_one({"_id": doc["_id"]}, {"$set": fields})
    return True


def add_question(test_id, teacher_id, question):
    doc = get(test_id, teacher_id)
    if not doc:
        return None
    question = dict(question)
    question["id"] = uuid.uuid4().hex
    db.tests.update_one(
        {"_id": doc["_id"]},
        {"$push": {"questions": question}, "$set": {"updated_at": datetime.utcnow()}},
    )
    return question


def add_questions_bulk(test_id, teacher_id, questions):
    added = []
    for q in questions:
        result = add_question(test_id, teacher_id, q)
        if result:
            added.append(result)
    return added


def update_question(test_id, teacher_id, question_id, fields):
    doc = get(test_id, teacher_id)
    if not doc:
        return False
    for q in doc["questions"]:
        if q["id"] == question_id:
            q.update(fields)
            db.tests.update_one({"_id": doc["_id"]}, {"$set": {"questions": doc["questions"], "updated_at": datetime.utcnow()}})
            return True
    return False


def delete_question(test_id, teacher_id, question_id):
    doc = get(test_id, teacher_id)
    if not doc:
        return False
    new_questions = [q for q in doc["questions"] if q["id"] != question_id]
    if len(new_questions) == len(doc["questions"]):
        return False
    db.tests.update_one({"_id": doc["_id"]}, {"$set": {"questions": new_questions, "updated_at": datetime.utcnow()}})
    return True


def reorder_questions(test_id, teacher_id, ordered_ids):
    doc = get(test_id, teacher_id)
    if not doc:
        return False
    by_id = {q["id"]: q for q in doc["questions"]}
    new_order = [by_id[qid] for qid in ordered_ids if qid in by_id]
    if len(new_order) != len(doc["questions"]):
        return False
    db.tests.update_one({"_id": doc["_id"]}, {"$set": {"questions": new_order, "updated_at": datetime.utcnow()}})
    return True


def publish(test_id, teacher_id):
    doc = get(test_id, teacher_id)
    if not doc:
        return False, "Test not found."
    if not doc["questions"]:
        return False, "Add at least one question before publishing."
    db.tests.update_one(
        {"_id": doc["_id"]}, {"$set": {"status": "published", "updated_at": datetime.utcnow()}}
    )
    return True, None


def delete_test(test_id, teacher_id):
    doc = get(test_id, teacher_id)
    if not doc:
        return False
    db.tests.delete_one({"_id": doc["_id"]})
    db.test_attempts.delete_many({"test_id": test_id})
    return True


def questions_for_taking(questions):
    """Strips correct_answer before sending to the client during an attempt."""
    return [{k: v for k, v in q.items() if k != "correct_answer"} for q in questions]


def generate_ai_questions(subject, topic, n, q_type):
    """Returns candidate questions (not yet saved) for the teacher to
    accept/reject/edit individually."""
    n = max(1, min(n, 20))
    messages = prompts.generate_test_questions_prompt(subject, topic, n, q_type)
    raw = chat_json(messages)
    candidates = raw.get("questions", [])[:n]

    cleaned = []
    for q in candidates:
        if not q.get("text") or not q.get("correct_answer"):
            continue
        item = {
            "text": q["text"], "type": q.get("type", q_type),
            "correct_answer": q["correct_answer"], "marks": q.get("marks", 1 if q_type in ("mcq", "tf") else 5),
            "explanation": q.get("explanation", ""),
        }
        if q_type in ("mcq", "tf"):
            item["options"] = q.get("options", [])
        cleaned.append(item)
    return cleaned
