"""
Attempt lifecycle: start -> autosave (every 30s from the client) -> submit
(auto-grades MCQ/True-False immediately; Short/Long go to the teacher
review queue) -> grade (teacher) -> results.
"""
import csv
import io
from datetime import datetime

from bson import ObjectId
from bson.errors import InvalidId

from app.extensions import db
from . import test_service


def get_or_create_attempt(test_id, student_id):
    existing = db.test_attempts.find_one({
        "test_id": test_id, "student_id": student_id, "submitted_at": None,
    })
    if existing:
        return existing

    completed = db.test_attempts.find_one({"test_id": test_id, "student_id": student_id})
    if completed:
        return completed

    doc = {
        "test_id": test_id,
        "student_id": student_id,
        "answers": [],
        "total_marks": 0,
        "score": 0,
        "submitted_at": None,
        "started_at": datetime.utcnow(),
        "time_taken_sec": 0,
        "grading_status": "pending",
        "last_autosave": None,
    }
    result = db.test_attempts.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


def get_attempt(attempt_id, student_id=None):
    try:
        oid = ObjectId(attempt_id)
    except (InvalidId, TypeError):
        return None
    query = {"_id": oid}
    if student_id:
        query["student_id"] = student_id
    return db.test_attempts.find_one(query)


def autosave(attempt_id, student_id, answers):
    attempt = get_attempt(attempt_id, student_id)
    if not attempt or attempt.get("submitted_at"):
        return False
    db.test_attempts.update_one(
        {"_id": attempt["_id"]},
        {"$set": {"answers": answers, "last_autosave": datetime.utcnow()}},
    )
    return True


def _grade_objective(question, response):
    if question["type"] not in ("mcq", "tf"):
        return None
    correct = (question.get("correct_answer") or "").strip().lower()
    given = (response or "").strip().lower()
    return question["marks"] if given == correct else 0


def submit(attempt_id, student_id, answers, time_taken_sec):
    attempt = get_attempt(attempt_id, student_id)
    if not attempt:
        raise ValueError("Attempt not found.")
    if attempt.get("submitted_at"):
        raise ValueError("This test has already been submitted.")

    test = test_service.get(attempt["test_id"])
    if not test:
        raise ValueError("Test not found.")

    questions_by_id = {q["id"]: q for q in test["questions"]}
    graded_answers = []
    has_descriptive = False

    for qid, question in questions_by_id.items():
        response = answers.get(qid, "")
        marks_awarded = _grade_objective(question, response)
        if marks_awarded is None:
            has_descriptive = True
        graded_answers.append({
            "question_id": qid, "response": response,
            "marks_awarded": marks_awarded, "comment": "",
        })

    total_marks = sum(q["marks"] for q in test["questions"])
    auto_score = sum(a["marks_awarded"] for a in graded_answers if a["marks_awarded"] is not None)
    grading_status = "pending" if has_descriptive else "complete"

    db.test_attempts.update_one(
        {"_id": attempt["_id"]},
        {"$set": {
            "answers": graded_answers, "total_marks": total_marks, "score": auto_score,
            "submitted_at": datetime.utcnow(), "time_taken_sec": time_taken_sec,
            "grading_status": grading_status,
        }},
    )
    return get_attempt(str(attempt["_id"]))


def grade_answer(test_id, teacher_id, attempt_id, question_id, marks_awarded, comment):
    test = test_service.get(test_id, teacher_id)
    if not test:
        return False

    attempt = get_attempt(attempt_id)
    if not attempt or attempt["test_id"] != test_id:
        return False

    answers = attempt["answers"]
    found = False
    for a in answers:
        if a["question_id"] == question_id:
            a["marks_awarded"] = marks_awarded
            a["comment"] = comment
            found = True
            break
    if not found:
        return False

    new_score = sum(a["marks_awarded"] for a in answers if a["marks_awarded"] is not None)
    all_graded = all(a["marks_awarded"] is not None for a in answers)
    grading_status = "complete" if all_graded else "partial"

    db.test_attempts.update_one(
        {"_id": attempt["_id"]},
        {"$set": {"answers": answers, "score": new_score, "grading_status": grading_status}},
    )
    return True


def list_attempts_for_test(test_id):
    return list(db.test_attempts.find({"test_id": test_id, "submitted_at": {"$ne": None}}).sort("submitted_at", -1))


def get_review_queue(test_id):
    return list(db.test_attempts.find({
        "test_id": test_id, "submitted_at": {"$ne": None}, "grading_status": {"$ne": "complete"},
    }))


def get_class_stats(test_id):
    attempts = [a for a in list_attempts_for_test(test_id) if a["grading_status"] == "complete"]
    total_submissions = db.test_attempts.count_documents({"test_id": test_id, "submitted_at": {"$ne": None}})
    if not attempts:
        return {"average": 0, "highest": 0, "lowest": 0, "graded_count": 0, "total_submissions": total_submissions}
    scores = [a["score"] for a in attempts]
    return {
        "average": round(sum(scores) / len(scores), 1),
        "highest": max(scores), "lowest": min(scores),
        "graded_count": len(attempts), "total_submissions": total_submissions,
    }


def export_results_csv(test_id, teacher_id):
    test = test_service.get(test_id, teacher_id)
    if not test:
        return ""
    attempts = list_attempts_for_test(test_id)
    students = {s["student_id"]: s["name"] for s in db.students.find({}, {"student_id": 1, "name": 1})}

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Student ID", "Name", "Score", "Total Marks", "Time Taken (s)", "Status", "Submitted At"])
    for a in attempts:
        writer.writerow([
            a["student_id"], students.get(a["student_id"], ""), a["score"], a["total_marks"],
            a["time_taken_sec"], a["grading_status"],
            a["submitted_at"].strftime("%Y-%m-%d %H:%M") if a.get("submitted_at") else "",
        ])
    return buf.getvalue()
