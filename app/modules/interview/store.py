"""
Mongo data-access for Interview Prep. One document per session with an
embedded `turns[]` array — a session is always read/written as a whole
unit (the analysis page, the take page, and the report page each need
the entire session), so embedding avoids a join that every other module
in this app that has similar shape (co_po_results, test_attempts) also
avoids for the same reason.
"""
from datetime import datetime

from bson import ObjectId
from bson.errors import InvalidId

from app.extensions import db


def ensure_indexes() -> None:
    db.interview_sessions.create_index([("user_id", 1), ("created_at", -1)], name="user_created")
    db.interview_sessions.create_index([("status", 1)], name="status")


def _oid(id_str):
    try:
        return ObjectId(id_str)
    except (InvalidId, TypeError):
        return None


def create_session(user_id: str, jd_text: str, resume_text: str,
                    jd_filename: str = "", resume_filename: str = "") -> dict:
    doc = {
        "user_id": user_id,
        "jd_text": jd_text,
        "resume_text": resume_text,
        "jd_filename": jd_filename,
        "resume_filename": resume_filename,
        "jd_analysis": {},
        "resume_analysis": {},
        "job_fit": {},
        "interview_level": 1,
        "questions_this_level": 0,
        "last_answer_quality": None,
        "last_answer_text": "",
        "status": "analyzing",  # analyzing -> interviewing -> completed
        "turns": [],
        "report": None,
        "teacher_feedback": None,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "completed_at": None,
    }
    result = db.interview_sessions.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


def get_session(session_id: str):
    oid = _oid(session_id)
    if not oid:
        return None
    return db.interview_sessions.find_one({"_id": oid})


def get_owned_session(session_id: str, user_id: str):
    s = get_session(session_id)
    if s and s.get("user_id") == user_id:
        return s
    return None


def save_analysis(session_id: str, jd_analysis: dict, resume_analysis: dict, job_fit: dict) -> None:
    db.interview_sessions.update_one(
        {"_id": _oid(session_id)},
        {"$set": {
            "jd_analysis": jd_analysis,
            "resume_analysis": resume_analysis,
            "job_fit": job_fit,
            "status": "interviewing",
            "updated_at": datetime.utcnow(),
        }},
    )


def add_question(session_id: str, level: int, question: str, targets_competency: str, interviewer: str) -> str:
    turn_id = ObjectId()
    turn = {
        "turn_id": turn_id,
        "level": level,
        "interviewer": interviewer,
        "question": question,
        "targets_competency": targets_competency,
        "answer": None,
        "quality_score": None,
        "competency_tag": None,
        "what_was_good": None,
        "what_could_be_better": None,
        "ideal_direction": None,
        "asked_at": datetime.utcnow(),
        "answered_at": None,
    }
    db.interview_sessions.update_one(
        {"_id": _oid(session_id)},
        {"$push": {"turns": turn}, "$set": {"updated_at": datetime.utcnow()}},
    )
    return str(turn_id)


def save_answer_evaluation(session_id: str, turn_id: str, answer: str, evaluation: dict) -> None:
    db.interview_sessions.update_one(
        {"_id": _oid(session_id), "turns.turn_id": _oid(turn_id)},
        {"$set": {
            "turns.$.answer": answer,
            "turns.$.quality_score": evaluation.get("quality_score"),
            "turns.$.competency_tag": evaluation.get("competency_tag"),
            "turns.$.what_was_good": evaluation.get("what_was_good"),
            "turns.$.what_could_be_better": evaluation.get("what_could_be_better"),
            "turns.$.ideal_direction": evaluation.get("ideal_direction"),
            "turns.$.answered_at": datetime.utcnow(),
            "last_answer_quality": evaluation.get("quality_score"),
            "last_answer_text": answer,
            "updated_at": datetime.utcnow(),
        }},
    )


def update_interview_state(session_id: str, level: int, questions_this_level: int) -> None:
    db.interview_sessions.update_one(
        {"_id": _oid(session_id)},
        {"$set": {
            "interview_level": level,
            "questions_this_level": questions_this_level,
            "updated_at": datetime.utcnow(),
        }},
    )


def mark_completed(session_id: str, report: dict) -> None:
    db.interview_sessions.update_one(
        {"_id": _oid(session_id)},
        {"$set": {
            "status": "completed",
            "report": report,
            "completed_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }},
    )


def list_sessions_for_user(user_id: str) -> list:
    return list(db.interview_sessions.find({"user_id": user_id}).sort("created_at", -1))


def list_all_sessions() -> list:
    return list(db.interview_sessions.find({}).sort("created_at", -1))


def save_teacher_feedback(session_id: str, teacher_id: str, teacher_name: str, comment: str) -> None:
    db.interview_sessions.update_one(
        {"_id": _oid(session_id)},
        {"$set": {
            "teacher_feedback": {
                "teacher_id": teacher_id,
                "teacher_name": teacher_name,
                "comment": comment,
                "given_at": datetime.utcnow(),
            },
            "updated_at": datetime.utcnow(),
        }},
    )
