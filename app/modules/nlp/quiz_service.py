"""
Quiz generation, server-authoritative grading (never trust the client for
correctness), and history. Correct answers are stored in `quiz_questions`
and are never serialised into the quiz-taking page — only revealed on the
results page after grading.
"""
from datetime import datetime

from bson import ObjectId
from bson.errors import InvalidId

from app.extensions import db
from . import prompts
from .llm_client import chat_json

GRADE_THRESHOLDS = [(90, "A+"), (80, "A"), (65, "B"), (50, "C")]


def grade_for_percent(percent: float) -> str:
    for threshold, grade in GRADE_THRESHOLDS:
        if percent >= threshold:
            return grade
    return "F"


def generate_quiz(user_id: str, topic: str, subject: str, branch: str, year: str,
                   n_questions: int, timed: bool) -> dict:
    n_questions = max(5, min(20, n_questions))
    raw = chat_json(prompts.quiz_generation_prompt(topic, subject, branch, year, n_questions))
    questions = raw.get("questions", [])[:n_questions]

    cleaned = []
    for q in questions:
        options = q.get("options", [])
        if len(options) != 4:
            continue
        correct_index = q.get("correct_index")
        if not isinstance(correct_index, int) or not (0 <= correct_index < 4):
            continue
        cleaned.append({
            "question": q.get("question", "").strip(),
            "options": options,
            "correct_index": correct_index,
            "explanation": q.get("explanation", ""),
        })

    if not cleaned:
        raise ValueError("The AI didn't return any usable questions — try rephrasing the topic.")

    doc = {
        "user_id": user_id,
        "topic": topic,
        "subject": subject,
        "branch": branch,
        "year": year,
        "questions": cleaned,
        "num_questions": len(cleaned),
        "timed": timed,
        "created_at": datetime.utcnow(),
    }
    result = db.quiz_questions.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


def get_quiz(quiz_id: str, user_id: str) -> dict | None:
    try:
        oid = ObjectId(quiz_id)
    except (InvalidId, TypeError):
        return None
    return db.quiz_questions.find_one({"_id": oid, "user_id": user_id})


def questions_for_taking(quiz: dict) -> list[dict]:
    """Strips correct_index/explanation before sending to the client."""
    return [
        {"index": i, "question": q["question"], "options": q["options"]}
        for i, q in enumerate(quiz["questions"])
    ]


def submit_quiz(quiz_id: str, user_id: str, answers: dict, time_taken_sec: int) -> dict:
    """`answers` maps str(question_index) -> selected_option_index (or None if skipped)."""
    quiz = get_quiz(quiz_id, user_id)
    if not quiz:
        raise ValueError("Quiz not found.")

    review = []
    correct_count = 0
    for i, q in enumerate(quiz["questions"]):
        selected = answers.get(str(i))
        is_correct = selected is not None and int(selected) == q["correct_index"]
        if is_correct:
            correct_count += 1
        review.append({
            "question": q["question"],
            "options": q["options"],
            "correct_index": q["correct_index"],
            "selected_index": selected if selected is None else int(selected),
            "is_correct": is_correct,
            "explanation": q["explanation"],
        })

    total = len(quiz["questions"])
    score_percent = round((correct_count / total) * 100, 1) if total else 0.0
    grade = grade_for_percent(score_percent)

    result_doc = {
        "user_id": user_id,
        "quiz_id": str(quiz["_id"]),
        "topic": quiz["topic"],
        "subject": quiz["subject"],
        "review": review,
        "score_percent": score_percent,
        "correct_count": correct_count,
        "total": total,
        "grade": grade,
        "time_taken_sec": time_taken_sec,
        "timestamp": datetime.utcnow(),
    }
    result = db.quiz_results.insert_one(result_doc)
    result_doc["_id"] = result.inserted_id

    _log_to_mlflow(quiz["topic"], score_percent, grade, time_taken_sec)
    return result_doc


def _log_to_mlflow(topic: str, score_percent: float, grade: str, time_taken_sec: int) -> None:
    try:
        from flask import current_app
        import mlflow
        mlflow.set_tracking_uri(current_app.config["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(current_app.config["MLFLOW_EXPERIMENT"])
        with mlflow.start_run(run_name="quiz_attempt"):
            mlflow.log_param("topic", topic)
            mlflow.log_param("grade", grade)
            mlflow.log_metric("score_percent", score_percent)
            mlflow.log_metric("time_taken_sec", time_taken_sec)
    except Exception:  # noqa: BLE001
        pass


def get_result(result_id: str, user_id: str) -> dict | None:
    try:
        oid = ObjectId(result_id)
    except (InvalidId, TypeError):
        return None
    return db.quiz_results.find_one({"_id": oid, "user_id": user_id})


def list_history(user_id: str) -> list[dict]:
    return list(db.quiz_results.find({"user_id": user_id}).sort("timestamp", -1))
