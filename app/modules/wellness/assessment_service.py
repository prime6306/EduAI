"""
A short, PHQ-9-inspired self-check-in - not a diagnostic tool. Framed
throughout as a way to notice how the last two weeks have felt, with
clear next steps at every severity level rather than a clinical label.
"""
from app.extensions import db

QUESTIONS = [
    {
        "id": 1,
        "text": "Over the past 2 weeks, how much interest or pleasure have you had in things you usually enjoy?",
        "options": ["Same as usual", "A little less than usual", "Noticeably less", "Almost none"],
    },
    {
        "id": 2,
        "text": "How often have you been feeling down, low, or a bit hopeless?",
        "options": ["Rarely", "A few days", "More than half the days", "Nearly every day"],
    },
    {
        "id": 3,
        "text": "How hard has it been to concentrate on your studies or coursework?",
        "options": ["Not hard at all", "A little harder than usual", "Quite hard", "I can barely focus"],
    },
    {
        "id": 4,
        "text": "How often have you felt low on energy or unusually tired?",
        "options": ["Rarely", "Some days", "Most days", "Nearly every day"],
    },
    {
        "id": 5,
        "text": "How does your recent academic performance feel compared to what you expect of yourself?",
        "options": ["On track", "Slightly below what I'd like", "Noticeably behind", "Far below - I feel like I'm falling behind"],
    },
    {
        "id": 6,
        "text": "Have you had thoughts about dropping out or giving up on your studies?",
        "options": ["Not at all", "A little, in passing", "Fairly often", "It's on my mind a lot"],
    },
    {
        "id": 7,
        "text": "Do you feel you have people - friends, family, faculty - you can turn to right now?",
        "options": ["Yes, plenty", "Some, but could use more", "Very little", "I feel pretty alone in this"],
    },
]

SEVERITY_LEVELS = [
    {"min": 0, "max": 4, "level": "Minimal", "color": "success",
     "message": "Things seem to be feeling okay for you right now. Keep doing what's working."},
    {"min": 5, "max": 9, "level": "Mild", "color": "warning",
     "message": "Sounds like you're facing some real challenges lately. It might help to talk them through with someone."},
    {"min": 10, "max": 14, "level": "Moderate", "color": "warning",
     "message": "It sounds like you're going through a genuinely tough stretch. Please consider speaking with a counsellor - you don't have to carry this alone."},
    {"min": 15, "max": 21, "level": "Severe", "color": "danger",
     "message": "Thank you for being honest - that takes courage. Please reach out to a mental health professional or one of the helplines below soon."},
]


def score_assessment(answers: list[int]) -> dict:
    total = sum(answers)
    for level in SEVERITY_LEVELS:
        if level["min"] <= total <= level["max"]:
            return {"total": total, **level}
    return {"total": total, **SEVERITY_LEVELS[-1]}


def save_assessment(user_id: str, answers: list[int], scored: dict) -> dict:
    from datetime import datetime
    doc = {
        "user_id": user_id,
        "answers": answers,
        "total_score": scored["total"],
        "severity": scored["level"],
        "timestamp": datetime.utcnow(),
    }
    result = db.wellness_assessments.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


def get_latest_assessment(user_id: str) -> dict | None:
    return db.wellness_assessments.find_one({"user_id": user_id}, sort=[("timestamp", -1)])
