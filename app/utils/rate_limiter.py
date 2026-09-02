"""
Infra Feature 1 — API Rate Limiting.

A small, purpose-built daily counter backed directly by MongoDB
(`rate_limit_counters`), rather than routing through Flask-Limiter's
internal storage. Flask-Limiter doesn't cleanly expose "how many has this
user used so far today" for the usage-meter UI the spec calls for without
reaching into library internals, so this hand-rolled version trades that
generality for full control over exactly the behavior specified: a daily
per-role cap that resets at UTC midnight, a teacher-grantable 24-hour
bypass, and an honest usage count to show the user.

Admin role: not modelled anywhere else in this app yet (see the audit log
viewer's access-control note) — `_limit_for` returns None (unlimited) for
any role not in a feature's table, which already covers a future "admin"
role for free.
"""
from datetime import datetime, timedelta

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from app.extensions import db, logger

# feature -> {"student": N, "teacher": N}. Absent role or None value = unlimited.
LIMITS = {
    "doubt_chat":       {"student": 60, "teacher": 120},
    "study_material":   {"student": 8, "teacher": 20},
    "quiz_generate":    {"student": 15, "teacher": 30},
    "rag_ask":          {"student": 40, "teacher": 80},
    "wellness_chat":    {"student": 30, "teacher": None},
    "study_plan":       {"student": 5, "teacher": 10},
    "question_paper":   {"student": None, "teacher": 10},
    "plagiarism_check": {"student": None, "teacher": 20},
    # A single mock interview run is a much heavier pipeline than the
    # other AI features — 2 analysis calls + up to 11 question/evaluation
    # call pairs + a closing prep-plan call, spread across two LLM
    # providers — so it gets its own, smaller daily cap. Teachers aren't
    # expected to run their own mock interviews (they review student
    # reports instead), so they're unlimited here rather than modelled
    # with a number.
    "interview_session": {"student": 3, "teacher": None},
}

FEATURE_LABELS = {
    "doubt_chat": "doubt solver messages",
    "study_material": "study material pipelines",
    "quiz_generate": "quiz generations",
    "rag_ask": "RAG Q&A questions",
    "wellness_chat": "wellness chat messages",
    "study_plan": "study plans",
    "question_paper": "question paper generations",
    "plagiarism_check": "plagiarism checks",
    "interview_session": "mock interview sessions",
}


def ensure_indexes() -> None:
    try:
        db.rate_limit_counters.create_index(
            [("user_id", 1), ("feature", 1), ("date", 1)], unique=True, name="uniq_user_feature_date"
        )
    except Exception:  # noqa: BLE001
        logger.warning("Could not create rate_limit_counters index (non-fatal).")


def _today_key() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


def _limit_for(feature: str, role: str):
    return LIMITS.get(feature, {}).get(role)


def has_bypass(user_id: str) -> bool:
    try:
        user = db.users.find_one({"_id": ObjectId(user_id)}, {"rate_limit_bypass_until": 1})
    except (InvalidId, TypeError):
        return False
    until = user.get("rate_limit_bypass_until") if user else None
    return bool(until and until > datetime.utcnow())


def grant_bypass(user_id: str, hours: int = 24) -> None:
    db.users.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"rate_limit_bypass_until": datetime.utcnow() + timedelta(hours=hours)}},
    )


def revoke_bypass(user_id: str) -> None:
    db.users.update_one({"_id": ObjectId(user_id)}, {"$unset": {"rate_limit_bypass_until": ""}})


def get_usage(feature: str, user_id: str) -> int:
    doc = db.rate_limit_counters.find_one({"user_id": user_id, "feature": feature, "date": _today_key()})
    return doc["count"] if doc else 0


def get_status(feature: str, user_id: str, role: str) -> dict:
    """For rendering a usage meter without consuming a request."""
    limit = _limit_for(feature, role)
    if limit is None:
        return {"unlimited": True, "used": 0, "limit": None, "remaining": None}
    used = get_usage(feature, user_id)
    return {"unlimited": False, "used": used, "limit": limit, "remaining": max(0, limit - used)}


def limit_message(feature: str, status: dict) -> str:
    label = FEATURE_LABELS.get(feature, feature)
    return f"You've reached today's limit for {label} ({status.get('limit')}/day). Come back tomorrow."


def check_and_increment(feature: str, user_id: str, role: str) -> tuple[bool, dict]:
    """Atomically increments today's counter if under the limit.
    Returns (allowed, status_dict).

    Correctness here does not depend on the unique index on
    (user_id, feature, date) — it's only used to handle the rare race of
    two concurrent first-requests-of-the-day cleanly. Without it, the
    conditional-upsert-with-a-range-filter trick that's easy to reach for
    here would silently create duplicate counter rows past the cap
    instead of blocking, which is caught by this design instead.
    """
    limit = _limit_for(feature, role)
    if limit is None:
        return True, {"unlimited": True}

    if has_bypass(user_id):
        return True, {"unlimited": True, "bypass": True}

    today = _today_key()
    filt = {"user_id": user_id, "feature": feature, "date": today}
    under_limit_filt = {**filt, "count": {"$lt": limit}}

    result = db.rate_limit_counters.find_one_and_update(
        under_limit_filt, {"$inc": {"count": 1}}, return_document=ReturnDocument.AFTER
    )
    if result is not None:
        used = result["count"]
        return True, {"unlimited": False, "used": used, "limit": limit, "remaining": max(0, limit - used)}

    existing = db.rate_limit_counters.find_one(filt)
    if existing is None:
        try:
            db.rate_limit_counters.insert_one({**filt, "count": 1})
            return True, {"unlimited": False, "used": 1, "limit": limit, "remaining": max(0, limit - 1)}
        except DuplicateKeyError:
            # Lost a race with a concurrent first request today — retry
            # the conditional increment now that a document exists.
            result = db.rate_limit_counters.find_one_and_update(
                under_limit_filt, {"$inc": {"count": 1}}, return_document=ReturnDocument.AFTER
            )
            if result is not None:
                used = result["count"]
                return True, {"unlimited": False, "used": used, "limit": limit, "remaining": max(0, limit - used)}
            existing = db.rate_limit_counters.find_one(filt)

    used = existing["count"] if existing else limit
    return False, {"unlimited": False, "used": used, "limit": limit, "remaining": 0}
