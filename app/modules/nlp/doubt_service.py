"""
Mongo persistence for doubt_conversations. Kept separate from routes.py so
the streaming route can stay focused on the SSE mechanics.
"""
from datetime import datetime
from bson import ObjectId
from bson.errors import InvalidId

from app.extensions import db

MAX_MESSAGES = 50


def get_conversation(conversation_id: str, user_id: str) -> dict | None:
    try:
        oid = ObjectId(conversation_id)
    except (InvalidId, TypeError):
        return None
    return db.doubt_conversations.find_one({"_id": oid, "user_id": user_id})


def create_conversation(user_id: str, subject: str, level: str) -> dict:
    doc = {
        "user_id": user_id,
        "subject": subject,
        "level": level,
        "messages": [],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    result = db.doubt_conversations.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


def append_message(conversation_id, role: str, content: str) -> None:
    db.doubt_conversations.update_one(
        {"_id": ObjectId(conversation_id)},
        {
            "$push": {"messages": {"$each": [{
                "role": role, "content": content, "timestamp": datetime.utcnow()
            }], "$slice": -MAX_MESSAGES}},
            "$set": {"updated_at": datetime.utcnow()},
        },
    )


def list_recent_conversations(user_id: str, limit: int = 5) -> list[dict]:
    return list(
        db.doubt_conversations.find({"user_id": user_id})
        .sort("updated_at", -1)
        .limit(limit)
    )


def conversation_title(conv: dict) -> str:
    """First user message, trimmed, used as the sidebar label."""
    for msg in conv.get("messages", []):
        if msg["role"] == "user":
            text = msg["content"].strip().replace("\n", " ")
            return text[:40] + ("…" if len(text) > 40 else "")
    return conv.get("subject") or "New conversation"
