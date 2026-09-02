"""
Chat session management: crisis-keyword short-circuit (bypasses the LLM
entirely and always shows helplines, per spec), VADER sentiment on every
message, and severity-aware Groq replies with 10-turn context.
"""
from datetime import datetime

from bson import ObjectId
from bson.errors import InvalidId

from app.extensions import db
from app.modules.nlp.llm_client import chat_completion
from . import prompts

CRISIS_KEYWORDS = [
    "kill myself",
    "end my life",
    "want to die",
    "suicidal",
    "no reason to live",
    "better off dead",
    "hurt myself",
    "can't go on",
]

MAX_CONTEXT_TURNS = 10


def detect_crisis(message):
    lowered = message.lower()
    return any(kw in lowered for kw in CRISIS_KEYWORDS)


def analyze_sentiment(message):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(message)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {"compound": compound, "label": label}


def create_session(user_id, severity_context):
    doc = {
        "user_id": user_id,
        "severity_context": severity_context,
        "messages": [],
        "crisis_flagged": False,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    result = db.wellness_sessions.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


def get_session(session_id, user_id):
    try:
        oid = ObjectId(session_id)
    except (InvalidId, TypeError):
        return None
    return db.wellness_sessions.find_one({"_id": oid, "user_id": user_id})


def list_sessions(user_id):
    return list(db.wellness_sessions.find({"user_id": user_id}).sort("updated_at", -1))


def send_message(user_id, session_id, message, severity_context):
    session = get_session(session_id, user_id) if session_id else None
    if not session:
        session = create_session(user_id, severity_context)

    sentiment = analyze_sentiment(message)
    is_crisis = detect_crisis(message)

    user_msg = {"role": "user", "content": message, "sentiment": sentiment, "timestamp": datetime.utcnow()}

    if is_crisis:
        reply = prompts.CRISIS_RESPONSE
        assistant_msg = {"role": "assistant", "content": reply, "timestamp": datetime.utcnow()}
        db.wellness_sessions.update_one(
            {"_id": session["_id"]},
            {
                "$push": {"messages": {"$each": [user_msg, assistant_msg]}},
                "$set": {"crisis_flagged": True, "updated_at": datetime.utcnow()},
            },
        )
        return {
            "session_id": str(session["_id"]), "reply": reply, "sentiment": sentiment, "crisis": True,
        }

    history = session.get("messages", [])[-(MAX_CONTEXT_TURNS * 2):]
    llm_messages = [{"role": "system", "content": prompts.wellness_system_prompt(session.get("severity_context"))}]
    for m in history:
        llm_messages.append({"role": m["role"], "content": m["content"]})
    llm_messages.append({"role": "user", "content": message})

    reply = chat_completion(llm_messages, temperature=0.7, max_tokens=500)
    assistant_msg = {"role": "assistant", "content": reply, "timestamp": datetime.utcnow()}

    db.wellness_sessions.update_one(
        {"_id": session["_id"]},
        {
            "$push": {"messages": {"$each": [user_msg, assistant_msg]}},
            "$set": {"updated_at": datetime.utcnow()},
        },
    )

    return {
        "session_id": str(session["_id"]), "reply": reply, "sentiment": sentiment, "crisis": False,
    }
