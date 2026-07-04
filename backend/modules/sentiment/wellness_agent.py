"""
modules/sentiment/wellness_agent.py
PHQ-9 inspired mental wellness assessment + empathetic Groq chat agent.
Tracks sentiment scores over time in MongoDB.
"""
import json
from datetime import datetime

from groq import Groq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from backend.config import get_settings
from backend.db.mongodb import get_sync_db

_groq_client: Groq | None = None
_vader = SentimentIntensityAnalyzer()


def _groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=get_settings().groq_api_key)
    return _groq_client


# ── PHQ-9 inspired questionnaire ──────────────────────────

WELLNESS_QUESTIONS = [
    {
        "id": "q1",
        "text": "Over the past week, how often have you felt little interest or pleasure in doing things?",
        "options": ["Not at all (0)", "Several days (1)", "More than half the days (2)", "Nearly every day (3)"],
        "scores": [0, 1, 2, 3],
    },
    {
        "id": "q2",
        "text": "How often have you felt down, depressed, or hopeless?",
        "options": ["Not at all (0)", "Several days (1)", "More than half the days (2)", "Nearly every day (3)"],
        "scores": [0, 1, 2, 3],
    },
    {
        "id": "q3",
        "text": "How often have you had trouble concentrating on studying or academic work?",
        "options": ["Not at all (0)", "Several days (1)", "More than half the days (2)", "Nearly every day (3)"],
        "scores": [0, 1, 2, 3],
    },
    {
        "id": "q4",
        "text": "How often have you felt tired or had little energy?",
        "options": ["Not at all (0)", "Several days (1)", "More than half the days (2)", "Nearly every day (3)"],
        "scores": [0, 1, 2, 3],
    },
    {
        "id": "q5",
        "text": "How do you feel about your academic performance right now?",
        "options": [
            "Very satisfied — I'm doing great!",
            "Okay — could be better",
            "Stressed — struggling to keep up",
            "Overwhelmed — I feel like giving up",
        ],
        "scores": [0, 1, 2, 3],
    },
    {
        "id": "q6",
        "text": "Have you had thoughts that you would be better off not studying or dropping out?",
        "options": ["Not at all", "Once or twice", "Several times", "Almost every day"],
        "scores": [0, 1, 2, 3],
    },
    {
        "id": "q7",
        "text": "How would you describe your social support (friends, family, teachers)?",
        "options": [
            "Strong — I have people I can talk to",
            "Moderate — some support available",
            "Weak — I mostly deal with things alone",
            "None — I feel completely isolated",
        ],
        "scores": [0, 1, 2, 3],
    },
]

CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "don't want to live",
    "want to die", "self harm", "hurt myself", "no point in living",
]

CRISIS_RESPONSE = """I'm really glad you reached out. What you're feeling matters deeply, and you are not alone.
Please speak to someone right now:

📞 iCall (India): 9152987821
📞 Vandrevala Foundation: 1860-2662-345 (24/7)
📞 NIMHANS: 080-46110007

If you are in immediate danger, please call 112 (Emergency) or go to your nearest hospital.

I'm here with you. Would you like to talk about what's been making you feel this way?"""


def interpret_score(total: int) -> dict:
    if total <= 4:
        return {"level": "minimal", "color": "green", "message": "You seem to be doing well! Keep it up."}
    elif total <= 9:
        return {"level": "mild", "color": "yellow",
                "message": "You're experiencing some challenges. Let's talk about it."}
    elif total <= 14:
        return {"level": "moderate", "color": "orange",
                "message": "You're going through a tough time. I'm here to help."}
    else:
        return {"level": "severe", "color": "red",
                "message": "It sounds like you're really struggling. Please reach out for support."}


def score_answers(answers: list[int]) -> dict:
    total = sum(answers)
    return {"total_score": total, "max_score": 21, **interpret_score(total)}


def analyze_sentiment_text(text: str) -> dict:
    scores = _vader.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {"compound": compound, "label": label, "scores": scores}


def check_crisis(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in CRISIS_KEYWORDS)


def get_system_prompt(assessment: dict) -> str:
    level = assessment.get("level", "mild")
    level_context = {
        "minimal": "The student is generally doing well but wants to chat.",
        "mild": "The student is experiencing some stress and mild difficulties.",
        "moderate": "The student is going through a tough time academically and emotionally.",
        "severe": "The student is in significant distress. Be extra gentle, patient, and supportive.",
    }
    return f"""You are EduAI Wellness, a compassionate and empathetic AI wellness companion
for university students in India.

Student's current mental wellness assessment: {level.upper()}
Context: {level_context.get(level, "")}

Your role:
- Listen with empathy and validate feelings
- Offer gentle encouragement and practical coping strategies
- Motivate without being dismissive ("just be positive")
- Suggest study tips, stress management techniques when appropriate
- If the student mentions academic struggles, offer concrete advice
- NEVER diagnose conditions
- NEVER replace professional mental health support — always mention iCall (9152987821) for serious concerns
- Keep responses warm, human, and concise (2-4 sentences per turn)
- Use simple, supportive language appropriate for Indian university students

If you detect crisis language, IMMEDIATELY provide iCall helpline: 9152987821"""


def chat_with_wellness_agent(
    message: str,
    conversation_history: list[dict],
    assessment: dict,
    user_id: str,
    session_id: str,
) -> dict:
    """
    Stateless chat turn with wellness agent.
    conversation_history: list of {"role": "user"/"assistant", "content": "..."}
    Returns assistant response + sentiment analysis.
    """
    # Crisis detection
    if check_crisis(message):
        db = get_sync_db()
        db.wellness_sessions.update_one(
            {"session_id": session_id},
            {"$set": {"crisis_flag": True, "crisis_time": datetime.utcnow()}},
        )
        return {
            "response": CRISIS_RESPONSE,
            "sentiment": analyze_sentiment_text(message),
            "crisis_detected": True,
        }

    # Build messages
    messages = [{"role": "system", "content": get_system_prompt(assessment)}]
    messages.extend(conversation_history[-10:])  # last 10 turns for context
    messages.append({"role": "user", "content": message})

    resp = _groq().chat.completions.create(
        model=get_settings().groq_model,
        messages=messages,
        max_tokens=400,
        temperature=0.75,
    )
    response_text = resp.choices[0].message.content.strip()

    # Analyze user message sentiment
    sentiment = analyze_sentiment_text(message)

    # Save to MongoDB
    db = get_sync_db()
    db.wellness_sessions.update_one(
        {"session_id": session_id},
        {
            "$push": {
                "messages": {
                    "role": "user", "content": message,
                    "sentiment": sentiment, "timestamp": datetime.utcnow(),
                }
            },
            "$set": {"last_active": datetime.utcnow()},
        },
        upsert=True,
    )
    db.wellness_sessions.update_one(
        {"session_id": session_id},
        {"$push": {"messages": {"role": "assistant", "content": response_text,
                                "timestamp": datetime.utcnow()}}},
    )

    return {
        "response": response_text,
        "sentiment": sentiment,
        "crisis_detected": False,
    }


def start_session(user_id: str, assessment: dict) -> str:
    """Create a new wellness session and return session_id."""
    import uuid
    session_id = str(uuid.uuid4())
    db = get_sync_db()
    db.wellness_sessions.insert_one({
        "session_id": session_id,
        "user_id": user_id,
        "assessment": assessment,
        "messages": [],
        "crisis_flag": False,
        "created_at": datetime.utcnow(),
        "last_active": datetime.utcnow(),
    })
    return session_id
