"""
modules/sentiment/router.py – Wellness assessment + chat endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.auth.router import get_current_user
from backend.modules.sentiment import wellness_agent as wa

router = APIRouter(prefix="/wellness", tags=["Wellness"])


class AssessmentAnswers(BaseModel):
    answers: list[int]  # 7 integers, 0-3 each


class ChatMessage(BaseModel):
    message: str
    session_id: str
    conversation_history: list[dict] = []
    assessment: dict = {}


@router.get("/questions")
async def get_questions():
    """Return the wellness questionnaire."""
    return {"questions": wa.WELLNESS_QUESTIONS}


@router.post("/assess")
async def assess(req: AssessmentAnswers, user=Depends(get_current_user)):
    """Score the questionnaire and start a wellness session."""
    if len(req.answers) != len(wa.WELLNESS_QUESTIONS):
        raise HTTPException(400, f"Expected {len(wa.WELLNESS_QUESTIONS)} answers")
    assessment = wa.score_answers(req.answers)
    session_id = wa.start_session(str(user["_id"]), assessment)
    return {"assessment": assessment, "session_id": session_id}


@router.post("/chat")
async def chat(req: ChatMessage, user=Depends(get_current_user)):
    try:
        result = wa.chat_with_wellness_agent(
            message=req.message,
            conversation_history=req.conversation_history,
            assessment=req.assessment,
            user_id=str(user["_id"]),
            session_id=req.session_id,
        )
        return result
    except Exception as e:
        raise HTTPException(500, str(e))
