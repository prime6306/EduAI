"""
modules/nlp/router.py – NLP feature endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
import tempfile, os

from backend.auth.router import get_current_user
from backend.modules.nlp import pipeline
from backend.modules.mlflow_tracker import log_run

router = APIRouter(prefix="/nlp", tags=["NLP"])


class TopicRequest(BaseModel):
    topic: str
    subject: str
    branch: str = "ECE"
    year: str = "3rd year"


class QuizRequest(BaseModel):
    topic: str
    subject: str
    branch: str = "ECE"
    year: str = "3rd year"
    num_questions: int = 10


class PlagiarismRequest(BaseModel):
    submissions: dict  # {student_name: text}
    threshold: float = 0.72


class StudyPlanRequest(BaseModel):
    subjects: list[str]
    exam_date: str
    hours_per_day: int = 4
    branch: str = "ECE"
    year: str = "3rd year"


@router.post("/full-pipeline")
async def full_pipeline(req: TopicRequest, user=Depends(get_current_user)):
    try:
        result = pipeline.run_full_pipeline(
            req.topic, req.subject, req.branch, req.year, str(user["_id"])
        )
        log_run("nlp_pipeline", params={"topic": req.topic, "subject": req.subject},
                metrics={"processing_time": result.get("processing_time_sec", 0)})
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/quiz")
async def generate_quiz(req: QuizRequest, user=Depends(get_current_user)):
    try:
        questions = pipeline.generate_quiz_from_topic(
            req.topic, req.subject, req.branch, req.year, req.num_questions
        )
        return {"topic": req.topic, "questions": questions, "count": len(questions)}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/plagiarism")
async def check_plagiarism(req: PlagiarismRequest, user=Depends(get_current_user)):
    if user.get("role") not in ("teacher", "admin"):
        raise HTTPException(403, "Only teachers can check plagiarism")
    try:
        results = pipeline.detect_plagiarism(req.submissions, req.threshold)
        log_run("plagiarism_check",
                params={"num_students": len(req.submissions), "threshold": req.threshold},
                metrics={"suspicious_pairs": len(results)})
        return {"suspicious_pairs": results, "total_submissions": len(req.submissions)}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/study-plan")
async def study_plan(req: StudyPlanRequest, user=Depends(get_current_user)):
    try:
        plan = pipeline.generate_study_plan(
            req.subjects, req.exam_date, req.hours_per_day, req.branch, req.year
        )
        return plan
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/plagiarism/upload")
async def plagiarism_upload(
    files: list[UploadFile] = File(...),
    user=Depends(get_current_user),
):
    """Upload multiple .txt/.pdf/.docx files and check for plagiarism."""
    if user.get("role") not in ("teacher", "admin"):
        raise HTTPException(403, "Teachers only")
    text_dict = {}
    for f in files:
        suffix = os.path.splitext(f.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await f.read())
            tmp_path = tmp.name
        try:
            text = pipeline.read_text_from_file(tmp_path)
            text_dict[f.filename] = text
        finally:
            os.unlink(tmp_path)
    results = pipeline.detect_plagiarism(text_dict)
    return {"suspicious_pairs": results, "total_submissions": len(text_dict)}


def read_text_from_file(path: str) -> str:
    """Delegated to pipeline module."""
    return pipeline.read_text_from_file(path) if hasattr(pipeline, "read_text_from_file") else ""
