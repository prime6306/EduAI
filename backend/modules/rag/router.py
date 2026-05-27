"""
modules/rag/router.py – PDF ingestion + RAG Q&A endpoints.
"""
import os
import tempfile

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.auth.router import get_current_user
from backend.modules.mlflow_tracker import log_run
from backend.modules.rag import ingestion, retriever

router = APIRouter(prefix="/rag", tags=["RAG / Q&A"])

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}


class QuestionRequest(BaseModel):
    question: str
    pdf_id: str | None = None
    top_k: int = 5


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...), user=Depends(get_current_user)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = ingestion.ingest_pdf(tmp_path, str(user["_id"]), file.filename)
        log_run("rag_ingest",
                params={"filename": file.filename, "user": user.get("email")},
                metrics={"chunks": result.get("chunks", 0)})
        return {"status": "success", **result, "filename": file.filename}
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        os.unlink(tmp_path)


@router.post("/ask")
async def ask_question(req: QuestionRequest, user=Depends(get_current_user)):
    try:
        result = retriever.search_and_answer(
            req.question, str(user["_id"]), req.pdf_id, req.top_k
        )
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/my-pdfs")
async def my_pdfs(user=Depends(get_current_user)):
    return retriever.list_user_pdfs(str(user["_id"]))


@router.delete("/pdf/{pdf_id}")
async def delete_pdf(pdf_id: str, user=Depends(get_current_user)):
    success = retriever.delete_pdf(pdf_id, str(user["_id"]))
    if not success:
        raise HTTPException(404, "PDF not found")
    return {"status": "deleted", "pdf_id": pdf_id}
