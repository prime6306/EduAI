"""
Orchestrates the RAG pipeline: upload -> extract -> chunk -> embed -> store,
and ask -> retrieve -> threshold-check -> answer -> hallucination-score.
"""
import os
from datetime import datetime

from bson import ObjectId
from bson.errors import InvalidId
from flask import current_app

from app.extensions import db, logger
from . import document_processor as docproc
from . import vector_store as vs
from . import prompts
from app.modules.nlp.llm_client import chat_completion, chat_json

GROUNDED_THRESHOLD = 70
PARTIAL_THRESHOLD = 40


def upload_document(user_id: str, filename: str, filepath: str, file_size_bytes: int) -> dict:
    text = docproc.extract_text(filepath, filename)
    if not text.strip():
        raise ValueError("Couldn't extract any text from that file.")

    chash = docproc.content_hash(text)
    existing = db.pdfs.find_one({"user_id": user_id, "content_hash": chash})
    if existing:
        return existing  # dedup - same content already indexed

    chunks = docproc.chunk_text(
        text,
        current_app.config["RAG_CHUNK_SIZE_WORDS"],
        current_app.config["RAG_CHUNK_OVERLAP_WORDS"],
    )
    if not chunks:
        raise ValueError("That file has no usable text content.")

    oid = ObjectId()
    pdf_id = str(oid)

    vs.add_chunks(pdf_id, user_id, chunks)

    doc = {
        "_id": oid,
        "pdf_id": pdf_id,
        "user_id": user_id,
        "filename": filename,
        "content_hash": chash,
        "chunk_count": len(chunks),
        "file_size_bytes": file_size_bytes,
        "created_at": datetime.utcnow(),
    }
    db.pdfs.insert_one(doc)
    return doc


def list_documents(user_id: str) -> list[dict]:
    return list(db.pdfs.find({"user_id": user_id}).sort("created_at", -1))


def get_document(pdf_id: str, user_id: str) -> dict | None:
    return db.pdfs.find_one({"pdf_id": pdf_id, "user_id": user_id})


def delete_document(pdf_id: str, user_id: str) -> bool:
    doc = get_document(pdf_id, user_id)
    if not doc:
        return False
    vs.delete_pdf(pdf_id)
    db.pdfs.delete_one({"pdf_id": pdf_id, "user_id": user_id})
    db.qa_history.delete_many({"pdf_id": pdf_id, "user_id": user_id})
    return True


def _verdict_for_score(score: int) -> str:
    if score >= GROUNDED_THRESHOLD:
        return "grounded"
    if score >= PARTIAL_THRESHOLD:
        return "partial"
    return "hallucinated"


def get_related_questions(user_id: str, pdf_id: str, exclude_question: str, limit: int = 3) -> list[str]:
    """Most recent distinct past questions on this document, excluding the
    current one. A lightweight, practical stand-in for full semantic
    similarity search over question history."""
    seen = set()
    out = []
    cursor = db.qa_history.find({"user_id": user_id, "pdf_id": pdf_id}).sort("timestamp", -1).limit(30)
    for h in cursor:
        q = h.get("question", "").strip()
        if q and q != exclude_question and q not in seen:
            seen.add(q)
            out.append(q)
        if len(out) >= limit:
            break
    return out


def ask_question(user_id: str, pdf_id: str, question: str) -> dict:
    doc = get_document(pdf_id, user_id)
    if not doc:
        raise ValueError("Document not found.")

    threshold = current_app.config["RAG_SIMILARITY_THRESHOLD"]
    matches = vs.query_chunks(pdf_id, user_id, question, top_k=5)

    if not matches or matches[0]["similarity"] < threshold:
        related = get_related_questions(user_id, pdf_id, question)
        result = {
            "answer": "This topic is not in the uploaded material.",
            "grounding_score": 0,
            "verdict": "off_topic",
            "related_questions": related,
            "chunks_used": 0,
        }
        _save_history(user_id, pdf_id, question, result)
        return result

    context_texts = [m["text"] for m in matches]
    answer = chat_completion(prompts.rag_answer_prompt(question, context_texts), max_tokens=800)

    try:
        score_raw = chat_json(prompts.hallucination_score_prompt(question, answer, context_texts))
        score = int(score_raw.get("score", 50))
        score = max(0, min(100, score))
    except Exception:  # noqa: BLE001
        logger.warning("Hallucination scoring failed - defaulting to 50.")
        score = 50

    related = get_related_questions(user_id, pdf_id, question)
    result = {
        "answer": answer,
        "grounding_score": score,
        "verdict": _verdict_for_score(score),
        "related_questions": related,
        "chunks_used": len(matches),
    }
    _save_history(user_id, pdf_id, question, result)
    return result


def _save_history(user_id: str, pdf_id: str, question: str, result: dict) -> None:
    db.qa_history.insert_one({
        "user_id": user_id,
        "pdf_id": pdf_id,
        "question": question,
        "answer": result["answer"],
        "hallucination_score": result["grounding_score"],
        "chunks_used": result["chunks_used"],
        "timestamp": datetime.utcnow(),
    })


def get_history(user_id: str, pdf_id: str) -> list[dict]:
    return list(db.qa_history.find({"user_id": user_id, "pdf_id": pdf_id}).sort("timestamp", 1))
