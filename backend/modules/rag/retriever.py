"""
modules/rag/retriever.py
ChromaDB semantic search → Groq Llama 3 answer generation.
Includes similarity threshold, hallucination scoring, Q&A recommendations.
"""
from datetime import datetime

from groq import Groq

def _get_embedder() -> "SentenceTransformer":
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(get_settings().embedding_model)
    return _embedder

from backend.config import get_settings
from backend.db.chromadb_client import get_collection
from backend.db.mongodb import get_sync_db
from backend.modules.nlp.pipeline import score_hallucination, get_qa_recommendations

SIMILARITY_THRESHOLD = 0.35  # cosine distance (ChromaDB uses distance, lower = more similar)

_groq_client: Groq | None = None
_embedder: SentenceTransformer | None = None


def _groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=get_settings().groq_api_key)
    return _groq_client


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(get_settings().embedding_model)
    return _embedder


def _generate_answer(question: str, context: str) -> str:
    prompt = f"""You are an expert academic tutor. Answer the student's question strictly
using the provided context. If the answer is not in the context, say:
"This topic is not covered in the uploaded material."

Context:
{context}

Question: {question}

Provide a clear, well-structured educational answer."""
    resp = _groq().chat.completions.create(
        model=get_settings().groq_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
    )
    return resp.choices[0].message.content.strip()


def search_and_answer(question: str, user_id: str, pdf_id: str | None = None, top_k: int = 5) -> dict:
    """
    1. Embed question
    2. Search ChromaDB (optionally filtered by pdf_id)
    3. Threshold check
    4. Generate answer with Groq
    5. Score hallucination
    6. Save to MongoDB
    7. Return recommendations
    """
    db = get_sync_db()
    collection = get_collection("pdf_chunks")

    # Embed query
    q_emb = _get_embedder().encode([question], convert_to_numpy=True).tolist()

    # Build filter
    where = {"user_id": user_id}
    if pdf_id:
        where = {"$and": [{"user_id": user_id}, {"pdf_id": pdf_id}]}

    try:
        results = collection.query(
            query_embeddings=q_emb,
            n_results=top_k,
            where=where,
            include=["documents", "distances", "metadatas"],
        )
    except Exception:
        # If no documents match the filter, ChromaDB raises
        results = {"documents": [[]], "distances": [[]], "metadatas": [[]]}

    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not docs or (distances and distances[0] > (1 - SIMILARITY_THRESHOLD)):
        answer = "This topic is not covered in the uploaded material."
        hallucination = {"grounded_score": 0, "verdict": "no_context", "reason": "No relevant chunks found"}
        recommendations = []
    else:
        context = "\n\n---\n\n".join(docs[:top_k])
        answer = _generate_answer(question, context)
        hallucination = score_hallucination(question, answer, context)
        recommendations = get_qa_recommendations(question, user_id)

    # Persist to MongoDB
    qa_doc = {
        "user_id": user_id,
        "pdf_id": pdf_id,
        "question": question,
        "answer": answer,
        "chunks_used": len(docs),
        "hallucination": hallucination,
        "timestamp": datetime.utcnow(),
    }
    db.qa_history.insert_one(qa_doc)

    return {
        "question": question,
        "answer": answer,
        "chunks_used": len(docs),
        "hallucination_score": hallucination,
        "recommendations": recommendations,
    }


def list_user_pdfs(user_id: str) -> list[dict]:
    db = get_sync_db()
    pdfs = list(db.pdfs.find({"user_id": user_id}, {"_id": 0, "content_hash": 0}))
    return pdfs


def delete_pdf(pdf_id: str, user_id: str) -> bool:
    db = get_sync_db()
    doc = db.pdfs.find_one({"pdf_id": pdf_id, "user_id": user_id})
    if not doc:
        return False
    # Remove from ChromaDB
    collection = get_collection("pdf_chunks")
    try:
        collection.delete(where={"pdf_id": pdf_id})
    except Exception:
        pass
    db.pdfs.delete_one({"pdf_id": pdf_id})
    return True
