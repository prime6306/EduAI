"""
modules/rag/ingestion.py
PDF → text → chunks → sentence-transformer embeddings → ChromaDB
Also stores metadata in MongoDB.
"""
import hashlib
import re
import uuid
from datetime import datetime
from pathlib import Path

def _get_embedder() -> "SentenceTransformer":
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(get_settings().embedding_model)
    return _embedder

from backend.config import get_settings
from backend.db.chromadb_client import get_collection
from backend.db.mongodb import get_sync_db

_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(get_settings().embedding_model)
    return _embedder


# ── Text extraction ───────────────────────────────────────

def extract_text(path: str) -> str:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(str(p))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        from docx import Document
        doc = Document(str(p))
        return "\n".join(para.text for para in doc.paragraphs)
    if suffix in (".txt", ".md"):
        return p.read_text(encoding="utf-8", errors="ignore")
    return ""


# ── Chunking ─────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    """Sliding window chunker on sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current, current_len = [], [], 0
    for sent in sentences:
        words = sent.split()
        if current_len + len(words) > chunk_size and current:
            chunks.append(" ".join(current))
            # overlap: keep last N words
            overlap_words = current[-overlap:] if len(current) > overlap else current
            current = list(overlap_words)
            current_len = len(current)
        current.extend(words)
        current_len += len(words)
    if current:
        chunks.append(" ".join(current))
    return [c.strip() for c in chunks if len(c.strip()) > 30]


# ── Ingest ────────────────────────────────────────────────

def ingest_pdf(file_path: str, user_id: str, original_filename: str) -> dict:
    """
    Full ingestion pipeline:
    1. Extract text
    2. Chunk
    3. Embed
    4. Store in ChromaDB
    5. Store metadata in MongoDB
    Returns pdf_id and chunk count.
    """
    db = get_sync_db()
    text = extract_text(file_path)
    if not text.strip():
        raise ValueError("Could not extract text from file")

    # Deduplicate by content hash
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    existing = db.pdfs.find_one({"content_hash": content_hash, "user_id": user_id})
    if existing:
        return {"pdf_id": str(existing["_id"]), "chunks": existing["chunk_count"], "cached": True}

    chunks = chunk_text(text)
    embedder = _get_embedder()
    embeddings = embedder.encode(chunks, batch_size=32, convert_to_numpy=True).tolist()

    pdf_id = str(uuid.uuid4())
    collection = get_collection("pdf_chunks")

    # Batch upsert into ChromaDB
    ids = [f"{pdf_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"pdf_id": pdf_id, "user_id": user_id, "chunk_idx": i} for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)

    # MongoDB metadata
    doc = {
        "pdf_id": pdf_id,
        "user_id": user_id,
        "filename": original_filename,
        "content_hash": content_hash,
        "chunk_count": len(chunks),
        "char_count": len(text),
        "created_at": datetime.utcnow(),
    }
    db.pdfs.insert_one(doc)
    return {"pdf_id": pdf_id, "chunks": len(chunks), "cached": False}
