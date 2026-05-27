"""
db/chromadb_client.py – Persistent ChromaDB client.
Collections (ChromaDB calls them "collections"):
  pdf_chunks  – student-uploaded PDF content
  topic_notes – generated study notes
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from backend.config import get_settings

_chroma_client = None


def get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        cfg = get_settings()
        _chroma_client = chromadb.PersistentClient(
            path=cfg.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _chroma_client


def get_collection(name: str):
    """Get or create a named collection."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
