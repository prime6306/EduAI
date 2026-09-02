"""
Turns an uploaded file into overlapping text chunks ready for embedding.
Supports PDF, DOCX, TXT, and Markdown per the spec. Heavy parser imports
are deferred into each function so a missing optional dependency only
breaks that one file type, not the whole module.
"""
import hashlib
import os


def extract_text(filepath: str, filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return _extract_pdf(filepath)
    if ext == ".docx":
        return _extract_docx(filepath)
    if ext in (".txt", ".md"):
        return _extract_plain(filepath)
    raise ValueError(f"Unsupported file type '{ext}'. Upload a PDF, DOCX, TXT, or Markdown file.")


def _extract_pdf(filepath: str) -> str:
    from PyPDF2 import PdfReader
    reader = PdfReader(filepath)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _extract_docx(filepath: str) -> str:
    import docx
    doc = docx.Document(filepath)
    return "\n".join(p.text for p in doc.paragraphs)


def _extract_plain(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def chunk_text(text: str, chunk_size_words: int = 500, overlap_words: int = 80) -> list[str]:
    """500-word chunks with 80-word overlap, per spec."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(chunk_size_words - overlap_words, 1)
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size_words]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size_words >= len(words):
            break
    return chunks


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
