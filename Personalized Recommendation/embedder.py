# embedder.py
import io
import uuid
import json
from datetime import datetime

import numpy as np
from bson import Binary
from pymongo import MongoClient
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# -------------------------
# Mongo setup (same as yours)
# -------------------------
mongo_uri = "mongodb+srv://studentUser:1234@cluster0.zodzonx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_uri)
mongo_db = client["studentdb"]

pdfs_collection = mongo_db["pdfs"]          # metadata about PDFs
pdf_chunks_collection = mongo_db["pdf_chunks"]  # one doc per chunk

# -------------------------
# Embedding model
# -------------------------
print("📦 Loading embedding model BGE-M3 (SentenceTransformer)...")
embed_model = SentenceTransformer("BAAI/bge-m3")
print("✅ Embedding model loaded.")


# -------------------------
# Helper: serialize/deserialize embeddings
# -------------------------
def serialize_embedding(vec: np.ndarray) -> Binary:
    """
    Store as BSON Binary bytes; keep dtype and shape separately in doc.
    """
    return Binary(vec.astype(np.float32).tobytes())


def deserialize_embedding(bin_data: Binary, shape, dtype="float32") -> np.ndarray:
    arr = np.frombuffer(bin_data, dtype=np.float32)
    if isinstance(shape, (list, tuple)):
        return arr.reshape(tuple(shape))
    return arr


# -------------------------
# PDF text extraction
# -------------------------
def extract_text_from_pdf_fileobj(fileobj) -> str:
    reader = PdfReader(fileobj)
    text = ""
    for i, page in enumerate(reader.pages):
        content = page.extract_text()
        if content:
            text += content + "\n"
        else:
            print(f"[extract] page {i+1} has no text")
    return text


# -------------------------
# chunking (same config as original)
# -------------------------
def chunk_text(text, chunk_size=1000, chunk_overlap=250):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    return splitter.split_text(text)


# -------------------------
# Main: process uploaded PDF, generate & store embeddings
# -------------------------
def process_pdf_file(file_stream, pdf_name: str = None):
    """
    file_stream: binary file-like object (BytesIO or Flask file.stream)
    Returns: dict with pdf_id and summary
    """
    # 1) generate pdf_id
    pdf_id = str(uuid.uuid4())
    pdf_name = pdf_name or f"pdf_{pdf_id}"

    # 2) extract text
    text = extract_text_from_pdf_fileobj(file_stream)

    # 3) chunk
    chunks = chunk_text(text)
    num_chunks = len(chunks)
    print(f"[embedder] PDF has {num_chunks} chunks.")

    # 4) generate embeddings in batches (to avoid OOM)
    batch_size = 16
    docs_to_insert = []
    for i in range(0, num_chunks, batch_size):
        batch_chunks = chunks[i: i + batch_size]
        # SentenceTransformer expects list[str]
        emb = embed_model.encode(batch_chunks, convert_to_numpy=True, show_progress_bar=False)
        for j, chunk_str in enumerate(batch_chunks):
            chunk_id = i + j
            vec = emb[j].astype(np.float32)
            doc = {
                "pdf_id": pdf_id,
                "pdf_name": pdf_name,
                "chunk_id": int(chunk_id),
                "text": chunk_str,
                "embedding": serialize_embedding(vec),
                "embedding_shape": [int(vec.shape[0])],
                "embedding_dtype": str(vec.dtype),
                "timestamp": datetime.utcnow()
            }
            docs_to_insert.append(doc)


        # Bulk insert per batch to reduce memory pressure
        if len(docs_to_insert) >= 100:
            pdf_chunks_collection.insert_many(docs_to_insert)
            docs_to_insert = []

    # Insert any remaining
    if docs_to_insert:
        pdf_chunks_collection.insert_many(docs_to_insert)

    # 5) store PDF metadata
    pdfs_collection.insert_one({
        "pdf_id": pdf_id,
        "pdf_name": pdf_name,
        "num_chunks": num_chunks,
        "timestamp": datetime.utcnow()
    })

    print(f"[embedder] Saved {num_chunks} chunks for pdf_id={pdf_id}")

    return {"pdf_id": pdf_id, "pdf_name": pdf_name, "num_chunks": num_chunks}
