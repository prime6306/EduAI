"""
Thin wrapper around ChromaDB. One collection ("documents"), filtered by
`user_id` + `pdf_id` metadata rather than one collection per document —
simpler to manage and perfectly fine at this scale.

The embedding model is loaded once into a module-level singleton (it's a
~90MB download the first time; subsequent app starts load from the local
HuggingFace cache).
"""
import threading

_embedder_lock = threading.Lock()
_embedder_cache = {"model": None}

_chroma_lock = threading.Lock()
_chroma_cache = {"client": None, "collection": None}


def get_embedder():
    if _embedder_cache["model"] is None:
        with _embedder_lock:
            if _embedder_cache["model"] is None:
                from flask import current_app
                from sentence_transformers import SentenceTransformer
                _embedder_cache["model"] = SentenceTransformer(
                    current_app.config["EMBEDDING_MODEL"]
                )
    return _embedder_cache["model"]


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()


def _get_collection():
    if _chroma_cache["collection"] is None:
        with _chroma_lock:
            if _chroma_cache["collection"] is None:
                from flask import current_app
                import chromadb
                from chromadb import EmbeddingFunction, Documents, Embeddings

                class _NoOpEF(EmbeddingFunction):
                    """Stub EF — never called; we always supply pre-computed embeddings."""
                    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
                        raise NotImplementedError("Pre-computed embeddings must be provided.")

                client = chromadb.PersistentClient(path=current_app.config["CHROMA_PATH"])
                collection = client.get_or_create_collection(
                    "documents",
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=_NoOpEF(),
                )
                _chroma_cache["client"] = client
                _chroma_cache["collection"] = collection
    return _chroma_cache["collection"]


def add_chunks(pdf_id: str, user_id: str, chunks: list[str]) -> None:
    if not chunks:
        return
    collection = _get_collection()
    embeddings = embed_texts(chunks)
    ids = [f"{pdf_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"user_id": user_id, "pdf_id": pdf_id, "chunk_index": i} for i in range(len(chunks))]
    collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)


def query_chunks(pdf_id: str, user_id: str, query_text: str, top_k: int = 5) -> list[dict]:
    """Returns [{text, similarity}], similarity in [0, 1] (cosine-space)."""
    collection = _get_collection()
    query_embedding = embed_texts([query_text])[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"$and": [{"user_id": user_id}, {"pdf_id": pdf_id}]},
    )
    docs = result.get("documents", [[]])[0]
    distances = result.get("distances", [[]])[0]
    out = []
    for doc, dist in zip(docs, distances):
        similarity = max(0.0, 1.0 - dist / 2.0)  # cosine distance in [0,2] -> similarity in [0,1]
        out.append({"text": doc, "similarity": round(similarity, 3)})
    return out


def delete_pdf(pdf_id: str) -> None:
    collection = _get_collection()
    collection.delete(where={"pdf_id": pdf_id})
