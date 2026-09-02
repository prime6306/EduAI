"""
Dual-method similarity: character n-grams (structural/surface similarity)
+ sentence-embedding cosine similarity (semantic similarity), averaged.
Reuses the RAG module's text extraction and embedding helpers rather than
duplicating PDF/DOCX parsing or model loading.
"""
from flask import current_app

from app.extensions import logger
from app.modules.rag import document_processor as docproc
from app.modules.rag.vector_store import embed_texts

DEFAULT_THRESHOLD = 0.72


def ngram_similarity_matrix(texts: list[str]):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = CountVectorizer(analyzer="char", ngram_range=(5, 5))
    matrix = vectorizer.fit_transform(texts)
    return cosine_similarity(matrix)


def embedding_similarity_matrix(texts: list[str]):
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    embeddings = np.array(embed_texts(texts))
    return cosine_similarity(embeddings)


def find_suspicious_pairs(names: list[str], texts: list[str], threshold: float = DEFAULT_THRESHOLD) -> dict:
    n = len(texts)
    if n < 2:
        raise ValueError("Need at least 2 submissions to compare.")

    ngram_matrix = ngram_similarity_matrix(texts)
    embedding_matrix = embedding_similarity_matrix(texts)

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            ngram_score = float(ngram_matrix[i][j])
            embedding_score = float(embedding_matrix[i][j])
            combined = 0.5 * ngram_score + 0.5 * embedding_score
            if combined >= threshold:
                pairs.append({
                    "student_a": names[i], "student_b": names[j],
                    "ngram_score": round(ngram_score * 100, 1),
                    "embedding_score": round(embedding_score * 100, 1),
                    "combined_score": round(combined * 100, 1),
                    "severity": "red" if combined > 0.85 else "amber",
                })

    pairs.sort(key=lambda p: p["combined_score"], reverse=True)

    result = {
        "total_submissions": n,
        "suspicious_pairs": pairs,
        "threshold": threshold,
    }
    _log_to_mlflow(n, len(pairs), threshold)
    return result


def _log_to_mlflow(n_submissions: int, n_pairs: int, threshold: float) -> None:
    try:
        import mlflow
        mlflow.set_tracking_uri(current_app.config["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(current_app.config["MLFLOW_EXPERIMENT"])
        with mlflow.start_run(run_name="plagiarism_check"):
            mlflow.log_param("threshold", threshold)
            mlflow.log_metric("submissions", n_submissions)
            mlflow.log_metric("suspicious_pairs", n_pairs)
    except Exception:  # noqa: BLE001
        logger.info("MLflow not reachable - skipping plagiarism run log.")


def extract_text_from_upload(filepath: str, filename: str) -> str:
    return docproc.extract_text(filepath, filename)
