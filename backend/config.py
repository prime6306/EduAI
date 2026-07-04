"""
config.py – Central configuration using pydantic-settings.
All modules import settings from here.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM
    groq_api_key: str
    groq_model: str = "openai/gpt-oss-120b"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # MongoDB
    mongodb_uri: str
    mongodb_db: str = "eduai"

    # ChromaDB
    chroma_path: str = "./chromadb_data"

    # JWT
    jwt_secret: str = "change_me_in_production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  # 24 h

    # External APIs
    youtube_api_key: str = ""
    google_api_key: str = ""
    google_search_engine_id: str = ""

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_experiment: str = "EduAI"

    # Anti-spoof / Face
    antispoof_model_path: str = "./models/antispoof_fullmodels.pkl"
    face_encodings_path: str = "./models/ENcodedFile.p"

    # Backend URL (used by Streamlit)
    backend_url: str = "http://localhost:8000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
