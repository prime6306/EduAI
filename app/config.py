"""
Central configuration, loaded from environment variables (.env).
Keep all tunables here — no hardcoded values scattered through the app.
"""
import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class Config:
    # ── Flask ─────────────────────────────────────────────────
    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
    DEBUG = _bool("FLASK_DEBUG", True)
    PORT = int(os.environ.get("FLASK_PORT", 5000))

    # ── MongoDB ───────────────────────────────────────────────
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
    MONGODB_DB = os.environ.get("MONGODB_DB", "eduai")

    # ── Auth / sessions ──────────────────────────────────────
    REMEMBER_COOKIE_DURATION = timedelta(days=30)
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"

    # ── JWT ───────────────────────────────────────────────────
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "dev-jwt-secret-change-me")
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(
        hours=int(os.environ.get("JWT_ACCESS_TOKEN_EXPIRES_HOURS", 24))
    )
    # Session cookie (Flask-Login) drives normal in-app browsing + AJAX.
    # JWT is issued alongside it for any external/API-consuming client, and is
    # also dropped in an httponly cookie so first-party fetch() calls can use
    # either. JWT cookie CSRF protection is left to Flask-WTF's CSRF token,
    # which already guards every form/AJAX call in this app.
    JWT_TOKEN_LOCATION = ["headers", "cookies"]
    JWT_COOKIE_CSRF_PROTECT = False
    JWT_COOKIE_SECURE = not _bool("FLASK_DEBUG", False)

    # ── CSRF ──────────────────────────────────────────────────
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = None
    # Accept the CSRF token from the X-CSRFToken request header so that
    # AJAX calls using apiFetch() (which sends it as a header) are validated
    # correctly without requiring a hidden form field in the request body.
    WTF_CSRF_HEADERS = ["X-CSRFToken", "X-CSRF-Token"]

    # ── Groq LLM ──────────────────────────────────────────────
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    # ── Gemini LLM (Interview Prep — second "recruiter" persona) ──
    # Optional: Interview Prep degrades to running both interviewer
    # personas on Groq alone if this is left unset (see
    # app/modules/interview/llm_router.py).
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    # ── Embeddings / RAG ──────────────────────────────────────
    EMBEDDING_MODEL = os.environ.get(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    CHROMA_PATH = os.environ.get("CHROMA_PATH", os.path.join(BASE_DIR, "chromadb_data"))
    RAG_CHUNK_SIZE_WORDS = 500
    RAG_CHUNK_OVERLAP_WORDS = 80
    RAG_SIMILARITY_THRESHOLD = 0.35

    # ── External APIs (optional) ─────────────────────────────
    YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
    GOOGLE_SEARCH_ENGINE_ID = os.environ.get("GOOGLE_SEARCH_ENGINE_ID", "")

    # ── MLflow ────────────────────────────────────────────────
    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
    MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "EduAI")

    # ── Model paths ───────────────────────────────────────────
    # Anti-spoof model is a pretrained artifact supplied by the operator — never trained
    # by this app. We only ever load it.
    ANTISPOOF_MODEL_PATH = os.environ.get(
        "ANTISPOOF_MODEL_PATH", os.path.join(BASE_DIR, "models", "antispoof_fullmodels.pkl")
    )
    FACE_ENCODINGS_PATH = os.environ.get(
        "FACE_ENCODINGS_PATH", os.path.join(BASE_DIR, "models", "ENcodedFile.p")
    )
    DROPOUT_RF_MODEL_PATH = os.path.join(BASE_DIR, "models", "dropout_rf.pkl")
    DROPOUT_LR_MODEL_PATH = os.path.join(BASE_DIR, "models", "dropout_lr.pkl")
    DROPOUT_SELECTOR_PATH = os.path.join(BASE_DIR, "models", "dropout_selector.pkl")
    STUDENT_IMAGES_DIR = os.path.join(BASE_DIR, "models", "Images")

    # ── Uploads ───────────────────────────────────────────────
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB, matches attendance photo limit

    # ── Attendance ────────────────────────────────────────────
    ATTENDANCE_COOLDOWN_SECONDS = 15

    # ── Wellness ──────────────────────────────────────────────
    WELLNESS_CRISIS_HELPLINE = "iCall: 9152987821"

    # ── Interview Prep (AI mock interviews, dual-persona) ──────
    # Screening / Competency / Deep-Dive, same shape the module was
    # originally designed and verified against.
    INTERVIEW_QUESTIONS_PER_LEVEL = {1: 3, 2: 4, 3: 4}
    INTERVIEW_LEVEL_NAMES = {1: "Screening Interview", 2: "Competency Interview", 3: "Deep-Dive Interview"}

    # ── Rate limiting (Infra Feature 1) ──────────────────────
    RATE_LIMIT_DEFAULT = os.environ.get("RATE_LIMIT_DEFAULT", "60 per minute")
    RATE_LIMIT_AI = os.environ.get("RATE_LIMIT_AI", "15 per minute")

    # ── Mail (Weekly Digest — Module 18) ─────────────────────
    # Left unset by default; digest generation still works and is stored
    # in-app either way — email delivery is skipped gracefully if
    # MAIL_SERVER is empty (see digest_service.send_digest_email).
    MAIL_SERVER = os.environ.get("MAIL_SERVER", "")
    MAIL_PORT = int(os.environ.get("MAIL_PORT", 587))
    MAIL_USE_TLS = _bool("MAIL_USE_TLS", True)
    MAIL_USERNAME = os.environ.get("MAIL_USERNAME", "")
    MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD", "")
    MAIL_DEFAULT_SENDER = os.environ.get("MAIL_DEFAULT_SENDER", "EduAI Digest <noreply@eduai.local>")

    # ── Scheduler (Weekly Digest — Module 18) ────────────────
    # Runs the digest job every Sunday 23:00 inside the Flask process.
    # Disable for multi-worker deployments where only one process should
    # run the scheduler, or in tests.
    DIGEST_SCHEDULER_ENABLED = _bool("DIGEST_SCHEDULER_ENABLED", True)
