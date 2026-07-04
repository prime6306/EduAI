"""
backend/main.py – EduAI FastAPI application entry point.
"""
import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.auth.router import router as auth_router
from backend.modules.attendance.router import router as attendance_router
from backend.modules.dropout.router import router as dropout_router
from backend.modules.nlp.router import router as nlp_router
from backend.modules.rag.router import router as rag_router
from backend.modules.sentiment.router import router as wellness_router

app = FastAPI(
    title="EduAI API",
    description="AI-powered academic platform — LLM, RAG, Face Attendance, Wellness",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow Streamlit dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routers
app.include_router(auth_router)
app.include_router(nlp_router)
app.include_router(rag_router)
app.include_router(attendance_router)
app.include_router(dropout_router)
app.include_router(wellness_router)


@app.get("/", tags=["Health"])
async def root():
    return {"status": "running", "service": "EduAI API v1.0", "docs": "/docs"}


@app.get("/health", tags=["Health"])
async def health():
    """Quick health check — verifies DB connections."""
    from backend.db.mongodb import get_async_db
    try:
        db = get_async_db()
        await db.command("ping")
        mongo_status = "ok"
    except Exception as e:
        mongo_status = f"error: {e}"

    try:
        from backend.db.chromadb_client import get_chroma_client
        get_chroma_client().heartbeat()
        chroma_status = "ok"
    except Exception as e:
        chroma_status = f"error: {e}"

    return {
        "mongodb": mongo_status,
        "chromadb": chroma_status,
        "status": "healthy" if mongo_status == "ok" else "degraded",
    }


@app.on_event("startup")
async def startup_event():
    print("[EduAI] Starting up (lazy-load mode)...")
