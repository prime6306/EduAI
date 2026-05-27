"""
db/mongodb.py – Async MongoDB client (Motor).
Collections used across the platform:
  users, topic_pipelines, qa_history, pdf_chunks, pdfs,
  attendance_logs, dropout_predictions, wellness_sessions, plagiarism_reports
"""
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from backend.config import get_settings

_async_client: AsyncIOMotorClient | None = None
_sync_client: MongoClient | None = None


def get_async_client() -> AsyncIOMotorClient:
    global _async_client
    if _async_client is None:
        s = get_settings()
        _async_client = AsyncIOMotorClient(s.mongodb_uri)
    return _async_client


def get_async_db():
    s = get_settings()
    return get_async_client()[s.mongodb_db]


def get_sync_client() -> MongoClient:
    global _sync_client
    if _sync_client is None:
        s = get_settings()
        _sync_client = MongoClient(s.mongodb_uri)
    return _sync_client


def get_sync_db():
    s = get_settings()
    return get_sync_client()[s.mongodb_db]
