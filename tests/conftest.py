"""
Shared pytest fixtures for smoke-testing EduAI without the heavy native
dependencies (torch/dlib/face_recognition/chromadb/...) or a real MongoDB
Atlas cluster — the same approach used to verify every phase of this
build before it was handed over (see README → "Testing this phase
yourself"), just formalised into a real pytest fixture instead of a
scratch script.

None of the stubbed modules are imported at module load time anywhere in
the app (see e.g. app/modules/attendance/face_engine.py's docstring) —
they're only reached inside functions that this test suite never calls —
so stubbing them here only unblocks import-time chains like
dropout -> analytics -> digest, it doesn't hide bugs in those features.
"""
import os
import sys
import types

import bcrypt
import mongomock
import pytest

_HEAVY_STUBS = [
    "torch", "torchvision", "face_recognition", "dlib", "cv2",
    "chromadb", "sentence_transformers", "weasyprint", "mlflow",
    "vaderSentiment", "groq", "google", "google.generativeai",
    "googleapiclient", "googleapiclient.discovery",
]

for _mod_name in _HEAVY_STUBS:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = types.ModuleType(_mod_name)

os.environ.setdefault("FLASK_SECRET_KEY", "test-secret")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret")
os.environ.setdefault("MONGODB_URI", "mongodb://localhost/eduai_test")
os.environ.setdefault("MONGODB_DB", "eduai_test")
os.environ.setdefault("DIGEST_SCHEDULER_ENABLED", "false")
os.environ.setdefault("WTF_CSRF_ENABLED", "false")

from app import create_app  # noqa: E402
from app.extensions import db as _db  # noqa: E402


@pytest.fixture()
def app():
    flask_app = create_app()
    flask_app.config["WTF_CSRF_ENABLED"] = False
    flask_app.config["TESTING"] = True

    _db._db = mongomock.MongoClient()["eduai_test"]

    with flask_app.app_context():
        from app.modules.interview.store import ensure_indexes
        ensure_indexes()

    yield flask_app


@pytest.fixture()
def db(app):
    return _db


@pytest.fixture()
def client(app):
    return app.test_client()


def _make_user(db_, name, email, role, **extra):
    doc = {
        "name": name, "email": email,
        "password_hash": bcrypt.hashpw(b"pass1234", bcrypt.gensalt()).decode(),
        "role": role, "branch": "ECE", "year": 3 if role == "student" else None,
    }
    doc.update(extra)
    result = db_.users.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


@pytest.fixture()
def student(db):
    return _make_user(db, "Riya Sharma", "riya@example.com", "student", student_id="21ECE099")


@pytest.fixture()
def teacher(db):
    return _make_user(db, "Dr. Verma", "verma@example.com", "teacher")


def login(client, email, password="pass1234"):
    return client.post("/auth/login", data={"email": email, "password": password}, follow_redirects=True)
