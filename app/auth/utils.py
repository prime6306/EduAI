"""
User model (thin wrapper over a `users` Mongo document) + access-control
helpers shared across every blueprint.
"""
from functools import wraps

from bson import ObjectId
from bson.errors import InvalidId
from flask import abort, flash, redirect, url_for
from flask_login import UserMixin, current_user, login_manager as _unused  # noqa: F401
from flask_login import current_user  # re-import kept explicit for clarity

from app.extensions import db, login_manager


class User(UserMixin):
    """Adapts a `users` collection document to the Flask-Login interface."""

    def __init__(self, doc: dict):
        self._doc = doc

    # Flask-Login required property
    def get_id(self) -> str:
        return str(self._doc["_id"])

    @property
    def id(self) -> str:
        return str(self._doc["_id"])

    @property
    def name(self) -> str:
        return self._doc.get("name", "")

    @property
    def email(self) -> str:
        return self._doc.get("email", "")

    @property
    def role(self) -> str:
        return self._doc.get("role", "student")

    @property
    def branch(self) -> str:
        return self._doc.get("branch", "")

    @property
    def year(self) -> str:
        return self._doc.get("year", "")

    @property
    def student_id(self) -> str:
        return self._doc.get("student_id", "")

    @property
    def initials(self) -> str:
        parts = self.name.split()
        letters = "".join(p[0] for p in parts[:2] if p)
        return letters.upper() or "U"

    @property
    def is_teacher(self) -> bool:
        return self.role == "teacher"

    @property
    def is_student(self) -> bool:
        return self.role == "student"

    def to_doc(self) -> dict:
        return self._doc


@login_manager.user_loader
def load_user(user_id: str):
    try:
        oid = ObjectId(user_id)
    except (InvalidId, TypeError):
        return None
    doc = db.users.find_one({"_id": oid})
    return User(doc) if doc else None


def role_required(*roles):
    """
    Route decorator restricting access to the given role(s).
    Usage: @role_required('teacher')
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not current_user.is_authenticated:
                flash("Please sign in to continue.", "info")
                return redirect(url_for("auth.login"))
            if current_user.role not in roles:
                abort(403)
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def get_current_user() -> User:
    return current_user
