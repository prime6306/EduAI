"""
Permanent, append-only audit trail. Call `log_action` from any route that
performs a meaningful state change. Never raises — a logging failure must
never break the user-facing action it's recording.
"""
from datetime import datetime

from flask import request
from flask_login import current_user

from app.extensions import db, logger


def log_action(action: str, details: dict | None = None) -> None:
    try:
        actor_authenticated = getattr(current_user, "is_authenticated", False)
        db.audit_log.insert_one(
            {
                "timestamp": datetime.utcnow(),
                "actor_id": current_user.id if actor_authenticated else None,
                "actor_name": current_user.name if actor_authenticated else "anonymous",
                "actor_role": current_user.role if actor_authenticated else "anonymous",
                "action": action,
                "details": details or {},
                "ip_address": request.remote_addr,
                "session_id": request.cookies.get("session"),
            }
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to write audit log entry for action '%s'", action)
