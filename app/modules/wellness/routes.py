from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from flask_login import login_required, current_user

from app.extensions import logger
from app.utils.audit import log_action
from app.utils import rate_limiter
from app.modules.nlp.llm_client import LLMNotConfigured
from . import assessment_service as assess
from . import chat_service
from .chat_service import detect_crisis

wellness_bp = Blueprint("wellness", __name__, url_prefix="/wellness")
wellness_api_bp = Blueprint("wellness_api", __name__, url_prefix="/api/wellness")

HELPLINES = [
    {"name": "iCall (TISS)", "number": "+91 9152987821", "hours": "Mon-Sat, 10am-8pm"},
    {"name": "Tele-MANAS (Govt. of India)", "number": "14416 / 1800-891-4416", "hours": "24/7"},
    {"name": "Vandrevala Foundation", "number": "1860-266-2345", "hours": "24/7"},
]


@wellness_bp.route("")
@login_required
def index():
    latest = assess.get_latest_assessment(current_user.id)
    return render_template("wellness/index.html", latest=latest, helplines=HELPLINES)


@wellness_bp.route("/assess", methods=["GET", "POST"])
@login_required
def assess_view():
    if request.method == "POST":
        answers = []
        for q in assess.QUESTIONS:
            val = request.form.get(f"q{q['id']}")
            if val is None:
                flash("Please answer every question.", "danger")
                return redirect(url_for("wellness.assess_view"))
            answers.append(int(val))

        scored = assess.score_assessment(answers)
        assess.save_assessment(current_user.id, answers, scored)
        log_action("wellness.assessment_completed", {"severity": scored["level"]})
        return redirect(url_for("wellness.index"))

    return render_template("wellness/assess.html", questions=assess.QUESTIONS, helplines=HELPLINES)


@wellness_bp.route("/chat")
@login_required
def chat_page():
    latest = assess.get_latest_assessment(current_user.id)
    severity = latest["severity"] if latest else None
    return render_template("wellness/chat.html", severity=severity, helplines=HELPLINES)


@wellness_api_bp.route("/chat", methods=["POST"])
@login_required
def api_chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    session_id = data.get("session_id")

    if not message:
        return jsonify({"error": "Message is required."}), 400

    # Safety takes priority over quota: a crisis message must never be
    # blocked by the daily rate limit, so the limit is skipped entirely
    # when crisis keywords are present.
    if not detect_crisis(message):
        allowed, status = rate_limiter.check_and_increment("wellness_chat", current_user.id, current_user.role)
        if not allowed:
            return jsonify({
                "error": "You've reached today's message limit here. If you'd like to keep talking to "
                         "someone today, iCall (9152987821) and NIMHANS (080-46110007) are available now."
            }), 429

    latest = assess.get_latest_assessment(current_user.id)
    severity_context = latest["severity"] if latest else None

    try:
        result = chat_service.send_message(current_user.id, session_id, message, severity_context)
    except LLMNotConfigured as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:  # noqa: BLE001
        logger.exception("Wellness chat failed")
        return jsonify({"error": "Something went wrong. Please try again."}), 500

    if result.get("crisis"):
        log_action("wellness.crisis_flagged", {"session_id": result["session_id"]})

    return jsonify(result)


@wellness_bp.route("/sessions")
@login_required
def sessions():
    session_list = chat_service.list_sessions(current_user.id)
    return render_template("wellness/sessions.html", sessions=session_list)
