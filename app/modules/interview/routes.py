"""
Routes for Interview Prep. Both students and teachers can start their own
mock interview (`interview_session` rate limit applies per the spec Q&A);
teachers additionally get a class-wide view of every student's sessions
and can leave a personalised written comment on any report.
"""
import os
import uuid

from bson import ObjectId
from bson.errors import InvalidId
from flask import (
    Blueprint, render_template, request, jsonify, redirect, url_for, flash, abort, current_app,
)
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

from app.extensions import db, logger
from app.auth.utils import role_required
from app.utils.audit import log_action
from app.utils import rate_limiter
from app.modules.rag import document_processor
from . import store, analysis_service, interview_engine, evaluation_service, personas
from .llm_router import NoInterviewProviderConfigured

interview_bp = Blueprint("interview", __name__, url_prefix="/interview")
interview_api_bp = Blueprint("interview_api", __name__, url_prefix="/api/interview")

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def _extract_uploaded(file_storage):
    """Save an uploaded FileStorage to a temp path, extract its text,
    then delete the temp file. Returns (text, filename); text is empty
    on any failure (caller falls back to whatever was pasted, if any)."""
    filename = secure_filename(file_storage.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        flash(f"Unsupported file type '{ext}'. Upload PDF, DOCX, TXT, or Markdown.", "danger")
        return "", filename

    tmp_name = f"{uuid.uuid4().hex}{ext}"
    tmp_path = os.path.join(current_app.config["UPLOAD_FOLDER"], tmp_name)
    file_storage.save(tmp_path)
    try:
        text = document_processor.extract_text(tmp_path, filename)
    except Exception:  # noqa: BLE001
        logger.exception("Interview Prep: failed to extract text from uploaded file %s", filename)
        text = ""
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return text, filename


# ---------------------------------------------------------------- pages --

@interview_bp.route("")
@login_required
def index():
    my_sessions = store.list_sessions_for_user(current_user.id)
    usage_status = rate_limiter.get_status("interview_session", current_user.id, current_user.role)

    teacher_sessions = None
    if current_user.is_teacher:
        all_sessions = store.list_all_sessions()
        user_ids = {s["user_id"] for s in all_sessions}
        oids = [ObjectId(uid) for uid in user_ids if ObjectId.is_valid(uid)]
        users_by_id = {str(u["_id"]): u for u in db.users.find({"_id": {"$in": oids}})} if oids else {}
        for s in all_sessions:
            owner = users_by_id.get(s["user_id"])
            s["student_name"] = owner["name"] if owner else "Unknown"
        teacher_sessions = all_sessions

    return render_template(
        "interview/index.html",
        sessions=my_sessions,
        teacher_sessions=teacher_sessions,
        status=usage_status,
        usage_feature_label=rate_limiter.FEATURE_LABELS["interview_session"],
        recruiters=[personas.get(personas.RECRUITER_A), personas.get(personas.RECRUITER_B)],
    )


@interview_bp.route("/start", methods=["POST"])
@login_required
def start_session():
    jd_text = (request.form.get("jd_text") or "").strip()
    resume_text = (request.form.get("resume_text") or "").strip()
    jd_filename, resume_filename = "", ""

    jd_file = request.files.get("jd_file")
    if jd_file and jd_file.filename:
        extracted, jd_filename = _extract_uploaded(jd_file)
        if extracted:
            jd_text = extracted

    resume_file = request.files.get("resume_file")
    if resume_file and resume_file.filename:
        extracted, resume_filename = _extract_uploaded(resume_file)
        if extracted:
            resume_text = extracted

    if not jd_text or not resume_text:
        flash("Please provide both a job description and a resume — paste text or upload a file for each.", "danger")
        return redirect(url_for("interview.index"))

    allowed, rl_status = rate_limiter.check_and_increment("interview_session", current_user.id, current_user.role)
    if not allowed:
        flash(rate_limiter.limit_message("interview_session", rl_status), "warning")
        return redirect(url_for("interview.index"))

    session = store.create_session(current_user.id, jd_text, resume_text, jd_filename, resume_filename)
    sid = str(session["_id"])

    try:
        jd_analysis, resume_analysis, fit = analysis_service.run_full_analysis(jd_text, resume_text)
    except NoInterviewProviderConfigured as exc:
        flash(str(exc), "danger")
        return redirect(url_for("interview.index"))
    except Exception:  # noqa: BLE001
        logger.exception("Interview Prep: analysis pipeline failed for session %s", sid)
        flash("Something went wrong analysing your resume and job description. Please try again.", "danger")
        return redirect(url_for("interview.index"))

    store.save_analysis(sid, jd_analysis, resume_analysis, fit)
    log_action("interview.session_started", {"session_id": sid, "role_title": jd_analysis.get("role_title", "")})

    return redirect(url_for("interview.analysis_view", sid=sid))


@interview_bp.route("/<sid>/analysis")
@login_required
def analysis_view(sid):
    session = store.get_session(sid)
    if not session:
        flash("Interview session not found.", "danger")
        return redirect(url_for("interview.index"))
    if session["user_id"] != current_user.id and not current_user.is_teacher:
        abort(403)
    return render_template(
        "interview/analysis.html", session=session,
        is_owner=(session["user_id"] == current_user.id),
    )


@interview_bp.route("/<sid>/take")
@login_required
def take_view(sid):
    session = store.get_owned_session(sid, current_user.id)
    if not session:
        flash("Interview session not found.", "danger")
        return redirect(url_for("interview.index"))
    if session["status"] == "completed":
        return redirect(url_for("interview.report_view", sid=sid))
    if session["status"] == "analyzing":
        flash("This session's analysis isn't ready yet.", "info")
        return redirect(url_for("interview.analysis_view", sid=sid))

    levels_cfg = current_app.config["INTERVIEW_QUESTIONS_PER_LEVEL"]
    level_names = current_app.config["INTERVIEW_LEVEL_NAMES"]

    # Resume mid-interview (e.g. after a refresh): hand back the last
    # unanswered question instead of generating a duplicate one.
    resume_turn = None
    turns = session.get("turns", [])
    if turns and turns[-1].get("answer") is None:
        t = turns[-1]
        p = personas.get(t["interviewer"])
        resume_turn = {
            "turn_id": str(t["turn_id"]), "level": t["level"], "level_name": level_names[t["level"]],
            "question": t["question"], "questions_this_level": session["questions_this_level"],
            "questions_target_this_level": levels_cfg[t["level"]],
            "interviewer": {
                "key": t["interviewer"], "name": p["name"], "title": p["title"],
                "initials": p["avatar_initials"], "voice_pitch": p["voice_pitch"], "voice_rate": p["voice_rate"],
            },
        }

    return render_template("interview/take.html", session=session, resume_turn=resume_turn)


@interview_bp.route("/<sid>/report")
@login_required
def report_view(sid):
    session = store.get_session(sid)
    if not session:
        flash("Interview session not found.", "danger")
        return redirect(url_for("interview.index"))
    if session["user_id"] != current_user.id and not current_user.is_teacher:
        abort(403)

    if session["status"] != "completed":
        levels_cfg = current_app.config["INTERVIEW_QUESTIONS_PER_LEVEL"]
        total_expected = sum(levels_cfg.values())
        turns = session.get("turns", [])
        finished = len(turns) >= total_expected and all(t.get("quality_score") is not None for t in turns)
        if not finished:
            flash("This interview hasn't finished yet.", "info")
            if session["user_id"] == current_user.id:
                return redirect(url_for("interview.take_view", sid=sid))
            return redirect(url_for("interview.index"))

        try:
            report = evaluation_service.build_report(session)
        except NoInterviewProviderConfigured as exc:
            flash(str(exc), "danger")
            return redirect(url_for("interview.index"))
        except Exception:  # noqa: BLE001
            logger.exception("Interview Prep: report generation failed for session %s", sid)
            flash("Couldn't generate the report right now. Please try again.", "danger")
            return redirect(url_for("interview.index"))

        store.mark_completed(sid, report)
        log_action("interview.completed", {"session_id": sid, "overall_score": report["overall_score"]})
        session = store.get_session(sid)

    student_name = None
    if current_user.is_teacher and session["user_id"] != current_user.id:
        try:
            owner = db.users.find_one({"_id": ObjectId(session["user_id"])})
        except (InvalidId, TypeError):
            owner = None
        student_name = owner["name"] if owner else "Unknown student"

    return render_template(
        "interview/report.html", session=session, report=session["report"], student_name=student_name,
    )


# ----------------------------------------------------------------- api --

@interview_api_bp.route("/<sid>/start", methods=["POST"])
@login_required
def api_start(sid):
    session = store.get_owned_session(sid, current_user.id)
    if not session:
        return jsonify({"error": "Interview session not found."}), 404
    if session["status"] == "completed":
        return jsonify({"error": "This interview has already finished."}), 400

    try:
        payload = interview_engine.start(sid)
    except NoInterviewProviderConfigured as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:  # noqa: BLE001
        logger.exception("Interview Prep: start failed for session %s", sid)
        return jsonify({"error": "Couldn't start the interview. Please try again."}), 500

    return jsonify(payload)


@interview_api_bp.route("/<sid>/answer", methods=["POST"])
@login_required
def api_answer(sid):
    session = store.get_owned_session(sid, current_user.id)
    if not session:
        return jsonify({"error": "Interview session not found."}), 404

    body = request.get_json(silent=True) or {}
    turn_id = body.get("turn_id")
    answer_text = (body.get("answer") or "").strip()
    if not turn_id or not answer_text:
        return jsonify({"error": "turn_id and answer are required."}), 400

    try:
        payload = interview_engine.submit_answer(sid, turn_id, answer_text)
    except NoInterviewProviderConfigured as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:  # noqa: BLE001
        logger.exception("Interview Prep: answer submission failed for session %s", sid)
        return jsonify({"error": "Couldn't process that answer. Please try again."}), 500

    return jsonify(payload)


@interview_api_bp.route("/<sid>/feedback", methods=["POST"])
@login_required
@role_required("teacher")
def api_feedback(sid):
    session = store.get_session(sid)
    if not session:
        return jsonify({"error": "Interview session not found."}), 404

    body = request.get_json(silent=True) or {}
    comment = (body.get("comment") or "").strip()
    if not comment:
        return jsonify({"error": "A comment is required."}), 400

    store.save_teacher_feedback(sid, current_user.id, current_user.name, comment)
    log_action("interview.teacher_feedback_added", {"session_id": sid, "student_id": session["user_id"]})
    return jsonify({"ok": True, "comment": comment, "teacher_name": current_user.name})
