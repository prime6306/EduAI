"""
Doubt Solver, Study Material Generator, and Quiz & Exam Simulator.
Three feature areas, one blueprint (nlp_bp) — they share the Groq client,
prompt builders, and the same "AI not configured" failure mode.
"""
import json

from flask import (
    Blueprint, render_template, request, redirect, url_for, flash,
    jsonify, Response, stream_with_context, send_file, current_app,
)
from flask_login import login_required, current_user

from app.auth.forms import BRANCH_CHOICES, YEAR_CHOICES
from app.extensions import logger
from app.utils.audit import log_action
from app.utils import rate_limiter
from . import doubt_service, study_service, quiz_service, planner_service, prompts
from .llm_client import chat_completion_stream, LLMNotConfigured

nlp_bp = Blueprint("nlp", __name__)

LEVELS = ["Simple", "Intermediate", "Advanced"]


# ═══════════════════════════════════════════════════════════════════
#  Doubt Solver
# ═══════════════════════════════════════════════════════════════════

@nlp_bp.route("/doubt-solver")
@login_required
def doubt_solver():
    recent = doubt_service.list_recent_conversations(current_user.id, limit=5)
    recent_view = [
        {"id": str(c["_id"]), "title": doubt_service.conversation_title(c)} for c in recent
    ]
    return render_template(
        "doubt/chat.html",
        recent_conversations=recent_view,
        levels=LEVELS,
    )


@nlp_bp.route("/api/doubt/new", methods=["POST"])
@login_required
def doubt_new():
    data = request.get_json(silent=True) or {}
    subject = (data.get("subject") or "").strip()
    level = data.get("level") or "Intermediate"
    conv = doubt_service.create_conversation(current_user.id, subject, level)
    return jsonify({"conversation_id": str(conv["_id"])})


@nlp_bp.route("/api/doubt/history")
@login_required
def doubt_history():
    recent = doubt_service.list_recent_conversations(current_user.id, limit=20)
    return jsonify([
        {
            "id": str(c["_id"]),
            "title": doubt_service.conversation_title(c),
            "subject": c.get("subject", ""),
            "updated_at": c["updated_at"].isoformat() if c.get("updated_at") else None,
        }
        for c in recent
    ])


@nlp_bp.route("/api/doubt/conversation/<conversation_id>")
@login_required
def doubt_conversation(conversation_id):
    conv = doubt_service.get_conversation(conversation_id, current_user.id)
    if not conv:
        return jsonify({"error": "not_found"}), 404
    return jsonify({
        "id": str(conv["_id"]),
        "subject": conv.get("subject", ""),
        "level": conv.get("level", "Intermediate"),
        "messages": [
            {"role": m["role"], "content": m["content"]} for m in conv.get("messages", [])
        ],
    })


@nlp_bp.route("/api/doubt/chat", methods=["POST"])
@login_required
def doubt_chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    conversation_id = data.get("conversation_id")
    subject = (data.get("subject") or "").strip()
    level = data.get("level") or "Intermediate"

    if not message:
        return jsonify({"error": "Message is required."}), 400

    allowed, status = rate_limiter.check_and_increment("doubt_chat", current_user.id, current_user.role)
    if not allowed:
        return jsonify({"error": rate_limiter.limit_message("doubt_chat", status)}), 429

    if conversation_id:
        conv = doubt_service.get_conversation(conversation_id, current_user.id)
        if not conv:
            return jsonify({"error": "Conversation not found."}), 404
    else:
        conv = doubt_service.create_conversation(current_user.id, subject, level)
        conversation_id = str(conv["_id"])

    doubt_service.append_message(conversation_id, "user", message)

    history = conv.get("messages", [])[-20:]  # bounded context window
    llm_messages = [
        {"role": "system", "content": prompts.doubt_solver_system_prompt(
            current_user.branch, current_user.year, conv.get("subject", subject), conv.get("level", level)
        )}
    ]
    for m in history:
        llm_messages.append({"role": m["role"], "content": m["content"]})
    llm_messages.append({"role": "user", "content": message})

    app = current_app._get_current_object()

    def generate():
        full_reply = []
        try:
            for delta in chat_completion_stream(llm_messages):
                full_reply.append(delta)
                yield f"data: {json.dumps({'delta': delta})}\n\n"
        except LLMNotConfigured as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            return
        except Exception:  # noqa: BLE001
            app.logger.exception("Doubt solver streaming failed")
            yield f"data: {json.dumps({'error': 'The AI tutor hit an error. Please try again.'})}\n\n"
            return

        final_text = "".join(full_reply)
        if final_text:
            with app.app_context():
                doubt_service.append_message(conversation_id, "assistant", final_text)
        yield f"data: {json.dumps({'done': True, 'conversation_id': conversation_id})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ═══════════════════════════════════════════════════════════════════
#  Study Material Generator
# ═══════════════════════════════════════════════════════════════════

@nlp_bp.route("/study-material", methods=["GET", "POST"])
@login_required
def study_material():
    if request.method == "POST":
        topic = (request.form.get("topic") or "").strip()
        subject = (request.form.get("subject") or "").strip()
        branch = request.form.get("branch") or current_user.branch
        year = request.form.get("year") or current_user.year

        if not topic or not subject:
            flash("Topic and subject are required.", "danger")
            return redirect(url_for("nlp.study_material"))

        allowed, rl_status = rate_limiter.check_and_increment("study_material", current_user.id, current_user.role)
        if not allowed:
            flash(rate_limiter.limit_message("study_material", rl_status), "warning")
            return redirect(url_for("nlp.study_material"))

        try:
            pipeline = study_service.run_pipeline(current_user.id, topic, subject, branch, year)
        except LLMNotConfigured as exc:
            flash(str(exc), "warning")
            return redirect(url_for("nlp.study_material"))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Study material pipeline failed")
            flash(f"Couldn't generate material: {exc}", "danger")
            return redirect(url_for("nlp.study_material"))

        log_action("study_material.generated", {"topic": topic, "subject": subject})
        return redirect(url_for("nlp.study_material_result", pipeline_id=str(pipeline["_id"])))

    recent_notes = study_service.list_notes(current_user.id)[:5]
    return render_template(
        "study/form.html",
        branches=BRANCH_CHOICES,
        years=YEAR_CHOICES,
        recent_notes=recent_notes,
        status=rate_limiter.get_status("study_material", current_user.id, current_user.role),
        usage_feature_label=rate_limiter.FEATURE_LABELS["study_material"],
    )


@nlp_bp.route("/study-material/result/<pipeline_id>")
@login_required
def study_material_result(pipeline_id):
    pipeline = study_service.get_pipeline(pipeline_id, current_user.id)
    if not pipeline:
        flash("That study material could not be found.", "danger")
        return redirect(url_for("nlp.study_material"))
    return render_template("study/result.html", p=pipeline)


@nlp_bp.route("/study-material/result/<pipeline_id>/pdf")
@login_required
def study_material_pdf(pipeline_id):
    pipeline = study_service.get_pipeline(pipeline_id, current_user.id)
    if not pipeline:
        flash("That study material could not be found.", "danger")
        return redirect(url_for("nlp.study_material"))
    try:
        pdf_bytes = study_service.render_pdf(pipeline)
    except Exception:  # noqa: BLE001
        logger.exception("PDF export failed")
        flash("PDF export isn't available in this environment right now.", "warning")
        return redirect(url_for("nlp.study_material_result", pipeline_id=pipeline_id))

    import io
    filename = f"{pipeline['topic'][:40].replace(' ', '_')}.pdf"
    return send_file(
        io.BytesIO(pdf_bytes), mimetype="application/pdf",
        as_attachment=True, download_name=filename,
    )


@nlp_bp.route("/study-material/notes")
@login_required
def study_material_notes():
    search = request.args.get("q", "").strip()
    notes = study_service.list_notes(current_user.id, search)
    return render_template("study/notes.html", notes=notes, search=search)


@nlp_bp.route("/api/notes/<note_id>", methods=["DELETE"])
@login_required
def delete_note(note_id):
    ok = study_service.delete_note(note_id, current_user.id)
    if ok:
        log_action("study_material.note_deleted", {"note_id": note_id})
    return jsonify({"deleted": ok})


# ═══════════════════════════════════════════════════════════════════
#  Quiz & Exam Simulator
# ═══════════════════════════════════════════════════════════════════

@nlp_bp.route("/quiz", methods=["GET", "POST"])
@login_required
def quiz():
    if request.method == "POST":
        topic = (request.form.get("topic") or "").strip()
        subject = (request.form.get("subject") or "").strip()
        branch = request.form.get("branch") or current_user.branch
        year = request.form.get("year") or current_user.year
        n_questions = int(request.form.get("num_questions") or 10)
        timed = request.form.get("timed") == "on"

        if not topic:
            flash("Topic is required.", "danger")
            return redirect(url_for("nlp.quiz"))

        allowed, rl_status = rate_limiter.check_and_increment("quiz_generate", current_user.id, current_user.role)
        if not allowed:
            flash(rate_limiter.limit_message("quiz_generate", rl_status), "warning")
            return redirect(url_for("nlp.quiz"))

        try:
            quiz_doc = quiz_service.generate_quiz(
                current_user.id, topic, subject, branch, year, n_questions, timed
            )
        except LLMNotConfigured as exc:
            flash(str(exc), "warning")
            return redirect(url_for("nlp.quiz"))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Quiz generation failed")
            flash(f"Couldn't generate the quiz: {exc}", "danger")
            return redirect(url_for("nlp.quiz"))

        log_action("quiz.generated", {"topic": topic, "num_questions": quiz_doc["num_questions"]})
        return redirect(url_for("nlp.quiz_take", quiz_id=str(quiz_doc["_id"])))

    prefill_topic = request.args.get("topic", "")
    prefill_subject = request.args.get("subject", "")
    return render_template(
        "quiz/setup.html",
        branches=BRANCH_CHOICES,
        years=YEAR_CHOICES,
        prefill_topic=prefill_topic,
        prefill_subject=prefill_subject,
        status=rate_limiter.get_status("quiz_generate", current_user.id, current_user.role),
        usage_feature_label=rate_limiter.FEATURE_LABELS["quiz_generate"],
    )


@nlp_bp.route("/quiz/take/<quiz_id>")
@login_required
def quiz_take(quiz_id):
    quiz_doc = quiz_service.get_quiz(quiz_id, current_user.id)
    if not quiz_doc:
        flash("That quiz could not be found.", "danger")
        return redirect(url_for("nlp.quiz"))
    return render_template(
        "quiz/take.html",
        quiz_id=str(quiz_doc["_id"]),
        topic=quiz_doc["topic"],
        timed=quiz_doc["timed"],
        questions=quiz_service.questions_for_taking(quiz_doc),
    )


@nlp_bp.route("/api/quiz/submit", methods=["POST"])
@login_required
def quiz_submit():
    data = request.get_json(silent=True) or {}
    quiz_id = data.get("quiz_id")
    answers = data.get("answers", {})
    time_taken_sec = int(data.get("time_taken_sec") or 0)

    if not quiz_id:
        return jsonify({"error": "quiz_id is required."}), 400

    try:
        result = quiz_service.submit_quiz(quiz_id, current_user.id, answers, time_taken_sec)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404

    log_action("quiz.submitted", {"quiz_id": quiz_id, "score_percent": result["score_percent"]})
    return jsonify({"result_id": str(result["_id"])})


@nlp_bp.route("/quiz/results/<result_id>")
@login_required
def quiz_results(result_id):
    result = quiz_service.get_result(result_id, current_user.id)
    if not result:
        flash("That result could not be found.", "danger")
        return redirect(url_for("nlp.quiz"))
    return render_template("quiz/results.html", r=result)


@nlp_bp.route("/quiz/history")
@login_required
def quiz_history():
    history = quiz_service.list_history(current_user.id)
    avg_score = round(sum(h["score_percent"] for h in history) / len(history), 1) if history else 0
    return render_template("quiz/history.html", history=history, avg_score=avg_score)


# ═══════════════════════════════════════════════════════════════════
#  Study Planner
# ═══════════════════════════════════════════════════════════════════

@nlp_bp.route("/planner", methods=["GET", "POST"])
@login_required
def planner():
    if request.method == "POST":
        subjects = [s.strip() for s in request.form.getlist("subjects[]") if s.strip()]
        exam_date = (request.form.get("exam_date") or "").strip()
        try:
            hours_per_day = int(request.form.get("hours_per_day") or 4)
        except ValueError:
            hours_per_day = 4
        hours_per_day = min(12, max(1, hours_per_day))

        if not subjects:
            flash("Add at least one subject.", "danger")
            return redirect(url_for("nlp.planner"))

        allowed, rl_status = rate_limiter.check_and_increment("study_plan", current_user.id, current_user.role)
        if not allowed:
            flash(rate_limiter.limit_message("study_plan", rl_status), "warning")
            return redirect(url_for("nlp.planner"))

        try:
            plan = planner_service.generate_plan(
                subjects, exam_date, hours_per_day, current_user.branch, current_user.year
            )
        except LLMNotConfigured as exc:
            flash(str(exc), "warning")
            return redirect(url_for("nlp.planner"))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Study planner pipeline failed")
            flash(f"Couldn't generate a plan: {exc}", "danger")
            return redirect(url_for("nlp.planner"))

        log_action("planner.generated", {"subjects": subjects, "hours_per_day": hours_per_day})
        return render_template("planner/result.html", plan=plan, saved=False, plan_id=None)

    recent_plans = planner_service.list_saved_plans(current_user.id)[:5]
    return render_template(
        "planner/form.html",
        recent_plans=recent_plans,
        status=rate_limiter.get_status("study_plan", current_user.id, current_user.role),
        usage_feature_label=rate_limiter.FEATURE_LABELS["study_plan"],
    )


@nlp_bp.route("/api/planner/save", methods=["POST"])
@login_required
def planner_save():
    data = request.get_json(silent=True) or {}
    plan = data.get("plan") or {}
    if not plan.get("schedule"):
        return jsonify({"error": "No plan data to save."}), 400

    doc = planner_service.save_plan(current_user.id, plan)
    log_action("planner.saved", {"plan_id": str(doc["_id"])})
    return jsonify({
        "saved": True,
        "plan_id": str(doc["_id"]),
        "redirect_url": url_for("nlp.planner_view", plan_id=str(doc["_id"])),
    })


@nlp_bp.route("/planner/saved")
@login_required
def planner_saved():
    plans = planner_service.list_saved_plans(current_user.id)
    return render_template("planner/saved.html", plans=plans)


@nlp_bp.route("/planner/saved/<plan_id>")
@login_required
def planner_view(plan_id):
    plan = planner_service.get_plan(plan_id, current_user.id)
    if not plan:
        flash("That study plan could not be found.", "danger")
        return redirect(url_for("nlp.planner_saved"))
    return render_template("planner/result.html", plan=plan, saved=True, plan_id=plan_id)


@nlp_bp.route("/api/planner/<plan_id>", methods=["DELETE"])
@login_required
def planner_delete(plan_id):
    ok = planner_service.delete_plan(plan_id, current_user.id)
    if ok:
        log_action("planner.deleted", {"plan_id": plan_id})
    return jsonify({"deleted": ok})


@nlp_bp.route("/planner/saved/<plan_id>/pdf")
@login_required
def planner_pdf(plan_id):
    plan = planner_service.get_plan(plan_id, current_user.id)
    if not plan:
        flash("That study plan could not be found.", "danger")
        return redirect(url_for("nlp.planner_saved"))
    try:
        pdf_bytes = planner_service.render_pdf(plan)
    except Exception:  # noqa: BLE001
        logger.exception("Planner PDF export failed")
        flash("PDF export isn't available in this environment right now.", "warning")
        return redirect(url_for("nlp.planner_view", plan_id=plan_id))

    import io
    filename = f"study_plan_{plan_id}.pdf"
    return send_file(
        io.BytesIO(pdf_bytes), mimetype="application/pdf",
        as_attachment=True, download_name=filename,
    )
