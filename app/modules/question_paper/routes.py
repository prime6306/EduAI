import io

from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, send_file
from flask_login import login_required, current_user

from app.auth.utils import role_required
from app.extensions import logger
from app.utils.audit import log_action
from app.utils import rate_limiter
from app.modules.nlp.llm_client import LLMNotConfigured
from . import qp_service as svc

question_paper_bp = Blueprint("question_paper", __name__, url_prefix="/question-paper")
question_paper_api_bp = Blueprint("question_paper_api", __name__, url_prefix="/api/question-paper")
question_bank_bp = Blueprint("question_bank", __name__, url_prefix="/question-bank")
question_bank_api_bp = Blueprint("question_bank_api", __name__, url_prefix="/api/question-bank")


@question_paper_bp.route("", methods=["GET", "POST"])
@login_required
@role_required("teacher")
def index():
    if request.method == "POST":
        subject = (request.form.get("subject") or "").strip()
        units = [u.strip() for u in request.form.getlist("units[]") if u.strip()]
        total_marks = int(request.form.get("total_marks") or 100)
        duration = request.form.get("duration", "3 hours")
        num_sets = int(request.form.get("num_sets") or 1)
        bloom_levels = request.form.getlist("bloom_levels[]")

        difficulty_split = {
            "easy": int(request.form.get("difficulty_easy") or 40),
            "medium": int(request.form.get("difficulty_medium") or 40),
            "hard": int(request.form.get("difficulty_hard") or 20),
        }

        question_types = request.form.getlist("question_types[]")
        marks_per_type = {
            "mcq": int(request.form.get("marks_per_mcq") or 1),
            "short": int(request.form.get("marks_per_short") or 5),
            "long": int(request.form.get("marks_per_long") or 15),
        }
        section_marks = {
            "mcq": int(request.form.get("section_marks_mcq") or 0),
            "short": int(request.form.get("section_marks_short") or 0),
            "long": int(request.form.get("section_marks_long") or 0),
        }

        if not subject or not question_types:
            flash("Subject and at least one question type are required.", "danger")
            return redirect(url_for("question_paper.index"))

        allowed, rl_status = rate_limiter.check_and_increment("question_paper", current_user.id, current_user.role)
        if not allowed:
            flash(rate_limiter.limit_message("question_paper", rl_status), "warning")
            return redirect(url_for("question_paper.index"))

        try:
            doc = svc.generate_paper(
                current_user.id, subject, units, total_marks, difficulty_split,
                question_types, marks_per_type, section_marks, num_sets, bloom_levels, duration,
            )
        except LLMNotConfigured as exc:
            flash(str(exc), "warning")
            return redirect(url_for("question_paper.index"))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Question paper generation failed")
            flash(f"Couldn't generate the paper: {exc}", "danger")
            return redirect(url_for("question_paper.index"))

        log_action("question_paper.generated", {"subject": subject, "paper_id": str(doc["_id"])})
        return redirect(url_for("question_paper.result", paper_id=str(doc["_id"])))

    return render_template(
        "question_paper/form.html",
        status=rate_limiter.get_status("question_paper", current_user.id, current_user.role),
        usage_feature_label=rate_limiter.FEATURE_LABELS["question_paper"],
    )


@question_paper_bp.route("/result/<paper_id>")
@login_required
@role_required("teacher")
def result(paper_id):
    doc = svc.get_paper(paper_id, current_user.id)
    if not doc:
        flash("Paper not found.", "danger")
        return redirect(url_for("question_paper.index"))
    return render_template("question_paper/result.html", doc=doc)


@question_paper_bp.route("/history")
@login_required
@role_required("teacher")
def history():
    papers = svc.list_papers(current_user.id)
    return render_template("question_paper/history.html", papers=papers)


@question_paper_bp.route("/result/<paper_id>/export")
@login_required
@role_required("teacher")
def export(paper_id):
    doc = svc.get_paper(paper_id, current_user.id)
    if not doc:
        flash("Paper not found.", "danger")
        return redirect(url_for("question_paper.index"))

    set_label = request.args.get("set", "Set A")
    fmt = request.args.get("format", "pdf")
    show_answers = request.args.get("answers") == "1"

    date_str = doc["created_at"].strftime("%Y%m%d")
    filename_base = f"{doc['subject'].replace(' ', '_')}_{date_str}_{set_label.replace(' ', '')}"

    try:
        if fmt == "docx":
            file_bytes = svc.render_docx(doc, set_label, show_answers)
            mimetype = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            filename = f"{filename_base}.docx"
        else:
            file_bytes = svc.render_pdf(doc, set_label, show_answers)
            mimetype = "application/pdf"
            filename = f"{filename_base}.pdf"
    except Exception:  # noqa: BLE001
        logger.exception("Question paper export failed")
        flash("Export isn't available in this environment right now.", "warning")
        return redirect(url_for("question_paper.result", paper_id=paper_id))

    return send_file(io.BytesIO(file_bytes), mimetype=mimetype, as_attachment=True, download_name=filename)


@question_paper_api_bp.route("/regenerate-question", methods=["POST"])
@login_required
@role_required("teacher")
def regenerate_question():
    data = request.get_json(silent=True) or {}
    paper_id = data.get("paper_id")
    set_label = data.get("set_label")
    question_index = data.get("question_index")

    new_q = svc.regenerate_question(paper_id, current_user.id, set_label, int(question_index))
    if new_q is None:
        return jsonify({"error": "Could not regenerate that question."}), 400
    return jsonify({"question": new_q})


@question_paper_api_bp.route("/edit-question", methods=["POST"])
@login_required
@role_required("teacher")
def edit_question():
    data = request.get_json(silent=True) or {}
    ok = svc.edit_question(
        data.get("paper_id"), current_user.id, data.get("set_label"),
        int(data.get("question_index")), data.get("fields", {}),
    )
    return jsonify({"updated": ok})


@question_paper_api_bp.route("/delete-question", methods=["POST"])
@login_required
@role_required("teacher")
def delete_question_route():
    data = request.get_json(silent=True) or {}
    ok = svc.delete_question(data.get("paper_id"), current_user.id, data.get("set_label"), int(data.get("question_index")))
    return jsonify({"deleted": ok})


@question_paper_api_bp.route("/add-question", methods=["POST"])
@login_required
@role_required("teacher")
def add_question_route():
    data = request.get_json(silent=True) or {}
    question = data.get("question", {})
    ok = svc.add_manual_question(data.get("paper_id"), current_user.id, data.get("set_label"), question)
    return jsonify({"added": ok})


@question_bank_bp.route("")
@login_required
@role_required("teacher")
def browse():
    subject = request.args.get("subject", "")
    q_type = request.args.get("type", "")
    difficulty = request.args.get("difficulty", "")
    questions = svc.list_question_bank(current_user.id, subject, q_type, difficulty)
    return render_template(
        "question_paper/bank.html", questions=questions, subject=subject, q_type=q_type, difficulty=difficulty,
    )


@question_bank_api_bp.route("/search")
@login_required
@role_required("teacher")
def search_bank():
    subject = request.args.get("subject", "")
    q_type = request.args.get("type", "")
    questions = svc.list_question_bank(current_user.id, subject, q_type)
    return jsonify([
        {
            "id": str(q["_id"]), "text": q["text"], "type": q["type"],
            "marks": q["marks"], "difficulty": q.get("difficulty", ""),
        }
        for q in questions[:50]
    ])


@question_bank_api_bp.route("/<question_id>", methods=["DELETE"])
@login_required
@role_required("teacher")
def delete_bank_question(question_id):
    ok = svc.delete_bank_question(question_id, current_user.id)
    return jsonify({"deleted": ok})
