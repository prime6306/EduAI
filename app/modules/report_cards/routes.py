import io

from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, send_file, Response
from flask_login import login_required, current_user

from app.auth.utils import role_required
from app.extensions import logger
from app.utils.audit import log_action
from . import report_card_service as svc

report_cards_bp = Blueprint("report_cards", __name__, url_prefix="/report-cards")
report_cards_api_bp = Blueprint("report_cards_api", __name__, url_prefix="/api/report-cards")


@report_cards_bp.route("")
@login_required
@role_required("teacher")
def index():
    batches = svc.list_for_teacher(current_user.id)
    return render_template("report_cards/list.html", batches=batches)


@report_cards_bp.route("/upload", methods=["GET", "POST"])
@login_required
@role_required("teacher")
def upload():
    if request.method == "POST":
        title = (request.form.get("title") or "").strip()
        subject = (request.form.get("subject") or "").strip()
        semester = request.form.get("semester", "")
        academic_year = request.form.get("academic_year", "")
        comp_names = request.form.getlist("component_name[]")
        comp_max = request.form.getlist("component_max[]")

        components = [
            {"name": n.strip(), "max_marks": float(m) if m else 100}
            for n, m in zip(comp_names, comp_max) if n.strip()
        ]

        if not title or not subject:
            flash("Title and subject are required.", "danger")
            return redirect(url_for("report_cards.upload"))
        if not components:
            flash("Add at least one assessment component (e.g. IA1, IA2, Assignment).", "danger")
            return redirect(url_for("report_cards.upload"))

        doc = svc.create_batch(current_user.id, title, subject, semester, academic_year, components)
        log_action("report_cards.batch_created", {"batch_id": str(doc["_id"])})
        return redirect(url_for("report_cards.marks", batch_id=str(doc["_id"])))

    return render_template("report_cards/upload.html")


@report_cards_bp.route("/<batch_id>/marks")
@login_required
@role_required("teacher")
def marks(batch_id):
    batch = svc.get(batch_id, current_user.id)
    if not batch:
        flash("Batch not found.", "danger")
        return redirect(url_for("report_cards.index"))
    return render_template("report_cards/marks.html", batch=batch)


@report_cards_bp.route("/<batch_id>/template-csv")
@login_required
@role_required("teacher")
def template_csv(batch_id):
    batch = svc.get(batch_id, current_user.id)
    if not batch:
        flash("Batch not found.", "danger")
        return redirect(url_for("report_cards.index"))
    csv_data = svc.generate_template_csv(batch)
    return Response(
        csv_data, mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={batch['subject']}_marks_template.csv"},
    )


@report_cards_bp.route("/<batch_id>/configure", methods=["GET", "POST"])
@login_required
@role_required("teacher")
def configure(batch_id):
    batch = svc.get(batch_id, current_user.id)
    if not batch:
        flash("Batch not found.", "danger")
        return redirect(url_for("report_cards.index"))

    if request.method == "POST":
        weightages = {
            "ia": float(request.form.get("weight_ia") or 0),
            "quiz": float(request.form.get("weight_quiz") or 0),
            "attendance": float(request.form.get("weight_attendance") or 0),
        }
        grade_names = request.form.getlist("grade_name[]")
        grade_mins = request.form.getlist("grade_min[]")
        grade_maxs = request.form.getlist("grade_max[]")
        grading_scheme = [
            {"min": float(mn), "max": float(mx), "grade": g}
            for g, mn, mx in zip(grade_names, grade_mins, grade_maxs) if g.strip()
        ] or svc.DEFAULT_GRADING_SCHEME

        remark_mode = request.form.get("remark_mode", "template")
        remark_template = request.form.get("remark_template") or "{name} performed {grade_word} this term."
        include_dropout_risk = request.form.get("include_dropout_risk") == "on"

        try:
            svc.compute_and_generate(
                batch_id, current_user.id, weightages, grading_scheme,
                remark_mode, remark_template, include_dropout_risk,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Report card computation failed")
            flash("Something went wrong generating report cards. Please try again.", "danger")
            return redirect(url_for("report_cards.configure", batch_id=batch_id))

        log_action("report_cards.computed", {"batch_id": batch_id})
        return redirect(url_for("report_cards.results", batch_id=batch_id))

    return render_template("report_cards/configure.html", batch=batch, default_scheme=svc.DEFAULT_GRADING_SCHEME)


@report_cards_bp.route("/<batch_id>/results")
@login_required
@role_required("teacher")
def results(batch_id):
    batch = svc.get(batch_id, current_user.id)
    if not batch:
        flash("Batch not found.", "danger")
        return redirect(url_for("report_cards.index"))
    return render_template("report_cards/results.html", batch=batch)


@report_cards_bp.route("/<batch_id>/generate-pdfs", methods=["POST"])
@login_required
@role_required("teacher")
def generate_pdfs(batch_id):
    try:
        count = svc.generate_all_pdfs(batch_id, current_user.id)
    except ValueError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("report_cards.results", batch_id=batch_id))
    except Exception:  # noqa: BLE001
        logger.exception("Report card PDF generation failed")
        flash("PDF generation isn't available in this environment right now.", "warning")
        return redirect(url_for("report_cards.results", batch_id=batch_id))

    log_action("report_cards.pdfs_generated", {"batch_id": batch_id, "count": count})
    flash(f"{count} report card PDF(s) generated.", "success")
    return redirect(url_for("report_cards.results", batch_id=batch_id))


@report_cards_bp.route("/<batch_id>/download")
@login_required
@role_required("teacher")
def download_zip(batch_id):
    batch = svc.get(batch_id, current_user.id)
    if not batch:
        flash("Batch not found.", "danger")
        return redirect(url_for("report_cards.index"))
    zip_bytes = svc.generate_zip(batch_id, current_user.id)
    if not zip_bytes:
        flash("No report cards generated yet.", "warning")
        return redirect(url_for("report_cards.results", batch_id=batch_id))
    return send_file(
        io.BytesIO(zip_bytes), mimetype="application/zip", as_attachment=True,
        download_name=f"{batch['subject'].replace(' ', '_')}_report_cards.zip",
    )


@report_cards_bp.route("/student/<batch_id>/<student_id>")
@login_required
@role_required("teacher")
def download_single(batch_id, student_id):
    pdf_bytes = svc.get_pdf_bytes(batch_id, current_user.id, student_id)
    if not pdf_bytes:
        flash("Report card not available for that student.", "warning")
        return redirect(url_for("report_cards.results", batch_id=batch_id))
    return send_file(
        io.BytesIO(pdf_bytes), mimetype="application/pdf", as_attachment=True,
        download_name=f"{student_id}_report_card.pdf",
    )


@report_cards_api_bp.route("/<batch_id>/import-csv", methods=["POST"])
@login_required
@role_required("teacher")
def import_csv(batch_id):
    file = request.files.get("csv_file")
    if not file:
        return jsonify({"error": "No CSV file provided."}), 400
    try:
        result = svc.import_csv(batch_id, current_user.id, file.stream)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    log_action("report_cards.csv_imported", {"batch_id": batch_id, **result})
    return jsonify(result)


@report_cards_api_bp.route("/<batch_id>/update-cell", methods=["POST"])
@login_required
@role_required("teacher")
def update_cell(batch_id):
    data = request.get_json(silent=True) or {}
    ok = svc.update_cell(batch_id, current_user.id, data.get("student_id"), data.get("component"), data.get("value"))
    return jsonify({"saved": ok})
