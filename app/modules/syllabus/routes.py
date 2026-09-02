import io
from datetime import datetime

from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, send_file
from flask_login import login_required, current_user

from app.auth.forms import BRANCH_CHOICES, YEAR_CHOICES
from app.auth.utils import role_required
from app.extensions import logger
from app.utils.audit import log_action
from . import syllabus_service as svc

syllabus_bp = Blueprint("syllabus", __name__, url_prefix="/syllabus")
syllabus_api_bp = Blueprint("syllabus_api", __name__, url_prefix="/api/syllabus")


@syllabus_bp.route("")
@login_required
@role_required("teacher")
def index():
    syllabuses = svc.list_for_teacher(current_user.id)
    items = []
    for s in syllabuses:
        progress = svc.compute_progress(s)
        items.append({"doc": s, "progress": progress})
    return render_template("syllabus/list.html", items=items)


@syllabus_bp.route("/create", methods=["GET", "POST"])
@login_required
@role_required("teacher")
def create():
    if request.method == "POST":
        subject = (request.form.get("subject") or "").strip()
        branch = request.form.get("branch", "")
        year = request.form.get("year", "")
        semester = request.form.get("semester", "")
        academic_year = request.form.get("academic_year", "")
        exam_date_str = request.form.get("exam_date")
        total_lectures = int(request.form.get("total_lectures") or 0)

        if not subject or not exam_date_str:
            flash("Subject and exam date are required.", "danger")
            return redirect(url_for("syllabus.create"))

        exam_date = datetime.strptime(exam_date_str, "%Y-%m-%d")

        unit_numbers = request.form.getlist("unit_number[]")
        unit_titles = request.form.getlist("unit_title[]")
        unit_lectures = request.form.getlist("unit_lectures[]")
        unit_topics_raw = request.form.getlist("unit_topics[]")

        units = []
        for num, title, lec, topics_raw in zip(unit_numbers, unit_titles, unit_lectures, unit_topics_raw):
            if not title.strip():
                continue
            topics = [
                {"text": t.strip(), "covered": False, "covered_date": None, "note": ""}
                for t in topics_raw.split("\n") if t.strip()
            ]
            units.append({
                "number": int(num) if num.isdigit() else len(units) + 1,
                "title": title.strip(),
                "estimated_lectures": int(lec) if lec.isdigit() else 0,
                "topics": topics,
            })

        if not units:
            flash("Add at least one unit with topics.", "danger")
            return redirect(url_for("syllabus.create"))

        doc = svc.create_syllabus(
            current_user.id, subject, branch, year, semester, academic_year,
            exam_date, total_lectures, units,
        )
        log_action("syllabus.created", {"subject": subject, "syllabus_id": str(doc["_id"])})
        flash(f"Syllabus for {subject} created.", "success")
        return redirect(url_for("syllabus.detail", syllabus_id=str(doc["_id"])))

    return render_template("syllabus/create.html", branches=BRANCH_CHOICES, years=YEAR_CHOICES)


@syllabus_bp.route("/<syllabus_id>")
@login_required
@role_required("teacher")
def detail(syllabus_id):
    doc = svc.get(syllabus_id, current_user.id)
    if not doc:
        flash("Syllabus not found.", "danger")
        return redirect(url_for("syllabus.index"))
    progress = svc.compute_progress(doc)
    return render_template("syllabus/detail.html", doc=doc, progress=progress)


@syllabus_api_bp.route("/<syllabus_id>/toggle", methods=["POST"])
@login_required
@role_required("teacher")
def toggle(syllabus_id):
    data = request.get_json(silent=True) or {}
    unit_index = data.get("unit_index")
    topic_index = data.get("topic_index")
    note = data.get("note")

    if unit_index is None or topic_index is None:
        return jsonify({"error": "unit_index and topic_index are required."}), 400

    topic = svc.toggle_topic(syllabus_id, current_user.id, int(unit_index), int(topic_index), note)
    if topic is None:
        return jsonify({"error": "Not found."}), 404

    doc = svc.get(syllabus_id, current_user.id)
    progress = svc.compute_progress(doc)
    log_action("syllabus.topic_toggled", {"syllabus_id": syllabus_id, "covered": topic["covered"]})
    return jsonify({"topic": {
        "covered": topic["covered"],
        "covered_date": topic["covered_date"].isoformat() if topic["covered_date"] else None,
    }, "progress": {
        "overall_pct": progress["overall_pct"], "pace_status": progress["pace_status"],
    }})


@syllabus_api_bp.route("/<syllabus_id>/pace")
@login_required
@role_required("teacher")
def pace(syllabus_id):
    doc = svc.get(syllabus_id, current_user.id)
    if not doc:
        return jsonify({"error": "Not found."}), 404
    progress = svc.compute_progress(doc)
    return jsonify({
        "overall_pct": progress["overall_pct"],
        "pace_status": progress["pace_status"],
        "heatmap": progress["heatmap"],
        "per_unit": progress["per_unit"],
    })


@syllabus_bp.route("/<syllabus_id>/export")
@login_required
@role_required("teacher")
def export(syllabus_id):
    doc = svc.get(syllabus_id, current_user.id)
    if not doc:
        flash("Syllabus not found.", "danger")
        return redirect(url_for("syllabus.index"))

    progress = svc.compute_progress(doc)
    summary = svc.generate_naac_summary(doc, progress)

    try:
        from weasyprint import HTML
        from flask import render_template as rt
        html_string = rt(
            "syllabus/naac_pdf.html", doc=doc, progress=progress, summary=summary,
            teacher_name=current_user.name,
        )
        pdf_bytes = HTML(string=html_string).write_pdf()
    except Exception:  # noqa: BLE001
        logger.exception("NAAC PDF export failed")
        flash("PDF export isn't available in this environment right now.", "warning")
        return redirect(url_for("syllabus.detail", syllabus_id=syllabus_id))

    filename = f"{doc['subject'].replace(' ', '_')}_NAAC_Report.pdf"
    return send_file(io.BytesIO(pdf_bytes), mimetype="application/pdf", as_attachment=True, download_name=filename)
