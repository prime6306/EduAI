import io

from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_login import login_required, current_user

from app.auth.utils import role_required
from app.utils.audit import log_action
from app.extensions import logger
from . import co_po_service as svc
from . import co_po_export as export_svc

co_po_bp = Blueprint("co_po", __name__, url_prefix="/co-po")
co_po_api_bp = Blueprint("co_po_api", __name__, url_prefix="/api/co-po")


# ═══════════════════════════════════════════════════════════════════
#  List + Create
# ═══════════════════════════════════════════════════════════════════

@co_po_bp.route("")
@login_required
@role_required("teacher")
def index():
    setups = svc.list_setups(current_user.id)
    return render_template("co_po/list.html", setups=setups)


@co_po_bp.route("/new", methods=["POST"])
@login_required
@role_required("teacher")
def new_setup():
    subject = (request.form.get("subject") or "").strip()
    semester = (request.form.get("semester") or "").strip()
    academic_year = (request.form.get("academic_year") or "").strip()
    if not subject:
        flash("Subject is required.", "danger")
        return redirect(url_for("co_po.index"))

    doc = svc.create_setup(current_user.id, subject, semester, academic_year)
    log_action("co_po.setup_created", {"setup_id": str(doc["_id"]), "subject": subject})
    return redirect(url_for("co_po.setup", subject_id=str(doc["_id"])))


@co_po_api_bp.route("/<subject_id>/delete", methods=["POST"])
@login_required
@role_required("teacher")
def delete(subject_id):
    ok = svc.delete_setup(subject_id, current_user.id)
    if ok:
        log_action("co_po.setup_deleted", {"setup_id": subject_id})
    return jsonify({"deleted": ok})


# ═══════════════════════════════════════════════════════════════════
#  Step 1 — Define Course Outcomes
# ═══════════════════════════════════════════════════════════════════

@co_po_bp.route("/setup/<subject_id>", methods=["GET", "POST"])
@login_required
@role_required("teacher")
def setup(subject_id):
    doc = svc.get_setup(subject_id, current_user.id)
    if not doc:
        flash("That subject setup could not be found.", "danger")
        return redirect(url_for("co_po.index"))

    if request.method == "POST":
        descriptions = request.form.getlist("co_description[]")
        targets = request.form.getlist("co_target[]")
        cos = []
        for i, (desc, target) in enumerate(zip(descriptions, targets), start=1):
            desc = desc.strip()
            if not desc:
                continue
            try:
                target_val = float(target)
            except (TypeError, ValueError):
                target_val = svc.DEFAULT_TARGET
            cos.append({"id": f"CO{i}", "description": desc, "target_attainment": round(target_val, 2)})

        if not cos:
            flash("Add at least one Course Outcome.", "danger")
            return redirect(url_for("co_po.setup", subject_id=subject_id))

        svc.save_course_outcomes(subject_id, cos)
        log_action("co_po.cos_saved", {"setup_id": subject_id, "count": len(cos)})
        return redirect(url_for("co_po.mapping", subject_id=subject_id))

    return render_template("co_po/setup.html", setup=doc, default_target=svc.DEFAULT_TARGET)


# ═══════════════════════════════════════════════════════════════════
#  Step 2 — CO-PO Mapping grid
# ═══════════════════════════════════════════════════════════════════

@co_po_bp.route("/mapping/<subject_id>")
@login_required
@role_required("teacher")
def mapping(subject_id):
    doc = svc.get_setup(subject_id, current_user.id)
    if not doc:
        flash("That subject setup could not be found.", "danger")
        return redirect(url_for("co_po.index"))
    if not doc.get("course_outcomes"):
        flash("Define your Course Outcomes first.", "info")
        return redirect(url_for("co_po.setup", subject_id=subject_id))
    return render_template("co_po/mapping.html", setup=doc, po_list=svc.PO_LIST, templates=list(svc.TEMPLATE_MAPPINGS.keys()))


@co_po_api_bp.route("/<subject_id>/mapping-cell", methods=["POST"])
@login_required
@role_required("teacher")
def mapping_cell(subject_id):
    data = request.get_json(silent=True) or {}
    co_id, po_id = data.get("co_id"), data.get("po_id")
    try:
        strength = int(data.get("strength", 0))
    except (TypeError, ValueError):
        strength = 0
    if not co_id or not po_id:
        return jsonify({"error": "Missing co_id or po_id."}), 400
    svc.set_mapping_cell(subject_id, co_id, po_id, strength)
    return jsonify({"saved": True})


@co_po_api_bp.route("/<subject_id>/apply-template", methods=["POST"])
@login_required
@role_required("teacher")
def apply_template(subject_id):
    data = request.get_json(silent=True) or {}
    template_name = data.get("template", "General")
    mapping_result = svc.apply_template(subject_id, template_name)
    log_action("co_po.template_applied", {"setup_id": subject_id, "template": template_name})
    return jsonify({"applied": True, "mapping": mapping_result})


# ═══════════════════════════════════════════════════════════════════
#  Step 3 — Map Assessments to COs
# ═══════════════════════════════════════════════════════════════════

@co_po_bp.route("/assessment-map/<subject_id>", methods=["GET", "POST"])
@login_required
@role_required("teacher")
def assessment_map(subject_id):
    doc = svc.get_setup(subject_id, current_user.id)
    if not doc:
        flash("That subject setup could not be found.", "danger")
        return redirect(url_for("co_po.index"))
    if not doc.get("po_mapping"):
        flash("Complete the CO-PO mapping grid first.", "info")
        return redirect(url_for("co_po.mapping", subject_id=subject_id))

    if request.method == "POST":
        assessments = svc.discover_assessments(doc["subject"])
        mapping_out = []
        for a in assessments:
            co_ids = request.form.getlist(f"co_ids__{a['assessment_id']}[]")
            if co_ids:
                mapping_out.append({**a, "co_ids": co_ids})
        svc.save_assessment_mapping(subject_id, mapping_out)
        log_action("co_po.assessments_mapped", {"setup_id": subject_id, "count": len(mapping_out)})
        return redirect(url_for("co_po.calculate", subject_id=subject_id))

    assessments = svc.discover_assessments(doc["subject"])
    existing = {m["assessment_id"]: m["co_ids"] for m in doc.get("assessment_mapping", [])}
    for a in assessments:
        a["selected_cos"] = existing.get(a["assessment_id"], [])
    return render_template("co_po/assessment_map.html", setup=doc, assessments=assessments)


# ═══════════════════════════════════════════════════════════════════
#  Step 4 — Calculate + Report
# ═══════════════════════════════════════════════════════════════════

@co_po_bp.route("/calculate/<subject_id>")
@login_required
@role_required("teacher")
def calculate(subject_id):
    doc = svc.get_setup(subject_id, current_user.id)
    if not doc:
        flash("That subject setup could not be found.", "danger")
        return redirect(url_for("co_po.index"))
    if not doc.get("assessment_mapping"):
        flash("Map at least one assessment to a Course Outcome first.", "info")
        return redirect(url_for("co_po.assessment_map", subject_id=subject_id))

    svc.calculate_attainment(doc)
    log_action("co_po.calculated", {"setup_id": subject_id})
    return redirect(url_for("co_po.report", subject_id=subject_id))


@co_po_bp.route("/report/<subject_id>")
@login_required
@role_required("teacher")
def report(subject_id):
    doc = svc.get_setup(subject_id, current_user.id)
    if not doc:
        flash("That subject setup could not be found.", "danger")
        return redirect(url_for("co_po.index"))
    result = svc.get_latest_result(subject_id)
    if not result:
        flash("Run the calculation first.", "info")
        return redirect(url_for("co_po.assessment_map", subject_id=subject_id))
    return render_template("co_po/report.html", setup=doc, result=result)


@co_po_bp.route("/report/<subject_id>/export-pdf")
@login_required
@role_required("teacher")
def export_pdf(subject_id):
    doc = svc.get_setup(subject_id, current_user.id)
    result = svc.get_latest_result(subject_id) if doc else None
    if not doc or not result:
        flash("Nothing to export yet — run the calculation first.", "danger")
        return redirect(url_for("co_po.index"))
    try:
        pdf_bytes = export_svc.render_pdf(doc, result)
    except Exception:  # noqa: BLE001
        logger.exception("CO-PO PDF export failed")
        flash("PDF export isn't available in this environment right now.", "warning")
        return redirect(url_for("co_po.report", subject_id=subject_id))
    log_action("co_po.exported_pdf", {"setup_id": subject_id})
    filename = f"CO_PO_Attainment_{doc['subject'].replace(' ', '_')}_{doc.get('semester', '')}.pdf"
    return send_file(io.BytesIO(pdf_bytes), mimetype="application/pdf", as_attachment=True, download_name=filename)


@co_po_bp.route("/report/<subject_id>/export-xlsx")
@login_required
@role_required("teacher")
def export_xlsx(subject_id):
    doc = svc.get_setup(subject_id, current_user.id)
    result = svc.get_latest_result(subject_id) if doc else None
    if not doc or not result:
        flash("Nothing to export yet — run the calculation first.", "danger")
        return redirect(url_for("co_po.index"))
    xlsx_bytes = export_svc.render_xlsx(doc, result)
    log_action("co_po.exported_xlsx", {"setup_id": subject_id})
    filename = f"CO_PO_Attainment_{doc['subject'].replace(' ', '_')}_{doc.get('semester', '')}.xlsx"
    return send_file(
        io.BytesIO(xlsx_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True, download_name=filename,
    )


# ═══════════════════════════════════════════════════════════════════
#  Department Summary
# ═══════════════════════════════════════════════════════════════════

@co_po_bp.route("/summary")
@login_required
@role_required("teacher")
def summary():
    # No separate HOD/admin role exists yet (tracked alongside the audit
    # log viewer's access-control question) — scoped to any teacher for now.
    rows = svc.department_summary()
    return render_template("co_po/summary.html", rows=rows)
