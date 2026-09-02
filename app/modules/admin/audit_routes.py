from datetime import datetime, timedelta

from flask import Blueprint, render_template, request, Response
from flask_login import login_required

from app.auth.utils import role_required
from . import audit_service as svc

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


def _parse_date(value: str, end_of_day: bool = False):
    if not value:
        return None
    try:
        dt = datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None
    return dt + timedelta(days=1, microseconds=-1) if end_of_day else dt


def _filters_from_request():
    return {
        "action": request.args.get("action") or None,
        "actor": request.args.get("actor") or None,
        "search": request.args.get("q") or None,
        "date_from": _parse_date(request.args.get("date_from", "")),
        "date_to": _parse_date(request.args.get("date_to", ""), end_of_day=True),
    }


@admin_bp.route("/audit-log")
@login_required
@role_required("teacher")
def audit_log():
    filters = _filters_from_request()
    page = request.args.get("page", 1, type=int)
    result = svc.search_entries(page=page, **filters)
    return render_template(
        "admin/audit_log.html",
        result=result, filters=request.args,
        actions=svc.distinct_actions(),
    )


@admin_bp.route("/audit-log/export")
@login_required
@role_required("teacher")
def export_audit_log():
    filters = _filters_from_request()
    csv_data = svc.export_csv(**filters)
    return Response(
        csv_data, mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=audit_log_export.csv"},
    )
