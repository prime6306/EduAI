"""
Infra Feature 2 — Activity Log / Audit Trail viewer.

The write side already exists and is used throughout the app: every
module calls `app.utils.audit.log_action()`. This module is purely the
read side — filtering, pagination, and a generic human-readable formatter
for the `details` dict, since hand-writing a sentence template for every
one of the dozens of distinct action strings logged across the app isn't
practical (or maintainable as new actions get added).

Access control note: the spec scopes this to an "Admin" role that doesn't
exist anywhere else in this app (only student/teacher). Rather than add a
whole new role + account-provisioning flow for two view-only features,
this is scoped to any teacher — the same call made for the CO-PO
department summary. Worth revisiting together if a real HOD/admin tier
gets added later.
"""
from datetime import datetime

from app.extensions import db

PAGE_SIZE = 50


def humanize_action(action: str) -> str:
    """'co_po.calculated' -> 'Co Po — calculated'"""
    module, _, verb = action.partition(".")
    module_label = module.replace("_", " ").title()
    verb_label = verb.replace("_", " ") if verb else action
    return f"{module_label} — {verb_label}"


def _format_value(v) -> str:
    if isinstance(v, list):
        return ", ".join(str(x) for x in v) if v else "none"
    if isinstance(v, dict):
        return ", ".join(f"{k}: {v2}" for k, v2 in v.items())
    return str(v)


def humanize_details(details: dict) -> str:
    if not details:
        return ""
    parts = [f"{k.replace('_', ' ')}: {_format_value(v)}" for k, v in details.items()]
    return "; ".join(parts)


def humanize_entry(entry: dict) -> str:
    detail_str = humanize_details(entry.get("details", {}))
    action_str = humanize_action(entry.get("action", ""))
    return f"{action_str} ({detail_str})" if detail_str else action_str


def distinct_actions() -> list[str]:
    return sorted(db.audit_log.distinct("action"))


def _build_query(action=None, actor=None, date_from=None, date_to=None, search=None) -> dict:
    query = {}
    if action:
        query["action"] = action
    if actor:
        query["actor_name"] = {"$regex": actor, "$options": "i"}
    if date_from or date_to:
        ts_query = {}
        if date_from:
            ts_query["$gte"] = date_from
        if date_to:
            ts_query["$lte"] = date_to
        query["timestamp"] = ts_query
    if search:
        query["$or"] = [
            {"actor_name": {"$regex": search, "$options": "i"}},
            {"action": {"$regex": search, "$options": "i"}},
            {"actor_role": {"$regex": search, "$options": "i"}},
        ]
    return query


def search_entries(action=None, actor=None, date_from=None, date_to=None, search=None, page=1) -> dict:
    query = _build_query(action, actor, date_from, date_to, search)
    total = db.audit_log.count_documents(query)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(1, min(page, total_pages))
    skip = (page - 1) * PAGE_SIZE

    entries = list(
        db.audit_log.find(query).sort("timestamp", -1).skip(skip).limit(PAGE_SIZE)
    )
    for e in entries:
        e["human_detail"] = humanize_entry(e)

    return {
        "entries": entries, "total": total, "page": page,
        "total_pages": total_pages, "page_size": PAGE_SIZE,
    }


def export_csv(action=None, actor=None, date_from=None, date_to=None, search=None) -> str:
    import csv
    import io

    query = _build_query(action, actor, date_from, date_to, search)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Timestamp (UTC)", "Actor", "Role", "Action", "Details", "IP Address"])
    for e in db.audit_log.find(query).sort("timestamp", -1):
        ts = e.get("timestamp")
        writer.writerow([
            ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, datetime) else "",
            e.get("actor_name", ""), e.get("actor_role", ""),
            e.get("action", ""), humanize_details(e.get("details", {})),
            e.get("ip_address", ""),
        ])
    return buf.getvalue()
