"""
CO-PO Attainment Mapping (Priority Feature) — NBA/NAAC accreditation
reporting. Course Outcomes (COs) map to Program Outcomes (PO1-PO12) with a
strength of 1-3; CO attainment comes from real scores already sitting in
quiz_results, test_attempts, and report_batches (IA/assignment marks) —
nothing needs to be re-entered by the teacher except which assessment
measures which CO.
"""
from datetime import datetime

from bson import ObjectId
from bson.errors import InvalidId

from app.extensions import db, logger

PO_LIST = [
    ("PO1", "Engineering Knowledge"), ("PO2", "Problem Analysis"),
    ("PO3", "Design/Development of Solutions"), ("PO4", "Conduct Investigations of Complex Problems"),
    ("PO5", "Modern Tool Usage"), ("PO6", "The Engineer and Society"),
    ("PO7", "Environment and Sustainability"), ("PO8", "Ethics"),
    ("PO9", "Individual and Team Work"), ("PO10", "Communication"),
    ("PO11", "Project Management and Finance"), ("PO12", "Life-long Learning"),
]
PO_NAMES = dict(PO_LIST)
DEFAULT_TARGET = 2.5
PO_THRESHOLD = 1.5  # institutional minimum — a PO below this is flagged red

# Starting points for "Apply Template" on the mapping grid — a convenience
# default the teacher is expected to adjust, not a rigorous accreditation
# mapping.
TEMPLATE_MAPPINGS = {
    "Core ECE": {"PO1": 3, "PO2": 2, "PO3": 2, "PO4": 1, "PO5": 1, "PO12": 1},
    "Core CS":  {"PO1": 3, "PO2": 2, "PO3": 3, "PO5": 2, "PO12": 1},
    "General":  {"PO1": 2, "PO2": 1, "PO12": 1},
}


# ═══════════════════════════════════════════════════════════════════
#  Setup CRUD
# ═══════════════════════════════════════════════════════════════════

def create_setup(teacher_id: str, subject: str, semester: str, academic_year: str) -> dict:
    doc = {
        "teacher_id": teacher_id, "subject": subject, "semester": semester,
        "academic_year": academic_year, "course_outcomes": [],
        "po_mapping": {}, "assessment_mapping": [],
        "created_at": datetime.utcnow(), "updated_at": datetime.utcnow(),
    }
    result = db.co_po_setups.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


def list_setups(teacher_id: str) -> list[dict]:
    setups = list(db.co_po_setups.find({"teacher_id": teacher_id}).sort("created_at", -1))
    for s in setups:
        s["status"] = setup_status(s)
    return setups


def get_setup(setup_id: str, teacher_id: str | None = None) -> dict | None:
    try:
        oid = ObjectId(setup_id)
    except (InvalidId, TypeError):
        return None
    query = {"_id": oid}
    if teacher_id is not None:
        query["teacher_id"] = teacher_id
    return db.co_po_setups.find_one(query)


def delete_setup(setup_id: str, teacher_id: str) -> bool:
    try:
        oid = ObjectId(setup_id)
    except (InvalidId, TypeError):
        return False
    db.co_po_results.delete_many({"setup_id": setup_id})
    result = db.co_po_setups.delete_one({"_id": oid, "teacher_id": teacher_id})
    return result.deleted_count > 0


def setup_status(setup: dict) -> str:
    if not setup.get("course_outcomes"):
        return "Draft"
    if not setup.get("po_mapping"):
        return "COs Defined"
    if not setup.get("assessment_mapping"):
        return "Mapped to POs"
    if db.co_po_results.count_documents({"setup_id": str(setup["_id"])}) == 0:
        return "Assessments Mapped"
    return "Calculated"


# ═══════════════════════════════════════════════════════════════════
#  Step 1 — Course Outcomes
# ═══════════════════════════════════════════════════════════════════

def save_course_outcomes(setup_id: str, cos: list[dict]) -> None:
    """cos: [{"id": "CO1", "description": "...", "target_attainment": 2.5}, ...]"""
    db.co_po_setups.update_one(
        {"_id": ObjectId(setup_id)},
        {"$set": {"course_outcomes": cos, "updated_at": datetime.utcnow()}},
    )


# ═══════════════════════════════════════════════════════════════════
#  Step 2 — CO-PO Mapping grid
# ═══════════════════════════════════════════════════════════════════

def set_mapping_cell(setup_id: str, co_id: str, po_id: str, strength: int) -> None:
    field = f"po_mapping.{co_id}.{po_id}"
    update = {"$set": {"updated_at": datetime.utcnow()}}
    if strength <= 0:
        update["$unset"] = {field: ""}
    else:
        update["$set"][field] = strength
    db.co_po_setups.update_one({"_id": ObjectId(setup_id)}, update)


def apply_template(setup_id: str, template_name: str) -> dict:
    setup = get_setup(setup_id)
    if not setup:
        return {}
    template = TEMPLATE_MAPPINGS.get(template_name, TEMPLATE_MAPPINGS["General"])
    mapping = {co["id"]: dict(template) for co in setup.get("course_outcomes", [])}
    db.co_po_setups.update_one(
        {"_id": ObjectId(setup_id)}, {"$set": {"po_mapping": mapping, "updated_at": datetime.utcnow()}}
    )
    return mapping


# ═══════════════════════════════════════════════════════════════════
#  Step 3 — Map Assessments to COs
# ═══════════════════════════════════════════════════════════════════

def discover_assessments(subject: str) -> list[dict]:
    """Everything gradeable already in the system for this subject —
    nothing needs to be entered manually, only tagged with which CO(s)
    it measures."""
    assessments = []
    for topic in db.quiz_questions.distinct("topic", {"subject": subject}):
        assessments.append({"assessment_type": "quiz", "assessment_id": topic, "label": f"Quiz: {topic}"})
    for t in db.tests.find({"subject": subject}, {"title": 1}):
        assessments.append({
            "assessment_type": "test", "assessment_id": str(t["_id"]),
            "label": f"Test: {t.get('title', 'Untitled Test')}",
        })
    for b in db.report_batches.find({"subject": subject}, {"title": 1, "components": 1}):
        for comp in b.get("components", []):
            assessments.append({
                "assessment_type": "grade_component", "assessment_id": f"{b['_id']}:{comp['name']}",
                "label": f"{b.get('title', 'Grade Batch')} — {comp['name']}",
            })
    return assessments


def save_assessment_mapping(setup_id: str, mapping: list[dict]) -> None:
    """mapping: [{"assessment_type", "assessment_id", "label", "co_ids": [...]}]
    — entries with no CO selected are dropped."""
    mapping = [m for m in mapping if m.get("co_ids")]
    db.co_po_setups.update_one(
        {"_id": ObjectId(setup_id)}, {"$set": {"assessment_mapping": mapping, "updated_at": datetime.utcnow()}}
    )


# ═══════════════════════════════════════════════════════════════════
#  Step 4 — Calculate Attainment
# ═══════════════════════════════════════════════════════════════════

def _score_band(pct: float) -> int:
    if pct >= 80:
        return 3
    if pct >= 65:
        return 2
    if pct >= 50:
        return 1
    return 0


def _assessment_average(m: dict) -> float | None:
    a_type, a_id = m["assessment_type"], m["assessment_id"]
    scores = []

    if a_type == "quiz":
        scores = [r.get("score_percent", 0) for r in db.quiz_results.find({"topic": a_id})]

    elif a_type == "test":
        for a in db.test_attempts.find({"test_id": a_id, "grading_status": "complete"}):
            total = a.get("total_marks") or 0
            if total:
                scores.append((a.get("score", 0) / total) * 100)

    elif a_type == "grade_component":
        batch_id, _, comp_name = a_id.partition(":")
        try:
            batch = db.report_batches.find_one({"_id": ObjectId(batch_id)})
        except (InvalidId, TypeError):
            batch = None
        if batch:
            max_marks = next((c["max_marks"] for c in batch.get("components", []) if c["name"] == comp_name), 100)
            for r in batch.get("student_records", []):
                mark = r.get("ia_marks", {}).get(comp_name)
                if mark is not None and max_marks:
                    scores.append((mark / max_marks) * 100)

    return round(sum(scores) / len(scores), 1) if scores else None


def calculate_attainment(setup: dict) -> dict:
    setup_id = str(setup["_id"])

    assessment_scores = {}
    for m in setup.get("assessment_mapping", []):
        assessment_scores[m["assessment_id"]] = {
            "label": m["label"], "average": _assessment_average(m), "co_ids": m["co_ids"],
        }

    co_attainment = []
    co_band_lookup = {}
    for co in setup.get("course_outcomes", []):
        co_id = co["id"]
        mapped = [v for v in assessment_scores.values() if co_id in v["co_ids"] and v["average"] is not None]
        attainment = round(sum(_score_band(v["average"]) for v in mapped) / len(mapped), 2) if mapped else None
        co_band_lookup[co_id] = attainment
        target = co.get("target_attainment", DEFAULT_TARGET)
        co_attainment.append({
            "co_id": co_id, "description": co.get("description", ""),
            "target": target, "attainment": attainment,
            "met": bool(attainment is not None and attainment >= target),
            "mapped_assessments": [{"label": v["label"], "average": v["average"]} for v in mapped],
        })

    po_mapping = setup.get("po_mapping", {})
    po_attainment = []
    for po_id, po_name in PO_LIST:
        weighted_sum = weight_total = target_weighted = target_total = 0.0
        for co in setup.get("course_outcomes", []):
            co_id = co["id"]
            strength = po_mapping.get(co_id, {}).get(po_id, 0)
            if not strength:
                continue
            attainment = co_band_lookup.get(co_id)
            if attainment is not None:
                weighted_sum += attainment * strength
                weight_total += strength
            target_weighted += co.get("target_attainment", DEFAULT_TARGET) * strength
            target_total += strength
        po_att = round(weighted_sum / weight_total, 2) if weight_total else None
        po_target = round(target_weighted / target_total, 2) if target_total else None
        po_attainment.append({
            "po_id": po_id, "po_name": po_name, "attainment": po_att, "target": po_target,
            "level": "red" if (po_att is not None and po_att < PO_THRESHOLD) else "ok",
        })

    gap_analysis = []
    for co in co_attainment:
        if co["attainment"] is not None and co["attainment"] < co["target"]:
            weakest = min(co["mapped_assessments"], key=lambda a: a["average"]) if co["mapped_assessments"] else None
            rec = f"{co['co_id']} is {round(co['target'] - co['attainment'], 2)} below target."
            if weakest:
                rec += f" Lowest scores on {weakest['label']} ({weakest['average']}%). Consider a revision session and a targeted follow-up quiz."
            gap_analysis.append({"type": "CO", "id": co["co_id"], "gap": round(co["target"] - co["attainment"], 2), "recommendation": rec})
    for po in po_attainment:
        if po["attainment"] is not None and po["attainment"] < PO_THRESHOLD:
            gap_analysis.append({
                "type": "PO", "id": po["po_id"],
                "gap": round(PO_THRESHOLD - po["attainment"], 2),
                "recommendation": f"{po['po_id']} ({po['po_name']}) is below the institutional threshold of {PO_THRESHOLD}.",
            })

    result_doc = {
        "setup_id": setup_id, "calculated_at": datetime.utcnow(),
        "co_attainment": co_attainment, "po_attainment": po_attainment, "gap_analysis": gap_analysis,
    }
    db.co_po_results.delete_many({"setup_id": setup_id})
    db.co_po_results.insert_one(result_doc)
    return result_doc


def get_latest_result(setup_id: str) -> dict | None:
    return db.co_po_results.find_one({"setup_id": setup_id}, sort=[("calculated_at", -1)])


# ═══════════════════════════════════════════════════════════════════
#  Department Summary
# ═══════════════════════════════════════════════════════════════════

def department_summary() -> list[dict]:
    rows = []
    for setup in db.co_po_setups.find():
        result = get_latest_result(str(setup["_id"]))
        po_values = [po["attainment"] for po in result["po_attainment"] if po["attainment"] is not None] if result else []
        avg_po = round(sum(po_values) / len(po_values), 2) if po_values else None

        teacher = None
        try:
            teacher = db.users.find_one({"_id": ObjectId(setup["teacher_id"])})
        except (InvalidId, TypeError):
            pass

        rows.append({
            "setup_id": str(setup["_id"]), "subject": setup["subject"],
            "semester": setup.get("semester", ""), "academic_year": setup.get("academic_year", ""),
            "teacher_name": teacher.get("name", "Unknown") if teacher else "Unknown",
            "avg_po_attainment": avg_po, "status": setup_status(setup),
        })

    rows.sort(key=lambda r: (r["avg_po_attainment"] is None, r["avg_po_attainment"] if r["avg_po_attainment"] is not None else 0))
    return rows
