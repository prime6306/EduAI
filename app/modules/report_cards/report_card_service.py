"""
Report cards are keyed to the attendance roster's `student_id` (not the
`users` collection's account id), since not every enrolled student
necessarily has a platform account yet. Where a score needs to be pulled
from account-scoped data (quiz results, dropout risk), we look up the
matching user account by its `student_id` field first and simply leave
that component blank if no account exists yet.
"""
import csv
import io
import os
import zipfile
from datetime import datetime

from bson import ObjectId
from bson.errors import InvalidId
from flask import current_app

from app.extensions import db, logger

DEFAULT_GRADING_SCHEME = [
    {"min": 90, "max": 100, "grade": "A+"},
    {"min": 80, "max": 89.99, "grade": "A"},
    {"min": 70, "max": 79.99, "grade": "B+"},
    {"min": 60, "max": 69.99, "grade": "B"},
    {"min": 50, "max": 59.99, "grade": "C"},
    {"min": 40, "max": 49.99, "grade": "D"},
    {"min": 0, "max": 39.99, "grade": "F"},
]
DEFAULT_WEIGHTAGES = {"ia": 40, "quiz": 30, "attendance": 30}


def create_batch(teacher_id, title, subject, semester, academic_year, components):
    students = list(db.students.find().sort("student_id", 1))
    student_records = [
        {
            "student_id": s["student_id"], "name": s["name"],
            "ia_marks": {}, "quiz_avg": None, "attendance_pct": None,
            "dropout_risk": None, "overall_score": None, "grade": None,
            "remark": "", "pdf_generated": False,
        }
        for s in students
    ]
    doc = {
        "teacher_id": teacher_id, "title": title, "subject": subject,
        "semester": semester, "academic_year": academic_year,
        "components": components,
        "weightages": dict(DEFAULT_WEIGHTAGES),
        "grading_scheme": DEFAULT_GRADING_SCHEME,
        "remark_mode": "template", "remark_template": "{name} performed {grade_word} this term.",
        "include_dropout_risk": False,
        "student_records": student_records,
        "status": "collecting",
        "created_at": datetime.utcnow(), "generated_at": None,
    }
    result = db.report_batches.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


def get(batch_id, teacher_id=None):
    try:
        oid = ObjectId(batch_id)
    except (InvalidId, TypeError):
        return None
    query = {"_id": oid}
    if teacher_id:
        query["teacher_id"] = teacher_id
    return db.report_batches.find_one(query)


def list_for_teacher(teacher_id):
    return list(db.report_batches.find({"teacher_id": teacher_id}).sort("created_at", -1))


def generate_template_csv(batch):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["student_id", "name"] + [c["name"] for c in batch["components"]])
    for r in batch["student_records"]:
        writer.writerow([r["student_id"], r["name"]] + [""] * len(batch["components"]))
    return buf.getvalue()


def import_csv(batch_id, teacher_id, csv_stream):
    batch = get(batch_id, teacher_id)
    if not batch:
        raise ValueError("Batch not found.")

    reader = csv.DictReader(io.StringIO(csv_stream.read().decode("utf-8")))
    component_names = {c["name"] for c in batch["components"]}
    records_by_id = {r["student_id"]: r for r in batch["student_records"]}

    updated, skipped = 0, 0
    for row in reader:
        sid = row.get("student_id", "").strip()
        record = records_by_id.get(sid)
        if not record:
            skipped += 1
            continue
        for comp in component_names:
            if comp in row and row[comp].strip():
                try:
                    record["ia_marks"][comp] = float(row[comp])
                except ValueError:
                    pass
        updated += 1

    db.report_batches.update_one(
        {"_id": batch["_id"]}, {"$set": {"student_records": batch["student_records"]}}
    )
    return {"updated": updated, "skipped": skipped}


def update_cell(batch_id, teacher_id, student_id, component, value):
    batch = get(batch_id, teacher_id)
    if not batch:
        return False
    for r in batch["student_records"]:
        if r["student_id"] == student_id:
            try:
                r["ia_marks"][component] = float(value) if value != "" else None
            except ValueError:
                return False
            db.report_batches.update_one(
                {"_id": batch["_id"]}, {"$set": {"student_records": batch["student_records"]}}
            )
            return True
    return False


def update_settings(batch_id, teacher_id, **fields):
    batch = get(batch_id, teacher_id)
    if not batch:
        return False
    db.report_batches.update_one({"_id": batch["_id"]}, {"$set": fields})
    return True


def _user_id_for_student(student_id):
    user = db.users.find_one({"student_id": student_id}, {"_id": 1})
    return str(user["_id"]) if user else None


def _grade_for_score(score, scheme):
    for tier in scheme:
        if tier["min"] <= score <= tier["max"]:
            return tier["grade"]
    return scheme[-1]["grade"] if scheme else "N/A"


def _remark_word(grade):
    return {
        "A+": "excellently", "A": "very well", "B+": "well", "B": "adequately",
        "C": "satisfactorily", "D": "below expectations", "F": "poorly",
    }.get(grade, "adequately")


def compute_and_generate(batch_id, teacher_id, weightages, grading_scheme, remark_mode,
                          remark_template, include_dropout_risk):
    batch = get(batch_id, teacher_id)
    if not batch:
        raise ValueError("Batch not found.")

    all_dates = {t.date() for t in db.attendance_logs.distinct("timestamp")}
    total_sessions = max(len(all_dates), 1)

    from app.modules.nlp.llm_client import chat_completion, LLMNotConfigured

    for record in batch["student_records"]:
        components = batch["components"]
        marks = record["ia_marks"]
        if components and marks:
            pct_values = [
                (marks[c["name"]] / c["max_marks"]) * 100
                for c in components if c["name"] in marks and c["max_marks"]
            ]
            ia_pct = round(sum(pct_values) / len(pct_values), 1) if pct_values else 0
        else:
            ia_pct = 0

        user_id = _user_id_for_student(record["student_id"])
        quiz_avg = None
        dropout_risk = None
        if user_id:
            quiz_results = list(db.quiz_results.find({"user_id": user_id}))
            if quiz_results:
                quiz_avg = round(sum(q["score_percent"] for q in quiz_results) / len(quiz_results), 1)
            if include_dropout_risk:
                latest = db.dropout_predictions.find_one({"user_id": user_id}, sort=[("created_at", -1)])
                dropout_risk = latest["risk_level"] if latest else None

        student_doc = db.students.find_one({"student_id": record["student_id"]})
        attendance_pct = (
            round(min(student_doc.get("total_attendance", 0) / total_sessions, 1) * 100, 1)
            if student_doc else 0
        )

        parts, weights = [], []
        if components:
            parts.append(ia_pct); weights.append(weightages.get("ia", 0))
        if quiz_avg is not None:
            parts.append(quiz_avg); weights.append(weightages.get("quiz", 0))
        parts.append(attendance_pct); weights.append(weightages.get("attendance", 0))

        total_weight = sum(weights) or 1
        overall_score = round(sum(p * w for p, w in zip(parts, weights)) / total_weight, 1)
        grade = _grade_for_score(overall_score, grading_scheme)

        if remark_mode == "ai":
            try:
                remark = chat_completion([
                    {"role": "system", "content": "Write one warm, specific, one-sentence report card remark. No markdown."},
                    {"role": "user", "content": (
                        f"Student {record['name']}: overall score {overall_score}%, grade {grade}, "
                        f"attendance {attendance_pct}%. Write the remark."
                    )},
                ], max_tokens=60)
            except (LLMNotConfigured, Exception):  # noqa: BLE001
                remark = remark_template.format(name=record["name"], grade=grade, grade_word=_remark_word(grade))
        elif remark_mode == "template":
            remark = remark_template.format(name=record["name"], grade=grade, grade_word=_remark_word(grade))
        else:
            remark = ""

        record.update({
            "quiz_avg": quiz_avg, "attendance_pct": attendance_pct, "dropout_risk": dropout_risk,
            "overall_score": overall_score, "grade": grade, "remark": remark,
        })

    db.report_batches.update_one(
        {"_id": batch["_id"]},
        {"$set": {
            "student_records": batch["student_records"], "weightages": weightages,
            "grading_scheme": grading_scheme, "remark_mode": remark_mode,
            "remark_template": remark_template, "include_dropout_risk": include_dropout_risk,
            "status": "generated", "generated_at": datetime.utcnow(),
        }},
    )
    _log_to_mlflow(batch["subject"], len(batch["student_records"]))
    return get(batch_id, teacher_id)


def _log_to_mlflow(subject, n_students):
    try:
        import mlflow
        mlflow.set_tracking_uri(current_app.config["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(current_app.config["MLFLOW_EXPERIMENT"])
        with mlflow.start_run(run_name="report_card_generation"):
            mlflow.log_param("subject", subject)
            mlflow.log_metric("students", n_students)
    except Exception:  # noqa: BLE001
        logger.info("MLflow not reachable - skipping report card run log.")


def _pdf_dir(batch_id):
    return os.path.join(current_app.config["UPLOAD_FOLDER"], "report_cards", batch_id)


def generate_single_pdf(batch, record):
    from weasyprint import HTML
    from flask import render_template

    html_string = render_template("report_cards/pdf.html", batch=batch, r=record)
    return HTML(string=html_string).write_pdf()


def generate_all_pdfs(batch_id, teacher_id):
    batch = get(batch_id, teacher_id)
    if not batch:
        raise ValueError("Batch not found.")

    dest_dir = _pdf_dir(batch_id)
    os.makedirs(dest_dir, exist_ok=True)

    count = 0
    for record in batch["student_records"]:
        if record.get("overall_score") is None:
            continue
        pdf_bytes = generate_single_pdf(batch, record)
        path = os.path.join(dest_dir, f"{record['student_id']}.pdf")
        with open(path, "wb") as f:
            f.write(pdf_bytes)
        record["pdf_generated"] = True
        count += 1

    db.report_batches.update_one(
        {"_id": batch["_id"]}, {"$set": {"student_records": batch["student_records"]}}
    )
    return count


def get_pdf_bytes(batch_id, teacher_id, student_id):
    batch = get(batch_id, teacher_id)
    if not batch:
        return None
    record = next((r for r in batch["student_records"] if r["student_id"] == student_id), None)
    if not record or record.get("overall_score") is None:
        return None

    path = os.path.join(_pdf_dir(batch_id), f"{student_id}.pdf")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return generate_single_pdf(batch, record)


def generate_zip(batch_id, teacher_id):
    batch = get(batch_id, teacher_id)
    if not batch:
        return None

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for record in batch["student_records"]:
            if record.get("overall_score") is None:
                continue
            pdf_bytes = get_pdf_bytes(batch_id, teacher_id, record["student_id"])
            if pdf_bytes:
                zf.writestr(f"{record['student_id']}_{record['name'].replace(' ', '_')}.pdf", pdf_bytes)
    buf.seek(0)
    return buf.getvalue()
