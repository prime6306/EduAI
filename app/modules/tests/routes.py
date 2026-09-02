from datetime import datetime

from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, Response
from flask_login import login_required, current_user

from app.auth.utils import role_required
from app.extensions import db, logger
from app.utils.audit import log_action
from app.modules.nlp.llm_client import LLMNotConfigured
from . import test_service as svc
from . import attempt_service as attempt_svc

tests_bp = Blueprint("tests", __name__, url_prefix="/tests")
tests_api_bp = Blueprint("tests_api", __name__, url_prefix="/api/tests")


def _parse_dt(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


@tests_bp.route("")
@login_required
def index():
    if current_user.is_teacher:
        tests = svc.list_for_teacher(current_user.id)
        return render_template("tests/list_teacher.html", tests=tests)

    tests = svc.list_for_student(current_user.student_id) if current_user.student_id else []
    attempts = {
        a["test_id"]: a for a in db.test_attempts.find({"student_id": current_user.student_id})
    } if current_user.student_id else {}
    return render_template("tests/list_student.html", tests=tests, attempts=attempts)


@tests_bp.route("/create", methods=["GET", "POST"])
@login_required
@role_required("teacher")
def create():
    if request.method == "POST":
        title = (request.form.get("title") or "").strip()
        subject = (request.form.get("subject") or "").strip()
        instructions = request.form.get("instructions", "")
        time_limit_raw = request.form.get("time_limit_minutes", "")
        time_limit = int(time_limit_raw) if time_limit_raw.isdigit() else None

        if not title or not subject:
            flash("Title and subject are required.", "danger")
            return redirect(url_for("tests.create"))

        doc = svc.create_draft(current_user.id, title, subject, instructions, time_limit)
        log_action("tests.draft_created", {"test_id": str(doc["_id"])})
        return redirect(url_for("tests.build", test_id=str(doc["_id"])))

    return render_template("tests/create.html")


@tests_bp.route("/<test_id>/build")
@login_required
@role_required("teacher")
def build(test_id):
    doc = svc.get(test_id, current_user.id)
    if not doc:
        flash("Test not found.", "danger")
        return redirect(url_for("tests.index"))
    students = list(db.students.find({}, {"student_id": 1, "name": 1}))
    return render_template("tests/build.html", doc=doc, students=students)


@tests_bp.route("/<test_id>/preview")
@login_required
@role_required("teacher")
def preview(test_id):
    doc = svc.get(test_id, current_user.id)
    if not doc:
        flash("Test not found.", "danger")
        return redirect(url_for("tests.index"))
    return render_template("tests/preview.html", doc=doc)


@tests_bp.route("/take/<test_id>")
@login_required
def take(test_id):
    if current_user.is_teacher:
        return redirect(url_for("tests.preview", test_id=test_id))

    doc = svc.get(test_id)
    if not doc or doc["status"] != "published":
        flash("This test isn't available.", "danger")
        return redirect(url_for("tests.index"))
    if not svc._is_available_now(doc):
        flash("This test isn't open right now.", "warning")
        return redirect(url_for("tests.index"))
    assigned = doc.get("assigned_to") or []
    if assigned and current_user.student_id not in assigned:
        flash("This test isn't assigned to you.", "danger")
        return redirect(url_for("tests.index"))

    attempt = attempt_svc.get_or_create_attempt(test_id, current_user.student_id)
    if attempt.get("submitted_at"):
        return redirect(url_for("tests.student_result", attempt_id=str(attempt["_id"])))

    questions = doc["questions"]
    if doc.get("shuffle_questions"):
        import random
        questions = questions[:]
        random.shuffle(questions)
    if doc.get("shuffle_options"):
        import random
        shuffled = []
        for q in questions:
            if q["type"] == "mcq" and q.get("options"):
                q = dict(q)
                q["options"] = random.sample(q["options"], len(q["options"]))
            shuffled.append(q)
        questions = shuffled

    saved = attempt.get("answers", [])
    saved_answers = {a["question_id"]: a["response"] for a in saved} if isinstance(saved, list) else {}

    return render_template(
        "tests/take.html", doc=doc, attempt_id=str(attempt["_id"]),
        questions=svc.questions_for_taking(questions),
        saved_answers=saved_answers,
    )


@tests_bp.route("/results/<attempt_id>")
@login_required
def student_result(attempt_id):
    student_id = None if current_user.is_teacher else current_user.student_id
    attempt = attempt_svc.get_attempt(attempt_id, student_id)
    if not attempt:
        flash("Result not found.", "danger")
        return redirect(url_for("tests.index"))
    test = svc.get(attempt["test_id"])
    return render_template("tests/student_result.html", attempt=attempt, test=test)


@tests_bp.route("/<test_id>/results")
@login_required
@role_required("teacher")
def results(test_id):
    doc = svc.get(test_id, current_user.id)
    if not doc:
        flash("Test not found.", "danger")
        return redirect(url_for("tests.index"))
    attempts = attempt_svc.list_attempts_for_test(test_id)
    stats = attempt_svc.get_class_stats(test_id)
    students = {s["student_id"]: s["name"] for s in db.students.find({}, {"student_id": 1, "name": 1})}
    return render_template("tests/results.html", doc=doc, attempts=attempts, stats=stats, students=students)


@tests_bp.route("/<test_id>/results/export")
@login_required
@role_required("teacher")
def export_results(test_id):
    csv_data = attempt_svc.export_results_csv(test_id, current_user.id)
    return Response(
        csv_data, mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=test_{test_id}_results.csv"},
    )


@tests_bp.route("/<test_id>/review")
@login_required
@role_required("teacher")
def review(test_id):
    doc = svc.get(test_id, current_user.id)
    if not doc:
        flash("Test not found.", "danger")
        return redirect(url_for("tests.index"))

    queue = attempt_svc.get_review_queue(test_id)
    attempt_id = request.args.get("attempt_id")

    students = {s["student_id"]: s["name"] for s in db.students.find({}, {"student_id": 1, "name": 1})}
    total_submissions = db.test_attempts.count_documents({"test_id": test_id, "submitted_at": {"$ne": None}})
    graded_count = total_submissions - len(queue)

    current_attempt = None
    descriptive_answers = []
    if attempt_id:
        current_attempt = attempt_svc.get_attempt(attempt_id)
        if current_attempt:
            questions_by_id = {q["id"]: q for q in doc["questions"]}
            for a in current_attempt["answers"]:
                q = questions_by_id.get(a["question_id"])
                if q and q["type"] in ("short", "long"):
                    descriptive_answers.append({"question": q, "answer": a})

    return render_template(
        "tests/review.html", doc=doc, queue=queue, students=students,
        graded_count=graded_count, total_submissions=total_submissions,
        current_attempt=current_attempt, descriptive_answers=descriptive_answers,
    )


@tests_bp.route("/<test_id>/delete", methods=["POST"])
@login_required
@role_required("teacher")
def delete(test_id):
    ok = svc.delete_test(test_id, current_user.id)
    if ok:
        log_action("tests.deleted", {"test_id": test_id})
        flash("Test deleted.", "success")
    return redirect(url_for("tests.index"))


# ═══════════════════════════════════════════════════════════════════
#  API
# ═══════════════════════════════════════════════════════════════════

@tests_api_bp.route("/<test_id>/publish", methods=["POST"])
@login_required
@role_required("teacher")
def publish(test_id):
    data = request.get_json(silent=True) or {}
    fields = {}
    if "available_from" in data:
        fields["available_from"] = _parse_dt(data["available_from"])
    if "available_until" in data:
        fields["available_until"] = _parse_dt(data["available_until"])
    if "assigned_to" in data:
        fields["assigned_to"] = data["assigned_to"]
    if "shuffle_questions" in data:
        fields["shuffle_questions"] = bool(data["shuffle_questions"])
    if "shuffle_options" in data:
        fields["shuffle_options"] = bool(data["shuffle_options"])
    if fields:
        svc.update_metadata(test_id, current_user.id, **fields)

    ok, error = svc.publish(test_id, current_user.id)
    if not ok:
        return jsonify({"error": error}), 400
    log_action("tests.published", {"test_id": test_id})
    return jsonify({"published": True})


@tests_api_bp.route("/<test_id>/questions/generate", methods=["POST"])
@login_required
@role_required("teacher")
def generate_questions(test_id):
    data = request.get_json(silent=True) or {}
    doc = svc.get(test_id, current_user.id)
    if not doc:
        return jsonify({"error": "Test not found."}), 404

    topic = data.get("topic", doc["subject"])
    n = int(data.get("n", 5))
    q_type = data.get("q_type", "mcq")

    try:
        candidates = svc.generate_ai_questions(doc["subject"], topic, n, q_type)
    except LLMNotConfigured as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:  # noqa: BLE001
        logger.exception("Test question generation failed")
        return jsonify({"error": "Couldn't generate questions. Please try again."}), 500

    return jsonify({"candidates": candidates})


@tests_api_bp.route("/<test_id>/questions/add", methods=["POST"])
@login_required
@role_required("teacher")
def add_question(test_id):
    data = request.get_json(silent=True) or {}
    question = data.get("question", {})
    if not question.get("text") or not question.get("type"):
        return jsonify({"error": "Question text and type are required."}), 400

    added = svc.add_question(test_id, current_user.id, question)
    if not added:
        return jsonify({"error": "Test not found."}), 404
    return jsonify({"question": added})


@tests_api_bp.route("/<test_id>/questions/from-bank", methods=["POST"])
@login_required
@role_required("teacher")
def add_from_bank(test_id):
    from bson import ObjectId

    data = request.get_json(silent=True) or {}
    question_ids = data.get("question_ids", [])

    added = []
    for qid in question_ids:
        try:
            oid = ObjectId(qid)
        except Exception:  # noqa: BLE001
            continue
        bank_q = db.question_bank.find_one({"_id": oid, "teacher_id": current_user.id})
        if not bank_q:
            continue
        q_type = "mcq" if bank_q["type"] == "mcq" else "long" if bank_q["type"] == "long" else "short"
        question = {
            "text": bank_q["text"], "type": q_type, "marks": bank_q["marks"],
            "correct_answer": bank_q.get("answer", ""), "explanation": "",
        }
        if q_type == "mcq":
            question["options"] = bank_q.get("options", [])
            question["correct_answer"] = (
                bank_q["options"][bank_q["correct_index"]] if bank_q.get("options") else ""
            )
        result = svc.add_question(test_id, current_user.id, question)
        if result:
            added.append(result)

    return jsonify({"added": added})


@tests_api_bp.route("/<test_id>/questions/delete", methods=["POST"])
@login_required
@role_required("teacher")
def delete_question(test_id):
    data = request.get_json(silent=True) or {}
    ok = svc.delete_question(test_id, current_user.id, data.get("question_id"))
    return jsonify({"deleted": ok})


@tests_api_bp.route("/<test_id>/questions/reorder", methods=["POST"])
@login_required
@role_required("teacher")
def reorder_questions(test_id):
    data = request.get_json(silent=True) or {}
    ok = svc.reorder_questions(test_id, current_user.id, data.get("ordered_ids", []))
    return jsonify({"reordered": ok})


@tests_api_bp.route("/autosave", methods=["POST"])
@login_required
def autosave():
    data = request.get_json(silent=True) or {}
    attempt_id = data.get("attempt_id")
    answers = data.get("answers", {})
    ok = attempt_svc.autosave(attempt_id, current_user.student_id, answers)
    return jsonify({"saved": ok})


@tests_api_bp.route("/submit", methods=["POST"])
@login_required
def submit():
    data = request.get_json(silent=True) or {}
    attempt_id = data.get("attempt_id")
    answers = data.get("answers", {})
    time_taken_sec = int(data.get("time_taken_sec") or 0)

    try:
        attempt = attempt_svc.submit(attempt_id, current_user.student_id, answers, time_taken_sec)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    log_action("tests.submitted", {"attempt_id": attempt_id, "score": attempt["score"]})
    return jsonify({"attempt_id": str(attempt["_id"])})


@tests_api_bp.route("/grade", methods=["POST"])
@login_required
@role_required("teacher")
def grade():
    data = request.get_json(silent=True) or {}
    ok = attempt_svc.grade_answer(
        data.get("test_id"), current_user.id, data.get("attempt_id"),
        data.get("question_id"), int(data.get("marks_awarded", 0)), data.get("comment", ""),
    )
    if ok:
        log_action("tests.graded", {"attempt_id": data.get("attempt_id"), "question_id": data.get("question_id")})
    return jsonify({"graded": ok})
