from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import tempfile

import features  # <-- your updated main.py module (renamed to features.py)

app = Flask(__name__)
CORS(app)

# Maximum upload size (200MB)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"txt", "pdf", "docx"}

# ------------------------------
# Helpers
# ------------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ------------------------------
# Root check
# ------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "API running", "version": "1.0"})


# ------------------------------
# FULL PIPELINE ENDPOINT
# ------------------------------
@app.route("/generate", methods=["POST"])
def generate():
    """
    Input JSON:
    {
        "topic": "...",
        "subject": "...",
        "branch": "...",
        "year": "..."
    }

    Returns:
        Full pipeline output (subtopics, explanations, MCQs, articles, MongoDB IDs)
    """
    try:
        data = request.json or {}

        required = ["topic", "subject", "branch", "year"]
        if not all(k in data for k in required):
            return jsonify({"error": "Missing required fields: topic, subject, branch, year"}), 400

        result = features.process_topic_pipeline(
            data["topic"],
            data["subject"],
            data["branch"],
            data["year"]
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------
# STUDENT ASSIGNMENT UPLOAD + PLAGIARISM
# ------------------------------
@app.route("/uploadassignment", methods=["POST"])
def upload_assignment():
    """
    Students upload multiple files.  
    Plagiarism is automatically detected using the features module.

    Example form-data:
    files: (upload multiple files)
    course_id: CS101
    assignment_id: A1
    """
    try:
        files = request.files.getlist("files")

        if not files:
            return jsonify({"error": "No files uploaded"}), 400

        # metadata optional
        course_id = request.form.get("course_id", "unknown_course")
        assignment_id = request.form.get("assignment_id", "assignment")

        temp_dir = tempfile.mkdtemp(prefix="assign_")

        student_texts = {}
        saved_files = []

        for f in files:
            filename = secure_filename(f.filename)
            if not filename or not allowed_file(filename):
                continue

            path = os.path.join(temp_dir, filename)
            f.save(path)

            # extract text
            text = features.read_text_from_file(path)
            roll = os.path.splitext(filename)[0]

            student_texts[roll] = text
            saved_files.append({"roll": roll, "filename": filename})

        # run plagiarism checker
        result = features.detect_plagiarism_from_texts(student_texts)

        # save report in DB
        report_id = features.save_pipeline_results(
            "plagiarism_reports",
            {
                "course_id": course_id,
                "assignment_id": assignment_id,
                "files": saved_files,
                "plagiarism_result": result
            }
        )

        return jsonify({
            "report_id": report_id,
            "suspicious_pairs": result,
            "uploaded_files": saved_files
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------
# Start server
# ------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
