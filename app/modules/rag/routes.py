import os
import uuid

from flask import Blueprint, render_template, request, jsonify, current_app, flash, redirect, url_for
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

from app.extensions import logger
from app.utils.audit import log_action
from app.utils import rate_limiter
from . import qa_service
from app.modules.nlp.llm_client import LLMNotConfigured

rag_bp = Blueprint("rag", __name__, url_prefix="/rag")

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@rag_bp.route("")
@login_required
def index():
    documents = qa_service.list_documents(current_user.id)
    return render_template(
        "rag/index.html", documents=documents,
        status=rate_limiter.get_status("rag_ask", current_user.id, current_user.role),
        usage_feature_label=rate_limiter.FEATURE_LABELS["rag_ask"],
    )


@rag_bp.route("/library")
@login_required
def library():
    documents = qa_service.list_documents(current_user.id)
    total_bytes = sum(d.get("file_size_bytes", 0) for d in documents)
    return render_template("rag/library.html", documents=documents, total_bytes=total_bytes)


@rag_bp.route("/api/rag/upload", methods=["POST"])
@login_required
def upload():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "No file provided."}), 400

    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported file type. Upload PDF, DOCX, TXT, or Markdown."}), 400

    tmp_name = f"{uuid.uuid4().hex}{ext}"
    tmp_path = os.path.join(current_app.config["UPLOAD_FOLDER"], tmp_name)
    file.save(tmp_path)
    file_size = os.path.getsize(tmp_path)

    try:
        doc = qa_service.upload_document(current_user.id, filename, tmp_path, file_size)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:  # noqa: BLE001
        logger.exception("RAG upload/indexing failed")
        return jsonify({"error": "Couldn't process that file. Please try again."}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    log_action("rag.document_uploaded", {"filename": filename, "pdf_id": doc["pdf_id"]})
    return jsonify({
        "pdf_id": doc["pdf_id"],
        "filename": doc["filename"],
        "chunk_count": doc["chunk_count"],
    })


@rag_bp.route("/api/rag/ask", methods=["POST"])
@login_required
def ask():
    data = request.get_json(silent=True) or {}
    pdf_id = data.get("pdf_id")
    question = (data.get("question") or "").strip()

    if not pdf_id or not question:
        return jsonify({"error": "pdf_id and question are required."}), 400

    allowed, status = rate_limiter.check_and_increment("rag_ask", current_user.id, current_user.role)
    if not allowed:
        return jsonify({"error": rate_limiter.limit_message("rag_ask", status)}), 429

    try:
        result = qa_service.ask_question(current_user.id, pdf_id, question)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except LLMNotConfigured as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:  # noqa: BLE001
        logger.exception("RAG ask failed")
        return jsonify({"error": "Something went wrong answering that question."}), 500

    return jsonify(result)


@rag_bp.route("/api/rag/pdf/<pdf_id>", methods=["DELETE"])
@login_required
def delete_pdf(pdf_id):
    ok = qa_service.delete_document(pdf_id, current_user.id)
    if ok:
        log_action("rag.document_deleted", {"pdf_id": pdf_id})
    return jsonify({"deleted": ok})
