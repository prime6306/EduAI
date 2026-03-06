# flask_app.py
import io
from flask import Flask, request, jsonify
from flask_cors import CORS

from embedder import process_pdf_file
from retriever import search_query

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return "🚀 Embedding & Retrieval API running"


# ---------------------------------------
# 1) Upload PDF -> generate embeddings -> save to Mongo
# ---------------------------------------
# Expect: multipart/form-data with 'file' and optional 'pdf_name'
@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Missing file"}), 400

        file = request.files["file"]
        pdf_name = request.form.get("pdf_name", file.filename)
        # Read file into BytesIO so PyPDF2 can use it
        file_bytes = io.BytesIO(file.read())
        result = process_pdf_file(file_bytes, pdf_name=pdf_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------
# 2) Search -> retrieve chunks from Mongo -> generate answer
# ---------------------------------------
# Expect JSON: { "input": "...", "pdf_id": optional, "k": optional }
@app.route("/search", methods=["POST"])
def search():
    try:
        payload = request.get_json()
        if not payload or "input" not in payload:
            return jsonify({"error": "Missing 'input' in JSON body"}), 400

        query = payload["input"]
        pdf_id = payload.get("pdf_id", None)
        k = int(payload.get("k", 5))

        response = search_query(query, pdf_id=pdf_id, top_k=k)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # for dev only. For production use gunicorn/uvicorn + proper workers
    app.run(debug=False, host="0.0.0.0", port=5000)
