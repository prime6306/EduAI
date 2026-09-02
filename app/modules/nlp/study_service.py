"""
Orchestrates the Study Material Generator pipeline:
  generate_subtopics -> generate_explanations -> generate_summary
  -> get_youtube_videos -> get_google_article -> save to Mongo -> log to MLflow
"""
import time
from datetime import datetime

from bson import ObjectId
from bson.errors import InvalidId
from flask import current_app, render_template

from app.extensions import db, logger
from . import external_apis, prompts
from .llm_client import chat_completion, chat_json


def run_pipeline(user_id: str, topic: str, subject: str, branch: str, year: str) -> dict:
    start = time.time()

    subtopics_raw = chat_json(prompts.subtopics_prompt(topic, subject, branch, year))
    subtopics = subtopics_raw.get("subtopics", [])[:6]
    if not subtopics:
        raise ValueError("The AI didn't return any subtopics — try rephrasing the topic.")

    explanations = []
    youtube = []
    articles = []
    for st in subtopics:
        title = st.get("title", "Untitled")
        key_points = st.get("key_points", [])

        explanation_text = chat_completion(
            prompts.explanation_prompt(topic, title, key_points), max_tokens=900
        )
        explanations.append({"title": title, "explanation": explanation_text})

        search_query = f"{title} {subject}".strip()
        youtube.append({"title": title, "videos": external_apis.get_youtube_videos(search_query, 3)})
        articles.append({"title": title, "article": external_apis.get_google_article(search_query)})

    summary = chat_completion(
        prompts.summary_prompt(topic, subject, [s.get("title", "") for s in subtopics]),
        max_tokens=400,
    )

    processing_time = round(time.time() - start, 1)

    doc = {
        "user_id": user_id,
        "topic": topic,
        "subject": subject,
        "branch": branch,
        "year": year,
        "subtopics": subtopics,
        "explanations": explanations,
        "summary": summary,
        "youtube": youtube,
        "articles": articles,
        "processing_time_sec": processing_time,
        "created_at": datetime.utcnow(),
    }
    result = db.topic_pipelines.insert_one(doc)
    doc["_id"] = result.inserted_id

    _log_to_mlflow(topic, subject, processing_time)
    return doc


def _log_to_mlflow(topic: str, subject: str, processing_time: float) -> None:
    try:
        import mlflow
        mlflow.set_tracking_uri(current_app.config["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(current_app.config["MLFLOW_EXPERIMENT"])
        with mlflow.start_run(run_name="study_material_pipeline"):
            mlflow.log_param("topic", topic)
            mlflow.log_param("subject", subject)
            mlflow.log_metric("processing_time_sec", processing_time)
    except Exception:  # noqa: BLE001
        logger.info("MLflow not reachable — skipping study-material run log.")


def get_pipeline(pipeline_id: str, user_id: str) -> dict | None:
    try:
        oid = ObjectId(pipeline_id)
    except (InvalidId, TypeError):
        return None
    return db.topic_pipelines.find_one({"_id": oid, "user_id": user_id})


def list_notes(user_id: str, search: str = "") -> list[dict]:
    query = {"user_id": user_id}
    if search:
        query["$or"] = [
            {"topic": {"$regex": search, "$options": "i"}},
            {"subject": {"$regex": search, "$options": "i"}},
        ]
    return list(db.topic_pipelines.find(query).sort("created_at", -1))


def delete_note(pipeline_id: str, user_id: str) -> bool:
    try:
        oid = ObjectId(pipeline_id)
    except (InvalidId, TypeError):
        return False
    result = db.topic_pipelines.delete_one({"_id": oid, "user_id": user_id})
    return result.deleted_count > 0


def render_pdf(pipeline: dict) -> bytes:
    """Renders the print-optimised template through WeasyPrint. Imports are
    deferred so the whole app doesn't hard-fail to boot if WeasyPrint's
    native deps (cairo/pango) aren't installed in a given environment."""
    import markdown as md
    from weasyprint import HTML

    summary_html = md.markdown(pipeline.get("summary", ""))
    explanations_html = [
        md.markdown(e.get("explanation", "")) for e in pipeline.get("explanations", [])
    ]

    html_string = render_template(
        "study/pdf.html",
        pipeline=pipeline,
        summary_html=summary_html,
        explanations_html=explanations_html,
    )
    return HTML(string=html_string).write_pdf()
