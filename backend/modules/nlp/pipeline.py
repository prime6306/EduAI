"""
modules/nlp/pipeline.py
All NLP features powered by Groq (Llama 3):
  - Subtopic generation
  - Detailed explanations
  - Summary
  - MCQ generation
  - Plagiarism detection (n-gram + embedding)
  - Study planner
  - Hallucination scoring
  - YouTube & Google search
"""
import json
import re
import time
from datetime import datetime
import docx
import PyPDF2
import os


import nltk
import numpy as np
from groq import Groq
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def _embedder() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(get_settings().embedding_model)
    return _embed_model
  
from sklearn.feature_extraction.text import CountVectorizer

from backend.config import get_settings
from backend.db.mongodb import get_sync_db

# ─── NLTK bootstrap ───────────────────────────────────────
for resource in ["punkt", "stopwords", "wordnet"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# ─── Singleton models ──────────────────────────────────────
_groq_client: Groq | None = None
_embed_model: SentenceTransformer | None = None


def _groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=get_settings().groq_api_key)
    return _groq_client


def _embedder() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(get_settings().embedding_model)
    return _embed_model


def _chat(prompt: str, max_tokens: int = 1000, json_mode: bool = False) -> str:
    """Call Groq with optional JSON mode."""
    kwargs: dict = dict(
        model=get_settings().groq_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = _groq().chat.completions.create(**kwargs)
    return resp.choices[0].message.content.strip()


# ═══════════════════════════════════════════════════════════
# 1. SUBTOPIC GENERATION
# ═══════════════════════════════════════════════════════════

def generate_subtopics(topic: str, subject: str, branch: str, year: str) -> dict:
    prompt = f"""You are an expert university professor for {branch} engineering students ({year}).
Generate exactly 6 subtopics for the topic "{topic}" in subject "{subject}".

Return ONLY valid JSON in this exact format:
{{
  "subtopics": [
    {{"title": "Subtopic Name", "key_points": ["point1", "point2", "point3"]}},
    ...
  ]
}}"""
    raw = _chat(prompt, max_tokens=800, json_mode=True)
    data = json.loads(raw)
    result = {}
    for item in data.get("subtopics", []):
        result[item["title"]] = item.get("key_points", [])
    return result


# ═══════════════════════════════════════════════════════════
# 2. EXPLANATIONS
# ═══════════════════════════════════════════════════════════

def generate_explanations(subtopics: dict, year: str, branch: str) -> dict:
    explanations = {}
    for subtopic, points in subtopics.items():
        points_text = "\n".join(f"- {p}" for p in points) if points else ""
        prompt = f"""You are a {branch} professor teaching {year} students.
Write a detailed, clear educational explanation (minimum 200 words) for:

Topic: {subtopic}
Key Points:
{points_text}

Include real-world examples where helpful. Write in plain text, no markdown."""
        explanations[subtopic] = _chat(prompt, max_tokens=900)
    return explanations


# ═══════════════════════════════════════════════════════════
# 3. SUMMARY
# ═══════════════════════════════════════════════════════════

def generate_summary(explanations: dict) -> str:
    combined = "\n\n".join(
        f"## {k}\n{v}" for k, v in list(explanations.items())[:6]
    )
    prompt = f"""Summarize the following lecture content into 6–8 clear, concise sentences
suitable for a student revision card. Focus on the most important concepts.

{combined}

Write only the summary, no headings."""
    return _chat(prompt, max_tokens=500)


# ═══════════════════════════════════════════════════════════
# 4. MCQ GENERATION
# ═══════════════════════════════════════════════════════════

def generate_mcqs(subtopic: str, explanation: str, num: int = 5) -> list[dict]:
    prompt = f"""Create exactly {num} multiple-choice questions from this explanation.

Topic: {subtopic}
Content: {explanation[:1500]}

Return ONLY valid JSON:
{{
  "questions": [
    {{
      "question": "...",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "answer": "A",
      "explanation": "Brief reason why A is correct"
    }}
  ]
}}"""
    raw = _chat(prompt, max_tokens=2000, json_mode=True)
    data = json.loads(raw)
    return data.get("questions", [])


def generate_quiz_from_topic(topic: str, subject: str, branch: str, year: str, num_questions: int = 10) -> list[dict]:
    """Generate quiz directly from topic without pre-existing explanation."""
    prompt = f"""Create exactly {num_questions} MCQs on "{topic}" for {branch} {year} students (subject: {subject}).

Return ONLY valid JSON:
{{
  "questions": [
    {{
      "question": "...",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "answer": "A",
      "explanation": "..."
    }}
  ]
}}"""
    raw = _chat(prompt, max_tokens=3000, json_mode=True)
    data = json.loads(raw)
    return data.get("questions", [])


# ═══════════════════════════════════════════════════════════
# 5. HALLUCINATION DETECTION
# ═══════════════════════════════════════════════════════════

def score_hallucination(question: str, answer: str, context: str) -> dict:
    """
    Ask Groq to evaluate whether the answer is grounded in context.
    Returns a score 0-100 and a brief verdict.
    """
    prompt = f"""You are a fact-checking assistant. Evaluate if the answer is supported by the context.

Question: {question}
Context: {context[:1200]}
Answer: {answer[:600]}

Return ONLY valid JSON:
{{
  "grounded_score": <integer 0-100>,
  "verdict": "grounded|partial|hallucinated",
  "reason": "one sentence explanation"
}}"""
    try:
        raw = _chat(prompt, max_tokens=200, json_mode=True)
        return json.loads(raw)
    except Exception:
        return {"grounded_score": 50, "verdict": "unknown", "reason": "Could not evaluate"}


# ═══════════════════════════════════════════════════════════
# 6. PLAGIARISM DETECTION
# ═══════════════════════════════════════════════════════════

def read_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".pdf":
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text

    elif ext == ".docx":
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    else:
        raise ValueError(f"Unsupported file type: {ext}")

def _preprocess(text: str) -> str:
    text = re.sub(r"[^a-zA-Z\s]", " ", text).lower()
    tokens = nltk.word_tokenize(text)
    return " ".join(
        LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS and t.isalpha()
    )


def ngram_similarity(t1: str, t2: str, n: int = 5) -> float:
    p1, p2 = _preprocess(t1), _preprocess(t2)
    try:
        vec = CountVectorizer(analyzer="char", ngram_range=(n, n))
        mat = vec.fit_transform([p1, p2]).toarray()
        num = float(np.dot(mat[0], mat[1]))
        denom = np.linalg.norm(mat[0]) * np.linalg.norm(mat[1])
        return num / denom if denom else 0.0
    except Exception:
        return 0.0


def embedding_similarity(t1: str, t2: str) -> float:
    embs = _embedder().encode([t1, t2], convert_to_numpy=True)
    cos = float(np.dot(embs[0], embs[1]) / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1])))
    return max(0.0, cos)


def detect_plagiarism(text_dict: dict, threshold: float = 0.72) -> list[dict]:
    """
    text_dict: {student_name: assignment_text}
    Returns list of suspicious pairs with ngram + embedding scores.
    """
    students = list(text_dict.keys())
    suspicious = []
    for i in range(len(students)):
        for j in range(i + 1, len(students)):
            s1, s2 = students[i], students[j]
            ng = ngram_similarity(text_dict[s1], text_dict[s2])
            em = embedding_similarity(text_dict[s1], text_dict[s2])
            combined = 0.5 * ng + 0.5 * em
            if combined >= threshold:
                suspicious.append({
                    "student1": s1, "student2": s2,
                    "ngram_score": round(ng, 3),
                    "embedding_score": round(em, 3),
                    "combined_score": round(combined, 3),
                    "flagged": True,
                })
    return suspicious


# ═══════════════════════════════════════════════════════════
# 7. STUDY PLANNER
# ═══════════════════════════════════════════════════════════

def generate_study_plan(subjects: list[str], exam_date: str, hours_per_day: int, branch: str, year: str) -> dict:
    subjects_str = ", ".join(subjects)
    prompt = f"""Create a detailed study plan for a {branch} engineering student ({year}).

Subjects: {subjects_str}
Exam Date: {exam_date}
Available study hours per day: {hours_per_day}

Return ONLY valid JSON:
{{
  "plan": [
    {{
      "day": "Day 1 - Monday",
      "date": "...",
      "sessions": [
        {{"subject": "...", "topic": "...", "duration_hours": 2, "priority": "high"}}
      ],
      "total_hours": 4
    }}
  ],
  "tips": ["tip1", "tip2", "tip3"]
}}
Generate a realistic 7-day plan."""
    raw = _chat(prompt, max_tokens=2500, json_mode=True)
    return json.loads(raw)


# ═══════════════════════════════════════════════════════════
# 8. Q&A RECOMMENDATION (based on past query similarity)
# ═══════════════════════════════════════════════════════════

def get_qa_recommendations(current_question: str, user_id: str, top_k: int = 3) -> list[dict]:
    """Fetch similar past Q&A from MongoDB using embedding similarity."""
    db = get_sync_db()
    past = list(db.qa_history.find({"user_id": user_id}, {"question": 1, "answer": 1}).limit(50))
    if not past:
        return []
    questions = [p["question"] for p in past]
    q_emb = _embedder().encode([current_question] + questions, convert_to_numpy=True)
    sims = np.dot(q_emb[1:], q_emb[0]) / (
        np.linalg.norm(q_emb[1:], axis=1) * np.linalg.norm(q_emb[0]) + 1e-9
    )
    top_idx = np.argsort(-sims)[:top_k]
    return [
        {
            "question": past[i]["question"],
            "answer": past[i]["answer"][:300] + "...",
            "similarity": round(float(sims[i]), 3),
        }
        for i in top_idx if sims[i] > 0.4
    ]


# ═══════════════════════════════════════════════════════════
# 9. YOUTUBE & GOOGLE SEARCH
# ═══════════════════════════════════════════════════════════

def get_youtube_videos(query: str, max_results: int = 3) -> list[dict]:
    s = get_settings()
    if not s.youtube_api_key:
        return []
    try:
        from googleapiclient.discovery import build
        yt = build("youtube", "v3", developerKey=s.youtube_api_key)
        res = yt.search().list(q=query, part="snippet", type="video", maxResults=max_results).execute()
        return [
            {"title": i["snippet"]["title"], "url": f"https://youtube.com/watch?v={i['id']['videoId']}"}
            for i in res.get("items", []) if i["id"].get("videoId")
        ]
    except Exception as e:
        print(f"[YouTube] {e}")
        return []


def get_google_article(query: str) -> dict:
    s = get_settings()
    if not s.google_api_key:
        return {}
    try:
        from googleapiclient.discovery import build
        svc = build("customsearch", "v1", developerKey=s.google_api_key)
        res = svc.cse().list(q=query, cx=s.google_search_engine_id, num=1).execute()
        item = res.get("items", [{}])[0]
        return {"title": item.get("title"), "link": item.get("link"), "snippet": item.get("snippet")}
    except Exception as e:
        print(f"[Google] {e}")
        return {}


# ═══════════════════════════════════════════════════════════
# 10. MASTER PIPELINE
# ═══════════════════════════════════════════════════════════

def run_full_pipeline(topic: str, subject: str, branch: str, year: str, user_id: str) -> dict:
    t0 = time.time()
    db = get_sync_db()

    subtopics = generate_subtopics(topic, subject, branch, year)
    explanations = generate_explanations(subtopics, year, branch)
    summary = generate_summary(explanations)

    quizzes = {}
    for st in list(subtopics.keys())[:3]:  # limit to 3 for speed
        quizzes[st] = generate_mcqs(st, explanations[st])

    youtube = {st: get_youtube_videos(st) for st in list(subtopics.keys())[:3]}
    articles = {st: get_google_article(st) for st in list(subtopics.keys())[:3]}

    doc = {
        "user_id": user_id,
        "topic": topic, "subject": subject, "branch": branch, "year": year,
        "subtopics": subtopics, "explanations": explanations, "summary": summary,
        "quizzes": quizzes, "youtube": youtube, "articles": articles,
        "processing_time_sec": round(time.time() - t0, 2),
        "created_at": datetime.utcnow(),
    }
    result = db.topic_pipelines.insert_one(doc)

    return {
        "pipeline_id": str(result.inserted_id),
        "topic": topic, "subject": subject,
        "summary": summary,
        "subtopics": subtopics,
        "explanations": explanations,
        "quizzes": quizzes,
        "youtube": youtube,
        "articles": articles,
        "processing_time_sec": doc["processing_time_sec"],
    }
