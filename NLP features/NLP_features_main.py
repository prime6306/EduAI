"""
features.py
Reusable pipeline module for:
- Subtopic generation
- Explanations
- MCQs
- YouTube + Google articles
- Summary
- Plagiarism detection
- MongoDB storage

Used by flaskapi.py
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path

from huggingface_hub import InferenceClient
from pymongo import MongoClient
from googleapiclient.discovery import build

# NLP + ML
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv

# -------------------------
# Load .env
# -------------------------
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_SUBTOPICS = os.getenv("HF_MODEL_SUBTOPICS", "meta-llama/Llama-3.1-8B-Instruct")
HF_MODEL_EXPLANATION = os.getenv("HF_MODEL_EXPLANATION", "mistralai/Mistral-7B-Instruct-v0.3")

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

MONGO_URI = os.getenv("MONGODB_URI")

if not HF_TOKEN:
    raise Exception("HF_TOKEN missing from .env")
if not MONGO_URI:
    raise Exception("MONGODB_URI missing from .env")

# -------------------------
# Initialize APIs
# -------------------------

hf_client = InferenceClient(
    model=HF_MODEL_SUBTOPICS,
    token=HF_TOKEN
)

mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client.get_default_database() or mongo_client["studentdb"]

# YouTube
youtube_service = None
if YOUTUBE_API_KEY:
    try:
        youtube_service = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    except Exception:
        youtube_service = None

# Google Search
google_service = None
if GOOGLE_API_KEY:
    try:
        google_service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    except Exception:
        google_service = None

# -------------------------
# NLTK Setup
# -------------------------
try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except:
    nltk.download("stopwords")
try:
    nltk.data.find("corpora/wordnet")
except:
    nltk.download("wordnet")

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# ============================================================
# 1. SUBTOPIC GENERATION
# ============================================================

def generate_subtopics(topic, subject, branch, year):
    prompt = f"""
You are an experienced university professor.
Generate 5–7 concise subtopics ONLY.

Subject: {subject}
Topic: {topic}
Branch: {branch}
Year: {year}

Output exactly:
Subtopic 1: ...
Subtopic 2: ...
"""
    try:
        res = hf_client.chat.completions.create(
            model=HF_MODEL_SUBTOPICS,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
        )
        return res.choices[0].message["content"].strip()
    except Exception as e:
        raise RuntimeError(f"Subtopic generation failed: {e}")

def parse_subtopics(raw_text: str):
    """
    Converts LLM text into:
    { "Title": ["point1", "point2", ...], ... }
    """
    parts = re.split(r"\n?Subtopic\s*\d+\s*:?\s*", raw_text.strip())
    parts = [p.strip() for p in parts if p.strip()]

    result = {}
    for p in parts:
        lines = p.split("\n")
        lines = [l.strip() for l in lines if l.strip()]

        if not lines:
            continue

        title = lines[0]
        bullet_points = [
            re.sub(r"^[-•]\s*", "", l) for l in lines[1:]
        ]

        result[title] = bullet_points

    return result

# ============================================================
# 2. EXPLANATIONS
# ============================================================

def describe_topics(subtopics_dict, year, branch):
    explanations = {}
    for subtopic, points in subtopics_dict.items():
        points_text = "\n".join(points) if points else ""

        prompt = f"""
You are an experienced professor.
Explain the topic: {subtopic}
Year: {year}
Branch: {branch}

Points:
{points_text}

Write a detailed explanation (min 150 words).
"""

        try:
            res = hf_client.chat.completions.create(
                model=HF_MODEL_EXPLANATION,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=900,
            )
            explanations[subtopic] = res.choices[0].message["content"].strip()
        except Exception as e:
            explanations[subtopic] = f"Error generating explanation: {e}"

    return explanations

# ============================================================
# 3. SUMMARY
# ============================================================

def generate_summary(explanations_dict):
    text = "\n\n".join(explanations_dict.values())

    prompt = f"""
Summarize the following lecture explanations into 5–7 clear sentences:

{text}
"""
    try:
        res = hf_client.chat.completions.create(
            model=HF_MODEL_SUBTOPICS,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        return res.choices[0].message["content"].strip()
    except Exception as e:
        return f"Error generating summary: {e}"

# ============================================================
# 4. YouTube Suggestions
# ============================================================

def get_youtube_suggestions(query, max_results=3):
    if not youtube_service:
        return []
    try:
        req = youtube_service.search().list(
            q=query, part="snippet", type="video", maxResults=max_results
        )
        res = req.execute()
        videos = []
        for item in res.get("items", []):
            vid = item["id"].get("videoId")
            title = item["snippet"]["title"]
            if vid:
                videos.append({"title": title, "url": f"https://www.youtube.com/watch?v={vid}"})
        return videos
    except Exception:
        return []

# ============================================================
# 5. Google Articles
# ============================================================

def fetch_top_google_article(query):
    if not google_service or not GOOGLE_SEARCH_ENGINE_ID:
        return {"title": None, "link": None, "snippet": None}

    try:
        res = google_service.cse().list(
            q=query, cx=GOOGLE_SEARCH_ENGINE_ID, num=1
        ).execute()
        item = res.get("items", [{}])[0]
        return {
            "title": item.get("title"),
            "link": item.get("link"),
            "snippet": item.get("snippet"),
        }
    except Exception:
        return {"title": None, "link": None, "snippet": None}

# ============================================================
# 6. MCQs
# ============================================================

def generate_mcq(subtopic, explanation, num_questions=5):
    prompt = f"""
Create {num_questions} MCQs from the following explanation:

Topic: {subtopic}
Explanation:
{explanation}

Return STRICTLY in JSON:
[
  {{
    "question": "...",
    "options": ["A", "B", "C", "D"],
    "answer": "A",
    "explanation": "..."
  }}
]
"""
    try:
        res = hf_client.chat.completions.create(
            model=HF_MODEL_SUBTOPICS,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
        )

        text = res.choices[0].message["content"].strip()
        return json.loads(text)

    except Exception as e:
        return [{"error": f"MCQ generation failed: {e}"}]

# ============================================================
# 7. File Reading for Assignments
# ============================================================

def read_text_from_file(path: str):
    path = Path(path)
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    if path.suffix.lower() == ".docx":
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)

    if path.suffix.lower() == ".pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    return ""

# ============================================================
# 8. Plagiarism Detection
# ============================================================

def preprocess_text(text):
    # Remove symbols, lowercase
    text = re.sub(r"[^a-zA-Z\s]", " ", text).lower()
    tokens = nltk.word_tokenize(text)
    tokens = [
        LEMMATIZER.lemmatize(t)
        for t in tokens
        if t not in STOP_WORDS and t.isalpha()
    ]
    return " ".join(tokens)

def ngram_similarity(text1, text2, n=5):
    t1, t2 = preprocess_text(text1), preprocess_text(text2)

    vectorizer = CountVectorizer(analyzer="char", ngram_range=(n, n))
    try:
        mat = vectorizer.fit_transform([t1, t2]).toarray()
    except:
        return 0.0

    num = np.dot(mat[0], mat[1])
    denom = np.linalg.norm(mat[0]) * np.linalg.norm(mat[1])

    return float(num / denom) if denom else 0.0

def detect_plagiarism_from_texts(text_dict, threshold=0.77, n=5):
    students = list(text_dict.keys())
    suspicious = []

    for i in range(len(students)):
        for j in range(i + 1, len(students)):
            s1, s2 = students[i], students[j]
            sim = ngram_similarity(text_dict[s1], text_dict[s2], n=n)
            if sim >= threshold:
                suspicious.append({
                    "student1": s1,
                    "student2": s2,
                    "similarity": round(sim, 3),
                })
    return suspicious

# ============================================================
# 9. MongoDB Helper
# ============================================================

def save_pipeline_results(collection, doc):
    coll = mongo_db[collection]
    res = coll.insert_one(doc)
    return str(res.inserted_id)

# ============================================================
# 10. FULL PIPELINE MASTER FUNCTION
# ============================================================

def process_topic_pipeline(topic, subject, branch, year):
    """
    Runs the FULL workflow and returns dict for API use.
    """
    start_time = datetime.utcnow().isoformat()

    # 1. Subtopics
    raw = generate_subtopics(topic, subject, branch, year)
    parsed = parse_subtopics(raw)

    # 2. YouTube
    youtube_map = {st: get_youtube_suggestions(st) for st in parsed}

    # 3. Explanations
    explanations = describe_topics(parsed, year, branch)

    # 4. Summary
    summary = generate_summary(explanations)

    # 5. Articles
    articles = {st: fetch_top_google_article(st) for st in parsed}

    # 6. MCQs
    quizzes = {st: generate_mcq(st, explanations[st]) for st in parsed}

    # Save to DB
    pipeline_doc = {
        "topic": topic,
        "subject": subject,
        "branch": branch,
        "year": year,
        "raw_subtopics": raw,
        "subtopics": parsed,
        "youtube": youtube_map,
        "explanations": explanations,
        "summary": summary,
        "articles": articles,
        "quizzes": quizzes,
        "created_at": datetime.utcnow()
    }

    pipeline_id = save_pipeline_results("topic_pipelines", pipeline_doc)

    # Return preview for API
    return {
        "pipeline_id": pipeline_id,
        "topic": topic,
        "subject": subject,
        "branch": branch,
        "year": year,
        "summary": summary,
        "subtopics": parsed,
        "explanations_preview": {k: v[:300] + "..." for k, v in explanations.items()},
        "youtube": youtube_map,
        "articles": articles,
        "created_at": start_time
    }
