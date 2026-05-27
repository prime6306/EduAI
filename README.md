# 🎓 EduAI — Deep Learning Academic Platform

> **Full-stack AI-powered academic platform integrating computer vision, semantic embeddings, and transformer-based LLMs for automated academic workflows.**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue)](https://docker.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.13-orange)](https://mlflow.org)

---

## 🏗️ Architecture

```
EduAI/
├── backend/                    # FastAPI application
│   ├── auth/                   # JWT authentication
│   ├── db/                     # MongoDB + ChromaDB clients
│   └── modules/
│       ├── nlp/                # Subtopics, MCQ, Plagiarism, Study Planner
│       ├── rag/                # PDF ingestion + ChromaDB Q&A
│       ├── attendance/         # Face recognition + Anti-spoofing
│       ├── dropout/            # Risk classifier (RF + LR)
│       └── sentiment/          # Wellness assessment + chat agent
├── frontend/                   # Streamlit multi-page app
│   ├── Home.py                 # Login / Register
│   └── pages/
│       ├── 1_Dashboard.py
│       ├── 2_Doubt_Solver.py
│       ├── 3_Study_Material.py
│       ├── 4_Quiz.py
│       ├── 5_RAG_QA.py
│       ├── 6_Attendance.py
│       ├── 7_Dropout_Risk.py
│       ├── 8_Wellness.py
│       └── 9_Teacher_Dashboard.py
├── tests/                      # Pytest unit tests
├── scripts/                    # DB setup + startup scripts
├── docker/                     # Dockerfiles
└── docker-compose.yml
```

---

## 🚀 Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Groq API — **Llama 3 70B** |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| RAG Pipeline | LangChain + **ChromaDB** |
| Primary DB | **MongoDB Atlas** |
| Face Recognition | `face-recognition` + **dlib** |
| Anti-Spoofing | **MobileNetV2 + ResNet18** ensemble (PyTorch) |
| ML Classifier | **Random Forest + Logistic Regression** (scikit-learn) |
| Experiment Tracking | **MLflow** |
| Backend | **FastAPI** (async, JWT auth) |
| Frontend | **Streamlit** |
| Containerisation | **Docker Compose** |
| CI/CD | **GitHub Actions** |
| Plagiarism | N-gram + **SentenceTransformer** embedding similarity |

---

## ⚡ Quick Start (Local)

### 1. Clone & setup environment
```bash
git clone https://github.com/your-username/EduAI.git
cd EduAI
cp .env.example .env
# Edit .env with your credentials
```

### 2. Add your model files
```bash
# Copy your trained models to:
cp /path/to/antispoof_fullmodels.pkl models/
cp /path/to/ENcodedFile.p models/
```

### 3. Run with startup script
```bash
chmod +x scripts/start_dev.sh
./scripts/start_dev.sh
```

Or manually:
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/setup_db.py
uvicorn backend.main:app --reload &
streamlit run frontend/Home.py
```

### 4. Access the platform
| Service | URL |
|---------|-----|
| 🌐 Frontend | http://localhost:8501 |
| 🔌 API Docs | http://localhost:8000/docs |
| 📊 MLflow | http://localhost:5001 |

---

## 🐳 Docker Deployment

```bash
# Build and start all services
docker compose up --build -d

# View logs
docker compose logs -f backend

# Stop
docker compose down
```

---

## 🔑 Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq API key (get free at console.groq.com) |
| `MONGODB_URI` | MongoDB Atlas connection string |
| `JWT_SECRET` | Random secret for JWT tokens |
| `YOUTUBE_API_KEY` | Google YouTube Data API v3 |
| `GOOGLE_API_KEY` | Google Custom Search API |
| `GOOGLE_SEARCH_ENGINE_ID` | Custom Search Engine ID |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📋 Features

| Feature | Description |
|---------|-------------|
| 💬 **AI Doubt Solver** | Conversational Llama 3 70B tutor with subject context |
| 📚 **Study Material** | Auto-generate subtopics, explanations, summaries, YouTube links |
| ❓ **Quiz Simulator** | Timed MCQ exams with scoring, review, and MLflow logging |
| 📄 **RAG Q&A** | Upload PDFs → semantic search → grounded answers with hallucination scoring |
| 📷 **Face Attendance** | Anti-spoof protected group/individual attendance marking |
| ⚠️ **Dropout Risk** | ML classifier with risk gauge and personalised recommendations |
| 💙 **Wellness Chat** | PHQ-9 assessment + empathetic Groq companion with crisis detection |
| 📅 **Study Planner** | AI-generated personalised weekly schedule |
| 🔍 **Plagiarism Detector** | N-gram + embedding dual-method for teachers |
| 👩‍🏫 **Teacher Dashboard** | Class analytics, attendance heatmaps, batch dropout screening |
| 📊 **MLflow Tracking** | All model runs logged (params, metrics, artifacts) |

---

## ⚠️ Security Notes

- **Never commit `.env`** — it's in `.gitignore`
- Rotate your Groq API key at [console.groq.com](https://console.groq.com) after this setup
- Change MongoDB password at Atlas → Database Access
- Set a strong `JWT_SECRET` in production

---

## 📌 Adding New Students to Attendance

1. Add their photo to `models/Images/{student_id}.jpg`
2. Run: `python backend/modules/attendance/encode_faces.py`
3. New `ENcodedFile.p` will be generated

---

*Built with ❤️ — MMMUT ECE • 2025*
