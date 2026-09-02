# EduAI

EduAI is a full-stack educational operations and learning platform built with Flask and MongoDB Atlas. It unifies classroom administrative workflows (biometric attendance, syllabus tracking, question paper generation, grading) with machine learning tools (anti-spoof liveness detection, student dropout risk prediction, document Q&A via RAG, and AI-assisted mock interviews).

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
  - [1. Attendance & Biometric Verification](#1-attendance--biometric-verification)
  - [2. Academic & Teaching Tools](#2-academic--teaching-tools)
  - [3. Assessment & Examination](#3-assessment--examination)
  - [4. Student Risk & Analytics](#4-student-risk--analytics)
  - [5. Learning Assistants & Career Prep](#5-learning-assistants--career-prep)
- [System Architecture](#system-architecture)
- [Directory Structure](#directory-structure)
- [Installation & Setup](#installation--setup)
  - [Prerequisites](#prerequisites)
  - [Local Setup](#local-setup)
  - [Windows-Specific Notes](#windows-specific-notes)
- [Configuration Reference (.env)](#configuration-reference-env)
- [Machine Learning & Model Training](#machine-learning--model-training)
  - [Anti-Spoofing Model (MobileNetV2 + ResNet18)](#anti-spoofing-model-mobilenetv2--resnet18)
  - [Dropout Prediction Models](#dropout-prediction-models)
  - [RAG & Vector Search](#rag--vector-search)
- [CLI Tools](#cli-tools)
- [Docker Deployment](#docker-deployment)
- [Troubleshooting](#troubleshooting)

---

## Overview

EduAI is designed for colleges and universities to reduce manual administrative overhead while providing students with targeted learning and assessment tools. The platform enforces strict role-based access control (Student, Teacher, Admin), audit-logs sensitive operations, and gracefully degrades when optional external API keys (e.g. YouTube, Google Search, Gemini) are not provided.

---

## Key Features

### 1. Attendance & Biometric Verification
- **Webcam & Batch Attendance:** Captures single-student or group photos and detects/recognizes faces using `face_recognition` (dlib) against a cached encoding bank (`models/ENcodedFile.p`).
- **Anti-Spoof Liveness Detection:** Employs a dual-backbone deep neural network (MobileNetV2 + ResNet18) that validates presentation liveness on 160×160 RGB face crops to prevent photo, screen, and replay attacks.
- **Attendance Correction Workflow:** Students can file formal attendance dispute requests with strict operational guardrails (maximum 30-day lookback, no duplicate marks, 2 requests/week limit). Teachers review, approve, or reject with mandatory audit remarks.
- **Roster Management:** Web-based and CLI student onboarding, face image enrollment, re-encoding pipelines, and attendance log CSV export.

### 2. Academic & Teaching Tools
- **Syllabus Tracker:** Module-by-module syllabus completion tracker with hours logged, faculty pacing metrics, and automated NAAC-compliant PDF export.
- **Question Paper Generator:** Multi-set exam generator mapping questions across Bloom's Taxonomy levels (Remember, Understand, Apply, Analyze, Evaluate) with automated deduplication and DOCX/PDF export.
- **Announcements Engine:** Targeted announcements with rich text formatting, scheduled release dates, audience scoping (branch/year/role), and unread indicators.
- **Report Card Generator:** Automated grade sheet generator with configurable weightages for internal assessments, quizzes, and attendance. Supports inline autosaving mark entry grids, batch CSV imports, and ZIP archives containing individualized student PDFs.
- **CO-PO Attainment:** Calculates direct and indirect Course Outcome (CO) and Program Outcome (PO) attainment percentages against NBA/NAAC threshold targets.

### 3. Assessment & Examination
- **Custom Test Creator:** Build quizzes and tests manually, pull from the shared question bank, or generate questions via LLM.
- **Student Exam Interface:** Distraction-free, timed examination UI featuring autosave, question palette navigation, and client-side heartbeat.
- **Hybrid Grading Queue:** Instant auto-grading for objective questions (MCQs, True/False) combined with a teacher evaluation queue for subjective short/long answers.

### 4. Student Risk & Analytics
- **Early Dropout Risk Prediction:** Evaluates academic performance, attendance records, parental education, and socio-economic signals using an ensemble of Random Forest and Logistic Regression classifiers. Highlights key risk drivers and tailored intervention recommendations.
- **Faculty Analytics Dashboard:** Real-time metrics on class performance distributions, attendance anomalies, at-risk student cohorts, and exam statistics.
- **Plagiarism Checker:** Dual-engine similarity inspection combining fuzzy n-gram lexical matching with dense semantic embeddings from `sentence-transformers`.

### 5. Learning Assistants & Career Prep
- **Interactive Doubt Solver:** Real-time conversational tutoring with server-sent events (SSE) streaming, Markdown rendering, and LaTeX formula support via KaTeX.
- **RAG Document Assistant:** Upload lecture notes and textbooks (PDF, DOCX, TXT) into a local ChromaDB vector store. Provides cited answers along with retrieval hallucination scores.
- **Study Material Generator:** Produces structured study guides, flashcards, summaries, and reading lists with optional embedded YouTube lectures and web references.
- **AI Mock Interview Simulator:** Dual-recruiter technical and behavioral interview preparation:
  - **HR Round:** Managed by conversational recruiter persona on Groq.
  - **Technical Round:** Managed by technical lead persona on Gemini.
  - Tracks answer clarity, technical depth, filler word frequency, and generates a structured performance report.
- **Wellness Companion:** Guided student support with strict safety guardrails that bypass external LLMs to immediately display verified national helpline contacts upon detecting crisis triggers.

---

## System Architecture

```
                                  +-----------------------+
                                  |   Browser / Client    |
                                  +-----------+-----------+
                                              |
                                              | HTTP / JSON / SSE
                                              v
+---------------------------------------------------------------------------------+
|                               EduAI Flask Application                           |
|                                                                                 |
|  +-------------------+  +--------------------+  +----------------------------+  |
|  |   Auth & RBAC     |  |   Core Blueprints  |  |     ML & Vision Pipelines  |  |
|  | Flask-Login / JWT |  | Attendance, Tests, |  | dlib (Face Recognition)    |  |
|  | CSRFProtect       |  | RAG, Syllabus, etc |  | PyTorch (Anti-Spoof PAD)   |  |
|  +---------+---------+  +---------+----------+  | Scikit-Learn (Dropout)     |  |
|            |                      |             +--------------+-------------+  |
+------------|----------------------|----------------------------|----------------+
             |                      |                            |
             v                      v                            v
   +-------------------+  +--------------------+       +-------------------+
   |   MongoDB Atlas   |  |  ChromaDB (Vector) |       | Pretrained Models |
   |  Users, Logs,     |  |  Document chunks   |       | .pkl / .p weights |
   |  Rosters, Quizzes |  |  & embeddings      |       | on local disk     |
   +-------------------+  +--------------------+       +-------------------+
```

---

## Directory Structure

```
eduai/
├── app/
│   ├── auth/                 # Authentication, user loaders, RBAC decorators
│   ├── dashboard/            # Role-specific routing (Student, Teacher, Admin)
│   ├── modules/
│   │   ├── admin/            # Audit logs and system administration
│   │   ├── analytics/        # Class performance & attendance trends
│   │   ├── announcements/    # Notice board with scheduling
│   │   ├── attendance/       # Face matching, liveness check, correction flows
│   │   ├── co_po/            # NBA/NAAC outcome attainment calculators
│   │   ├── digest/           # Automated email reporting
│   │   ├── dropout/          # ML dropout risk prediction & batch scoring
│   │   ├── interview/        # Dual-persona AI mock interview engine
│   │   ├── nlp/              # Doubt solver, study notes, quiz generator
│   │   ├── plagiarism/       # Lexical & semantic text comparison
│   │   ├── question_paper/   # Exam question paper compilation & export
│   │   ├── rag/              # Document upload, vector indexing, retrieval
│   │   ├── report_cards/     # Weighted marks calculation & PDF generation
│   │   ├── syllabus/         # Course tracking & NAAC report export
│   │   ├── tests/            # Exam authoring, test runner, evaluation
│   │   └── wellness/         # Mental wellness assistant & safety triggers
│   ├── static/               # CSS stylesheets, vanilla JS modules, SVG icons
│   ├── templates/            # Jinja2 templates organized by module
│   ├── utils/                # Audit logger, export helpers, decorators
│   ├── config.py             # Central application configuration
│   └── extensions.py         # Mongo wrapper, login manager, CSRF, mail
├── docker/                   # Dockerfile and docker-compose configurations
├── models/                   # Serialized ML models (.pkl, .p)
├── scripts/
│   ├── add_student.py        # CLI for roster registration & face enrollment
│   ├── fetch_datasets.py     # Automated dataset downloader (Kaggle / direct)
│   ├── create_dataset.py     # Video/image frame extraction & 160x160 face cropper
│   └── train_antispoof.py    # PyTorch training script for anti-spoof ensemble
├── run.py                    # Application entrypoint
├── requirements.txt          # Python dependencies
└── .env.example              # Environment variables template
```

---

## Installation & Setup

### Prerequisites
- **Python:** 3.10 to 3.12
- **MongoDB:** A MongoDB Atlas URI or a local MongoDB instance (v6.0+)
- **Build Tools (Windows):** Visual Studio C++ Build Tools (required for `dlib`)
- **Git & Git LFS:** For cloning repository and downloading model weights

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/prime6306/EduAI.git
   cd EduAI
   git lfs pull
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Linux/macOS
   python3 -m venv .venv
   source .venv/bin/activate

   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   Open `.env` and configure your database URI, secret keys, and API credentials (see [Configuration Reference](#configuration-reference-env)).

5. **Start the application:**
   ```bash
   python run.py
   ```
   Open your browser at `http://localhost:5000`.

---

### Windows-Specific Notes

- **dlib DLL Initialization:** On Windows, `dlib`'s native C++ runtime can encounter thread initialization issues if imported inside secondary Flask worker threads. `run.py` and `app/__init__.py` include eager initialization at process launch to ensure `_dlib_pybind11` loads on the primary thread.
- **C++ Build Tools:** If installing `dlib` from source on Windows, install the "Desktop development with C++" workload from Visual Studio Installer. Pre-built wheels for Python 3.10–3.12 can also be installed directly.

---

## Configuration Reference (.env)

| Variable | Required | Default | Purpose |
| :--- | :--- | :--- | :--- |
| `FLASK_SECRET_KEY` | **Yes** | — | Flask session encryption key |
| `FLASK_DEBUG` | No | `false` | Enable/disable Flask debug mode |
| `FLASK_PORT` | No | `5000` | Port for the web application |
| `MONGODB_URI` | **Yes** | — | MongoDB Atlas connection string |
| `MONGODB_DB` | No | `eduai` | Database name |
| `JWT_SECRET_KEY` | **Yes** | — | Secret key for JWT signing |
| `JWT_ACCESS_TOKEN_EXPIRES_HOURS` | No | `24` | Token lifespan in hours |
| `GROQ_API_KEY` | Recommended | — | Key for doubt solver, study notes, quiz generator |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | Model ID for Groq completions |
| `GEMINI_API_KEY` | Optional | — | Enables Interview Prep's technical interviewer |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | Model ID for Google Gemini |
| `EMBEDDING_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` | SentenceTransformer model name |
| `CHROMA_PATH` | No | `./chromadb_data` | Persistent path for local vector database |
| `ANTISPOOF_MODEL_PATH` | No | `./models/antispoof_fullmodels.pkl` | Path to trained anti-spoof model weights |
| `FACE_ENCODINGS_PATH` | No | `./models/ENcodedFile.p` | Path to cached student face encodings |
| `YOUTUBE_API_KEY` | Optional | — | Fetches video lecture recommendations |
| `GOOGLE_API_KEY` | Optional | — | Enables article lookup in Study Material |
| `GOOGLE_SEARCH_ENGINE_ID`| Optional | — | Custom search engine ID |
| `MAIL_SERVER` | Optional | — | SMTP host for weekly email digests |
| `MAIL_PORT` | Optional | `587` | SMTP port |
| `MAIL_USERNAME` | Optional | — | SMTP account username |
| `MAIL_PASSWORD` | Optional | — | SMTP account password / app password |
| `DIGEST_SCHEDULER_ENABLED`| No | `true` | Starts background weekly digest scheduler |

---

## Machine Learning & Model Training

EduAI includes standalone scripts to train and evaluate the machine learning models.

### Anti-Spoofing Model (MobileNetV2 + ResNet18)

The anti-spoofing module prevents presentation attacks using an ensemble of MobileNetV2 and ResNet18. Input crops must be 160×160 RGB images normalized with standard ImageNet statistics.

#### 1. Download Public PAD Datasets
Use `scripts/fetch_datasets.py` to retrieve public datasets (requires Kaggle API credentials):
```bash
# Download Real vs Fake Video Anti-Spoofing dataset
python scripts/fetch_datasets.py --dataset real-vs-fake --dest ./datasets/real_vs_fake

# Download CelebA-Spoof mirror
python scripts/fetch_datasets.py --dataset celeba-spoof --dest ./datasets/celeba_spoof
```

#### 2. Process & Crop Faces (160×160)
Extract faces from raw videos or photos into normalized train/val splits:
```bash
# Process video files (reads metadata CSVs or folder structures)
python scripts/create_dataset.py --mode videos --input-dir ./datasets/real_vs_fake --output-dir ./datasets/processed_pad --frame-interval 8

# Create a balanced subset from CelebA-Spoof
python scripts/create_dataset.py --mode celeba-subset --input-dir ./datasets/celeba_spoof --output-dir ./datasets/processed_pad --max-per-class 2500
```

#### 3. Train the Ensemble
Train locally or on Google Colab with GPU acceleration:
```bash
python scripts/train_antispoof.py \
  --data-dir ./datasets/processed_pad \
  --model ensemble \
  --epochs 6 \
  --batch-size 32 \
  --lr 0.0003 \
  --output ./models/antispoof_fullmodels.pkl
```
The script evaluates standard Presentation Attack Detection metrics (Accuracy, APCER, BPCER, and ACER) and automatically restores the best validation checkpoint before saving.

---

### Dropout Prediction Models

The dropout module predicts student risk levels (`Low`, `Medium`, `High`) and suggests remedial interventions:
- **Algorithms:** Random Forest and Logistic Regression trained on academic history, attendance statistics, and demographic features.
- **Retraining:** Teachers can retrain the model directly from the web interface (`/dropout/retrain`) or let the application auto-initialize baseline weights on first startup (`app.modules.dropout.model.ensure_models_trained`).
- **Artifacts:**
  - `models/dropout_rf.pkl`
  - `models/dropout_lr.pkl`
  - `models/dropout_selector.pkl`

---

### RAG & Vector Search

- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (runs locally, 384-dimensional dense vectors).
- **Storage:** Persistent ChromaDB collections stored at `CHROMA_PATH`.
- **Query Pipeline:** Cosine similarity retrieval with top-k ranking, context assembly, grounding validation, and hallucination scoring.

---

## CLI Tools

### Student Onboarding & Face Registration
Enroll students and build the facial encodings database without using the browser:

```bash
# Register a student and enroll their face photo
python scripts/add_student.py --id 2024CS101 --name "Rahul Sharma" --branch CS --year 2 --photo ./student_photos/rahul.jpg

# List all enrolled students
python scripts/add_student.py --list

# List students with enrolled face vectors
python scripts/add_student.py --list-faces
```

---

## Docker Deployment

To spin up EduAI and a local MLflow tracking server using Docker Compose:

```bash
cd docker
docker compose up --build -d
```

- **Web Application:** `http://localhost:5000`
- **MLflow Tracking Dashboard:** `http://localhost:5001`

*Note: The Docker setup connects to your MongoDB Atlas cluster configured in `.env`.*

---

## Troubleshooting

### "The CSRF session token is missing" on JSON API calls
Ensure your requests include the CSRF token in the `X-CSRFToken` header. The frontend helper `window.apiFetch()` automatically reads the `<meta name="csrf-token">` tag and attaches this header to all `fetch` calls.

### Anti-Spoof Model Loading (`_pickle.UnpicklingError` / `weights_only`)
PyTorch 2.6+ changed the default parameter of `torch.load` to `weights_only=True`. The loader in `app/modules/attendance/antispoof.py` explicitly handles `weights_only=False` with backward-compatible fallbacks.

### OpenCV Headless in Colab
If you encounter `AttributeError: module 'cv2' has no attribute 'CascadeClassifier'` in headless server environments, run:
```bash
pip install --upgrade --force-reinstall opencv-python-headless
```
`scripts/create_dataset.py` includes a safe fallback to smart portrait center-cropping if Haar cascades are unavailable.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
