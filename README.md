# EduAI

An AI-powered educational platform for MMMUT: doubt solving, study material generation,
quizzes, RAG-based document Q&A, face-recognition attendance, dropout risk prediction,
a wellness companion, plagiarism detection, teacher analytics, and NBA-style CO-PO
attainment reporting — built as a Flask app on MongoDB Atlas, per `EduAI_PROJECT_SPEC_v2-1.md`
and `EduAI_DESIGN_SPEC.md`.

## Build status — Phase 7 of N

This is a large, multi-phase build. **Phases 1–7 are fully working end-to-end:**

**Phase 1 — Foundation:** app factory, Mongo wrapper, auth (bcrypt + Flask-Login +
JWT + roles + audit log), role-aware dashboards, full design system, auto-training
Dropout classifiers, Docker/compose.

**Phase 2 — Core AI:** Doubt Solver (SSE streaming chat), Study Material Generator
(full pipeline + PDF export), Quiz & Exam Simulator (server-graded, timed mode).

**Phase 3 — RAG + Attendance:** PDF/DOCX/TXT/MD Q&A with ChromaDB + grounding
scores; face-recognition attendance (group photo + webcam) with anti-spoof gating
and cooldown; full Student Management (enroll, re-encode, delete, CSV export, CLI).

**Phase 4 — Dropout UI, Wellness, Plagiarism:** full prediction form + batch CSV +
retraining wired to the real trained models; Wellness Companion with verified
crisis helplines and LLM-bypassing crisis detection; dual-method Plagiarism
Detector reusing RAG's text/embedding pipeline.

**Phase 5 — Announcements, Syllabus Tracker, Question Paper Generator:** rich-text
announcements with live scheduling and an unread badge; syllabus pace tracking
with a verified working NAAC PDF export; multi-set question papers with verified
PDF + DOCX export and an auto-populating, deduplicated question bank.

**Phase 6 — Custom Test Creator:** AI/bank/manual question builder, drag-reorder,
publishing with per-student assignment and shuffling, a distraction-free timed
test-taking UI with autosave, instant auto-grading for MCQ/True-False, and a
teacher review queue for descriptive answers with live score recomputation.

**Phase 7 — Attendance Corrections + Report Card Generator**
- ✅ **Attendance Correction Requests** (`/attendance/corrections`) — a structured
  replacement for WhatsApp-based fix requests. Every guard from the spec is
  enforced server-side, not just suggested in the UI: no request for a date
  more than 30 days old, no request for a date already marked present, and a
  hard cap of 2 requests/week — **all three verified with actual boundary-testing
  requests that were expected to fail and did**. Approval creates a real
  attendance log entry tagged "Manual correction" (verified the attendance
  count actually incremented), rejection requires a reason, bulk approve/reject
  work from the teacher queue, and both a sidebar-adjacent dashboard chip and a
  student dashboard notification are wired end-to-end
- ✅ **Report Card Generator** (`/report-cards`) — batch creation prefilled from
  the live student roster, marks entry via CSV import or an inline-autosaving
  grid, configurable score weightages (IA/quiz/attendance) with live-updating
  grading tiers, template or AI-generated remarks, and PDF + ZIP export that
  I verified by **hand-checking the actual computed numbers** — a synthetic
  student with 95%/100% IA scores, a 90% quiz average, and 90% attendance
  computed to exactly 93.0% overall and grade A+ under a 40/30/30 weighting,
  matching the math by hand before trusting the code
- ✅ **A real, previously-undiscovered bug got caught and fixed this round**:
  the teacher dashboard's attendance chart silently 500'd whenever real
  attendance data existed, because a dict key named `values` collided with
  Python's `dict.values()` method under Jinja's attribute-lookup rules — every
  earlier regression pass had missed it because no test had ever populated
  real attendance numbers before. Renamed the key, verified the chart renders.

**Not yet implemented** (placeholder pages): Study Planner, CO-PO Attainment
Mapping, Weekly Digest, Teacher Analytics detail pages, rate limiting, and the
audit-log viewer UI.

What's left is CO-PO Attainment Mapping and Weekly Digest (the last two teacher
reporting tools), plus the smaller Study Planner, Teacher Analytics detail pages,
rate limiting, and the audit-log viewer — ask me to continue with whichever
matters most next.

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env — at minimum set MONGODB_URI to your Atlas connection string
# and FLASK_SECRET_KEY / JWT_SECRET_KEY to random values

python run.py
# → http://localhost:5000
```

The app boots and serves pages even if some `.env` values are left blank —
Mongo-backed reads degrade to empty states, and AI-backed modules will simply
say so once you reach them, rather than crashing the whole app.

### Anti-spoof model

`app/modules/attendance/antispoof.py` only ever **loads** the anti-spoof ensemble —
it does not train one. Drop your pretrained `.pkl` at the path in
`ANTISPOOF_MODEL_PATH` (default: `./models/antispoof_fullmodels.pkl`).

### AI features (Doubt Solver, Study Material, Quiz, RAG Q&A)

These call the Groq API and, for Study Material, optionally the YouTube Data API
and Google Custom Search. Set `GROQ_API_KEY` in `.env` to enable them — without it,
each feature shows a clear "AI features are not configured" message instead of
erroring. `YOUTUBE_API_KEY` / `GOOGLE_API_KEY` + `GOOGLE_SEARCH_ENGINE_ID` are
optional; Study Material works fine without them, just without embedded videos/
articles.

### RAG Q&A

First use downloads the `sentence-transformers` embedding model (~90MB) and
initialises a local ChromaDB store at `CHROMA_PATH` (default `./chromadb_data`) —
both fully local after that, no API key needed for retrieval itself (only for the
answer-generation and hallucination-scoring calls, which use Groq).

### Attendance / face recognition

`face_recognition` (which wraps `dlib`) and `torch` are native/heavy dependencies.
This build's routes and business logic (matching, cooldown, anti-spoof gating,
enrollment, re-encoding) are fully implemented and were verified end-to-end with
mocked face/anti-spoof functions, since this development sandbox couldn't install
`dlib`/`torch` itself — you'll want to smoke-test the real `face_recognition` and
anti-spoof inference paths in your own environment before relying on them in
production. If your anti-spoof `.pkl`'s structure differs from the dict-of-models
or single-model convention assumed in `app/modules/attendance/antispoof.py`, that
file's `predict_is_real()` is the only place you'll need to adjust.

### Wellness Companion

Crisis keyword detection short-circuits the LLM entirely and always shows the
helplines in `app/modules/wellness/routes.py` (`HELPLINES`) — verified in testing
that a crisis message never reaches Groq. The helpline numbers were checked
against current public sources as of this build, not just copied from the spec;
worth a periodic re-check since helpline numbers do change.

### Fonts

`static/css/main.css` expects a self-hosted Inter variable font at
`static/fonts/Inter/Inter-Variable.woff2` (and JetBrains Mono similarly) for full
offline support. Until you add those files, the browser falls back to the
platform UI font, which looks close but isn't pixel-identical to the spec.

## Docker

```bash
cd docker
docker compose up --build
# app on :5000, local MLflow UI on :5001
```

MongoDB itself is expected to be Atlas (cloud) per the spec, so it's not a
compose service — point `MONGODB_URI` in `.env` at your cluster. See the
comment in `docker/docker-compose.yml` if you'd rather run Mongo locally.

## Project layout

```
app/
  auth/            # register/login/logout/profile
  dashboard/        # role-aware dashboard
  modules/          # one folder per feature (nlp, rag, attendance, dropout, ...)
  static/           # css/js/icons/fonts
  templates/        # Jinja templates, mirrors modules/
  utils/            # audit log + other cross-cutting helpers
  config.py
  extensions.py     # Mongo wrapper, Flask-Login, JWT, CSRF
  __init__.py       # app factory
run.py
requirements.txt
docker/
```

## Interview Prep — AI mock interviews (`app/modules/interview/`)

Integrated from a standalone Interview Accelerator prototype into a full EduAI
module: EduAI design system, login-gated, MongoDB-backed, rate-limited and
audit-logged like every other AI feature here.

**The two-recruiter design.** Rather than one chatbot playing two roles, the
mock interview is run by two personas on two *different* model providers, so
they never end up sounding like the same voice twice:

- **Priya Menon** — Talent Acquisition Partner — runs on **Groq** (warm,
  conversational). Owns the Screening round end-to-end, half the Competency
  round, and comes back for the closing question.
- **Arjun Rao** — Senior Technical Lead — runs on **Gemini** (precise,
  probing). Owns the technical half of Competency and all of Deep-Dive.

Persona routing is deterministic (`personas.persona_for_slot`), not left to
the LLM to decide: Screening = A,A,A · Competency = A,B,A,B · Deep-Dive =
B,B,B,**A** (HR closes it out) — mirroring a real HR-screen → panel →
technical-deep-dive → HR-wrapup loop. Whichever persona asks a question also
grades that answer, so judgement work splits naturally across both models;
one-off tasks (JD/resume analysis, job-fit score, prep plan) are split
explicitly between the two in `analysis_service.py` / `evaluation_service.py`.

**Resilience.** `llm_router.py` tries a persona's assigned provider first and
falls back to the *other* configured provider (same persona/prompt, different
model) if that one is missing or errors — the interview only actually stops
if neither `GROQ_API_KEY` nor `GEMINI_API_KEY` works. Prompts in
`prompts.py` were written specifically to avoid the "reads like a form"
failure mode of LLM interview questions — verbatim anti-repetition, in-persona
example questions, a cliché ban list, and reactions grounded in the
candidate's actual last answer text, not just its score.

**Access model:** both students and teachers can start their own practice
interview (`interview_session` rate limit: 3/day student, unlimited teacher).
Teachers additionally get a class-wide "Class Interview Reports" table
(`GET /interview`) and can leave a saved personalised comment on any
student's completed report (`POST /api/interview/<id>/feedback`).

Env vars: `GEMINI_API_KEY` / `GEMINI_MODEL` (optional — degrades to running
both personas on Groq alone if unset, see `llm_router.py`).

**Known gap:** voice (Web Speech API, `interview-voice.js`) only works in
Chrome/Edge — Firefox/Safari users fall back to typing automatically, but
there's no in-UI notice explaining *why* the mic button is disabled for them
yet.

## Testing this phase yourself

```bash
pip install mongomock pytest   # only needed to run the test suite / smoke-test without a real Atlas cluster
pytest tests/                  # full suite, including tests/test_interview.py
```

`tests/conftest.py` stubs the heavy native deps (torch/dlib/face_recognition/
chromadb/...) that this sandbox can't install and swaps in `mongomock` for
Mongo — none of those stubs are imported at module load time anywhere in the
app, only inside functions this suite never calls, so stubbing them doesn't
hide bugs in those features. `tests/test_interview.py` drives the Interview
Prep module end-to-end (analysis → all 11 questions across both personas →
report → teacher feedback) with the LLM calls mocked, plus unit tests for the
Groq/Gemini fallback logic. This is exactly how every phase of this build,
including this one, was verified before being handed to you.
