"""
End-to-end tests for Interview Prep, run against mongomock with the LLM
calls mocked (see tests/conftest.py for the stubbing approach). These
mirror the manual smoke tests this module was verified with before
shipping — see the module's section in README.md.
"""
import io
from unittest.mock import patch

from tests.conftest import login

FAKE_JD_ANALYSIS = {
    "role_title": "Backend Engineer", "seniority": "Junior",
    "key_responsibilities": ["Build REST APIs", "Own database schema design"],
    "required_skills": ["Python", "Flask", "MongoDB"], "preferred_skills": ["Docker"],
    "technical_competencies": ["API design"], "behavioural_competencies": ["Ownership"],
    "experience_expectations": "0-2 years", "important_keywords": ["Python", "REST"],
}
FAKE_RESUME_ANALYSIS = {
    "candidate_key_skills": ["Python", "Flask"], "relevant_experience": ["Backend intern at Acme"],
    "relevant_projects": ["EduAI platform"], "relevant_achievements": ["Built a Flask app"],
    "strengths_against_jd": ["Python"], "missing_skills": ["Docker"],
    "weak_or_insufficient_areas": ["Testing"], "claims_needing_further_questioning": ["EduAI platform"],
}
FAKE_JOB_FIT = {
    "job_fit_percent": 72, "strong_match": ["Python"], "partial_match": ["MongoDB"],
    "missing_or_weak": ["Docker"], "rationale": "Good language fit, limited deployment experience.",
}
FAKE_EVAL = {
    "quality_score": 8, "competency_tag": "role_fit",
    "what_was_good": "Specific and enthusiastic.", "what_could_be_better": "Could quantify impact.",
    "ideal_direction": "Mention metrics.",
}
FAKE_PREP = {"priorities": [{"rank": 1, "topic": "Docker", "why": "Missing skill vs JD",
                             "review_items": ["Learn Dockerfile basics"]}]}


def _fake_complete_json(provider, system, user, temperature=0.7):
    if "job-fit" in system or "scoring how well" in system:
        return FAKE_JOB_FIT
    if "structured breakdown" in system:
        return FAKE_JD_ANALYSIS
    if "comparing a candidate" in system:
        return FAKE_RESUME_ANALYSIS
    if "evaluating the answer" in system:
        return FAKE_EVAL
    if "preparation plan" in system:
        return FAKE_PREP
    if "asking the next question" in system:
        return {"question": f"Question via {provider}", "targets_competency": "role_fit", "reasoning": "x"}
    raise AssertionError(f"Unexpected prompt reached the fake LLM: {system[:100]}")


def _patched_router():
    return (
        patch("app.modules.interview.analysis_service.llm_router.complete_json", side_effect=_fake_complete_json),
        patch("app.modules.interview.interview_engine.llm_router.complete_json", side_effect=_fake_complete_json),
        patch("app.modules.interview.evaluation_service.llm_router.complete_json", side_effect=_fake_complete_json),
    )


TOTAL_QUESTIONS = 3 + 4 + 4  # Screening + Competency + Deep-Dive


def _run_full_interview(client, sid):
    """Drives all 11 questions to completion, returns the persona sequence.

    `start()` asks question 1, so answering all TOTAL_QUESTIONS questions
    (including the final one, which is when `is_final` flips True) takes
    exactly TOTAL_QUESTIONS `/answer` calls — not TOTAL_QUESTIONS - 1.
    """
    r = client.post(f"/api/interview/{sid}/start")
    assert r.status_code == 200
    payload = r.get_json()
    sequence = [payload["interviewer"]["key"]]
    for _ in range(TOTAL_QUESTIONS):
        r = client.post(f"/api/interview/{sid}/answer", json={
            "turn_id": payload["turn_id"], "answer": "I led the backend design end to end.",
        })
        assert r.status_code == 200, r.get_json()
        payload = r.get_json()
        if payload.get("is_final"):
            break
        sequence.append(payload["interviewer"]["key"])
    return sequence


def test_full_flow_and_persona_routing(client, db, student):
    login(client, student["email"])
    p1, p2, p3 = _patched_router()
    with p1, p2, p3:
        r = client.post("/interview/start", data={
            "jd_text": "We need a backend engineer skilled in Python, Flask, MongoDB.",
            "resume_text": "Riya Sharma. Built EduAI platform using Flask and MongoDB.",
        }, follow_redirects=True)
        assert r.status_code == 200

        session_doc = db.interview_sessions.find_one({})
        assert session_doc["status"] == "interviewing"
        assert session_doc["jd_analysis"]["role_title"] == "Backend Engineer"
        assert session_doc["job_fit"]["job_fit_percent"] == 72
        sid = str(session_doc["_id"])

        assert client.get(f"/interview/{sid}/analysis").status_code == 200
        assert client.get(f"/interview/{sid}/take").status_code == 200

        sequence = _run_full_interview(client, sid)
        labels = ["A" if p == "recruiter_a" else "B" for p in sequence]
        # Screening (A,A,A) -> Competency mixed (A,B,A,B) -> Deep-Dive (B,B,B) + HR closing (A)
        assert labels == ["A", "A", "A", "A", "B", "A", "B", "B", "B", "B", "A"]

        session_doc = db.interview_sessions.find_one({"_id": session_doc["_id"]})
        assert len(session_doc["turns"]) == 11

        r = client.get(f"/interview/{sid}/report")
        assert r.status_code == 200

        session_doc = db.interview_sessions.find_one({"_id": session_doc["_id"]})
        assert session_doc["status"] == "completed"
        assert session_doc["report"]["overall_score"] == 80.0  # every answer scored 8/10
        assert session_doc["report"]["readiness"]["label"] == "Strong Candidate"


def test_rate_limit_caps_student_at_three_sessions_per_day(client, db, student):
    login(client, student["email"])
    p1, p2, p3 = _patched_router()
    with p1, p2, p3:
        for _ in range(3):
            client.post("/interview/start", data={
                "jd_text": "Need a backend engineer.", "resume_text": "Some resume text here.",
            }, follow_redirects=True)
        assert db.interview_sessions.count_documents({"user_id": str(student["_id"])}) == 3

        r = client.post("/interview/start", data={
            "jd_text": "Need a backend engineer.", "resume_text": "Some resume text here.",
        }, follow_redirects=True)
        assert b"limit" in r.data.lower()
        assert db.interview_sessions.count_documents({"user_id": str(student["_id"])}) == 3


def test_jd_and_resume_file_upload_extraction(client, db, student):
    login(client, student["email"])
    with patch("app.modules.interview.analysis_service.llm_router.complete_json", side_effect=_fake_complete_json):
        jd_file = (io.BytesIO(b"We need someone skilled in Kubernetes and Go."), "jd.txt")
        resume_file = (io.BytesIO(b"My name is Riya. I know Kubernetes and Go."), "resume.txt")
        r = client.post("/interview/start", data={
            "jd_text": "", "resume_text": "", "jd_file": jd_file, "resume_file": resume_file,
        }, content_type="multipart/form-data", follow_redirects=True)
        assert r.status_code == 200

        session_doc = db.interview_sessions.find_one({})
        assert "Kubernetes" in session_doc["jd_text"]
        assert "Kubernetes" in session_doc["resume_text"]
        assert session_doc["jd_filename"] == "jd.txt"


def test_start_requires_both_jd_and_resume(client, student):
    login(client, student["email"])
    r = client.post("/interview/start", data={"jd_text": "", "resume_text": ""}, follow_redirects=True)
    assert b"provide both" in r.data.lower()


def test_teacher_sees_class_reports_and_can_leave_feedback(client, db, student, teacher):
    login(client, student["email"])
    p1, p2, p3 = _patched_router()
    with p1, p2, p3:
        client.post("/interview/start", data={
            "jd_text": "Need a backend engineer.", "resume_text": "Riya, backend dev.",
        }, follow_redirects=True)
        session_doc = db.interview_sessions.find_one({})
        sid = str(session_doc["_id"])
        _run_full_interview(client, sid)
        client.get(f"/interview/{sid}/report")  # triggers report build + completion

    client.get("/auth/logout")
    login(client, teacher["email"])

    r = client.get("/interview")
    assert r.status_code == 200
    assert b"Class Interview Reports" in r.data
    assert b"Riya Sharma" in r.data

    r = client.get(f"/interview/{sid}/report")
    assert r.status_code == 200

    r = client.post(f"/api/interview/{sid}/feedback",
                     json={"comment": "Strong on Python, brush up on Docker."})
    assert r.status_code == 200

    session_doc = db.interview_sessions.find_one({"_id": session_doc["_id"]})
    assert session_doc["teacher_feedback"]["comment"].startswith("Strong on Python")
    assert session_doc["teacher_feedback"]["teacher_name"] == teacher["name"]


def test_student_cannot_leave_teacher_feedback(client, db, student, teacher):
    login(client, student["email"])
    p1, p2, p3 = _patched_router()
    with p1, p2, p3:
        client.post("/interview/start", data={
            "jd_text": "Need a backend engineer.", "resume_text": "Riya, backend dev.",
        }, follow_redirects=True)
    session_doc = db.interview_sessions.find_one({})
    sid = str(session_doc["_id"])

    r = client.post(f"/api/interview/{sid}/feedback", json={"comment": "nope"})
    assert r.status_code == 403


def test_llm_router_falls_back_to_the_other_provider(app):
    from app.modules.interview import llm_router
    from app.modules.interview.gemini_client import GeminiNotConfigured

    with app.app_context():
        app.config["GROQ_API_KEY"] = ""
        app.config["GEMINI_API_KEY"] = ""
        try:
            llm_router.complete_json("groq", "sys", "usr")
            assert False, "expected NoInterviewProviderConfigured"
        except llm_router.NoInterviewProviderConfigured:
            pass

        with patch("app.modules.interview.llm_router._call_groq", side_effect=RuntimeError("groq down")), \
             patch("app.modules.interview.llm_router._call_gemini", return_value={"ok": True}):
            assert llm_router.complete_json("groq", "sys", "usr") == {"ok": True}

        with patch("app.modules.interview.llm_router._call_gemini", side_effect=GeminiNotConfigured("no key")), \
             patch("app.modules.interview.llm_router._call_groq", return_value={"ok": True}):
            assert llm_router.complete_json("gemini", "sys", "usr") == {"ok": True}


def test_gemini_client_parses_fenced_json():
    from app.modules.interview.gemini_client import _parse_json_loose
    raw = '```json\n{"a": 1, "b": [1, 2, 3]}\n```'
    assert _parse_json_loose(raw) == {"a": 1, "b": [1, 2, 3]}
