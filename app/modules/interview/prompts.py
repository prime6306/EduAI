"""
Prompt construction for the Interview Prep module. Two personas ask
questions on two different model providers, so each persona's identity
and speaking style are baked directly into the system prompt for every
call it makes — including the one-off analysis/judgement calls, so those
don't read like a third, generic voice.

This file exists to fix the most common failure mode of LLM-generated
interview questions: they read like a form. Every builder below:

  - names the persona and gives it a short behavioural style, not just
    "you are an interviewer"
  - passes the FULL verbatim text of every question already asked (not
    just topics/tags) so the model can't repeat a phrasing pattern even
    if the topic is technically different
  - includes 1-3 in-persona example questions, lightly filled in with
    the actual candidate/JD specifics, purely as a style anchor (the
    model is told not to reuse them verbatim)
  - explicitly bans stock interview-prep-book clichés
  - is told to react to what the candidate's last answer actually said
    (verbatim), not just its numeric score, so transitions feel like a
    real conversation instead of a question dump
"""
import json
import random

from . import personas

_JSON_ONLY = (
    "Respond with ONLY a single valid JSON object. No markdown fences, "
    "no commentary before or after. If a field has no value, use an empty "
    "string, empty list, or 0 as appropriate — never omit a key."
)

_CLICHE_BAN = (
    "Never open with 'Tell me about yourself', 'What is your biggest weakness', "
    "'Where do you see yourself in 5 years', or any other stock interview-prep-book "
    "line — those read as a form, not a real conversation."
)

_COMPETENCY_ENUM = (
    "one of: role_fit, technical_knowledge, problem_solving, communication, "
    "confidence, depth_of_understanding, behavioural_fit"
)


def _persona_header(persona_key: str) -> str:
    p = personas.get(persona_key)
    return f"You are {p['name']}, {p['title']}, conducting a real interview. You are {p['style']}\n\n"


def _sample(items, fallback: str) -> str:
    items = [i for i in (items or []) if i]
    return random.choice(items) if items else fallback


def _example_questions(persona_key: str, jd_summary: dict, resume_summary: dict) -> list:
    p = personas.get(persona_key)
    fill = {
        "sample_project": _sample(resume_summary.get("relevant_projects"), "one of your recent projects"),
        "sample_experience": _sample(resume_summary.get("relevant_experience"), "your most recent role"),
        "sample_skill": _sample(
            jd_summary.get("required_skills") or resume_summary.get("candidate_key_skills"),
            "the tools on your resume",
        ),
        "sample_scenario": _sample(jd_summary.get("key_responsibilities"), "a production issue under time pressure"),
    }
    out = []
    for template in p["example_questions"]:
        try:
            out.append(template.format(**fill))
        except (KeyError, IndexError):
            out.append(template)
    return out


# ---------------------------------------------------------------- analysis --

def jd_analysis_prompt(jd_text: str, persona_key: str):
    system = (
        _persona_header(persona_key)
        + "Right now you are reading a job description before the interview and extracting a "
        "precise, structured breakdown of what the role actually requires — the same prep work "
        "a real recruiter does before ever meeting a candidate. " + _JSON_ONLY
    )
    schema = {
        "role_title": "string",
        "seniority": "string (e.g. Intern / Junior / Mid / Senior)",
        "key_responsibilities": ["string", "..."],
        "required_skills": ["string", "..."],
        "preferred_skills": ["string", "..."],
        "technical_competencies": ["string", "..."],
        "behavioural_competencies": ["string", "..."],
        "experience_expectations": "string",
        "important_keywords": ["string", "..."],
    }
    user = (
        f"Job Description:\n---\n{jd_text}\n---\n\n"
        f"Extract the structured breakdown. JSON schema (fill it in):\n{json.dumps(schema, indent=2)}"
    )
    return system, user


def resume_analysis_prompt(resume_text: str, jd_summary: dict, persona_key: str):
    system = (
        _persona_header(persona_key)
        + "You are comparing a candidate's resume against this specific job's requirements before "
        "the interview begins. Be specific and evidence-based — cite what's actually in the resume, "
        "never invent experience that isn't there. " + _JSON_ONLY
    )
    schema = {
        "candidate_key_skills": ["string", "..."],
        "relevant_experience": ["string", "..."],
        "relevant_projects": ["string", "..."],
        "relevant_achievements": ["string", "..."],
        "strengths_against_jd": ["string", "..."],
        "missing_skills": ["string", "..."],
        "weak_or_insufficient_areas": ["string", "..."],
        "claims_needing_further_questioning": ["string", "..."],
    }
    user = (
        f"Job requires (summary): {json.dumps(jd_summary)}\n\n"
        f"Candidate Resume:\n---\n{resume_text}\n---\n\n"
        f"Analyse the candidate against the JD. JSON schema:\n{json.dumps(schema, indent=2)}"
    )
    return system, user


def job_fit_prompt(jd_summary: dict, resume_summary: dict, persona_key: str):
    system = (
        _persona_header(persona_key)
        + "You are scoring how well a candidate currently fits this role, 0-100, based on the JD "
        "and resume analysis your colleague on the panel already produced. Be realistic, not "
        "flattering — this score calibrates how hard to push in the actual interview. " + _JSON_ONLY
    )
    schema = {
        "job_fit_percent": "integer 0-100",
        "strong_match": ["string", "..."],
        "partial_match": ["string", "..."],
        "missing_or_weak": ["string", "..."],
        "rationale": "string, 1-2 sentences, in your own voice",
    }
    user = (
        f"JD analysis: {json.dumps(jd_summary)}\n\n"
        f"Resume analysis: {json.dumps(resume_summary)}\n\n"
        f"Produce the job-fit assessment. JSON schema:\n{json.dumps(schema, indent=2)}"
    )
    return system, user


# ---------------------------------------------------------------- interview --

def next_question_prompt(persona_key: str, context: dict):
    """
    context keys: level, level_name, jd_summary, resume_summary, job_fit,
    history (list of {interviewer, question, answer, targets_competency,
    quality_score}), questions_asked_this_level, last_answer_quality,
    last_answer_text, is_closing_question.
    """
    examples = _example_questions(persona_key, context["jd_summary"], context["resume_summary"])
    examples_block = "\n".join(f'  - "{q}"' for q in examples)

    closing_note = ""
    if context.get("is_closing_question"):
        closing_note = (
            "\n\nThis is the FINAL question of the entire interview — you are closing it out. Ask "
            "one warm, genuine closing question (motivation, culture fit, or what questions the "
            "candidate has for the panel) rather than another technical probe."
        )

    system = (
        _persona_header(persona_key)
        + "You are live, mid-interview, asking the next question. You ALWAYS personalise questions "
        "using specifics from the candidate's actual resume and this JD — never generic phrasing. "
        "You remember everything asked and answered so far (given to you verbatim below) and never "
        "repeat a question or its exact structure. " + _CLICHE_BAN + "\n\n"
        "Style anchors — questions in your voice sound like this (do NOT reuse these verbatim, "
        "they're only the tone/structure to match):\n" + examples_block + "\n\n"
        "Level behaviour:\n"
        "- Screening: resume walkthrough, motivation, basic role fit, communication. Friendly, warm-up tone.\n"
        "- Competency: job-specific technical & behavioural competencies, grounded in the candidate's actual projects.\n"
        "- Deep-Dive: challenge vague or weak answers, probe resume claims, ask 'why' and 'how', "
        "introduce scenarios, surface inconsistencies.\n\n"
        "Adaptivity rule: if the candidate's last answer was strong (quality >= 7/10), go deeper on "
        "that same thread or raise difficulty. If it was weak (<= 4/10), don't just move on — ask a "
        "clarifying or simpler follow-up on the same gap before changing topic. If their actual last "
        "answer text is given below, open with a short, natural reaction to something specific they "
        "said (one short clause — a real interviewer acknowledges what was just said) before asking "
        "the next question. Keep the whole thing to 1-3 sentences, spoken naturally — not a "
        "monologue, and never invent resume/JD details that aren't given to you." + closing_note
        + " " + _JSON_ONLY
    )
    schema = {
        "question": "string — the next interview question, spoken naturally, 1-3 sentences",
        "targets_competency": _COMPETENCY_ENUM,
        "reasoning": "string — one sentence, why this question now (internal, not shown to the candidate)",
    }
    user = (
        f"Current level: {context['level']} ({context['level_name']})\n"
        f"Questions asked so far at this level: {context['questions_asked_this_level']}\n"
        f"Candidate's last answer (verbatim, if any): "
        f"{context.get('last_answer_text') or '(none yet — this is the first question)'}\n"
        f"Last answer quality score: {context.get('last_answer_quality')}\n\n"
        f"JD summary: {json.dumps(context['jd_summary'])}\n\n"
        f"Resume summary: {json.dumps(context['resume_summary'])}\n\n"
        f"Job fit gaps: {json.dumps(context.get('job_fit', {}))}\n\n"
        f"Full interview transcript so far (verbatim — never repeat these questions or their "
        f"structure):\n{json.dumps(context['history'], indent=2)}\n\n"
        f"Generate the next question. JSON schema:\n{json.dumps(schema, indent=2)}"
    )
    return system, user


def evaluate_answer_prompt(persona_key: str, question: str, answer: str, context: dict):
    system = (
        _persona_header(persona_key)
        + "You are evaluating the answer you just heard, in real time, as part of the interview. "
        "Be specific and actionable — never generic feedback like 'improve your communication'. "
        "Ground every point in what the candidate actually said. " + _JSON_ONLY
    )
    schema = {
        "quality_score": "integer 0-10",
        "competency_tag": _COMPETENCY_ENUM,
        "what_was_good": "string, specific",
        "what_could_be_better": "string, specific",
        "ideal_direction": "string — what a stronger answer would have covered",
    }
    user = (
        f"JD summary: {json.dumps(context.get('jd_summary', {}))}\n\n"
        f"This question was targeting competency: {context.get('targets_competency', 'unspecified')}\n\n"
        f"Question you asked: {question}\n\n"
        f"Candidate's answer: {answer}\n\n"
        f"Evaluate it. JSON schema:\n{json.dumps(schema, indent=2)}"
    )
    return system, user


def prep_plan_prompt(jd_summary: dict, weaknesses: list, missing_skills: list, persona_key: str):
    system = (
        _persona_header(persona_key)
        + "The interview is over. You're now building a prioritised interview-preparation plan for "
        "this candidate to work through before their real interview. Priority 1 is the single most "
        "important gap to close. " + _JSON_ONLY
    )
    schema = {
        "priorities": [
            {
                "rank": "integer starting at 1",
                "topic": "string",
                "why": "string, 1 sentence",
                "review_items": ["string", "..."],
            }
        ]
    }
    user = (
        f"JD summary: {json.dumps(jd_summary)}\n\n"
        f"Weak areas observed in interview: {json.dumps(weaknesses)}\n\n"
        f"Missing skills vs JD: {json.dumps(missing_skills)}\n\n"
        f"Build a 3-5 item prioritised prep plan. JSON schema:\n{json.dumps(schema, indent=2)}"
    )
    return system, user
