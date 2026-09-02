"""
Orchestrates the pre-interview pipeline: JD analysis -> resume analysis
-> job fit. JD/resume analysis run on Priya's persona (Groq, fast) since
that's genuinely "recruiter prep work"; job-fit scoring runs on Arjun's
persona (Gemini) as a deliberate second opinion from the technical side
of the panel, rather than the same model grading its own analysis
end-to-end.
"""
from . import prompts, personas, llm_router


def analyze_jd(jd_text: str) -> dict:
    system, user = prompts.jd_analysis_prompt(jd_text, personas.RECRUITER_A)
    return llm_router.complete_json("groq", system, user, temperature=0.4)


def analyze_resume(resume_text: str, jd_summary: dict) -> dict:
    system, user = prompts.resume_analysis_prompt(resume_text, jd_summary, personas.RECRUITER_A)
    return llm_router.complete_json("groq", system, user, temperature=0.4)


def job_fit(jd_summary: dict, resume_summary: dict) -> dict:
    system, user = prompts.job_fit_prompt(jd_summary, resume_summary, personas.RECRUITER_B)
    return llm_router.complete_json("gemini", system, user, temperature=0.4)


def run_full_analysis(jd_text: str, resume_text: str) -> tuple:
    jd_analysis = analyze_jd(jd_text)
    resume_analysis = analyze_resume(resume_text, jd_analysis)
    fit = job_fit(jd_analysis, resume_analysis)
    return jd_analysis, resume_analysis, fit
