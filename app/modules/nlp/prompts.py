"""
Prompt construction, kept separate from routes/services so tone and
instructions can be tuned in one place without touching business logic.
"""

LEVEL_INSTRUCTIONS = {
    "Simple": "Explain in plain, beginner-friendly language. Avoid jargon; "
              "define any technical term you must use. Use everyday analogies.",
    "Intermediate": "Assume the student has covered the basics. Use standard "
                     "technical terminology, with brief reminders of prerequisite concepts.",
    "Advanced": "Assume strong prior knowledge. Be precise and rigorous, "
                "reference relevant theorems/derivations, and don't over-explain basics.",
}


def doubt_solver_system_prompt(branch: str, year: str, subject: str, level: str) -> str:
    level_instr = LEVEL_INSTRUCTIONS.get(level, LEVEL_INSTRUCTIONS["Intermediate"])
    subject_line = f'Constrain your answers to the subject "{subject}".' if subject else ""
    year_label = f"{year}th-year" if str(year).isdigit() else (year or "undergraduate")
    return (
        "You are an expert, patient AI tutor for engineering students at an Indian "
        f"technical university. The student is a {year_label} {branch or 'engineering'} "
        f"branch student. {subject_line}\n"
        f"{level_instr}\n"
        "Use Markdown. Wrap code in fenced code blocks with a language tag. "
        "Wrap math in LaTeX: inline as \\(...\\) and block equations as \\[...\\]. "
        "Be concise but complete — prefer worked examples over abstract description. "
        "If a question is ambiguous, ask one clarifying question before answering."
    )


def subtopics_prompt(topic: str, subject: str, branch: str, year: str) -> list[dict]:
    system = (
        "You are a curriculum designer for engineering education. Respond only with "
        "valid JSON, no prose, no markdown fences."
    )
    user = (
        f'Break the topic "{topic}" (subject: "{subject}", branch: {branch}, year: {year}) '
        "into exactly 6 logically ordered subtopics suitable for a study guide. "
        'For each, give 3-5 short key points. Respond as JSON: '
        '{"subtopics": [{"title": "...", "key_points": ["...", "..."]}, ...]} '
        "with exactly 6 items in the array."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def explanation_prompt(topic: str, subtopic_title: str, key_points: list[str]) -> list[dict]:
    system = (
        "You are an expert engineering educator writing a study-guide section. "
        "Write clearly, in Markdown, using \\(...\\) for inline math and \\[...\\] for block math."
    )
    points = "\n".join(f"- {p}" for p in key_points)
    user = (
        f'Write a detailed explanation (250-350 words) of "{subtopic_title}", '
        f'a subtopic of "{topic}". Cover these key points:\n{points}\n'
        "Include at least one worked example or illustrative case where relevant."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def summary_prompt(topic: str, subject: str, subtopic_titles: list[str]) -> list[dict]:
    system = "You are an expert educator writing a concise revision summary. Respond in Markdown."
    titles = ", ".join(subtopic_titles)
    user = (
        f'Write a 6-8 sentence revision summary of "{topic}" (subject: "{subject}") '
        f"covering these subtopics at a high level: {titles}. "
        "This is the first thing a student reads before the full material — make it a "
        "useful standalone overview."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def planner_prompt(
    subjects: list[str], exam_date: str, days_until_exam: int | None,
    hours_per_day: int, branch: str, year: str,
) -> list[dict]:
    system = (
        "You are an academic study coach for engineering students. Respond only with "
        "valid JSON, no prose, no markdown fences."
    )
    subjects_line = ", ".join(subjects)
    exam_line = ""
    if exam_date and days_until_exam is not None:
        if days_until_exam >= 0:
            exam_line = f" The exam is on {exam_date}, {days_until_exam} day(s) from today."
        else:
            exam_line = f" The given exam date ({exam_date}) has already passed — treat this as a general revision cycle."
    year_label = f"{year}th-year" if str(year).isdigit() else (year or "undergraduate")
    user = (
        f"Build a repeating 7-day (Monday-Sunday) weekly study schedule for a {year_label} "
        f"{branch or 'engineering'} student preparing for: {subjects_line}. They have "
        f"{hours_per_day} hour(s) available per day.{exam_line}\n"
        "Assume subjects listed first are higher priority and should get more frequent or "
        "longer sessions. Not every subject needs to appear every day, and it's fine to leave "
        "some short gaps — never schedule more total time on a single day than the stated "
        "hours allow (converted to minutes). Give each session a specific topic or activity, "
        "not just the subject name. Also write 3-5 short, practical study tips tailored to this "
        "schedule (e.g. spacing, interleaving, revision timing).\n"
        'Respond as JSON: {"schedule": [{"day": "Monday", "sessions": [{"subject": "...", '
        '"topic": "specific topic or activity, e.g. \'Fourier Series - practice problems\'", '
        '"duration_minutes": 60, "priority": "high|medium|low"}, ...]}, ...] with exactly 7 '
        'entries covering Monday through Sunday in order (a day can have an empty sessions '
        'list if it should be a rest day), "tips": ["...", "..."]} with 3-5 tips.'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def quiz_generation_prompt(topic: str, subject: str, branch: str, year: str, n_questions: int) -> list[dict]:
    system = (
        "You are an exam-question writer for engineering education. Respond only with "
        "valid JSON, no prose, no markdown fences. Every question must have exactly 4 "
        "options with exactly one correct answer."
    )
    user = (
        f'Generate {n_questions} multiple-choice questions on "{topic}" '
        f'(subject: "{subject}", branch: {branch}, year: {year}). '
        "Vary difficulty across the set (roughly 40% easy, 40% medium, 20% hard). "
        'Respond as JSON: {"questions": [{"question": "...", "options": ["A text", "B text", '
        '"C text", "D text"], "correct_index": 0, "explanation": "why this is correct, '
        f'1-2 sentences"}}, ...]}} with exactly {n_questions} items."'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
