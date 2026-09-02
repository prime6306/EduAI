def question_paper_prompt(
    subject, units, mcq_count, short_count, long_count,
    marks_mcq, marks_short, marks_long,
    difficulty_split, bloom_levels,
):
    system = (
        "You are an experienced university exam-paper setter. Respond only with valid JSON, "
        "no prose, no markdown fences. Every MCQ must have exactly 4 options with exactly one correct answer."
    )

    units_str = ", ".join(units) if units else "the subject's core syllabus"
    bloom_str = f" Target these Bloom's Taxonomy levels where possible: {', '.join(bloom_levels)}." if bloom_levels else ""
    difficulty_str = (
        f"Aim for roughly {difficulty_split.get('easy', 40)}% Easy, "
        f"{difficulty_split.get('medium', 40)}% Medium, {difficulty_split.get('hard', 20)}% Hard questions."
    )

    parts = [f'Generate a university exam paper for "{subject}", covering units/topics: {units_str}. {difficulty_str}{bloom_str}']
    schema_parts = []

    if mcq_count > 0:
        parts.append(f"Include exactly {mcq_count} multiple-choice questions, {marks_mcq} mark(s) each.")
        schema_parts.append(
            '"mcq": [{"text": "...", "options": ["A", "B", "C", "D"], "correct_index": 0, '
            '"difficulty": "Easy|Medium|Hard", "bloom_level": "Remember|Understand|Apply|Analyse", "topic": "..."}]'
        )
    if short_count > 0:
        parts.append(f"Include exactly {short_count} short-answer questions, {marks_short} mark(s) each.")
        schema_parts.append(
            '"short": [{"text": "...", "answer": "model answer, 2-4 sentences", '
            '"difficulty": "Easy|Medium|Hard", "bloom_level": "...", "topic": "..."}]'
        )
    if long_count > 0:
        parts.append(f"Include exactly {long_count} long-answer questions, {marks_long} mark(s) each.")
        schema_parts.append(
            '"long": [{"text": "...", "answer": "model answer, 5-8 sentences or key points", '
            '"difficulty": "Easy|Medium|Hard", "bloom_level": "...", "topic": "..."}]'
        )

    user = " ".join(parts) + f' Respond as JSON: {{{", ".join(schema_parts)}}}'
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def single_question_prompt(subject, topic, q_type, marks, difficulty):
    system = "You are an exam-paper setter. Respond only with valid JSON, no prose, no markdown fences."
    if q_type == "mcq":
        schema = '{"text": "...", "options": ["A","B","C","D"], "correct_index": 0}'
    else:
        schema = '{"text": "...", "answer": "model answer"}'
    user = (
        f'Generate one {difficulty} difficulty, {marks}-mark {q_type} exam question on "{topic}" '
        f'(subject: "{subject}"). Respond as JSON: {schema}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
