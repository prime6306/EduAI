def generate_test_questions_prompt(subject, topic, n, q_type):
    system = "You are an exam-question writer. Respond only with valid JSON, no prose, no markdown fences."

    if q_type == "mcq":
        schema = (
            '{"questions": [{"text": "...", "type": "mcq", "options": ["A","B","C","D"], '
            '"correct_answer": "the correct option text", "marks": 1, "explanation": "..."}]}'
        )
    elif q_type == "tf":
        schema = (
            '{"questions": [{"text": "...", "type": "tf", "options": ["True", "False"], '
            '"correct_answer": "True or False", "marks": 1, "explanation": "..."}]}'
        )
    else:
        schema = (
            f'{{"questions": [{{"text": "...", "type": "{q_type}", "correct_answer": "model answer", '
            '"marks": 5, "explanation": "grading guidance for the teacher"}]}'
        )

    user = (
        f'Generate {n} {q_type} exam questions on "{topic}" (subject: "{subject}"). '
        f"Respond as JSON: {schema} with exactly {n} items in the array."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
