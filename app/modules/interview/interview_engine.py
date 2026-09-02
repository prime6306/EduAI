"""
The adaptive interview engine. This module owns *when* things happen
(level advancement, ending the interview, which persona speaks next);
the LLM decides *what* to ask. Ported from the original Interview
Accelerator's state machine, adapted for MongoDB storage and dual-persona
routing (see personas.py for exactly how turns are split between Priya
and Arjun).
"""
from flask import current_app

from . import prompts, personas, llm_router, store


def start(session_id: str) -> dict:
    session = store.get_session(session_id)
    levels_cfg = current_app.config["INTERVIEW_QUESTIONS_PER_LEVEL"]
    level_names = current_app.config["INTERVIEW_LEVEL_NAMES"]

    persona_key = personas.persona_for_slot(1, 1, levels_cfg[1])
    question_payload = _generate_question(
        session, level=1, history=[], persona_key=persona_key,
        questions_asked_this_level=0, is_closing=False,
    )
    turn_id = store.add_question(
        session_id, level=1, question=question_payload["question"],
        targets_competency=question_payload.get("targets_competency", ""),
        interviewer=persona_key,
    )
    store.update_interview_state(session_id, level=1, questions_this_level=1)

    return _turn_response(turn_id, 1, level_names[1], 1, levels_cfg[1],
                           question_payload["question"], persona_key, is_final=False)


def submit_answer(session_id: str, turn_id: str, answer_text: str) -> dict:
    session = store.get_session(session_id)
    levels_cfg = current_app.config["INTERVIEW_QUESTIONS_PER_LEVEL"]
    level_names = current_app.config["INTERVIEW_LEVEL_NAMES"]

    current_turn = next((t for t in session["turns"] if str(t["turn_id"]) == str(turn_id)), None)
    if current_turn is None:
        raise ValueError("Unknown turn_id for this interview session.")

    eval_persona = current_turn["interviewer"]
    system, user = prompts.evaluate_answer_prompt(
        eval_persona, current_turn["question"], answer_text,
        {"jd_summary": session.get("jd_analysis", {}), "targets_competency": current_turn.get("targets_competency")},
    )
    evaluation = llm_router.complete_json(personas.provider_for(eval_persona), system, user, temperature=0.3)
    store.save_answer_evaluation(session_id, turn_id, answer_text, evaluation)

    level = session["interview_level"]
    asked_this_level = session["questions_this_level"]
    target_for_level = levels_cfg[level]

    advance_level = asked_this_level >= target_for_level
    next_level = level + 1 if advance_level else level

    if next_level > 3:
        return {
            "evaluation": evaluation, "is_final": True,
            "level": level, "level_name": level_names[level],
        }

    session = store.get_session(session_id)  # refresh with the evaluation just saved
    questions_at_next_level_so_far = sum(1 for t in session["turns"] if t["level"] == next_level)
    next_question_number = questions_at_next_level_so_far + 1
    persona_key = personas.persona_for_slot(next_level, next_question_number, levels_cfg[next_level])
    is_closing = next_level == 3 and next_question_number >= levels_cfg[3]

    question_payload = _generate_question(
        session, level=next_level, history=session["turns"], persona_key=persona_key,
        questions_asked_this_level=questions_at_next_level_so_far, is_closing=is_closing,
    )
    new_turn_id = store.add_question(
        session_id, level=next_level, question=question_payload["question"],
        targets_competency=question_payload.get("targets_competency", ""),
        interviewer=persona_key,
    )
    store.update_interview_state(session_id, level=next_level, questions_this_level=next_question_number)

    response = _turn_response(
        new_turn_id, next_level, level_names[next_level], next_question_number, levels_cfg[next_level],
        question_payload["question"], persona_key, is_final=False,
    )
    response["evaluation"] = evaluation
    response["level_changed"] = advance_level
    return response


def _generate_question(session, level, history, persona_key, questions_asked_this_level, is_closing) -> dict:
    level_names = current_app.config["INTERVIEW_LEVEL_NAMES"]
    context = {
        "level": level,
        "level_name": level_names[level],
        "jd_summary": session.get("jd_analysis", {}),
        "resume_summary": session.get("resume_analysis", {}),
        "job_fit": session.get("job_fit", {}),
        "history": [
            {
                "interviewer": personas.get(t["interviewer"])["name"],
                "question": t["question"],
                "answer": t.get("answer"),
                "targets_competency": t.get("targets_competency"),
                "quality_score": t.get("quality_score"),
            }
            for t in history
        ],
        "questions_asked_this_level": questions_asked_this_level,
        "last_answer_quality": session.get("last_answer_quality"),
        "last_answer_text": session.get("last_answer_text"),
        "is_closing_question": is_closing,
    }
    system, user = prompts.next_question_prompt(persona_key, context)
    return llm_router.complete_json(personas.provider_for(persona_key), system, user, temperature=0.75)


def _turn_response(turn_id, level, level_name, questions_this_level, questions_target_this_level,
                    question, persona_key, is_final) -> dict:
    p = personas.get(persona_key)
    return {
        "turn_id": turn_id,
        "level": level,
        "level_name": level_name,
        "questions_this_level": questions_this_level,
        "questions_target_this_level": questions_target_this_level,
        "question": question,
        "interviewer": {
            "key": persona_key, "name": p["name"], "title": p["title"],
            "initials": p["avatar_initials"], "voice_pitch": p["voice_pitch"], "voice_rate": p["voice_rate"],
        },
        "is_final": is_final,
    }
