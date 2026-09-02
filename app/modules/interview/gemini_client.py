"""
Thin wrapper around Google's Gemini API — the second LLM provider used
for the Interview Prep module's "Arjun Rao" persona and for a couple of
one-off judgement calls (job-fit scoring, prep plan). Mirrors
app/modules/nlp/llm_client.py's shape (a JSON-mode chat call that returns
a parsed dict, with a clear "not configured" exception) so the two
providers are interchangeable from llm_router.py's point of view.
"""
import json
import re

from flask import current_app


class GeminiNotConfigured(RuntimeError):
    """Raised when GEMINI_API_KEY is missing — callers should fall back
    to the other provider rather than surface this directly."""


def _model():
    api_key = current_app.config.get("GEMINI_API_KEY")
    if not api_key:
        raise GeminiNotConfigured(
            "GEMINI_API_KEY is not set. Add it to .env to enable the Technical Lead interviewer persona."
        )
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name=current_app.config.get("GEMINI_MODEL", "gemini-2.5-flash"),
        generation_config={"response_mime_type": "application/json"},
    )


def chat_json(system: str, user: str, temperature: float = 0.7) -> dict:
    """Single JSON-mode completion. Raises GeminiNotConfigured or
    RuntimeError on failure — never returns malformed data silently."""
    model = _model()
    try:
        response = model.generate_content(
            [{"role": "user", "parts": [f"{system}\n\n{user}"]}],
            generation_config={"response_mime_type": "application/json", "temperature": temperature},
        )
        raw = response.text
    except GeminiNotConfigured:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Gemini request failed: {exc}") from exc

    return _parse_json_loose(raw)


def _parse_json_loose(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from Gemini response: {raw[:300]}")
