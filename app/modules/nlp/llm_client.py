"""
Thin wrapper around the Groq chat-completions API, shared by every AI
feature (Doubt Solver, Study Material Generator, Quiz Generator, and later
RAG/Wellness/Plagiarism). Centralising this here means:
  - one place to swap models/providers later
  - one place to handle a missing/invalid API key gracefully
  - one place to parse JSON-mode responses defensively (LLMs occasionally
    wrap JSON in prose or code fences despite instructions)
"""
import json
import re
from flask import current_app


class LLMNotConfigured(RuntimeError):
    """Raised when GROQ_API_KEY is missing — callers should show the user
    a clear 'AI features are not configured' message rather than a 500."""


def _client():
    api_key = current_app.config.get("GROQ_API_KEY")
    if not api_key:
        raise LLMNotConfigured(
            "GROQ_API_KEY is not set. Add it to .env to enable AI features."
        )
    from groq import Groq
    return Groq(api_key=api_key)


def chat_completion(messages: list[dict], temperature: float = 0.6, max_tokens: int = 2048) -> str:
    """Single non-streamed completion. Returns the assistant's text."""
    client = _client()
    resp = client.chat.completions.create(
        model=current_app.config["GROQ_MODEL"],
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def chat_completion_stream(messages: list[dict], temperature: float = 0.6, max_tokens: int = 2048):
    """Generator yielding text deltas as they arrive, for SSE streaming."""
    client = _client()
    stream = client.chat.completions.create(
        model=current_app.config["GROQ_MODEL"],
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def chat_json(messages: list[dict], temperature: float = 0.4, max_tokens: int = 4096) -> dict | list:
    """
    Completion with Groq's JSON mode. Falls back to extracting the first
    {...} or [...] block if the model still wraps the JSON in prose,
    since JSON mode compliance isn't 100% guaranteed across models.
    """
    client = _client()
    resp = client.chat.completions.create(
        model=current_app.config["GROQ_MODEL"],
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content
    return _parse_json_loose(raw)


def _parse_json_loose(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from LLM response: {raw[:300]}")
