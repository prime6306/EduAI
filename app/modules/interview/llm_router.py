"""
Routes a (system, user) prompt pair to whichever real LLM provider a
persona is assigned to — Groq for Priya, Gemini for Arjun — with a
same-persona fallback to the *other* configured provider if the assigned
one is missing or errors.

This is the graceful-degradation behaviour the rest of EduAI's AI
features already follow (spec: "all external API calls wrapped in
try/except with graceful fallback ... application never crashes due to
API failure"), extended to a two-provider feature: if only one API key
is set, both interviewer personas simply run on that one model instead
— the candidate still gets two distinct personalities/prompts, just
voiced by the same underlying LLM until the second key is added. The
interview only actually stops if *neither* key works.
"""
import logging

from app.modules.nlp.llm_client import chat_json as _groq_chat_json, LLMNotConfigured
from . import gemini_client
from .gemini_client import GeminiNotConfigured

logger = logging.getLogger("eduai.interview")


class NoInterviewProviderConfigured(RuntimeError):
    """Raised when neither GROQ_API_KEY nor GEMINI_API_KEY works."""


def _call_groq(system: str, user: str, temperature: float) -> dict:
    return _groq_chat_json(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
    )


def _call_gemini(system: str, user: str, temperature: float) -> dict:
    return gemini_client.chat_json(system, user, temperature=temperature)


def complete_json(provider: str, system: str, user: str, temperature: float = 0.7) -> dict:
    """
    `provider` is the persona's *preferred* provider ("groq" or
    "gemini"). Tries that first, then the other one, before giving up.
    """
    primary, fallback = (_call_groq, _call_gemini) if provider == "groq" else (_call_gemini, _call_groq)

    try:
        return primary(system, user, temperature)
    except (LLMNotConfigured, GeminiNotConfigured, RuntimeError, ValueError) as exc:
        logger.warning("Interview: primary provider '%s' unavailable (%s) — trying the other provider.",
                        provider, exc)

    try:
        return fallback(system, user, temperature)
    except (LLMNotConfigured, GeminiNotConfigured, RuntimeError, ValueError) as exc:
        logger.error("Interview: both LLM providers failed/unconfigured: %s", exc)
        raise NoInterviewProviderConfigured(
            "Interview Prep needs at least one of GROQ_API_KEY or GEMINI_API_KEY configured."
        ) from exc
