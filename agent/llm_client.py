"""
LLM client — the single, authoritative gateway for all LLM calls.

Architecture
------------
Primary provider:  Groq
  - role="plan"     → llama-3.3-70b-versatile  (or override via ``model`` arg)
  - role="codegen"  → llama-3.3-70b-versatile  (or override via ``model`` arg)
  - role="critique" → openai/gpt-oss-120b       (or override via ``model`` arg)
  - role="observe"  → llama-3.3-70b-versatile  (or override via ``model`` arg)

Fallback provider: Google Gemini (gemini-2.0-flash)
  - Used automatically when Groq fails after one retry.

Retry / fallback sequence
--------------------------
1. Call Groq.
2. On failure → wait GROQ_RETRY_BACKOFF_SECONDS → retry Groq once.
3. If retry also fails → fall back to Gemini.
4. If Gemini also fails → raise LLMUnavailableError.

The ``role`` parameter is used ONLY for structured logging — it does not
change which provider is called; provider selection is always Groq-first.

Usage
-----
    from agent.llm_client import call_llm
    response = call_llm("Analyse these rows …", role="critique")
"""

from __future__ import annotations

import logging
import os
import time
from typing import Literal

from dotenv import load_dotenv

from agent.config import (
    GEMINI_FALLBACK_MODEL,
    GROQ_CRITIQUE_MODEL,
    GROQ_DEFAULT_MODEL,
    GROQ_RETRY_BACKOFF_SECONDS,
    VALID_ROLES,
)

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

Role = Literal["plan", "codegen", "critique", "observe"]


class LLMUnavailableError(RuntimeError):
    """
    Raised when both Groq and Gemini fail to return a response.

    Attributes:
        groq_error:   The exception raised by Groq (after retry).
        gemini_error: The exception raised by Gemini.
    """

    def __init__(
        self,
        message: str,
        groq_error: Exception | None = None,
        gemini_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.groq_error = groq_error
        self.gemini_error = gemini_error


# ---------------------------------------------------------------------------
# Default model selector
# ---------------------------------------------------------------------------


def _default_model_for_role(role: str) -> str:
    """
    Return the default Groq model identifier for *role*.

    Args:
        role: One of the defined role constants (``"plan"``, ``"codegen"``,
              ``"critique"``, ``"observe"``).

    Returns:
        Groq model identifier string.
    """
    if role == "critique":
        return GROQ_CRITIQUE_MODEL
    return GROQ_DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Provider helpers — imported lazily so missing packages give clear errors
# ---------------------------------------------------------------------------


def _call_groq(prompt: str, model: str, api_key: str) -> str:
    """
    Send *prompt* to Groq and return the assistant message content.

    Args:
        prompt:  Full prompt string.
        model:   Groq model identifier.
        api_key: Groq API key.

    Returns:
        Assistant reply as a plain string.

    Raises:
        Exception: Any Groq SDK or network exception (caller decides retry).
    """
    try:
        from groq import Groq  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "groq package not installed. Run: uv add groq"
        ) from exc

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    content = response.choices[0].message.content
    return (content or "").strip()


def _call_gemini(prompt: str, model: str, api_key: str) -> str:
    """
    Send *prompt* to Google Gemini and return the text response.

    Args:
        prompt:  Full prompt string.
        model:   Gemini model identifier.
        api_key: Google Gemini API key.

    Returns:
        Model reply as a plain string.

    Raises:
        Exception: Any Gemini SDK or network exception.
    """
    try:
        import google.generativeai as genai  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "google-generativeai not installed. Run: uv add google-generativeai"
        ) from exc

    genai.configure(api_key=api_key)
    model_client = genai.GenerativeModel(model)
    response = model_client.generate_content(prompt)
    return (response.text or "").strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def call_llm(
    prompt: str,
    role: str,
    model: str | None = None,
) -> str:
    """
    Send *prompt* to the best-available LLM and return the response.

    Provider selection follows the Groq-first, Gemini-fallback policy:

    1. Call Groq with the resolved model.
    2. On any failure: wait :data:`~agent.config.GROQ_RETRY_BACKOFF_SECONDS`,
       then retry Groq once.
    3. If retry also fails: fall back to Gemini
       (:data:`~agent.config.GEMINI_FALLBACK_MODEL`).
    4. If Gemini also fails: raise :class:`LLMUnavailableError`.

    The ``role`` parameter is used only for structured log output so that
    log consumers can see which pipeline stage made each call. It does NOT
    change which provider or model is selected (use ``model`` for that).

    Args:
        prompt: The complete prompt to send (caller builds the full message).
        role:   Pipeline stage label for logging — one of ``"plan"``,
                ``"codegen"``, ``"critique"``, ``"observe"``.
        model:  Optional Groq model override. When ``None``, the default for
                *role* is used (see :func:`_default_model_for_role`).

    Returns:
        The LLM's response as a plain string.

    Raises:
        ValueError: If *role* is not a recognised value.
        LLMUnavailableError: If both Groq and Gemini fail.
    """
    if role not in VALID_ROLES:
        raise ValueError(
            f"role must be one of {sorted(VALID_ROLES)}, got {role!r}"
        )

    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    resolved_model = model or _default_model_for_role(role)

    logger.info(
        "[call_llm] role=%s model=%s prompt_chars=%d",
        role,
        resolved_model,
        len(prompt),
    )

    # ── Groq attempt 1 ───────────────────────────────────────────────────────
    groq_error: Exception | None = None
    try:
        result = _call_groq(prompt, resolved_model, groq_api_key)
        logger.info("[call_llm] served by groq model=%s role=%s", resolved_model, role)
        return result
    except Exception as exc:  # noqa: BLE001
        groq_error = exc
        logger.warning(
            "[call_llm] Groq attempt 1 failed (role=%s): %s — retrying in %.1fs",
            role,
            exc,
            GROQ_RETRY_BACKOFF_SECONDS,
        )

    # ── Groq retry (attempt 2) ────────────────────────────────────────────────
    time.sleep(GROQ_RETRY_BACKOFF_SECONDS)
    try:
        result = _call_groq(prompt, resolved_model, groq_api_key)
        logger.info(
            "[call_llm] served by groq (retry) model=%s role=%s", resolved_model, role
        )
        return result
    except Exception as exc:  # noqa: BLE001
        groq_error = exc
        logger.warning(
            "[call_llm] Groq retry failed (role=%s): %s — falling back to Gemini",
            role,
            exc,
        )

    # ── Gemini fallback ───────────────────────────────────────────────────────
    gemini_error: Exception | None = None
    try:
        result = _call_gemini(prompt, GEMINI_FALLBACK_MODEL, gemini_api_key)
        logger.info(
            "[call_llm] served by gemini (fallback) model=%s role=%s",
            GEMINI_FALLBACK_MODEL,
            role,
        )
        return result
    except Exception as exc:  # noqa: BLE001
        gemini_error = exc
        logger.error(
            "[call_llm] Gemini fallback also failed (role=%s): %s", role, exc
        )

    raise LLMUnavailableError(
        f"Both Groq and Gemini failed for role={role!r}. "
        f"Groq: {groq_error}. Gemini: {gemini_error}.",
        groq_error=groq_error,
        gemini_error=gemini_error,
    )
