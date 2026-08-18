"""
LLM client — the single, authoritative gateway for all LLM calls.

Architecture
------------
Three-tier provider chain with per-role routing::

    plan     : Groq → OpenRouter
    codegen  : Groq (Codestral) → Cerebras → OpenRouter (Codestral)
    critique : Groq (gpt-oss-120b) → Cerebras (same) → Gemini
    default  : Groq → OpenRouter

Each provider in the chain is retried once (with backoff) before falling to
the next.  Fallback is triggered on 429, 5xx, timeout, or general SDK error.
400 and 401 are never retried — they surface immediately.

The ``role`` parameter drives which chain is selected when *model* is None.
When *model* is provided explicitly, the default chain is used.
"""

from __future__ import annotations

import json as _json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Literal

from dotenv import load_dotenv

from agent.config import (
    CEREBRAS_BASE_URL,
    CEREBRAS_CODEGEN_FALLBACK_MODEL,
    CEREBRAS_CRITIQUE_FALLBACK_MODEL,
    GEMINI_FALLBACK_MODEL,
    GROQ_CODEGEN_MODEL,
    GROQ_CRITIQUE_MODEL,
    GROQ_DEFAULT_MODEL,
    OPENROUTER_BASE_URL,
    OPENROUTER_CODEGEN_FALLBACK_MODEL,
    OPENROUTER_PLAN_FALLBACK_MODEL,
    OPENROUTER_REFERER,
    RETRY_BACKOFF_SECONDS,
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
    Raised when every provider in the chain fails.

    Attributes:
        errors: List of (provider_name, exception) tuples for each failed level.
    """

    def __init__(
        self, message: str, errors: list[tuple[str, Exception]] | None = None
    ) -> None:
        super().__init__(message)
        self.errors = errors or []


# ---------------------------------------------------------------------------
# Chain builder — per-role routing
# ---------------------------------------------------------------------------

# A chain level: (provider_label, callable, model_string)
ChainLevel = tuple[str, Callable[..., str], str]


def _build_chain(role: str, model_override: str | None) -> list[ChainLevel]:
    """
    Build the provider chain for *role*.

    Each element is ``(provider_name, call_fn, model)``.  The chain runner
    iterates through levels, retrying each once, until one succeeds.
    """
    if role == "plan":
        return [
            ("groq", _call_groq, model_override or GROQ_DEFAULT_MODEL),
            (
                "openrouter",
                _call_openrouter,
                model_override or OPENROUTER_PLAN_FALLBACK_MODEL,
            ),
        ]
    if role == "codegen":
        return [
            ("groq", _call_groq, model_override or GROQ_CODEGEN_MODEL),
            (
                "cerebras",
                _call_cerebras,
                model_override or CEREBRAS_CODEGEN_FALLBACK_MODEL,
            ),
            (
                "openrouter",
                _call_openrouter,
                model_override or OPENROUTER_CODEGEN_FALLBACK_MODEL,
            ),
        ]
    if role == "critique":
        return [
            ("groq", _call_groq, model_override or GROQ_CRITIQUE_MODEL),
            (
                "cerebras",
                _call_cerebras,
                model_override or CEREBRAS_CRITIQUE_FALLBACK_MODEL,
            ),
            ("gemini", _call_gemini, model_override or GEMINI_FALLBACK_MODEL),
        ]
    # observe and any other role → Groq first, OpenRouter fallback
    resolved = model_override or GROQ_DEFAULT_MODEL
    return [
        ("groq", _call_groq, resolved),
        ("openrouter", _call_openrouter, resolved),
    ]


# ---------------------------------------------------------------------------
# Retry / fallback logic
# ---------------------------------------------------------------------------


def _should_fallback(exc: Exception) -> bool:
    """
    Return True if the exception justifies a retry or provider fallback.

    Falls back on: HTTP 429 (rate-limit), HTTP 5xx (server error),
    connection timeouts, and generic SDK exceptions.
    Does NOT fall back on: HTTP 400 (bad request) or HTTP 401/403 (auth).
    """
    msg = str(exc).lower()
    # Catch HTTP errors via urllib
    if isinstance(exc, urllib.error.HTTPError):
        code = exc.code
        return code >= 500 or code == 429
    # Catch connection / timeout errors
    if isinstance(exc, (urllib.error.URLError, TimeoutError, OSError)):
        return True
    # Groq SDK exceptions surface status codes in messages
    if any(
        tag in msg
        for tag in (
            "rate limit",
            "rate_limit",
            "429",
            "server error",
            "503",
            "502",
            "500",
        )
    ):
        return True
    if any(tag in msg for tag in ("timeout", "timed out", "connection")):
        return True
    # 404 model_not_found — the model isn't available on this provider, try next
    if "404" in msg and "model_not_found" in msg:
        return True
    # Other 404s (e.g. wrong endpoint) are config errors
    if "404" in msg:
        return False
    # 400 / 401 / 403 are config/auth errors — do NOT fall back
    if any(
        tag in msg for tag in ("400", "401", "403", "invalid api key", "unauthorized")
    ):
        return False
    # Default: fall back for unknown errors (safer than silently accepting)
    return True


def _try_with_retry(
    provider_name: str,
    call_fn: Callable[..., str],
    prompt: str,
    model: str,
) -> tuple[str | None, Exception | None]:
    """
    Call *call_fn* once; on fallback-worthy failure, retry after backoff.

    Returns:
        (result, None) on success, or (None, last_error) on failure.
    """
    # ── Attempt 1 ──────────────────────────────────────────────────────────
    try:
        result = call_fn(prompt, model)
        logger.info("[call_llm] served by %s model=%s", provider_name, model)
        return result, None
    except Exception as exc:  # noqa: BLE001
        if not _should_fallback(exc):
            # Auth / config error — surface immediately
            logger.error(
                "[call_llm] %s fatal error (not retrying): %s", provider_name, exc
            )
            return None, exc
        logger.warning(
            "[call_llm] %s attempt 1 failed: %s — retrying in %.1fs",
            provider_name,
            exc,
            RETRY_BACKOFF_SECONDS,
        )

    # ── Attempt 2 (retry) ──────────────────────────────────────────────────
    time.sleep(RETRY_BACKOFF_SECONDS)
    try:
        result = call_fn(prompt, model)
        logger.info("[call_llm] served by %s (retry) model=%s", provider_name, model)
        return result, None
    except Exception as exc:  # noqa: BLE001
        logger.warning("[call_llm] %s retry failed: %s", provider_name, exc)
        return None, exc


# ---------------------------------------------------------------------------
# Provider helpers — imported lazily so missing packages give clear errors
# ---------------------------------------------------------------------------


def _call_groq(prompt: str, model: str) -> str:
    """
    Send *prompt* to Groq and return the assistant message content.

    Args:
        prompt: Full prompt string.
        model:  Groq model identifier.

    Returns:
        Assistant reply as a plain string.

    Raises:
        RuntimeError: If groq package is not installed.
        Exception:    Any Groq SDK or network error.
    """
    try:
        from groq import Groq  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("groq package not installed. Run: uv add groq") from exc

    api_key = os.environ.get("GROQ_API_KEY", "")
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    content = response.choices[0].message.content
    return (content or "").strip()


def _call_cerebras(prompt: str, model: str) -> str:
    """
    Send *prompt* to Cerebras (OpenAI-compatible) and return the response.

    Uses the cerebras-cloud-sdk which is OpenAI-compatible.

    Args:
        prompt: Full prompt string.
        model:  Cerebras model identifier.

    Returns:
        Model reply as a plain string.

    Raises:
        RuntimeError: If cerebras-cloud-sdk is not installed.
        Exception:    Any SDK or network error.
    """
    try:
        from cerebras.cloud.sdk import Cerebras  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "cerebras-cloud-sdk not installed. Run: uv add cerebras-cloud-sdk"
        ) from exc

    api_key = os.environ.get("CEREBRAS_API_KEY", "")
    client = Cerebras(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    content = response.choices[0].message.content
    return (content or "").strip()


def _call_openrouter(prompt: str, model: str) -> str:
    """
    Send *prompt* to OpenRouter (OpenAI-compatible HTTP API) and return the response.

    Uses stdlib urllib — no extra SDK required.

    Args:
        prompt: Full prompt string.
        model:  OpenRouter model identifier (e.g. "mistralai/codestral-2501:free").

    Returns:
        Model reply as a plain string.

    Raises:
        RuntimeError: If OPENROUTER_API_KEY is not set.
        urllib.error.HTTPError: On HTTP errors.
        urllib.error.URLError:  On connection errors.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment")

    payload = _json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": OPENROUTER_REFERER,
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        body = _json.loads(resp.read())
        content = body["choices"][0]["message"]["content"]
        return (content or "").strip()


def _call_gemini(prompt: str, model: str) -> str:
    """
    Send *prompt* to Google Gemini and return the text response.

    Args:
        prompt: Full prompt string.
        model:  Gemini model identifier.

    Returns:
        Model reply as a plain string.

    Raises:
        RuntimeError: If google-genai is not installed.
        Exception:    Any Gemini SDK or network exception.
    """
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai not installed. Run: uv add google-genai"
        ) from exc

    api_key = os.environ.get("GEMINI_API_KEY", "")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=prompt)
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

    Provider chain depends on *role* (see module docstring).  Each level is
    retried once with backoff before falling to the next level.  If every
    provider in the chain fails, :class:`LLMUnavailableError` is raised.

    When *model* is provided, it overrides the default model for every level
    in the chain.  The chain itself is determined by *role*.

    Args:
        prompt: The complete prompt to send.
        role:   Pipeline stage label — one of ``"plan"``, ``"codegen"``,
                ``"critique"``, ``"observe"``.
        model:  Optional model override for all chain levels.

    Returns:
        The LLM's response as a plain string.

    Raises:
        ValueError:            If *role* is not a recognised value.
        LLMUnavailableError:   If every provider in the chain fails.
    """
    if role not in VALID_ROLES:
        raise ValueError(f"role must be one of {sorted(VALID_ROLES)}, got {role!r}")

    chain = _build_chain(role, model)
    resolved_model = model or chain[0][2]

    logger.info(
        "[call_llm] role=%s model=%s prompt_chars=%d providers=%s",
        role,
        resolved_model,
        len(prompt),
        "→".join(name for name, _, _ in chain),
    )

    errors: list[tuple[str, Exception]] = []
    for provider_name, call_fn, provider_model in chain:
        result, error = _try_with_retry(provider_name, call_fn, prompt, provider_model)
        if result is not None:
            return result
        if error is not None:
            errors.append((provider_name, error))
        # If the error isn't fallback-worthy (auth), the helper returned it
        # but we still try next providers in case one has valid credentials.

    raise LLMUnavailableError(
        f"All providers failed for role={role!r}. "
        + "; ".join(f"{n}: {e}" for n, e in errors),
        errors=errors,
    )
