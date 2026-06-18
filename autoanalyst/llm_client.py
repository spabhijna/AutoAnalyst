"""
LLM client — the single, authoritative gateway for all LLM calls.

Rules:
- All code in the project MUST call :func:`call_llm` rather than
  instantiating Groq or Gemini clients directly.
- Routing: ``role="generate"`` → Groq; ``role="critique"`` → Gemini.
  This keeps the two stages on genuinely independent providers so the
  critique cannot share cached reasoning with the generation.
- Pass an explicit *model* to override the default for that role.
- All errors are caught narrowly and re-raised as :class:`LLMError`.
"""

from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

Role = Literal["generate", "critique"]


class LLMError(RuntimeError):
    """Raised when an LLM call fails after all retry attempts."""


# ---------------------------------------------------------------------------
# Internal helpers — import lazily to avoid hard failures when keys are absent
# ---------------------------------------------------------------------------


def _call_groq(prompt: str, model: str, api_key: str) -> str:
    """
    Send *prompt* to Groq and return the assistant message content.

    Args:
        prompt:  The full prompt string to send as a user message.
        model:   Groq model identifier (e.g. ``"llama-3.3-70b-versatile"``).
        api_key: Groq API key.

    Returns:
        The assistant's reply as a plain string.

    Raises:
        LLMError: On any Groq API or network error.
    """
    try:
        from groq import Groq  # type: ignore[import-untyped]

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""
    except ImportError as exc:
        raise LLMError("groq package is not installed. Run: pip install groq") from exc
    except Exception as exc:  # noqa: BLE001
        raise LLMError(f"Groq call failed: {exc}") from exc


def _call_gemini(prompt: str, model: str, api_key: str) -> str:
    """
    Send *prompt* to Google Gemini and return the text response.

    Args:
        prompt:  The full prompt string.
        model:   Gemini model identifier (e.g. ``"gemini-2.0-flash"``).
        api_key: Google Gemini API key.

    Returns:
        The model's reply as a plain string.

    Raises:
        LLMError: On any Gemini API or network error.
    """
    try:
        import google.generativeai as genai  # type: ignore[import-untyped]

        genai.configure(api_key=api_key)
        model_client = genai.GenerativeModel(model)
        response = model_client.generate_content(prompt)
        return response.text or ""
    except ImportError as exc:
        raise LLMError(
            "google-generativeai package is not installed. "
            "Run: pip install google-generativeai"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise LLMError(f"Gemini call failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def call_llm(
    prompt: str,
    role: Role,
    *,
    model: str | None = None,
    config: "PipelineConfig | None" = None,  # type: ignore[name-defined]  # noqa: F821
) -> str:
    """
    Route *prompt* to the appropriate LLM based on *role*.

    - ``role="generate"`` → Groq (fast, used for PLAN / CODE-GEN / EXECUTE stages)
    - ``role="critique"`` → Google Gemini (independent provider for CRITIQUE stage)

    The function reads credentials from *config* when supplied, otherwise
    falls back to importing :func:`autoanalyst.config.PipelineConfig.from_env`.
    This allows callers in tests to inject a mock config without touching the
    environment.

    Args:
        prompt: The complete prompt to send. The caller is responsible for
                building the full message (system + context + instruction).
        role:   ``"generate"`` or ``"critique"`` — controls which provider
                and default model are used.
        model:  Optional override for the model identifier. When *None*, the
                default from the active :class:`PipelineConfig` is used.
        config: Optional :class:`PipelineConfig` instance. When *None*, one
                is constructed from the current environment.

    Returns:
        The LLM's response as a plain string (stripped of leading/trailing
        whitespace).

    Raises:
        LLMError: If the underlying provider call fails.
        ValueError: If *role* is not one of the allowed values.
    """
    if role not in ("generate", "critique"):
        raise ValueError(f"role must be 'generate' or 'critique', got {role!r}")

    # Resolve config — import here to avoid circular imports at module level
    if config is None:
        from autoanalyst.config import PipelineConfig

        config = PipelineConfig.from_env()

    # Dry-run mode: return a canned stub so the pipeline can be exercised
    # end-to-end without live credentials.
    if config.dry_run:
        stub = f"[DRY-RUN stub for role={role!r}] Received {len(prompt)} chars."
        logger.debug("dry-run: skipping LLM call, returning stub")
        return stub

    if role == "generate":
        resolved_model = model or config.groq_model
        logger.debug("Calling Groq model=%s, prompt_len=%d", resolved_model, len(prompt))
        result = _call_groq(prompt, resolved_model, config.groq_api_key)

    else:  # role == "critique"
        resolved_model = model or config.gemini_model
        logger.debug(
            "Calling Gemini model=%s, prompt_len=%d", resolved_model, len(prompt)
        )
        result = _call_gemini(prompt, resolved_model, config.gemini_api_key)

    logger.debug("LLM response_len=%d", len(result))
    return result.strip()


# Avoid circular import at type-checking time
from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    from autoanalyst.config import PipelineConfig  # noqa: F401
