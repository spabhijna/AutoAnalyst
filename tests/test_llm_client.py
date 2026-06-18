"""
Tests for agent.llm_client — the single LLM gateway.

Contract under test
-------------------
1. Happy path: Groq succeeds on attempt 1 → result returned, Gemini not called.
2. Retry: Groq fails on attempt 1 → backoff → retry → Groq succeeds on attempt 2.
   Gemini must NOT be called.
3. Fallback: Groq fails on both attempts → Gemini called → result returned.
4. Total failure: Both Groq (both attempts) and Gemini fail →
   LLMUnavailableError raised with both exception references attached.
5. Role is for logging only — all roles resolve correctly to their default model.
6. model= override is passed through to Groq (not the default for the role).
7. Invalid role raises ValueError before any provider is called.
8. LLMUnavailableError carries both provider errors as attributes.

Notes
-----
- `time.sleep` is patched throughout so tests don't actually wait.
- `_call_groq` and `_call_gemini` are patched at the agent.llm_client module
  level (not at the import source) per Python mock best practices.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from agent.config import (
    GEMINI_FALLBACK_MODEL,
    GROQ_CRITIQUE_MODEL,
    GROQ_DEFAULT_MODEL,
)
from agent.llm_client import LLMUnavailableError, call_llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GROQ_PATH = "agent.llm_client._call_groq"
GEMINI_PATH = "agent.llm_client._call_gemini"
SLEEP_PATH = "agent.llm_client.time.sleep"


# ---------------------------------------------------------------------------
# 1. Happy path — Groq succeeds immediately
# ---------------------------------------------------------------------------


class TestGroqSuccessOnFirstAttempt:
    def test_returns_groq_response(self) -> None:
        with patch(GROQ_PATH, return_value="groq answer") as mock_groq, \
             patch(GEMINI_PATH) as mock_gemini, \
             patch(SLEEP_PATH):
            result = call_llm("prompt", role="plan")
        assert result == "groq answer"
        mock_groq.assert_called_once()
        mock_gemini.assert_not_called()

    def test_no_sleep_on_success(self) -> None:
        with patch(GROQ_PATH, return_value="ok"), \
             patch(GEMINI_PATH), \
             patch(SLEEP_PATH) as mock_sleep:
            call_llm("prompt", role="codegen")
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Retry — Groq fails once, succeeds on retry; Gemini never called
# ---------------------------------------------------------------------------


class TestGroqRetry:
    def test_retry_on_first_failure(self) -> None:
        """Groq: fail → sleep → succeed.  Gemini: not called."""
        groq_responses = [Exception("timeout"), "groq retry answer"]

        def groq_side_effect(*args, **kwargs):
            resp = groq_responses.pop(0)
            if isinstance(resp, Exception):
                raise resp
            return resp

        with patch(GROQ_PATH, side_effect=groq_side_effect) as mock_groq, \
             patch(GEMINI_PATH) as mock_gemini, \
             patch(SLEEP_PATH) as mock_sleep:
            result = call_llm("prompt", role="plan")

        assert result == "groq retry answer"
        assert mock_groq.call_count == 2
        mock_gemini.assert_not_called()
        mock_sleep.assert_called_once()  # exactly one backoff

    def test_sleep_duration_matches_config(self) -> None:
        """The backoff sleep must use GROQ_RETRY_BACKOFF_SECONDS from config."""
        from agent.config import GROQ_RETRY_BACKOFF_SECONDS

        groq_responses = [Exception("err"), "ok"]

        def side_effect(*a, **k):
            r = groq_responses.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        with patch(GROQ_PATH, side_effect=side_effect), \
             patch(GEMINI_PATH), \
             patch(SLEEP_PATH) as mock_sleep:
            call_llm("prompt", role="codegen")

        mock_sleep.assert_called_once_with(GROQ_RETRY_BACKOFF_SECONDS)

    def test_no_fallback_when_retry_succeeds(self) -> None:
        """Gemini must not be called if the Groq retry succeeds."""
        seq = [Exception("fail"), "success"]

        def side(*a, **k):
            r = seq.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        with patch(GROQ_PATH, side_effect=side), \
             patch(GEMINI_PATH) as mock_gemini, \
             patch(SLEEP_PATH):
            call_llm("prompt", role="observe")

        mock_gemini.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Fallback — both Groq attempts fail → Gemini called
# ---------------------------------------------------------------------------


class TestGeminiFallback:
    def test_gemini_called_after_two_groq_failures(self) -> None:
        with patch(GROQ_PATH, side_effect=Exception("groq down")) as mock_groq, \
             patch(GEMINI_PATH, return_value="gemini answer") as mock_gemini, \
             patch(SLEEP_PATH):
            result = call_llm("prompt", role="critique")

        assert result == "gemini answer"
        assert mock_groq.call_count == 2   # attempt 1 + retry
        mock_gemini.assert_called_once()

    def test_gemini_uses_fallback_model(self) -> None:
        """Gemini must be called with GEMINI_FALLBACK_MODEL, not the Groq model."""
        with patch(GROQ_PATH, side_effect=Exception("down")), \
             patch(GEMINI_PATH, return_value="ok") as mock_gemini, \
             patch(SLEEP_PATH):
            call_llm("prompt", role="plan")

        # _call_gemini(prompt, model, api_key) — model is positional arg 2
        _, called_model, _ = mock_gemini.call_args[0]
        assert called_model == GEMINI_FALLBACK_MODEL


# ---------------------------------------------------------------------------
# 4. Total failure — LLMUnavailableError
# ---------------------------------------------------------------------------


class TestTotalFailure:
    def test_raises_llm_unavailable_error(self) -> None:
        with patch(GROQ_PATH, side_effect=Exception("groq dead")), \
             patch(GEMINI_PATH, side_effect=Exception("gemini dead")), \
             patch(SLEEP_PATH):
            with pytest.raises(LLMUnavailableError):
                call_llm("prompt", role="plan")

    def test_error_carries_both_exceptions(self) -> None:
        groq_exc = RuntimeError("groq 503")
        gemini_exc = RuntimeError("gemini 429")

        with patch(GROQ_PATH, side_effect=groq_exc), \
             patch(GEMINI_PATH, side_effect=gemini_exc), \
             patch(SLEEP_PATH):
            with pytest.raises(LLMUnavailableError) as exc_info:
                call_llm("prompt", role="codegen")

        err = exc_info.value
        assert err.groq_error is groq_exc
        assert err.gemini_error is gemini_exc

    def test_error_message_mentions_both_providers(self) -> None:
        with patch(GROQ_PATH, side_effect=Exception("groq dead")), \
             patch(GEMINI_PATH, side_effect=Exception("gemini dead")), \
             patch(SLEEP_PATH):
            with pytest.raises(LLMUnavailableError, match="Groq"):
                call_llm("prompt", role="plan")

    def test_groq_called_twice_before_raising(self) -> None:
        """Must attempt Groq exactly twice (initial + 1 retry) before giving up."""
        with patch(GROQ_PATH, side_effect=Exception("down")) as mock_groq, \
             patch(GEMINI_PATH, side_effect=Exception("down")), \
             patch(SLEEP_PATH):
            with pytest.raises(LLMUnavailableError):
                call_llm("prompt", role="plan")

        assert mock_groq.call_count == 2


# ---------------------------------------------------------------------------
# 5. Role → model routing (role is for logging only, but determines default)
# ---------------------------------------------------------------------------


class TestRoleModelRouting:
    def test_plan_uses_default_groq_model(self) -> None:
        with patch(GROQ_PATH, return_value="ok") as mock_groq, \
             patch(SLEEP_PATH):
            call_llm("p", role="plan")
        _, called_model, _ = mock_groq.call_args[0]
        assert called_model == GROQ_DEFAULT_MODEL

    def test_codegen_uses_default_groq_model(self) -> None:
        with patch(GROQ_PATH, return_value="ok") as mock_groq, \
             patch(SLEEP_PATH):
            call_llm("p", role="codegen")
        _, called_model, _ = mock_groq.call_args[0]
        assert called_model == GROQ_DEFAULT_MODEL

    def test_observe_uses_default_groq_model(self) -> None:
        with patch(GROQ_PATH, return_value="ok") as mock_groq, \
             patch(SLEEP_PATH):
            call_llm("p", role="observe")
        _, called_model, _ = mock_groq.call_args[0]
        assert called_model == GROQ_DEFAULT_MODEL

    def test_critique_uses_critique_groq_model(self) -> None:
        with patch(GROQ_PATH, return_value="ok") as mock_groq, \
             patch(SLEEP_PATH):
            call_llm("p", role="critique")
        _, called_model, _ = mock_groq.call_args[0]
        assert called_model == GROQ_CRITIQUE_MODEL


# ---------------------------------------------------------------------------
# 6. model= override
# ---------------------------------------------------------------------------


class TestModelOverride:
    def test_override_passed_to_groq(self) -> None:
        with patch(GROQ_PATH, return_value="ok") as mock_groq, \
             patch(SLEEP_PATH):
            call_llm("p", role="plan", model="custom-model-xyz")
        _, called_model, _ = mock_groq.call_args[0]
        assert called_model == "custom-model-xyz"

    def test_override_does_not_affect_gemini_model(self) -> None:
        """Even with a Groq model override, Gemini fallback uses GEMINI_FALLBACK_MODEL."""
        with patch(GROQ_PATH, side_effect=Exception("down")), \
             patch(GEMINI_PATH, return_value="ok") as mock_gemini, \
             patch(SLEEP_PATH):
            call_llm("p", role="plan", model="custom-model-xyz")
        _, called_model, _ = mock_gemini.call_args[0]
        assert called_model == GEMINI_FALLBACK_MODEL


# ---------------------------------------------------------------------------
# 7. Invalid role
# ---------------------------------------------------------------------------


class TestInvalidRole:
    def test_raises_value_error_for_unknown_role(self) -> None:
        with pytest.raises(ValueError, match="role must be one of"):
            call_llm("prompt", role="invent")  # type: ignore[arg-type]

    def test_no_provider_called_on_bad_role(self) -> None:
        with patch(GROQ_PATH) as mock_groq, \
             patch(GEMINI_PATH) as mock_gemini:
            with pytest.raises(ValueError):
                call_llm("prompt", role="badvalue")  # type: ignore[arg-type]
        mock_groq.assert_not_called()
        mock_gemini.assert_not_called()
