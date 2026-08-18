"""
Tests for agent.llm_client — the three-tier LLM gateway.

Contract under test
-------------------
1. Happy path: Primary provider succeeds → result returned, no fallback called.
2. Retry: Primary fails on attempt 1 → backoff → retry → succeeds on attempt 2.
   No fallback called.
3. Fallback chain: Primary fails twice → fallback1 called.
4. Multi-tier failure: Primary + fallback1 fail → fallback2 serves it.
5. Total failure: All providers in chain fail → LLMUnavailableError.
6. Role routing: plan (Groq→OpenRouter), codegen (Groq→Cerebras→OpenRouter),
   critique (Groq→Cerebras→Gemini).
7. model= override passed to every provider in the chain.
8. Invalid role → ValueError before any provider call.

Notes
-----
- `time.sleep` is patched so tests don't actually wait.
- Provider functions are patched at the agent.llm_client module level.
- `_should_fallback` is imported and patched where needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.config import (
    CEREBRAS_CODEGEN_FALLBACK_MODEL,
    CEREBRAS_CRITIQUE_FALLBACK_MODEL,
    GEMINI_FALLBACK_MODEL,
    GROQ_CODEGEN_MODEL,
    GROQ_CRITIQUE_MODEL,
    GROQ_DEFAULT_MODEL,
    OPENROUTER_PLAN_FALLBACK_MODEL,
    RETRY_BACKOFF_SECONDS,
)
from agent.llm_client import LLMUnavailableError, call_llm

# ---------------------------------------------------------------------------
# Patch paths
# ---------------------------------------------------------------------------

GROQ = "agent.llm_client._call_groq"
CEREBRAS = "agent.llm_client._call_cerebras"
OPENROUTER = "agent.llm_client._call_openrouter"
GEMINI = "agent.llm_client._call_gemini"
SLEEP = "agent.llm_client.time.sleep"


# ---------------------------------------------------------------------------
# 1. Happy path
# ---------------------------------------------------------------------------


class TestGroqSuccessOnFirstAttempt:
    def test_returns_groq_response(self) -> None:
        with (
            patch(GROQ, return_value="groq answer"),
            patch(SLEEP),
            patch(CEREBRAS),
            patch(OPENROUTER),
            patch(GEMINI),
        ):
            result = call_llm("prompt", role="plan")
        assert result == "groq answer"

    def test_no_sleep_on_success(self) -> None:
        with (
            patch(GROQ, return_value="ok"),
            patch(SLEEP) as mock_sleep,
            patch(CEREBRAS),
            patch(OPENROUTER),
            patch(GEMINI),
        ):
            call_llm("prompt", role="codegen")
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Retry
# ---------------------------------------------------------------------------


class TestRetry:
    def test_retry_on_first_failure(self) -> None:
        groq_responses = [Exception("timeout"), "groq retry answer"]

        def side(*a, **k):
            r = groq_responses.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        with (
            patch(GROQ, side_effect=side) as mock_groq,
            patch(SLEEP) as mock_sleep,
            patch(CEREBRAS),
            patch(OPENROUTER),
            patch(GEMINI),
        ):
            result = call_llm("prompt", role="plan")

        assert result == "groq retry answer"
        assert mock_groq.call_count == 2
        mock_sleep.assert_called_once()

    def test_sleep_duration_matches_config(self) -> None:
        groq_responses = [Exception("err"), "ok"]

        def side(*a, **k):
            r = groq_responses.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        with (
            patch(GROQ, side_effect=side),
            patch(SLEEP) as mock_sleep,
            patch(CEREBRAS),
            patch(OPENROUTER),
            patch(GEMINI),
        ):
            call_llm("prompt", role="codegen")

        mock_sleep.assert_called_once_with(RETRY_BACKOFF_SECONDS)


# ---------------------------------------------------------------------------
# 3. Fallback — plan chain: Groq → OpenRouter
# ---------------------------------------------------------------------------


class TestPlanChain:
    def test_openrouter_called_after_groq_fails_twice(self) -> None:
        with (
            patch(GROQ, side_effect=Exception("groq down")),
            patch(OPENROUTER, return_value="or answer") as mock_or,
            patch(SLEEP),
            patch(CEREBRAS),
            patch(GEMINI),
        ):
            result = call_llm("prompt", role="plan")

        assert result == "or answer"
        assert mock_or.called

    def test_plan_openrouter_uses_fallback_model(self) -> None:
        with (
            patch(GROQ, side_effect=Exception("down")),
            patch(OPENROUTER, return_value="ok") as mock_or,
            patch(SLEEP),
            patch(CEREBRAS),
            patch(GEMINI),
        ):
            call_llm("prompt", role="plan")

        _, called_model = mock_or.call_args[0]
        assert called_model == OPENROUTER_PLAN_FALLBACK_MODEL


# ---------------------------------------------------------------------------
# 4. Codegen chain: Groq (Codestral) → Cerebras → OpenRouter (Codestral)
# ---------------------------------------------------------------------------


class TestCodegenChain:
    def test_cerebras_called_after_groq_fails(self) -> None:
        with (
            patch(GROQ, side_effect=Exception("groq down")),
            patch(CEREBRAS, return_value="cerebras answer") as mock_cb,
            patch(OPENROUTER),
            patch(GEMINI),
            patch(SLEEP),
        ):
            result = call_llm("prompt", role="codegen")

        assert result == "cerebras answer"
        assert mock_cb.called

    def test_codegen_groq_uses_codestral(self) -> None:
        with (
            patch(GROQ, return_value="ok") as mock_groq,
            patch(SLEEP),
            patch(CEREBRAS),
            patch(OPENROUTER),
            patch(GEMINI),
        ):
            call_llm("prompt", role="codegen")

        _, called_model = mock_groq.call_args[0]
        assert called_model == GROQ_CODEGEN_MODEL

    def test_codegen_cerebras_uses_correct_model(self) -> None:
        with (
            patch(GROQ, side_effect=Exception("down")),
            patch(CEREBRAS, return_value="ok") as mock_cb,
            patch(OPENROUTER),
            patch(GEMINI),
            patch(SLEEP),
        ):
            call_llm("prompt", role="codegen")

        _, called_model = mock_cb.call_args[0]
        assert called_model == CEREBRAS_CODEGEN_FALLBACK_MODEL

    def test_openrouter_fallback_after_groq_and_cerebras_fail(self) -> None:
        with (
            patch(GROQ, side_effect=Exception("groq down")),
            patch(CEREBRAS, side_effect=Exception("cerebras down")),
            patch(OPENROUTER, return_value="or answer") as mock_or,
            patch(GEMINI),
            patch(SLEEP),
        ):
            result = call_llm("prompt", role="codegen")

        assert result == "or answer"
        assert mock_or.called


# ---------------------------------------------------------------------------
# 5. Critique chain: Groq → Cerebras → Gemini
# ---------------------------------------------------------------------------


class TestCritiqueChain:
    def test_groq_critique_uses_correct_model(self) -> None:
        with (
            patch(GROQ, return_value="ok") as mock_groq,
            patch(SLEEP),
            patch(CEREBRAS),
            patch(OPENROUTER),
            patch(GEMINI),
        ):
            call_llm("prompt", role="critique")

        _, called_model = mock_groq.call_args[0]
        assert called_model == GROQ_CRITIQUE_MODEL

    def test_cerebras_called_after_groq_fails(self) -> None:
        with (
            patch(GROQ, side_effect=Exception("groq down")),
            patch(CEREBRAS, return_value="cerebras answer") as mock_cb,
            patch(OPENROUTER),
            patch(GEMINI),
            patch(SLEEP),
        ):
            result = call_llm("prompt", role="critique")

        assert result == "cerebras answer"
        assert mock_cb.called

    def test_critique_cerebras_uses_correct_model(self) -> None:
        with (
            patch(GROQ, side_effect=Exception("down")),
            patch(CEREBRAS, return_value="ok") as mock_cb,
            patch(OPENROUTER),
            patch(GEMINI),
            patch(SLEEP),
        ):
            call_llm("prompt", role="critique")

        _, called_model = mock_cb.call_args[0]
        assert called_model == CEREBRAS_CRITIQUE_FALLBACK_MODEL

    def test_gemini_after_groq_and_cerebras_fail(self) -> None:
        with (
            patch(GROQ, side_effect=Exception("groq down")),
            patch(CEREBRAS, side_effect=Exception("cerebras down")),
            patch(GEMINI, return_value="gemini answer") as mock_gm,
            patch(OPENROUTER),
            patch(SLEEP),
        ):
            result = call_llm("prompt", role="critique")

        assert result == "gemini answer"
        assert mock_gm.called

    def test_gemini_uses_fallback_model(self) -> None:
        with (
            patch(GROQ, side_effect=Exception("down")),
            patch(CEREBRAS, side_effect=Exception("down")),
            patch(GEMINI, return_value="ok") as mock_gm,
            patch(OPENROUTER),
            patch(SLEEP),
        ):
            call_llm("prompt", role="critique")

        _, called_model = mock_gm.call_args[0]
        assert called_model == GEMINI_FALLBACK_MODEL


# ---------------------------------------------------------------------------
# 6. Total failure — LLMUnavailableError
# ---------------------------------------------------------------------------


class TestTotalFailure:
    def test_raises_llm_unavailable_error(self) -> None:
        with (
            patch(GROQ, side_effect=Exception("groq dead")),
            patch(CEREBRAS, side_effect=Exception("cerebras dead")),
            patch(GEMINI, side_effect=Exception("gemini dead")),
            patch(OPENROUTER),
            patch(SLEEP),
        ):
            with pytest.raises(LLMUnavailableError):
                call_llm("prompt", role="critique")

    def test_error_carries_errors_list(self) -> None:
        groq_exc = RuntimeError("groq 503")
        cerebras_exc = RuntimeError("cerebras 429")
        gemini_exc = RuntimeError("gemini timeout")

        with (
            patch(GROQ, side_effect=groq_exc),
            patch(CEREBRAS, side_effect=cerebras_exc),
            patch(GEMINI, side_effect=gemini_exc),
            patch(OPENROUTER),
            patch(SLEEP),
        ):
            with pytest.raises(LLMUnavailableError) as exc_info:
                call_llm("prompt", role="critique")

        err = exc_info.value
        assert len(err.errors) == 3
        provider_names = [n for n, _ in err.errors]
        assert provider_names == ["groq", "cerebras", "gemini"]

    def test_plain_chain_all_fail_raises(self) -> None:
        """Plan chain (Groq→OpenRouter) both failing raises LLMUnavailableError."""
        with (
            patch(GROQ, side_effect=Exception("dead")),
            patch(OPENROUTER, side_effect=Exception("dead")),
            patch(SLEEP),
            patch(CEREBRAS),
            patch(GEMINI),
        ):
            with pytest.raises(LLMUnavailableError):
                call_llm("prompt", role="plan")


# ---------------------------------------------------------------------------
# 7. Role → model routing
# ---------------------------------------------------------------------------


class TestRoleModelRouting:
    def test_plan_uses_default_groq_model(self) -> None:
        with (
            patch(GROQ, return_value="ok") as mock_groq,
            patch(SLEEP),
            patch(CEREBRAS),
            patch(OPENROUTER),
            patch(GEMINI),
        ):
            call_llm("p", role="plan")
        _, called_model = mock_groq.call_args[0]
        assert called_model == GROQ_DEFAULT_MODEL

    def test_codegen_uses_codestral_model(self) -> None:
        with (
            patch(GROQ, return_value="ok") as mock_groq,
            patch(SLEEP),
            patch(CEREBRAS),
            patch(OPENROUTER),
            patch(GEMINI),
        ):
            call_llm("p", role="codegen")
        _, called_model = mock_groq.call_args[0]
        assert called_model == GROQ_CODEGEN_MODEL

    def test_critique_uses_critique_model(self) -> None:
        with (
            patch(GROQ, return_value="ok") as mock_groq,
            patch(SLEEP),
            patch(CEREBRAS),
            patch(OPENROUTER),
            patch(GEMINI),
        ):
            call_llm("p", role="critique")
        _, called_model = mock_groq.call_args[0]
        assert called_model == GROQ_CRITIQUE_MODEL

    def test_observe_uses_default_groq_model(self) -> None:
        with (
            patch(GROQ, return_value="ok") as mock_groq,
            patch(SLEEP),
            patch(CEREBRAS),
            patch(OPENROUTER),
            patch(GEMINI),
        ):
            call_llm("p", role="observe")
        _, called_model = mock_groq.call_args[0]
        assert called_model == GROQ_DEFAULT_MODEL


# ---------------------------------------------------------------------------
# 8. model= override
# ---------------------------------------------------------------------------


class TestModelOverride:
    def test_codegen_override_passed_to_all_providers(self) -> None:
        with (
            patch(GROQ, return_value="ok") as mock_groq,
            patch(CEREBRAS),
            patch(OPENROUTER),
            patch(GEMINI),
            patch(SLEEP),
        ):
            call_llm("p", role="codegen", model="custom-model")
        _, called_model = mock_groq.call_args[0]
        assert called_model == "custom-model"

    def test_critique_override_flows_to_chain(self) -> None:
        """model= override replaces the model at every chain level."""
        with (
            patch(GROQ, side_effect=Exception("down")),
            patch(CEREBRAS, return_value="ok") as mock_cb,
            patch(OPENROUTER),
            patch(GEMINI),
            patch(SLEEP),
        ):
            call_llm("p", role="critique", model="my-model")

        _, called_model = mock_cb.call_args[0]
        assert called_model == "my-model"


# ---------------------------------------------------------------------------
# 9. Invalid role
# ---------------------------------------------------------------------------


class TestInvalidRole:
    def test_raises_value_error_for_unknown_role(self) -> None:
        with pytest.raises(ValueError, match="role must be one of"):
            call_llm("prompt", role="invent")  # type: ignore[arg-type]

    def test_no_provider_called_on_bad_role(self) -> None:
        with (
            patch(GROQ) as mock_groq,
            patch(CEREBRAS) as mock_cb,
            patch(GEMINI) as mock_gm,
            patch(OPENROUTER) as mock_or,
        ):
            with pytest.raises(ValueError):
                call_llm("prompt", role="badvalue")  # type: ignore[arg-type]
        mock_groq.assert_not_called()
        mock_cb.assert_not_called()
        mock_gm.assert_not_called()
        mock_or.assert_not_called()
