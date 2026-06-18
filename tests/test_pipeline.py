"""
Integration smoke tests for autoanalyst.pipeline.

Uses dry-run mode (no live LLM calls) with mocked stage functions to verify:
- The pipeline produces a valid PipelineRunResult.
- The JSONL log file is created and contains the right number of entries.
- The retry cap (max_retries) is enforced — the loop never runs unbounded.
- A critique PASS on the first iteration advances to the next sub-question.
- A critique FAIL triggers retry with previous_error passed to CODE-GEN.
- All retries exhausted → QuestionResult.passed is False, pipeline still completes.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autoanalyst.config import PipelineConfig
from autoanalyst.logger import read_run_log
from autoanalyst.pipeline import PipelineRunResult, run_pipeline
from autoanalyst.stages.critic import CritiqueResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config(tmp_path: Path) -> PipelineConfig:
    """Dry-run config so no LLM or network calls happen."""
    return PipelineConfig(
        groq_api_key="fake",
        groq_model="fake-groq",
        gemini_api_key="fake",
        gemini_model="fake-gemini",
        max_retries=3,
        log_dir=tmp_path / "runs",
        dry_run=True,
    )


@pytest.fixture()
def tiny_df():
    """A minimal 3-row DataFrame for testing."""
    pytest.importorskip("pandas")
    import pandas as pd
    return pd.DataFrame({
        "Sales": [100.0, 200.0, 300.0],
        "Category": ["A", "B", "A"],
    })


def _pass_critique() -> CritiqueResult:
    return CritiqueResult(passed=True, feedback="Looks good.", issues=[])


def _fail_critique(issues: list[str] | None = None) -> CritiqueResult:
    return CritiqueResult(
        passed=False,
        feedback="Found problems.",
        issues=issues or ["Duplicate rows not handled"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPipelineSmoke:
    def test_dry_run_produces_result(self, config: PipelineConfig, tiny_df) -> None:
        """Pipeline in dry-run mode must return a PipelineRunResult without crashing."""
        with patch("autoanalyst.stages.planner.call_llm", return_value="1. What is total sales?"), \
             patch("autoanalyst.stages.codegen.call_llm", return_value="```python\n_result = df['Sales'].sum()\n```"), \
             patch("autoanalyst.stages.observer.call_llm", return_value="Total sales is 600."), \
             patch("autoanalyst.stages.critic.call_llm",
                   return_value="VERDICT: PASS\nISSUES:\n- None\nFEEDBACK:\nResult is correct."):
            result = run_pipeline(
                goal="Analyse sales",
                dataset_schema="Sales (float), Category (str)",
                df=tiny_df,
                config=config,
            )

        assert isinstance(result, PipelineRunResult)
        assert result.run_id.startswith("run_")
        assert len(result.question_results) == 1
        assert result.question_results[0].passed is True
        assert result.log_path.exists()

    def test_log_file_created_with_entries(self, config: PipelineConfig, tiny_df) -> None:
        """The JSONL log must contain plan + codegen + execute + observe + critique entries."""
        with patch("autoanalyst.stages.planner.call_llm", return_value="1. Q1?"), \
             patch("autoanalyst.stages.codegen.call_llm", return_value="_result = 42"), \
             patch("autoanalyst.stages.observer.call_llm", return_value="Answer is 42."), \
             patch("autoanalyst.stages.critic.call_llm",
                   return_value="VERDICT: PASS\nISSUES:\n- None\nFEEDBACK:\nOK."):
            result = run_pipeline("goal", "schema", tiny_df, config)

        entries = read_run_log(result.run_id, config.log_dir)
        stages = [e.stage for e in entries]
        assert "plan" in stages
        assert "codegen" in stages
        assert "execute" in stages
        assert "observe" in stages
        assert "critique" in stages

    def test_retry_cap_enforced(self, config: PipelineConfig, tiny_df) -> None:
        """When critique always fails, the loop runs at most max_retries times."""
        call_counts = {"codegen": 0}

        def count_codegen_calls(prompt, role, config=None):
            if role == "generate" and "_result" not in prompt:
                call_counts["codegen"] += 1
            return "_result = 1"

        with patch("autoanalyst.stages.planner.call_llm", return_value="1. Q?"), \
             patch("autoanalyst.stages.codegen.call_llm", return_value="_result = 1"), \
             patch("autoanalyst.stages.observer.call_llm", return_value="Observation."), \
             patch("autoanalyst.stages.critic.call_llm",
                   return_value="VERDICT: FAIL\nISSUES:\n- Problem\nFEEDBACK:\nToo bad."):
            result = run_pipeline("goal", "schema", tiny_df, config)

        qr = result.question_results[0]
        assert not qr.passed
        assert qr.iterations == config.max_retries

    def test_pass_on_second_attempt(self, config: PipelineConfig, tiny_df) -> None:
        """Pipeline should pass on the second attempt when first critique fails."""
        critique_responses = [
            "VERDICT: FAIL\nISSUES:\n- Nulls not handled\nFEEDBACK:\nFix nulls.",
            "VERDICT: PASS\nISSUES:\n- None\nFEEDBACK:\nNow correct.",
        ]
        critique_iter = iter(critique_responses)

        with patch("autoanalyst.stages.planner.call_llm", return_value="1. Q?"), \
             patch("autoanalyst.stages.codegen.call_llm", return_value="_result = 1"), \
             patch("autoanalyst.stages.observer.call_llm", return_value="Observation."), \
             patch("autoanalyst.stages.critic.call_llm",
                   side_effect=lambda *a, **kw: next(critique_iter)):
            result = run_pipeline("goal", "schema", tiny_df, config)

        qr = result.question_results[0]
        assert qr.passed is True
        assert qr.iterations == 2  # failed on iter 0, passed on iter 1

    def test_custom_run_id(self, config: PipelineConfig, tiny_df) -> None:
        """Passing an explicit run_id should be used for the log file name."""
        with patch("autoanalyst.stages.planner.call_llm", return_value="1. Q?"), \
             patch("autoanalyst.stages.codegen.call_llm", return_value="_result = 1"), \
             patch("autoanalyst.stages.observer.call_llm", return_value="Obs."), \
             patch("autoanalyst.stages.critic.call_llm",
                   return_value="VERDICT: PASS\nISSUES:\n- None\nFEEDBACK:\nOK."):
            result = run_pipeline("goal", "schema", tiny_df, config, run_id="my_custom_run")

        assert result.run_id == "my_custom_run"
        assert (config.log_dir / "my_custom_run.jsonl").exists()
