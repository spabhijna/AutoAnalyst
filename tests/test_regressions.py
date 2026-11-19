"""
Phase-0 safety net for the cleanup refactor (see REFACTOR_PLAN.md).

These tests were added in Phase 0 as ``xfail(strict=True)`` regressions for the
bugs found in the audit (D1, D2, D3, D5, D6, D8); each fixing phase removed its
marker, and they now guard against the bugs coming back.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest

import agent.loop as loop
from agent.parsing import parse_json_list
from agent.artifact_store import ArtifactStore
from agent.loop import MAX_PLAN_REVISIONS, run
from data.generate_data import generate_sales_data

CALL_LLM = "agent.loop.call_llm"
RUN_SANDBOXED = "agent.loop.run_sandboxed"
WRITE_LOG = "agent.loop.write_log_entry"


@pytest.fixture()
def sales_df() -> pd.DataFrame:
    return generate_sales_data(n_rows=200, seed=42)


# ---------------------------------------------------------------------------
# Dry-run smoke test (passes today; must keep passing through every phase)
# ---------------------------------------------------------------------------


def test_dry_run_smoke(sales_df, tmp_path) -> None:
    state = run("any goal", sales_df, dry_run=True, log_dir=str(tmp_path))

    assert not state.aborted
    assert len(state.working_memory.findings) == 5
    assert len(state.final_results) == 5
    assert state.summary


# ---------------------------------------------------------------------------
# D1: _parse_json_list fallback regex is malformed
# ---------------------------------------------------------------------------


def test_parse_json_list_numbered_fallback() -> None:
    result = parse_json_list("1. First question\n2) Second question")
    assert result == ["First question", "Second question"]


# ---------------------------------------------------------------------------
# D2: deterministic artifacts are registered twice
# ---------------------------------------------------------------------------


def test_deterministic_artifacts_registered_once(sales_df, tmp_path) -> None:
    keys: list[str] = []
    original = ArtifactStore.register

    def spy(self, key, *args, **kwargs):
        keys.append(key)
        return original(self, key, *args, **kwargs)

    with patch.object(ArtifactStore, "register", spy):
        run("any goal", sales_df, dry_run=True, log_dir=str(tmp_path))

    assert keys, "dry run should register artifacts"
    duplicates = {k for k in keys if keys.count(k) > 1}
    assert not duplicates, f"registered more than once: {sorted(duplicates)}"


# ---------------------------------------------------------------------------
# D3: dry-run permanently replaces agent.loop.call_llm
# ---------------------------------------------------------------------------


def test_dry_run_does_not_leak(sales_df, tmp_path) -> None:
    original = loop.call_llm
    run("any goal", sales_df, dry_run=True, log_dir=str(tmp_path))
    assert loop.call_llm is original


# ---------------------------------------------------------------------------
# D5: plan-revision cap must abort, not advance past an incomplete task
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("plan", ['["Q1?"]', '["Q1?", "Q2?", "Q3?"]'])
def test_plan_revision_cap_aborts(sales_df, tmp_path, plan) -> None:
    revise = json.dumps({"trust": "low", "reason": ".", "action": "revise_plan"})
    calls = {"n": 0}

    def fake_llm(prompt: str, role: str, *a, **k) -> str:
        calls["n"] += 1
        if calls["n"] > 60:  # guard: a hung loop must fail the test, not freeze it
            raise AssertionError("loop did not terminate after revision cap")
        if role == "plan":
            return plan
        if role == "codegen":
            return "result = 1"
        if role == "critique":
            return revise
        return "Summary."

    ok = {"success": True, "result": "1", "stdout": "", "error": None, "error_type": None}
    with (
        patch(CALL_LLM, side_effect=fake_llm),
        patch(RUN_SANDBOXED, return_value=ok),
        patch(WRITE_LOG),
    ):
        state = run("goal", sales_df, log_dir=str(tmp_path))

    assert state.plan_revisions == MAX_PLAN_REVISIONS
    assert state.aborted
    assert "revision" in state.abort_reason.lower()


# ---------------------------------------------------------------------------
# D6: step-specific semantic checks reject correct capability outputs
# ---------------------------------------------------------------------------


def test_dry_run_is_fully_deterministic(sales_df, tmp_path) -> None:
    state = run("any goal", sales_df, dry_run=True, log_dir=str(tmp_path))

    assert all(a.code == "(deterministic executor)" for a in state.debug_history)
    assert set(state.artifact_store.available_keys()) == {
        "quarter_df",
        "region_revenue",
        "top_region",
        "monthly_trend",
        "revenue_share",
    }


# ---------------------------------------------------------------------------
# D8: critique_prompt drops the observer's facts
# ---------------------------------------------------------------------------


def test_critique_prompt_includes_observation_text() -> None:
    from agent.prompts import critique_prompt

    prompt = critique_prompt(
        "goal",
        "sub question",
        "result = 1",
        {"success": True, "result": "1", "stdout": "", "error": None},
        "(no history)",
        observation_text="SENTINEL_OBS",
    )
    assert "SENTINEL_OBS" in prompt


# ---------------------------------------------------------------------------
# Phase 3: injectable LLM and call counting
# ---------------------------------------------------------------------------


def test_explicit_llm_callable_is_used_and_counted(sales_df, tmp_path) -> None:
    from agent.dry_run import dry_run_llm

    seen: list[str] = []

    def spy(prompt: str, role: str, *a, **k) -> str:
        seen.append(role)
        return dry_run_llm(prompt, role)

    state = run("any goal", sales_df, log_dir=str(tmp_path), llm=spy)

    assert not state.aborted
    assert seen and seen[0] == "plan"
    assert state.llm_call_count == len(seen)
