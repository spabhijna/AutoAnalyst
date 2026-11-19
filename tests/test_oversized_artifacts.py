"""
Artifacts over the sandbox size limit.

On large data a step can legitimately produce a DataFrame too big to send back
(for example a filter over 3M rows). That used to be dropped silently and then
reported as "no DataFrame artifact was produced", so every retry failed. Now the
sandbox describes the frame, the contract accepts it unstored, and later steps
recompute from ``df``.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd

import agent.sandbox as sandbox
from agent.loop import run
from agent.state import TaskNode
from agent.validator import CapabilityValidator

ACCEPT = json.dumps({"trust": "high", "action": "accept", "reason": "ok"})


def _frame(rows: int = 1000) -> pd.DataFrame:
    return pd.DataFrame({"a": range(rows), "b": [1.5] * rows})


# ---------------------------------------------------------------------------
# Sandbox: describe instead of silently dropping
# ---------------------------------------------------------------------------


def test_small_artifact_is_stored_with_info() -> None:
    frame, info = sandbox._extract_artifact("small", _frame())

    assert frame is not None
    assert info["stored"] is True
    assert info["rows"] == 1000
    assert info["columns"] == ["a", "b"]
    assert info["reason"] is None


def test_oversized_artifact_is_described_not_stored(monkeypatch) -> None:
    monkeypatch.setattr(sandbox, "_ARTIFACT_SIZE_LIMIT_BYTES", 1_000)

    frame, info = sandbox._extract_artifact("big", _frame())

    assert frame is None
    assert info["stored"] is False
    assert info["rows"] == 1000
    assert info["columns"] == ["a", "b"]
    assert info["bytes"] > 1_000
    assert "too large" in info["reason"]


def test_sandbox_result_carries_artifact_info() -> None:
    out = sandbox.run_sandboxed(
        "quarter_df = df.head(3)\nresult = 1",
        _frame(10),
        extract_artifact="quarter_df",
        timeout_seconds=30,
    )

    assert out["success"], out["error"]
    assert out["artifact_info"]["stored"] is True
    assert out["artifact_info"]["rows"] == 3


# ---------------------------------------------------------------------------
# Contract: oversized satisfies it, schema still checked
# ---------------------------------------------------------------------------


def _task(schema: list[str] | None = None) -> TaskNode:
    return TaskNode(
        task_id="step_1",
        question="q",
        produces_artifact="quarter_df",
        output_schema=schema or [],
    )


def _oversized_info(columns: list[str]) -> dict:
    return {
        "name": "quarter_df",
        "rows": 3_000_000,
        "columns": columns,
        "bytes": 400_000_000,
        "stored": False,
        "reason": "too large to store (400 MB > 52 MB limit)",
    }


def test_contract_accepts_oversized_artifact_and_flags_it() -> None:
    result = CapabilityValidator().validate_semantic_contract(
        _task(["a", "b"]), None, _oversized_info(["b", "a"])
    )

    assert result.success, result.reasons
    assert "too large" in result.metadata["artifact_not_stored"]


def test_contract_still_checks_columns_of_oversized_artifact() -> None:
    result = CapabilityValidator().validate_semantic_contract(
        _task(["a", "b"]), None, _oversized_info(["a", "z"])
    )

    assert not result.success
    assert "missing: ['b']" in result.reasons[0]


def test_contract_still_fails_when_nothing_was_produced() -> None:
    result = CapabilityValidator().validate_semantic_contract(_task(), None, None)

    assert not result.success
    assert "no DataFrame artifact was produced" in result.reasons[0]


# ---------------------------------------------------------------------------
# End to end: the run continues and later steps recompute from df
# ---------------------------------------------------------------------------


def test_run_continues_past_an_oversized_artifact(tmp_path) -> None:
    df = _frame(50)
    plan = json.dumps(
        [
            {"question": "Filter the data", "capability": None,
             "produces_artifact": "quarter_df", "consumes_artifacts": [],
             "output_schema": ["a", "b"]},
            {"question": "Sum b over the filtered data", "capability": None,
             "produces_artifact": None, "consumes_artifacts": ["quarter_df"],
             "output_schema": []},
        ]
    )
    oversized = {
        "success": True, "result": "50", "stdout": "", "error": None,
        "error_type": None, "error_category": "none", "artifact": None,
        "artifact_key": "quarter_df", "artifact_info": _oversized_info(["a", "b"]),
    }
    downstream = {
        "success": True, "result": "75.0", "stdout": "", "error": None,
        "error_type": None, "error_category": "none", "artifact": None,
        "artifact_key": None, "artifact_info": None,
    }
    seen_artifacts: list[dict] = []
    critique_prompts: list[str] = []

    def fake_sandbox(code, df, artifacts=None, extract_artifact=None, **kwargs):
        seen_artifacts.append(dict(artifacts or {}))
        return dict(oversized if extract_artifact == "quarter_df" else downstream)

    def llm(prompt: str, role: str, *a, **k) -> str:
        if role == "critique":
            critique_prompts.append(prompt)
        return {"plan": plan, "codegen": "result = 1", "critique": ACCEPT}.get(role, "s")

    with patch("agent.loop.run_sandboxed", side_effect=fake_sandbox):
        state = run("goal", df, llm=llm, log_dir=str(tmp_path))

    assert not state.aborted
    assert len(state.debug_history) == 2  # each step accepted on its first attempt
    assert "quarter_df" not in state.artifact_store.available_keys()
    assert "quarter_df" not in seen_artifacts[1]  # step 2 must recompute from df
    assert "not stored" in critique_prompts[0]  # the critic is told why
