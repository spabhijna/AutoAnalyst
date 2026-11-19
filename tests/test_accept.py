"""Unit tests for the loop's single accept path (_accept / _try_deterministic)."""

from __future__ import annotations

import pandas as pd
import pytest

from agent.execution_graph import ExecutionGraph
from agent.loop import AgentRun, _accept, _try_deterministic
from agent.state import AttemptRecord, TaskNode


@pytest.fixture()
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "region": ["North", "South", "North"],
            "revenue": [10.0, 20.0, 30.0],
            "product_category": ["A", "B", "A"],
        }
    )


def _state(tmp_path, tasks: list[TaskNode]) -> AgentRun:
    state = AgentRun(goal="g", schema_summary="s", log_dir=str(tmp_path))
    state.tasks = tasks
    state.plan = [t.question for t in tasks]
    state.execution_graph = ExecutionGraph.from_tasks(tasks)
    return state


def _two_tasks(first_capability: str | None = None) -> list[TaskNode]:
    return [
        TaskNode(
            task_id="step_1",
            question="Filter df to the last quarter and return quarter_df",
            capability=first_capability,
            produces_artifact="quarter_df",
        ),
        TaskNode(
            task_id="step_2",
            question="Revenue by region",
            depends_on=["step_1"],
            consumes_artifacts=["quarter_df"],
        ),
    ]


def test_accept_registers_artifact_with_metadata(tmp_path, df) -> None:
    state = _state(tmp_path, _two_tasks())
    task = state.tasks[0]
    sandbox_result = {"success": True, "result": "3 rows", "artifact": df}
    attempt = AttemptRecord(
        task_id="step_1",
        sub_question=task.question,
        iteration=0,
        code="x",
        sandbox_result=sandbox_result,
        critique="ok",
        action="accept",
        trust="high",
        error_category="none",
    )

    finding = _accept(
        state,
        task,
        attempt,
        sandbox_result,
        "high",
        "looks right",
        attempt_count=1,
        had_execution_errors=False,
        provenance=["SomeCapability", "step_1"],
    )

    artifact = state.artifact_store.get_artifact("quarter_df")
    assert artifact is not None
    assert artifact.producer == "step_1"
    assert artifact.consumers == ["step_2"]
    assert artifact.is_frozen is True
    assert artifact.confidence == "high"
    assert artifact.provenance == ["SomeCapability", "step_1"]
    assert artifact.validated_at is not None
    assert finding.produced_artifact_name == "quarter_df"
    assert task.status == "accepted"
    assert "step_1" in state.working_memory.findings
    assert state.final_results == [finding]


def test_accept_without_artifact_registers_nothing(tmp_path) -> None:
    state = _state(tmp_path, _two_tasks())
    task = state.tasks[1]  # declares no produces_artifact
    attempt = AttemptRecord(
        task_id="step_2",
        sub_question=task.question,
        iteration=0,
        code="x",
        sandbox_result={"success": True, "result": "1"},
        critique="ok",
        action="accept",
        trust="high",
        error_category="none",
    )
    finding = _accept(
        state, task, attempt, {"success": True, "result": "1"}, "high", "ok", 1, False
    )
    assert state.artifact_store.available_keys() == []
    assert finding.produced_artifact_name is None


@pytest.mark.parametrize("declared", ["FilterByDateRange", None])
def test_deterministic_provenance_uses_resolved_capability(
    tmp_path, df, declared
) -> None:
    """Even without task.capability (e.g. replanned tasks) provenance is set."""
    state = _state(tmp_path, _two_tasks(first_capability=declared))
    task = state.tasks[0]
    assert task.capability == declared

    assert _try_deterministic(state, task, df) is True

    artifact = state.artifact_store.get_artifact("quarter_df")
    assert artifact is not None
    assert artifact.provenance == ["FilterByDateRange", "step_1"]
    assert artifact.consumers == ["step_2"]
