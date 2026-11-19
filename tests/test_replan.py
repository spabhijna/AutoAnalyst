"""D4: replanning keeps completed tasks and produces structured, uniquely-numbered tasks."""

from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd

from agent.execution_graph import ExecutionGraph
from agent.loop import AgentRun, run
from agent.planning import apply_replan, next_task_index
from agent.prompts import replan_prompt
from agent.state import Finding, TaskNode


def _finding(task_id: str) -> Finding:
    return Finding(
        task_id=task_id,
        question="q",
        answer="a",
        answer_confidence="high",
        evidence="e",
        artifact_key=f"k_{task_id}",
    )


def _state_with_completed_first_task() -> AgentRun:
    done = TaskNode(
        task_id="step_1",
        question="Filter",
        status="accepted",
        produces_artifact="quarter_df",
        capability="FilterByDateRange",
        output_schema=["date", "revenue"],
    )
    failed = TaskNode(
        task_id="step_2",
        question="Bad question",
        depends_on=["step_1"],
        status="failed",
        consumes_artifacts=["quarter_df"],
    )
    state = AgentRun(goal="g", schema_summary="s")
    state.tasks = [done, failed]
    state.plan = [t.question for t in state.tasks]
    state.execution_graph = ExecutionGraph.from_tasks(state.tasks)
    state.working_memory.add_finding(_finding("step_1"))
    return state


def test_apply_replan_keeps_completed_tasks_unchanged() -> None:
    state = _state_with_completed_first_task()
    original = state.tasks[0]
    snapshot = (
        original.task_id,
        original.status,
        original.produces_artifact,
        original.capability,
        list(original.output_schema),
    )
    specs = [
        {
            "question": "Revenue by region",
            "capability": "GroupRevenueByRegion",
            "produces_artifact": "region_revenue",
            "consumes_artifacts": ["quarter_df"],
            "output_schema": ["region", "total_revenue"],
        },
        {"question": "Top region?", "produces_artifact": None},
    ]

    new_tasks = apply_replan(state, specs)

    assert state.tasks[0] is original
    assert (
        original.task_id,
        original.status,
        original.produces_artifact,
        original.capability,
        list(original.output_schema),
    ) == snapshot
    # failed task dropped; new IDs continue after the highest existing step_N
    assert [t.task_id for t in new_tasks] == ["step_3", "step_4"]
    assert [t.task_id for t in state.tasks] == ["step_1", "step_3", "step_4"]
    assert state.plan == ["Filter", "Revenue by region", "Top region?"]
    # structure carried over, dependencies chain from the kept task
    assert new_tasks[0].capability == "GroupRevenueByRegion"
    assert new_tasks[0].produces_artifact == "region_revenue"
    assert new_tasks[0].consumes_artifacts == ["quarter_df"]
    assert new_tasks[0].output_schema == ["region", "total_revenue"]
    assert new_tasks[0].depends_on == ["step_1"]
    assert new_tasks[1].depends_on == ["step_1", "step_3"]
    assert state.execution_graph.task_index("step_4") == 2


def test_new_ids_never_reuse_existing_ones() -> None:
    state = _state_with_completed_first_task()
    assert next_task_index(state.tasks) == 2
    first = apply_replan(state, [{"question": "A"}])
    assert first[0].task_id == "step_3"
    # a second replan (nothing new completed) skips past the dropped IDs again
    second = apply_replan(state, [{"question": "B"}])
    assert second[0].task_id == "step_4"


def test_replan_prompt_requests_structured_tasks() -> None:
    prompt = replan_prompt("goal", "schema", "found X", "bad scope", "ART_DESC")
    for field in ("question", "capability", "produces_artifact", "consumes_artifacts"):
        assert f'"{field}"' in prompt
    assert "JSON array of task objects" in prompt
    assert "ART_DESC" in prompt
    assert "found X" in prompt and "bad scope" in prompt


def test_run_replan_with_structured_response(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "region": ["North", "South", "East"],
            "revenue": [100.0, 200.0, 150.0],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        }
    )
    revise = json.dumps({"trust": "low", "reason": "scope", "action": "revise_plan"})
    accept = json.dumps({"trust": "high", "reason": "ok", "action": "accept"})
    replan = json.dumps(
        [
            {
                "question": "New question?",
                "capability": None,
                "produces_artifact": "new_df",
                "consumes_artifacts": [],
                "output_schema": [],
            }
        ]
    )
    ok = {"success": True, "result": "1", "stdout": "", "error": None, "error_type": None}
    responses = iter(
        [
            '["Q1?", "Q2?"]',  # plan
            "result = 1",  # codegen Q1
            accept,  # critique Q1
            "result = 2",  # codegen Q2
            revise,  # critique Q2 -> revise_plan
            replan,  # structured replan
            "result = 3",  # codegen new
            accept,  # critique new
            "Summary.",
        ]
    )
    with (
        patch("agent.loop.call_llm", side_effect=lambda *a, **k: next(responses)),
        patch("agent.loop.run_sandboxed", return_value=ok),
        patch("agent.loop.write_log_entry"),
    ):
        state = run("goal", df, log_dir=str(tmp_path))

    assert not state.aborted
    assert [t.task_id for t in state.tasks] == ["step_1", "step_3"]
    assert state.tasks[0].status == "accepted"
    assert state.tasks[1].produces_artifact == "new_df"
    assert set(state.working_memory.findings) == {"step_1", "step_3"}
