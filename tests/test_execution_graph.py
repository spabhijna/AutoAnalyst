"""
Tests for the artifact-driven execution graph.
"""

from __future__ import annotations

from agent.execution_graph import ExecutionGraph
from agent.state import TaskNode


def test_ready_tasks_respect_dependencies() -> None:
    tasks = [
        TaskNode(task_id="step_1", question="filter", produces_artifact="quarter_df"),
        TaskNode(
            task_id="step_2",
            question="aggregate",
            depends_on=["step_1"],
            consumes_artifacts=["quarter_df"],
        ),
        TaskNode(
            task_id="step_3",
            question="summarize",
            depends_on=["step_2"],
            consumes_artifacts=["quarter_df"],
        ),
    ]
    graph = ExecutionGraph.from_tasks(tasks)

    assert [task.task_id for task in graph.ready_tasks(set())] == ["step_1"]
    assert [task.task_id for task in graph.ready_tasks({"step_1"})] == ["step_2"]
    assert graph.next_ready_task({"step_1"}).task_id == "step_2"


def test_artifact_consumers_are_derived_from_tasks() -> None:
    tasks = [
        TaskNode(task_id="step_1", question="filter", produces_artifact="quarter_df"),
        TaskNode(
            task_id="step_2",
            question="aggregate",
            depends_on=["step_1"],
            consumes_artifacts=["quarter_df"],
        ),
        TaskNode(
            task_id="step_3",
            question="other",
            depends_on=["step_1"],
            consumes_artifacts=["quarter_df"],
        ),
    ]
    graph = ExecutionGraph.from_tasks(tasks)

    assert graph.consumers_for("quarter_df") == ["step_2", "step_3"]
    assert graph.output_schema_for("step_2") == []
    assert graph.task_index("step_3") == 2
