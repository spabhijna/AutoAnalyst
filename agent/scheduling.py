"""
Scheduling helpers: pick the next ready task, check dependencies, and mark task
statuses when a run aborts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.errors import PipelineAbortedError
from agent.state import TaskNode

if TYPE_CHECKING:
    from agent.loop import AgentRun


def next_ready_task(state: "AgentRun") -> TaskNode:
    """Return the next ready task, or abort if the scheduler is stuck."""
    task = state.execution_graph.next_ready_task(
        set(state.working_memory.findings.keys())
    )
    if task is None:
        raise PipelineAbortedError(
            sub_question="(scheduler)",
            last_reason="No ready task was available before the graph completed.",
            run_id=state.run_id,
            kind="scheduling failure",
        )
    return task


def ensure_dependencies_available(state: "AgentRun", task: TaskNode) -> None:
    """
    Abort predictably if a task's prerequisites are missing.

    This should rarely trigger because the loop aborts on failed steps, but it
    protects replans and future non-linear task graphs from running with
    incomplete working memory.
    """
    missing = state.working_memory.dependency_failures(task)
    if not missing:
        return
    reason = f"Missing accepted dependencies for {task.task_id}: {', '.join(missing)}"
    task.status = "skipped"
    task.failure_reason = reason
    raise PipelineAbortedError(task.question, reason, state.run_id)


def mark_failed_and_skipped(
    state: "AgentRun",
    failed_question: str,
    reason: str,
) -> None:
    """Mark the failed task and all downstream tasks with predictable statuses."""
    failed_index: int | None = None
    for index, task in enumerate(state.tasks):
        if task.question == failed_question:
            failed_index = index
            task.status = "failed"
            task.failure_reason = reason
            break
    if failed_index is None:
        return
    for task in state.tasks[failed_index + 1 :]:
        task.status = "skipped"
        task.failure_reason = f"Upstream dependency failed: {failed_question}"
