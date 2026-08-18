"""
Execution graph helpers for artifact-driven scheduling.

The graph wraps the planner-produced TaskNode list and provides ready-node
selection plus artifact lineage queries. The loop can keep its public API
while relying on this module for dependency resolution instead of positional
iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.state import TaskNode


@dataclass
class ExecutionGraph:
    """Dependency-aware view over a plan's TaskNode list."""

    tasks: list[TaskNode] = field(default_factory=list)

    @classmethod
    def from_tasks(cls, tasks: list[TaskNode]) -> "ExecutionGraph":
        return cls(tasks=list(tasks))

    @property
    def task_ids(self) -> list[str]:
        return [task.task_id for task in self.tasks]

    def task_index(self, task_id: str) -> int:
        for index, task in enumerate(self.tasks):
            if task.task_id == task_id:
                return index
        return -1

    def task_for_id(self, task_id: str) -> TaskNode | None:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def ready_tasks(self, completed_task_ids: set[str]) -> list[TaskNode]:
        """Return pending tasks whose dependencies are satisfied."""
        ready: list[TaskNode] = []
        for task in self.tasks:
            if task.task_id in completed_task_ids:
                continue
            if task.status != "pending":
                continue
            if all(dep in completed_task_ids for dep in task.depends_on):
                ready.append(task)
        return ready

    def next_ready_task(self, completed_task_ids: set[str]) -> TaskNode | None:
        ready = self.ready_tasks(completed_task_ids)
        return ready[0] if ready else None

    def all_completed(self) -> bool:
        return all(
            task.status in {"accepted", "failed", "skipped"} for task in self.tasks
        )

    def artifact_consumers(self) -> dict[str, list[str]]:
        """Return artifact name → downstream task IDs that consume it."""
        consumers: dict[str, list[str]] = {}
        for task in self.tasks:
            for artifact_name in task.consumes_artifacts:
                consumers.setdefault(artifact_name, []).append(task.task_id)
        return consumers

    def consumers_for(self, artifact_name: str) -> list[str]:
        return self.artifact_consumers().get(artifact_name, [])

    def output_schema_for(self, task_id: str) -> list[str]:
        task = self.task_for_id(task_id)
        return list(task.output_schema) if task is not None else []
