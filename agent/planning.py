"""
Planning stage: build task nodes from the plan and run the PLAN / RE-PLAN LLM calls.

The LLM callable and the logger are passed in by ``agent.loop`` (resolved at call
time) so tests and dry-run can swap them without this module importing the loop.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable, Sequence

import pandas as pd

from agent import reporting
from agent.execution_graph import ExecutionGraph
from agent.logger import LogEntry
from agent.parsing import parse_plan_json
from agent.prompts import plan_prompt, replan_prompt
from agent.state import TaskNode

if TYPE_CHECKING:
    from agent.loop import AgentRun

LLM = Callable[..., str]


def task_id(index: int) -> str:
    """Return the stable task ID for a 0-based plan index."""
    return f"step_{index + 1}"


def task_order(tasks: list[TaskNode]) -> list[str]:
    """Return task IDs in current plan order."""
    return [task.task_id for task in tasks]


def next_task_index(tasks: list[TaskNode]) -> int:
    """Return the 0-based plan index for the next new task: one past the highest
    existing ``step_N`` number, so a new ID never reuses an existing one."""
    numbers = [
        int(m.group(1)) for t in tasks if (m := re.fullmatch(r"step_(\d+)", t.task_id))
    ]
    return max(numbers, default=0)


def build_tasks(
    plan: list[str],
    task_specs: list[dict] | None = None,
    *,
    first_index: int = 0,
    depends_on_prefix: Sequence[str] = (),
) -> list[TaskNode]:
    """
    Build task nodes from the ordered plan.

    When *task_specs* is provided (structured plan format), each TaskNode is
    populated with the planner-declared ``produces_artifact`` and
    ``consumes_artifacts`` values. When *task_specs* is absent or incomplete
    (flat string array / legacy format), these fields default to None / []
    and the pipeline operates without artifact declarations.

    The dependency graph (``depends_on``) still chains all prior tasks for
    safety, regardless of artifact declarations.

    ``first_index`` / ``depends_on_prefix`` let a replan append new tasks after
    existing ones: IDs start at ``task_id(first_index)`` and every new task also
    depends on the IDs in ``depends_on_prefix``.
    """
    tasks: list[TaskNode] = []
    for index, question in enumerate(plan):
        spec = task_specs[index] if task_specs and index < len(task_specs) else {}
        produces = spec.get("produces_artifact") or None
        consumes = spec.get("consumes_artifacts") or []
        output_schema = spec.get("output_schema") or []
        if isinstance(consumes, str):
            consumes = [consumes]  # defensive: normalize stray strings
        if isinstance(output_schema, str):
            output_schema = [output_schema]
        execution_fn = "llm_codegen"
        capability = spec.get("capability") or None
        tasks.append(
            TaskNode(
                task_id=task_id(first_index + index),
                question=question,
                depends_on=list(depends_on_prefix)
                + [task_id(first_index + i) for i in range(index)],
                produces_artifact=produces,
                consumes_artifacts=list(consumes),
                capability=capability,
                execution_fn=execution_fn,
                output_schema=list(output_schema),
            )
        )
    return tasks


def sync_task_capabilities(state: "AgentRun") -> None:
    """Resolve missing capability IDs using the registered capability set."""
    for task in state.tasks:
        if task.capability:
            continue
        capability = state.execution_engine.registry.resolve(task)
        if capability is not None:
            task.capability = capability.capability_id


def do_plan(
    state: "AgentRun", df: pd.DataFrame, llm: LLM, log: Callable[["AgentRun", LogEntry], None]
) -> list[str]:
    """Call the PLAN stage, parse the structured plan, and return sub-question strings."""
    reporting.plan_stage_start()
    prompt = plan_prompt(state.goal, state.schema_summary)
    raw = llm(prompt, role="plan")
    task_specs = parse_plan_json(raw)
    questions = [spec["question"] for spec in task_specs]
    state.plan = questions
    state.tasks = build_tasks(questions, task_specs)
    state.execution_graph = ExecutionGraph.from_tasks(state.tasks)
    sync_task_capabilities(state)
    reporting.plan_generated(questions)

    log(
        state,
        LogEntry(
            run_id=state.run_id,
            sub_question=state.goal,
            iteration=0,
            stage="plan",
            action="plan_generated",
            extra={"plan": questions},
        ),
    )
    return questions


def do_replan(state: "AgentRun", df: pd.DataFrame, llm: LLM, completed: str) -> list[dict]:
    """Re-plan from the current step forward and return structured task specs.

    The replan prompt asks for the same JSON task shape as the initial plan
    (question, capability, produces/consumes artifacts, output_schema), parsed
    with :func:`parse_plan_json`. Use :func:`apply_replan` to rebuild the tasks.
    """
    reporting.replan_stage_start()
    # Get the critique reason from the most recent debug attempt.
    last_reason = ""
    if state.debug_history:
        last_reason = state.debug_history[-1].critique

    prompt = replan_prompt(
        state.goal,
        state.schema_summary,
        completed,
        last_reason,
        available_artifacts=state.artifact_store.describe(),
    )
    raw = llm(prompt, role="plan")
    specs = parse_plan_json(raw)
    reporting.replan_generated([spec["question"] for spec in specs])
    return specs


def apply_replan(state: "AgentRun", specs: list[dict]) -> list[TaskNode]:
    """
    Rebuild ``state.tasks`` / ``plan`` / ``execution_graph`` after a replan.

    Completed tasks (those with a finding in working memory) are kept exactly as
    they are (same TaskNode objects, so their capability, artifact and schema
    metadata survive). Only the tail is new: the failed task and anything after
    it are dropped, and the new tasks get IDs that continue after the current
    highest ``step_N`` and depend on every kept task.

    Returns the new tail tasks.
    """
    kept = [t for t in state.tasks if t.task_id in state.working_memory.findings]
    new_tasks = build_tasks(
        [spec["question"] for spec in specs],
        specs,
        first_index=next_task_index(state.tasks),
        depends_on_prefix=[t.task_id for t in kept],
    )
    state.tasks = kept + new_tasks
    state.plan = [t.question for t in state.tasks]
    state.execution_graph = ExecutionGraph.from_tasks(state.tasks)
    sync_task_capabilities(state)
    return new_tasks
