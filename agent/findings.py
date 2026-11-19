"""
Helpers that turn attempts into findings and read working memory back out.

``AgentRun.debug_history`` holds every attempt; ``working_memory`` holds only
accepted findings. These helpers promote an accepted attempt into a
:class:`~agent.state.Finding` and build the prompt-facing views of that memory.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from agent.planning import task_order
from agent.state import AttemptRecord, ExecutionDifficulty, Finding, TaskNode

if TYPE_CHECKING:
    from agent.loop import AgentRun


def attempt_for_finding(
    state: "AgentRun",
    finding: Finding,
) -> AttemptRecord | None:
    """Return the accepted debug attempt that produced *finding*."""
    for attempt in reversed(state.debug_history):
        if (
            attempt.task_id == finding.task_id
            and attempt.action == "accept"
            and attempt.iteration == finding.source_iteration
        ):
            return attempt
    return None


def build_history_summary(state: "AgentRun") -> str:
    """
    Build a summary of accepted findings for the critique prompt.

    Includes the sub-question, result, AND the first 200 chars of the code
    so the critic can compare filter operations across steps rather than
    guessing whether scope differences are intentional. Failed intermediate
    iterations are excluded so the critic does not treat retry noise as
    analytical memory.
    """
    accepted_findings = state.final_results
    if not accepted_findings:
        return "(none yet)"
    lines = []
    for i, finding in enumerate(accepted_findings, 1):
        attempt = attempt_for_finding(state, finding)
        q = finding.question[:80]
        result_snippet = " → " + finding.answer[:100] if finding.answer else ""
        # Include enough of the code for scope comparison (date filters, groupby, etc.)
        code = (attempt.code if attempt else "").strip()
        code_preview = code[:200].replace("\n", " | ")
        if len(code) > 200:
            code_preview += "…"
        lines.append(
            f"[{i}] [accept/{finding.answer_confidence}] {finding.artifact_key}: {q}{result_snippet}\n"
            f"    code: {code_preview}"
        )
    return "\n".join(lines)


def final_results(state: "AgentRun") -> list[dict[str, str]]:
    """
    Build the sanitized final-results payload for the summary stage.

    The summary LLM must not see retry counts, failed attempts, exception text,
    raw critiques, or code. Those are debug history. It receives only final
    question/answer/trust triples, in plan order.

    Args:
        state: Current agent run.

    Returns:
        List of dicts with ``question``, ``answer``, and ``trust`` keys.
    """
    return state.working_memory.summary_payload(task_order(state.tasks))


def artifact_key(task: TaskNode) -> str:
    """Return a stable reusable artifact key for a task."""
    slug = re.sub(r"[^a-z0-9]+", "_", task.question.lower()).strip("_")
    slug = slug[:48] or task.task_id
    return f"{task.task_id}_{slug}"


def finding_from_attempt(
    attempt: AttemptRecord,
    task: TaskNode,
    attempt_count: int,
    had_execution_errors: bool,
    reason: str,
) -> Finding:
    """Promote an accepted attempt into validated working memory."""
    sandbox_result = attempt.sandbox_result or {}
    return Finding(
        task_id=task.task_id,
        question=task.question,
        answer=str(sandbox_result.get("result", "(no result)")),
        answer_confidence=attempt.trust,
        evidence=reason,
        artifact_key=artifact_key(task),
        dependencies=list(task.depends_on),
        execution_difficulty=ExecutionDifficulty(
            attempts=attempt_count,
            had_execution_errors=had_execution_errors,
            final_error_type=sandbox_result.get("error_type"),
        ),
        source_iteration=attempt.iteration,
    )


def build_working_memory_prompt(state: "AgentRun") -> str:
    """
    Return validated operational memory for downstream stages.

    Combines the compact working-memory bullet list with the artifact store
    description so CODE-GEN sees both the accepted text answers AND which
    pre-validated DataFrames are available for reuse.
    """
    memory_text = state.working_memory.as_prompt_text(task_order(state.tasks))
    artifact_text = state.artifact_store.describe()
    if artifact_text == "(no pre-computed artifacts available yet)":
        return memory_text
    return f"{memory_text}\n\n{artifact_text}"


def coerce_trust(value: Any) -> str:
    """Normalize arbitrary trust values to the supported trust labels."""
    if value in ("high", "low"):
        return str(value)
    return "unknown"
