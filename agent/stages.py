"""
Stage implementations used by the loop: prompts + LLM calls, OBSERVE/CRITIQUE,
and per-attempt bookkeeping.

Every function takes the :class:`~agent.loop.AgentRun` and reads the LLM from
``state.llm``. Sandbox execution stays in ``agent.loop`` next to the semantic
validation it is paired with.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from agent import reporting
from agent.findings import build_history_summary, coerce_trust, final_results
from agent.logger import LogEntry
from agent.observer import observe
from agent.parsing import parse_critique_json, strip_markdown_fences
from agent.prompts import (
    codegen_prompt,
    critique_prompt,
    repair_prompt,
    summary_prompt,
)
from agent.run_export import log_entry
from agent.state import SCHEMA_PLACEHOLDERS, AttemptRecord, TaskMemory, TaskNode

if TYPE_CHECKING:
    from agent.loop import AgentRun


def generate_code(
    state: "AgentRun",
    task: TaskNode,
    task_memory: TaskMemory,
    iteration: int,
    previous_code: str | None,
    previous_reason: str | None,
) -> str:
    """CODE-GEN on the first attempt, REPAIR afterwards."""
    reporting.codegen_start(iteration)
    if iteration == 0:
        code = do_codegen(state, task, task_memory)
    else:
        code = do_repair(
            state, task, previous_code or "", previous_reason or "", task_memory
        )
    reporting.codegen_done(len(code))
    return code


def observe_and_critique(
    state: "AgentRun",
    task: TaskNode,
    df: pd.DataFrame,
    code: str,
    sandbox_result: dict[str, Any],
    error_category: str,
) -> tuple[str, str, str, str, str]:
    """
    OBSERVE (deterministic data-quality facts) then CRITIQUE.

    The observer computes actual null counts, duplicates, and outliers so the
    critique LLM sees hard facts instead of speculating.

    Returns ``(raw_critique, trust, action, reason, error_category)``.
    """
    reporting.observe_start()
    obs_report = observe(
        df,
        sandbox_result,
        artifact_store_available_keys=state.artifact_store.available_keys(),
    )
    reporting.observe_result(obs_report)

    reporting.critique_start()
    critique_raw, parsed = do_critique(
        state,
        task.question,
        code,
        sandbox_result,
        build_history_summary(state),
        observation_text=obs_report.as_critique_text(),
    )
    trust = parsed.get("trust", "low")
    action = parsed.get("action", "revise_code")
    reason = parsed.get("reason", "")
    if action == "revise_code" and error_category == "none":
        error_category = "logic_error"
    reporting.critique_result(trust, action, reason)
    return critique_raw, trust, action, reason, error_category


def record_constraints(
    state: "AgentRun",
    task_memory: TaskMemory,
    error_category: str,
    reason: str,
) -> None:
    """Accumulate per-task constraints (and environment limits) from a critique."""
    if error_category == "capability_error":
        state.environment_manifest.add_limitation(f"Unsupported capability: {reason}")
        task_memory.add_constraint(f"Avoid capability error: {reason}")
    elif error_category == "execution_error":
        task_memory.add_constraint(f"Fix runtime error: {reason}")
    elif error_category == "logic_error":
        task_memory.add_constraint(f"Address critique: {reason}")


def record_attempt(
    state: "AgentRun",
    task: TaskNode,
    iteration: int,
    code: str,
    sandbox_result: dict[str, Any],
    critique_raw: str,
    trust: str,
    action: str,
    error_category: str,
) -> AttemptRecord:
    """Append this iteration to the debug history and the JSONL log."""
    attempt = AttemptRecord(
        task_id=task.task_id,
        sub_question=task.question,
        iteration=iteration,
        code=code,
        sandbox_result=sandbox_result,
        critique=critique_raw,
        action=action,
        trust=coerce_trust(trust),
        error_category=error_category,
    )
    state.debug_history.append(attempt)
    log_entry(
        state,
        LogEntry(
            run_id=state.run_id,
            sub_question=task.question,
            iteration=iteration,
            stage="codegen_execute_critique",
            action=action,
            code=code,
            sandbox_result=sandbox_result,
            critique=critique_raw,
            trust=trust,
        ),
    )
    return attempt


def artifact_contract_text(task: TaskNode) -> str:
    """
    Describe the DataFrame a task must produce, for the CODE-GEN / REPAIR prompts.

    The semantic validator requires a DataFrame under ``task.produces_artifact``
    (with the declared columns), so the model has to be told the name and
    columns. Returns ``""`` when the task declares no artifact; the columns line
    is omitted when ``output_schema`` is empty or a placeholder.
    """
    if not task.produces_artifact:
        return ""
    text = (
        "OUTPUT CONTRACT:\n"
        f"Assign your final DataFrame to a variable named `{task.produces_artifact}`"
    )
    schema = list(task.output_schema)
    if schema and schema not in SCHEMA_PLACEHOLDERS:
        text += f"\nwith columns: {', '.join(schema)} (any order)."
    else:
        text += "."
    return text + f"\nThen also set `result = {task.produces_artifact}`."


def do_codegen(
    state: "AgentRun",
    task: TaskNode,
    task_memory: TaskMemory,
) -> str:
    """Call CODE-GEN, strip markdown fences, and return the code string."""
    prompt = codegen_prompt(
        state.goal,
        task.question,
        state.schema_summary,
        environment_manifest=state.environment_manifest.as_prompt_text(),
        task_constraints=task_memory.as_prompt_text(),
        available_artifacts=state.artifact_store.describe(),
        artifact_contract=artifact_contract_text(task),
    )
    raw = state.llm(prompt, role="codegen")
    return strip_markdown_fences(raw)


def do_repair(
    state: "AgentRun",
    task: TaskNode,
    previous_code: str,
    previous_critique_reason: str,
    task_memory: TaskMemory,
) -> str:
    """Call REPAIR, strip markdown fences, and return the code string."""
    prompt = repair_prompt(
        state.goal,
        task.question,
        state.schema_summary,
        environment_manifest=state.environment_manifest.as_prompt_text(),
        task_constraints=task_memory.as_prompt_text(),
        previous_code=previous_code,
        error_reason=previous_critique_reason,
        available_artifacts=state.artifact_store.describe(),
        artifact_contract=artifact_contract_text(task),
    )
    raw = state.llm(prompt, role="codegen")
    return strip_markdown_fences(raw)


def do_critique(
    state: "AgentRun",
    sub_question: str,
    code: str,
    sandbox_result: dict[str, Any],
    history_summary: str,
    observation_text: str = "",
) -> tuple[str, dict[str, Any]]:
    """
    Call CRITIQUE and return (raw_string, parsed_dict).

    Parses the JSON response defensively — if parsing fails, returns a
    safe default that forces a code revision rather than silently accepting.

    Args:
        observation_text: Structured data-quality report from the OBSERVE stage
                          (ObservationReport.as_critique_text()). When provided,
                          the critique prompt includes hard computed facts about
                          nulls, duplicates, and outliers.
    """
    prompt = critique_prompt(
        state.goal,
        sub_question,
        code,
        sandbox_result,
        history_summary,
        observation_text=observation_text,
    )
    raw = state.llm(prompt, role="critique")
    parsed = parse_critique_json(raw)
    return raw, parsed


def do_summary(state: "AgentRun") -> str:
    """
    Generate the final cohesive summary from ACCEPTED findings only.

    Only the final accepted result for each sub-question is passed to the
    summary LLM. Intermediate failed iterations are internal loop state
    and must never appear in the summary narrative.
    """
    reporting.summary_stage_start()
    findings = final_results(state)
    prompt = summary_prompt(state.goal, findings)
    summary = state.llm(prompt, role="observe")

    log_entry(
        state,
        LogEntry(
            run_id=state.run_id,
            sub_question=state.goal,
            iteration=0,
            stage="summary",
            action="summary_generated",
            extra={"summary": summary, "finding_count": len(findings)},
        ),
    )
    return summary
