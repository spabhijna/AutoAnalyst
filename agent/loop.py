"""
Five-stage agent loop: PLAN → CODE-GEN → EXECUTE → CRITIQUE → (accept/revise).

The ``AgentRun`` class holds all mutable state for one run. The ``run()``
method drives the loop and returns the final state including the LLM-generated
summary.

Loop control
------------
For each sub-question:
  1. CODE-GEN: generate pandas code.
  2. EXECUTE:  run in sandbox.
  3. CRITIQUE: independent LLM evaluates the result.
  4a. accept        → log, move to next sub-question.
  4b. revise_code   → loop back to CODE-GEN (up to max_retries_per_step).
                      On exhaustion → ABORT the run (raising PipelineAbortedError).
                      Continuing on a foundationally broken step poisons all
                      subsequent steps — a clean stop is safer than cascading
                      low-trust results into the analysis.
  4c. revise_plan   → regenerate remaining plan (up to MAX_PLAN_REVISIONS).

After all sub-questions → SUMMARY: feed sanitized final results to the LLM.

Every accepted OR aborted step is logged to JSONL immediately.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from agent import config, reporting
from agent.artifact_store import ArtifactStore
from agent.execution_engine import ExecutionEngine
from agent.execution_graph import ExecutionGraph
from agent.llm_client import call_llm
from agent.observer import observe
from agent.parsing import parse_critique_json, strip_markdown_fences
from agent.planning import (
    build_tasks,
    do_plan,
    do_replan,
    task_id,
    task_order,
)
from agent.schema import build_schema_summary
from agent.logger import LogEntry, export_run_json, new_run_id, write_log_entry
from agent.prompts import (
    codegen_prompt,
    critique_prompt,
    repair_prompt,
    summary_prompt,
)
from agent.environment import EnvironmentManifest
from agent.sandbox import run_sandboxed
from agent.state import (
    AttemptRecord,
    ExecutionDifficulty,
    Finding,
    TaskNode,
    TaskMemory,
    WorkingMemory,
    utc_now_iso,
)
from agent.validator import CapabilityValidator

logger = logging.getLogger(__name__)

MAX_PLAN_REVISIONS: int = 2
SANDBOX_TIMEOUT: int = 10  # seconds per execution


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PipelineAbortedError(RuntimeError):
    """
    Raised when a sub-question exhausts all CODE-GEN retries without producing
    a trustworthy result.

    Continuing past this point would cascade an unverified (possibly wrong)
    finding into all downstream sub-questions, making the entire run
    unreliable. The run stops here and returns the partial state.

    Attributes:
        sub_question: The question that could not be answered.
        last_reason:  The critique's reason from the final attempt.
        run_id:       The run identifier for log correlation.
    """

    def __init__(self, sub_question: str, last_reason: str, run_id: str) -> None:
        self.sub_question = sub_question
        self.last_reason = last_reason
        self.run_id = run_id
        super().__init__(
            f"Pipeline aborted (run_id={run_id}): max retries exhausted on "
            f"sub_question={sub_question!r}. Last critique: {last_reason}"
        )


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------


@dataclass
class AgentRun:
    """
    Holds the full mutable state of one pipeline run.

    Attributes:
        goal:                 High-level analytical goal from the user.
        schema_summary:       Column names/dtypes/sample rows (NOT full data).
        plan:                 Ordered list of sub-questions.
        tasks:                Explicit task nodes with dependencies.
        debug_history:        Every attempt, including failures/retries.
        working_memory:       Accepted findings only.
        final_results:        Accepted findings in plan order.
        current_step:         Index into ``plan`` currently being executed.
        max_retries_per_step: Hard cap on CODE-GEN retries per sub-question.
        run_id:               Unique identifier for this run (auto-generated).
        plan_revisions:       Count of plan revisions so far (capped at MAX_PLAN_REVISIONS).
        summary:              Final LLM-generated summary (populated at end of run).
        log_dir:              Directory where JSONL logs are written.
    """

    goal: str
    schema_summary: str
    plan: list[str] = field(default_factory=list)
    tasks: list[TaskNode] = field(default_factory=list)
    debug_history: list[AttemptRecord] = field(default_factory=list)
    working_memory: WorkingMemory = field(default_factory=WorkingMemory)
    final_results: list[Finding] = field(default_factory=list)
    current_step: int = 0
    max_retries_per_step: int = config.MAX_PIPELINE_RETRIES
    run_id: str = field(default_factory=new_run_id)
    plan_revisions: int = 0
    summary: str = ""
    log_dir: str = config.LOG_DIR
    aborted: bool = False
    abort_reason: str = ""
    llm_call_count: int = 0
    environment_manifest: EnvironmentManifest = field(
        default_factory=EnvironmentManifest
    )
    task_memories: dict[str, TaskMemory] = field(default_factory=dict)
    artifact_store: ArtifactStore = field(default_factory=ArtifactStore)
    execution_engine: ExecutionEngine = field(default_factory=ExecutionEngine)
    execution_graph: ExecutionGraph = field(default_factory=ExecutionGraph)

    @property
    def history(self) -> list[dict[str, Any]]:
        """
        Backward-compatible debug-history view.

        New code should use ``debug_history`` for attempts and
        ``working_memory``/``final_results`` for accepted analytical memory.
        """
        return [attempt.to_legacy_dict() for attempt in self.debug_history]

    @history.setter
    def history(self, entries: list[dict[str, Any]]) -> None:
        """Set debug history from the legacy dict shape used by older tests."""
        self.tasks = build_tasks(self.plan)
        question_to_task_id = {task.question: task.task_id for task in self.tasks}
        self.debug_history = [
            AttemptRecord(
                task_id=str(
                    entry.get("task_id")
                    or question_to_task_id.get(str(entry.get("sub_question", "")))
                    or task_id(i)
                ),
                sub_question=str(entry.get("sub_question", "")),
                iteration=int(entry.get("iteration", 0)),
                code=str(entry.get("code", "")),
                sandbox_result=entry.get("sandbox_result") or {},
                critique=str(entry.get("critique", "")),
                action=str(entry.get("action", "")),
                trust=_coerce_trust(entry.get("trust")),
                timestamp=str(entry.get("timestamp", "")) or utc_now_iso(),
            )
            for i, entry in enumerate(entries)
        ]
        self.working_memory = WorkingMemory()
        self.final_results = []
        for attempt in self.debug_history:
            if attempt.action == "accept":
                finding = _finding_from_attempt(
                    attempt=attempt,
                    task=_task_for_attempt(self.tasks, attempt),
                    attempt_count=attempt.iteration + 1,
                    had_execution_errors=False,
                    reason=parse_critique_json(attempt.critique).get("reason", ""),
                )
                self.working_memory.add_finding(finding)
        self.final_results = self.working_memory.accepted_findings(
            task_order(self.tasks)
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(
    goal: str,
    df: pd.DataFrame,
    log_dir: str = config.LOG_DIR,
    max_retries_per_step: int = config.MAX_PIPELINE_RETRIES,
    dry_run: bool = False,
) -> AgentRun:
    """
    Execute the full five-stage loop for *goal* against *df*.

    At the end of every run (including aborted runs), writes a structured
    ``runs/<run_id>.json`` summary via :func:`agent.logger.export_run_json`.

    Args:
        goal:               High-level analytical question.
        df:                 The pandas DataFrame to analyse.
        log_dir:            Directory for JSONL and JSON run logs.
        max_retries_per_step: Hard cap on CODE-GEN retries per sub-question.
        dry_run:            When True, uses canned LLM responses so the
                            pipeline runs without API keys.  Defaults to False.

    Returns:
        Completed :class:`AgentRun` with full history and summary populated.
    """
    t_start = time.monotonic()
    schema_summary = build_schema_summary(df)
    state = AgentRun(
        goal=goal,
        schema_summary=schema_summary,
        log_dir=log_dir,
        max_retries_per_step=max_retries_per_step,
    )

    logger.info("=== AgentRun start: run_id=%s ===", state.run_id)
    reporting.run_banner(state.run_id, goal)

    # ── Inject dry-run mode ─────────────────────────────────────────────────
    if dry_run:
        import sys as _sys

        _m = _sys.modules[__name__]

        def _dry_call(prompt: str, role: str, model: str | None = None) -> str:
            plan_json = (
                '[{"question": "Filter to last quarter and return quarter_df", "capability": "FilterByDateRange", "produces_artifact": "quarter_df", "consumes_artifacts": [], "output_schema": ["transaction_id", "date", "region", "product_category", "units_sold", "unit_price", "revenue"]}, '
                '{"question": "What is total revenue by region in the last quarter?", "capability": "GroupRevenueByRegion", "produces_artifact": "region_revenue", "consumes_artifacts": ["quarter_df"], "output_schema": ["region", "total_revenue"]}, '
                '{"question": "Which region had highest revenue and which category drove it?", "capability": "FindMaximumRevenueRegion", "produces_artifact": "top_region", "consumes_artifacts": ["region_revenue"], "output_schema": ["region", "total_revenue"]}, '
                '{"question": "Check if top region performance was consistent month-over-month", "capability": "MonthlyRevenueTrend", "produces_artifact": "monthly_trend", "consumes_artifacts": ["quarter_df"], "output_schema": ["month", "total_revenue"]}, '
                '{"question": "What was the overall revenue trend by region over the quarter?", "capability": "RevenueShare", "produces_artifact": "revenue_share", "consumes_artifacts": ["region_revenue"], "output_schema": ["region", "total_revenue", "revenue_share"]}]'
            )
            code = 'result = df.describe() if len(df) > 0 else "(empty DataFrame)"'
            critique = (
                '{"trust": "high", "action": "accept", "reason": "Dry-run: verified."}'
            )
            summary = (
                "Dry-run summary: The pipeline executed successfully without LLM API keys. "
                "The planner produced a capability-based execution graph, and all five "
                "steps were resolved deterministically through registered capabilities."
            )
            if role == "plan":
                return plan_json
            if role == "codegen":
                return code
            if role == "critique":
                return critique
            return summary

        _m.call_llm = _dry_call
        logger.info("Dry-run mode enabled — using canned LLM responses")
        reporting.dry_run_notice()

    # ── PLAN ──────────────────────────────────────────────────────────────────
    do_plan(state, df, call_llm, _log)  # sets state.plan and state.tasks (with artifact declarations)

    # ── Per-sub-question loop ─────────────────────────────────────────────────
    try:
        while not state.execution_graph.all_completed():
            completed_task_ids = set(state.working_memory.findings.keys())
            task = state.execution_graph.next_ready_task(completed_task_ids)
            if task is None:
                raise PipelineAbortedError(
                    sub_question="(scheduler)",
                    last_reason="No ready task was available before the graph completed.",
                    run_id=state.run_id,
                )
            state.current_step = state.execution_graph.task_index(task.task_id)
            sub_question = task.question
            reporting.step_header(state.current_step, len(state.plan), sub_question)
            _ensure_dependencies_available(state, task)
            task.status = "running"

            revise_plan_requested = _run_question(state, task, df)

            if revise_plan_requested:
                if state.plan_revisions >= MAX_PLAN_REVISIONS:
                    logger.warning(
                        "Plan revision requested but cap (%d) reached — accepting current plan",
                        MAX_PLAN_REVISIONS,
                    )
                    state.current_step += 1
                else:
                    new_plan_tail = do_replan(
                        state, df, call_llm, _build_working_memory_prompt(state)
                    )
                    state.plan = state.plan[: state.current_step] + new_plan_tail
                    state.tasks = build_tasks(state.plan)
                    state.execution_graph = ExecutionGraph.from_tasks(state.tasks)
                    state.plan_revisions += 1
                    logger.info(
                        "Plan revised (revision %d/%d): %d new questions",
                        state.plan_revisions,
                        MAX_PLAN_REVISIONS,
                        len(new_plan_tail),
                    )
            else:
                state.current_step += 1

    except PipelineAbortedError as exc:
        state.aborted = True
        state.abort_reason = str(exc)
        _mark_failed_and_skipped(state, exc.sub_question, exc.last_reason)
        logger.error("Pipeline aborted: %s", exc)
        _log(
            state,
            LogEntry(
                run_id=state.run_id,
                sub_question=exc.sub_question,
                iteration=state.max_retries_per_step,
                stage="codegen_execute_critique",
                action="pipeline_aborted",
                extra={"abort_reason": exc.last_reason},
            ),
        )
        reporting.abort_banner(
            state.current_step + 1,
            len(state.plan),
            state.max_retries_per_step,
            exc.sub_question,
            exc.last_reason,
            state.run_id,
        )
        # ── Write structured run JSON even on abort ───────────────────────────
        _finalize_run(state, t_start)
        return state

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    state.summary = _do_summary(state)

    logger.info("=== AgentRun complete: run_id=%s ===", state.run_id)
    reporting.final_summary(state.summary)

    # ── Write structured run JSON ─────────────────────────────────────────────
    _finalize_run(state, t_start)
    return state


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------


def _run_question(
    state: AgentRun,
    task: TaskNode,
    df: pd.DataFrame,
) -> bool:
    """
    Run the CODE-GEN → EXECUTE → CRITIQUE loop for one sub-question.

    Before CODE-GEN, builds an artifact namespace from the ArtifactStore for
    all keys listed in ``task.consumes_artifacts``. This namespace is injected
    into the sandbox so generated code can read pre-validated DataFrames without
    re-deriving them.

    After CRITIQUE accepts a result, if ``task.produces_artifact`` is set, the
    named DataFrame is extracted from the sandbox result and registered in
    ``state.artifact_store`` for downstream steps.

    Returns:
        True if critique returned "revise_plan", False otherwise.
    """
    previous_code: str | None = None
    previous_critique_reason: str | None = None
    had_execution_errors = False
    sub_question = task.question

    if task.task_id not in state.task_memories:
        state.task_memories[task.task_id] = TaskMemory(task_id=task.task_id)
    task_memory = state.task_memories[task.task_id]

    # ── Pre-build artifact namespace for this task ───────────────────────────────────
    # If the task has explicit consumes_artifacts, fetch only those.
    # Otherwise, expose the full ArtifactStore so any prior artifact is reachable.
    if task.consumes_artifacts:
        artifact_namespace = state.artifact_store.as_namespace_for(
            task.consumes_artifacts
        )
    else:
        artifact_namespace = state.artifact_store.as_namespace()

    deterministic_result = state.execution_engine.execute(
        task,
        df,
        state.artifact_store,
        consumers=state.execution_graph.consumers_for(task.produces_artifact)
        if task.produces_artifact
        else [],
    )
    if deterministic_result is not None:
        reporting.deterministic_execute_start()
        sandbox_result = deterministic_result
        reporting.deterministic_execute_ok()
        reason = "Deterministic executor validated the reusable dataframe step."
        trust = "high"
        attempt = AttemptRecord(
            task_id=task.task_id,
            sub_question=sub_question,
            iteration=0,
            code="(deterministic executor)",
            sandbox_result=sandbox_result,
            critique=reason,
            action="accept",
            trust=_coerce_trust(trust),
            error_category="none",
        )
        state.debug_history.append(attempt)
        _log(
            state,
            LogEntry(
                run_id=state.run_id,
                sub_question=sub_question,
                iteration=0,
                stage="deterministic_execute",
                action="accept",
                code=attempt.code,
                sandbox_result=sandbox_result,
                critique=reason,
                trust=trust,
            ),
        )
        task.status = "accepted"
        finding = _finding_from_attempt(
            attempt=attempt,
            task=task,
            attempt_count=1,
            had_execution_errors=False,
            reason=reason,
        )
        state.working_memory.add_finding(finding)
        state.final_results = state.working_memory.accepted_findings(
            task_order(state.tasks)
        )
        if task.produces_artifact:
            extracted_df = sandbox_result.get("artifact")
            if isinstance(extracted_df, pd.DataFrame):
                finding.produced_artifact_name = task.produces_artifact
                state.artifact_store.register(
                    key=task.produces_artifact,
                    df=extracted_df,
                    producing_task_id=task.task_id,
                    consumers=state.execution_graph.consumers_for(
                        task.produces_artifact
                    ),
                    confidence="high",
                )
        return False

    for iteration in range(state.max_retries_per_step):
        # ── CODE-GEN ──────────────────────────────────────────────────────────
        reporting.codegen_start(iteration)
        if iteration == 0:
            code = _do_codegen(state, sub_question, task_memory)
        else:
            code = _do_repair(
                state,
                sub_question,
                previous_code or "",
                previous_critique_reason or "",
                task_memory,
            )
        reporting.codegen_done(len(code))

        # ── EXECUTE ───────────────────────────────────────────────────────────
        reporting.execute_start()
        sandbox_result = run_sandboxed(
            code,
            df,
            artifacts=artifact_namespace,
            extract_artifact=task.produces_artifact,
            timeout_seconds=SANDBOX_TIMEOUT,
        )
        had_execution_errors = had_execution_errors or not sandbox_result["success"]
        status = (
            "✓ OK"
            if sandbox_result["success"]
            else f"✗ {sandbox_result.get('error_type', 'ERROR')}"
        )
        reporting.execute_status(status, sandbox_result.get("stdout"))

        error_category = sandbox_result.get("error_category", "none")

        if sandbox_result["success"]:
            semantic_validation = CapabilityValidator().validate_semantic_contract(
                task.task_id, 
                sandbox_result.get("artifact"), 
                sandbox_result.get("result"), 
                state.artifact_store.as_namespace()
            )
            if not semantic_validation.success:
                sandbox_result["success"] = False
                sandbox_result["error"] = "Semantic Validation Failed: " + "; ".join(semantic_validation.reasons)
                sandbox_result["error_category"] = "logic_error"
                error_category = "logic_error"
                had_execution_errors = True
                reporting.semantic_error(sandbox_result['error'])

        # ── OBSERVE ───────────────────────────────────────────────────────────
        # Deterministic data-quality checks on the live DataFrame BEFORE critique.
        # The observer computes actual null counts, duplicates, and outliers so
        # the critique LLM sees hard facts instead of speculating.
        reporting.observe_start()
        obs_report = observe(
            df,
            sandbox_result,
            artifact_store_available_keys=state.artifact_store.available_keys(),
        )
        reporting.observe_result(obs_report)

        # ── CRITIQUE ──────────────────────────────────────────────────────────
        reporting.critique_start()
        history_summary = _build_history_summary(state)
        critique_raw, critique_parsed = _do_critique(
            state,
            sub_question,
            code,
            sandbox_result,
            history_summary,
            observation_text=obs_report.as_critique_text(),
        )
        trust = critique_parsed.get("trust", "low")
        action = critique_parsed.get("action", "revise_code")
        reason = critique_parsed.get("reason", "")
        if action == "revise_code" and error_category == "none":
            error_category = "logic_error"

        # ── ACCUMULATE CONSTRAINTS ─────────────────────────────────────────────
        if error_category == "capability_error":
            limitation = f"Unsupported capability: {reason}"
            state.environment_manifest.add_limitation(limitation)
            task_memory.add_constraint(f"Avoid capability error: {reason}")
        elif error_category == "execution_error":
            task_memory.add_constraint(f"Fix runtime error: {reason}")
        elif error_category == "logic_error":
            task_memory.add_constraint(f"Address critique: {reason}")

        reporting.critique_result(trust, action, reason)

        # ── LOG this iteration ─────────────────────────────────────────────────
        attempt = AttemptRecord(
            task_id=task.task_id,
            sub_question=sub_question,
            iteration=iteration,
            code=code,
            sandbox_result=sandbox_result,
            critique=critique_raw,
            action=action,
            trust=_coerce_trust(trust),
            error_category=error_category,
        )
        state.debug_history.append(attempt)

        _log(
            state,
            LogEntry(
                run_id=state.run_id,
                sub_question=sub_question,
                iteration=iteration,
                stage="codegen_execute_critique",
                action=action,
                code=code,
                sandbox_result=sandbox_result,
                critique=critique_raw,
                trust=trust,
            ),
        )

        # ── ROUTING ───────────────────────────────────────────────────────────
        if action == "accept":
            task.status = "accepted"
            finding = _finding_from_attempt(
                attempt=attempt,
                task=task,
                attempt_count=iteration + 1,
                had_execution_errors=had_execution_errors,
                reason=reason,
            )
            state.working_memory.add_finding(finding)
            state.final_results = state.working_memory.accepted_findings(
                task_order(state.tasks)
            )

            # ── Artifact registration ──────────────────────────────────────────
            # If the task declared it would produce an artifact, check whether
            # the sandbox extracted a DataFrame for it and register it.
            if task.produces_artifact:
                extracted_df = sandbox_result.get("artifact")
                if isinstance(extracted_df, pd.DataFrame):
                    registered = state.artifact_store.register(
                        key=task.produces_artifact,
                        df=extracted_df,
                        producing_task_id=task.task_id,
                        consumers=state.execution_graph.consumers_for(
                            task.produces_artifact
                        ),
                        confidence=trust,
                    )
                    if registered:
                        finding.produced_artifact_name = task.produces_artifact
                        logger.info(
                            "Artifact %r registered from %s (%d rows)",
                            task.produces_artifact,
                            task.task_id,
                            len(extracted_df),
                        )
                    else:
                        logger.warning(
                            "Task %s declared produces_artifact=%r but registration failed",
                            task.task_id,
                            task.produces_artifact,
                        )

            return False  # move to next sub-question

        if action == "revise_plan":
            task.status = "failed"
            task.failure_reason = reason
            return True  # signal caller to replan

        # action == "revise_code" → continue loop
        previous_code = code
        previous_critique_reason = reason

    # Retries exhausted — abort the pipeline.
    # Do NOT force-accept: a foundational step that can't produce a trustworthy
    # answer will corrupt every downstream step that depends on it.
    last_reason = previous_critique_reason or "(unknown)"
    logger.error(
        "Max retries (%d) exhausted for sub_question=%r — aborting pipeline",
        state.max_retries_per_step,
        sub_question[:60],
    )
    reporting.max_retries_exceeded(state.max_retries_per_step)
    raise PipelineAbortedError(
        sub_question=sub_question,
        last_reason=last_reason,
        run_id=state.run_id,
    )


def _do_codegen(
    state: AgentRun,
    sub_question: str,
    task_memory: TaskMemory,
) -> str:
    """Call CODE-GEN, strip markdown fences, and return the code string."""
    prompt = codegen_prompt(
        state.goal,
        sub_question,
        state.schema_summary,
        environment_manifest=state.environment_manifest.as_prompt_text(),
        task_constraints=task_memory.as_prompt_text(),
        available_artifacts=state.artifact_store.describe(),
    )
    raw = call_llm(prompt, role="codegen")
    return strip_markdown_fences(raw)


def _do_repair(
    state: AgentRun,
    sub_question: str,
    previous_code: str,
    previous_critique_reason: str,
    task_memory: TaskMemory,
) -> str:
    """Call REPAIR, strip markdown fences, and return the code string."""
    prompt = repair_prompt(
        state.goal,
        sub_question,
        state.schema_summary,
        environment_manifest=state.environment_manifest.as_prompt_text(),
        task_constraints=task_memory.as_prompt_text(),
        previous_code=previous_code,
        error_reason=previous_critique_reason,
        available_artifacts=state.artifact_store.describe(),
    )
    raw = call_llm(prompt, role="codegen")
    return strip_markdown_fences(raw)


def _do_critique(
    state: AgentRun,
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
    raw = call_llm(prompt, role="critique")
    parsed = parse_critique_json(raw)
    return raw, parsed


def _accepted_findings(state: AgentRun) -> list[dict]:
    """
    Return exactly one history entry per sub-question: the final accepted result.

    Intermediate failed iterations (action="revise_code") are internal loop
    state and must not appear in the summary. By the time this is called, every
    sub-question that reached the summary stage has exactly one accepted entry
    (the pipeline aborts before reaching _do_summary if any question exhausts
    retries without being accepted).

    Returns:
        Ordered list of accepted entries, one per sub-question.
    """
    return [
        _attempt_for_finding(state, finding).to_legacy_dict()
        for finding in state.final_results
        if _attempt_for_finding(state, finding) is not None
    ]


def _do_summary(state: AgentRun) -> str:
    """
    Generate the final cohesive summary from ACCEPTED findings only.

    Only the final accepted result for each sub-question is passed to the
    summary LLM. Intermediate failed iterations are internal loop state
    and must never appear in the summary narrative.
    """
    reporting.summary_stage_start()
    findings = _final_results(state)
    prompt = summary_prompt(state.goal, findings)
    summary = call_llm(prompt, role="observe")

    _log(
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_history_summary(state: AgentRun) -> str:
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
        attempt = _attempt_for_finding(state, finding)
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


def _final_results(state: AgentRun) -> list[dict[str, str]]:
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


def _task_for_attempt(tasks: list[TaskNode], attempt: AttemptRecord) -> TaskNode:
    """
    Return the task matching *attempt*.

    A fallback node is returned for legacy tests that seed history without
    constructing tasks first.
    """
    for task in tasks:
        if task.task_id == attempt.task_id or task.question == attempt.sub_question:
            return task
    return TaskNode(task_id=attempt.task_id, question=attempt.sub_question)


def _ensure_dependencies_available(state: AgentRun, task: TaskNode) -> None:
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


def _mark_failed_and_skipped(
    state: AgentRun,
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


def _artifact_key(task: TaskNode) -> str:
    """Return a stable reusable artifact key for a task."""
    slug = re.sub(r"[^a-z0-9]+", "_", task.question.lower()).strip("_")
    slug = slug[:48] or task.task_id
    return f"{task.task_id}_{slug}"


def _finding_from_attempt(
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
        artifact_key=_artifact_key(task),
        dependencies=list(task.depends_on),
        execution_difficulty=ExecutionDifficulty(
            attempts=attempt_count,
            had_execution_errors=had_execution_errors,
            final_error_type=sandbox_result.get("error_type"),
        ),
        source_iteration=attempt.iteration,
    )


def _attempt_for_finding(
    state: AgentRun,
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


def _build_working_memory_prompt(state: AgentRun) -> str:
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


def _coerce_trust(value: Any) -> str:
    """Normalize arbitrary trust values to the supported trust labels."""
    if value in ("high", "low"):
        return str(value)
    return "unknown"


def _log(state: AgentRun, entry: LogEntry) -> None:
    """Write a log entry, catching I/O errors so they don't crash the run."""
    try:
        write_log_entry(entry, state.log_dir)
    except OSError as exc:
        logger.error("Failed to write log entry for run_id=%s: %s", state.run_id, exc)


def _finalize_run(state: AgentRun, t_start: float) -> None:
    """
    Write the structured run JSON summary at the end of a run.

    Called from ``run()`` on both successful completion and abort.
    Aggregates metadata (goal, model versions, LLM call count, wall clock)
    + the full debug history into ``runs/<run_id>.json``.
    """
    from agent.config import (
        GROQ_CODEGEN_MODEL,
        GROQ_CRITIQUE_MODEL,
        GROQ_DEFAULT_MODEL,
    )

    wall_clock = time.monotonic() - t_start
    debug_entries = [attempt.to_legacy_dict() for attempt in state.debug_history]

    # Derive model versions from config (no live introspection needed)
    model_versions = {
        "plan": GROQ_DEFAULT_MODEL,
        "codegen": GROQ_CODEGEN_MODEL,
        "critique": GROQ_CRITIQUE_MODEL,
        "observe": GROQ_DEFAULT_MODEL,
    }

    json_path = export_run_json(
        run_id=state.run_id,
        goal=state.goal,
        dataset_label="clean" if not state.aborted else "clean (aborted)",
        model_versions=model_versions,
        debug_entries=debug_entries,
        final_summary=state.summary if not state.aborted else "",
        aborted=state.aborted,
        wall_clock_seconds=wall_clock,
        total_llm_calls=state.llm_call_count,
        log_dir=state.log_dir,
    )
    reporting.run_json_exported(json_path)
