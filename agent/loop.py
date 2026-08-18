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

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from agent import config
from agent.artifact_store import ArtifactStore
from agent.capabilities import CapabilityRegistry
from agent.execution_engine import ExecutionEngine
from agent.execution_graph import ExecutionGraph
from agent.llm_client import call_llm
from agent.observer import observe
from agent.logger import LogEntry, export_run_json, new_run_id, write_log_entry
from agent.prompts import (
    codegen_prompt,
    critique_prompt,
    plan_prompt,
    replan_prompt,
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
    capability_registry: CapabilityRegistry = field(
        default_factory=lambda: ExecutionEngine().registry
    )
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
        self.tasks = _build_tasks(self.plan)
        question_to_task_id = {task.question: task.task_id for task in self.tasks}
        self.debug_history = [
            AttemptRecord(
                task_id=str(
                    entry.get("task_id")
                    or question_to_task_id.get(str(entry.get("sub_question", "")))
                    or _task_id(i)
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
                    reason=_parse_critique_json(attempt.critique).get("reason", ""),
                )
                self.working_memory.add_finding(finding)
        self.final_results = self.working_memory.accepted_findings(
            _task_order(self.tasks)
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
    schema_summary = _build_schema_summary(df)
    state = AgentRun(
        goal=goal,
        schema_summary=schema_summary,
        log_dir=log_dir,
        max_retries_per_step=max_retries_per_step,
    )

    logger.info("=== AgentRun start: run_id=%s ===", state.run_id)
    print(f"\n{'=' * 60}")
    print(f"  run_id : {state.run_id}")
    print(f"  goal   : {goal}")
    print(f"{'=' * 60}\n")

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
        print("  [dry-run] LLM calls return canned responses")

    # ── PLAN ──────────────────────────────────────────────────────────────────
    _do_plan(state, df)  # sets state.plan and state.tasks (with artifact declarations)

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
            _print_step_header(state.current_step, len(state.plan), sub_question)
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
                    new_plan_tail = _do_replan(state, df)
                    state.plan = state.plan[: state.current_step] + new_plan_tail
                    state.tasks = _build_tasks(state.plan)
                    state.execution_graph = ExecutionGraph.from_tasks(state.tasks)
                    _sync_artifact_consumers(state)
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
        print(f"\n{'=' * 60}")
        print("  ✖  PIPELINE ABORTED")
        print(f"{'=' * 60}")
        print(f"  Step {state.current_step + 1}/{len(state.plan)} could not produce a")
        print(f"  trustworthy answer after {state.max_retries_per_step} attempts.")
        print(f"  Sub-question: {exc.sub_question}")
        print(f"  Last critique: {exc.last_reason}")
        print()
        print("  Completed steps are in the JSONL log. Fix the data or")
        print("  increase max_retries_per_step and re-run.")
        print(f"  Log: runs/{state.run_id}.jsonl")
        # ── Write structured run JSON even on abort ───────────────────────────
        _finalize_run(state, t_start)
        return state

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    state.summary = _do_summary(state)

    logger.info("=== AgentRun complete: run_id=%s ===", state.run_id)
    print(f"\n{'=' * 60}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(state.summary)

    # ── Write structured run JSON ─────────────────────────────────────────────
    _finalize_run(state, t_start)
    return state


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------


def _do_plan(state: AgentRun, df: pd.DataFrame) -> list[str]:
    """Call the PLAN stage, parse the structured plan, and return sub-question strings."""
    print("► PLAN stage …")
    prompt = plan_prompt(state.goal, state.schema_summary)
    raw = call_llm(prompt, role="plan")
    task_specs = _parse_plan_json(raw)
    questions = [spec["question"] for spec in task_specs]
    state.plan = questions
    state.tasks = _build_tasks(questions, task_specs)
    state.execution_graph = ExecutionGraph.from_tasks(state.tasks)
    _sync_task_capabilities(state)
    _sync_artifact_consumers(state)
    print(f"  → {len(questions)} sub-questions generated")
    for i, q in enumerate(questions, 1):
        print(f"     {i}. {q}")
    print()

    _log(
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


def _do_replan(state: AgentRun, df: pd.DataFrame) -> list[str]:
    """Re-plan from the current step forward, carrying completed findings.

    Replanning emits a flat string array (no artifact declarations) since the
    LLM doesn't know which artifacts have already been registered. Replanned
    tasks will still have access to the ArtifactStore via the `artifacts` sandbox
    global — the code-gen prompt lists all available artifacts regardless of
    whether the task declared explicit consumes_artifacts.
    """
    print("► RE-PLAN stage …")
    completed = _build_completed_findings(state)
    # Get the critique reason from the most recent debug attempt.
    last_reason = ""
    if state.debug_history:
        last_reason = state.debug_history[-1].critique

    prompt = replan_prompt(
        state.goal,
        state.schema_summary,
        completed,
        last_reason,
    )
    raw = call_llm(prompt, role="plan")
    # Replan uses flat string array for simplicity
    questions = _parse_json_list(raw)
    print(f"  → {len(questions)} new sub-questions")
    for i, q in enumerate(questions, 1):
        print(f"     {i}. {q}")
    print()
    return questions


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
        print("  ► DETERMINISTIC EXECUTE …")
        sandbox_result = deterministic_result
        print("    ✓ OK (deterministic)")
        obs_report = observe(
            df,
            sandbox_result,
            artifact_store_available_keys=state.artifact_store.available_keys(),
        )
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
            _task_order(state.tasks)
        )
        if task.produces_artifact:
            extracted_df = sandbox_result.get("artifact")
            if isinstance(extracted_df, pd.DataFrame):
                finding.produced_artifact_name = task.produces_artifact
                _sync_artifact_consumers(state)
        return False

    for iteration in range(state.max_retries_per_step):
        # ── CODE-GEN ──────────────────────────────────────────────────────────
        print(f"  ► CODE-GEN (iter {iteration}) …")
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
        print(f"    {len(code)} chars generated")

        # ── EXECUTE ───────────────────────────────────────────────────────────
        print("  ► EXECUTE …")
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
        print(f"    {status}")
        if sandbox_result.get("stdout"):
            # Print first 3 lines of stdout for visibility
            lines = sandbox_result["stdout"].strip().splitlines()[:3]
            for ln in lines:
                print(f"    | {ln}")

        error_category = sandbox_result.get("error_category", "none")

        # ── OBSERVE ───────────────────────────────────────────────────────────
        # Deterministic data-quality checks on the live DataFrame BEFORE critique.
        # The observer computes actual null counts, duplicates, and outliers so
        # the critique LLM sees hard facts instead of speculating.
        print("  ► OBSERVE …")
        obs_report = observe(
            df,
            sandbox_result,
            artifact_store_available_keys=state.artifact_store.available_keys(),
        )
        print(
            f"    ✓ nulls={obs_report.null_counts}, dupes={obs_report.duplicate_txn_ids}, "
            f"neg_rev={obs_report.negative_revenue_rows}, "
            f"outliers={len(obs_report.revenue_outliers_iqr)}"
        )

        # ── CRITIQUE ──────────────────────────────────────────────────────────
        print("  ► CRITIQUE …")
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

        trust_icon = "✓" if trust == "high" else "⚠"
        print(f"    {trust_icon} trust={trust} action={action}")
        print(f"    reason: {reason[:400]}")
        print()

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
                _task_order(state.tasks)
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
    print(
        f"  ✖ max_retries_exceeded after {state.max_retries_per_step} attempts — aborting\n"
    )
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
        completed_findings=_build_working_memory_prompt(state),
        environment_manifest=state.environment_manifest.as_prompt_text(),
        task_constraints=task_memory.as_prompt_text(),
        available_artifacts=state.artifact_store.describe(),
    )
    raw = call_llm(prompt, role="codegen")
    return _strip_markdown_fences(raw)


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
        completed_findings=_build_working_memory_prompt(state),
        environment_manifest=state.environment_manifest.as_prompt_text(),
        task_constraints=task_memory.as_prompt_text(),
        previous_code=previous_code,
        error_reason=previous_critique_reason,
        available_artifacts=state.artifact_store.describe(),
    )
    raw = call_llm(prompt, role="codegen")
    return _strip_markdown_fences(raw)


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
    parsed = _parse_critique_json(raw)
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
    print("► SUMMARY stage …")
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


def _build_schema_summary(df: pd.DataFrame) -> str:
    """
    Build a concise schema description for use in LLM prompts.

    Includes column names, dtypes, null percentages, and 3 sample values.
    Does NOT include full row data.

    The "last quarter" date window is computed here in Python and injected as
    concrete constants (LAST_QUARTER_START / LAST_QUARTER_END) so the LLM
    never has to derive date boundaries itself — eliminating the entire class
    of off-by-one critique loops that arise when the model re-derives the
    window independently on every attempt.

    Args:
        df: The DataFrame to describe.

    Returns:
        Multi-line schema string.
    """
    lines = [f"Rows: {len(df):,}  |  Columns: {len(df.columns)}"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_pct = df[col].isna().mean() * 100
        samples = df[col].dropna().head(3).tolist()
        sample_str = ", ".join(repr(v) for v in samples)
        lines.append(f"  • {col!r} ({dtype}) nulls={null_pct:.1f}% — e.g. {sample_str}")

    # ── Pre-compute the "last quarter" window in Python so LLMs never have to ──
    # "Last quarter" = the 3 most-recent complete calendar months in the data.
    # Example: if max date is 2024-06-15, the window is 2024-04-01 → 2024-06-15.
    last_quarter_start: str | None = None
    last_quarter_end: str | None = None
    if "date" in df.columns:
        try:
            date_series = pd.to_datetime(df["date"].dropna())
            if not date_series.empty:
                max_ts = date_series.max()  # pandas Timestamp
                # First day of the month that is 2 full months before max_ts month.
                # e.g. max=2024-06-15 → start=2024-04-01 (3 months including June).
                start_ts = (max_ts - pd.DateOffset(months=2)).replace(day=1)
                last_quarter_start = start_ts.strftime("%Y-%m-%d")
                last_quarter_end = max_ts.strftime("%Y-%m-%d")
        except Exception:  # noqa: BLE001 — schema build must never crash
            pass

    lines.append("")
    if last_quarter_start and last_quarter_end:
        lines.append(
            "PRE-COMPUTED DATE WINDOW (do NOT re-derive these; copy them verbatim):\n"
            f"  LAST_QUARTER_START = pd.Timestamp('{last_quarter_start}')  # inclusive\n"
            f"  LAST_QUARTER_END   = pd.Timestamp('{last_quarter_end}')    # inclusive\n"
            "\n"
            "Use EXACTLY this filter when the goal mentions 'last quarter':\n"
            "  mask = (pd.to_datetime(df['date']) >= LAST_QUARTER_START) & \\\n"
            "         (pd.to_datetime(df['date']) <= LAST_QUARTER_END)\n"
            "  quarter_df = df[mask]\n"
            "\n"
            "Do NOT use pd.Timestamp('today'), datetime.date objects, or\n"
            "pd.DateOffset arithmetic in your code — the boundaries are already\n"
            "computed above. Mismatching these constants is a logic error."
        )
    else:
        lines.append(
            'Time range note: "last quarter" means the most recent 3 calendar\n'
            "months in the data. Use df['date'].max() as reference, filter back\n"
            "2 full months to the first of that month, and do not use today's date."
        )
    return "\n".join(lines)


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


def _build_completed_findings(state: AgentRun) -> str:
    """Multi-line summary of high-trust completed findings for prompts."""
    return _build_working_memory_prompt(state)


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
    return state.working_memory.summary_payload(_task_order(state.tasks))


def _task_id(index: int) -> str:
    """Return the stable task ID for a 0-based plan index."""
    return f"step_{index + 1}"


def _task_order(tasks: list[TaskNode]) -> list[str]:
    """Return task IDs in current plan order."""
    return [task.task_id for task in tasks]


def _build_tasks(
    plan: list[str], task_specs: list[dict] | None = None
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
                task_id=_task_id(index),
                question=question,
                depends_on=[_task_id(i) for i in range(index)],
                produces_artifact=produces,
                consumes_artifacts=list(consumes),
                capability=capability,
                execution_fn=execution_fn,
                output_schema=list(output_schema),
            )
        )
    return tasks


def _sync_task_capabilities(state: AgentRun) -> None:
    """Resolve missing capability IDs using the registered capability set."""
    for task in state.tasks:
        if task.capability:
            continue
        capability = state.execution_engine.registry.resolve(task)
        if capability is not None:
            task.capability = capability.capability_id


def _sync_artifact_consumers(state: AgentRun) -> None:
    """Validate downstream artifact lineage without mutating frozen artifacts."""
    for task in state.tasks:
        if not task.produces_artifact:
            continue
        artifact = state.artifact_store.get_artifact(task.produces_artifact)
        if artifact is None:
            continue
        expected_consumers = state.execution_graph.consumers_for(task.produces_artifact)
        if artifact.consumers != expected_consumers:
            logger.debug(
                "Artifact %r consumers differ from graph lineage: store=%s graph=%s",
                task.produces_artifact,
                artifact.consumers,
                expected_consumers,
            )


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
    memory_text = state.working_memory.as_prompt_text(_task_order(state.tasks))
    artifact_text = state.artifact_store.describe()
    if artifact_text == "(no pre-computed artifacts available yet)":
        return memory_text
    return f"{memory_text}\n\n{artifact_text}"


def _coerce_trust(value: Any) -> str:
    """Normalize arbitrary trust values to the supported trust labels."""
    if value in ("high", "low"):
        return str(value)
    return "unknown"


def _parse_plan_json(raw: str) -> list[dict]:
    """
    Parse the plan LLM response into a list of task specification dicts.

    Handles two formats:

    **Structured format (new)** — a JSON array of task objects::

        [
          {"question": "...", "produces_artifact": "quarter_df", "consumes_artifacts": []},
          {"question": "...", "produces_artifact": null,          "consumes_artifacts": ["quarter_df"]}
        ]

    **Legacy flat format (backward-compatible)** — a JSON array of strings::

        ["Q1?", "Q2?", "Q3?"]

    In the legacy case, each string is wrapped in a minimal dict with
    ``produces_artifact=None`` and ``consumes_artifacts=[]`` so the rest
    of the loop can treat both formats identically.

    The function is intentionally defensive: if JSON parsing fails entirely,
    it falls back to line-splitting (same as the old ``_parse_json_list``).

    Args:
        raw: Raw LLM response string.

    Returns:
        List of dicts, each with at least ``{"question": str}``.
    """
    # Try to extract the outermost JSON array from the raw string.
    # Use greedy matching so nested [] (e.g. "consumes_artifacts": []) do not
    # cause an early match — we need the full outer array.
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list) and parsed:
                # Structured format: list of dicts with "question" key
                if all(
                    isinstance(item, dict) and "question" in item for item in parsed
                ):
                    return [
                        {
                            "question": str(item["question"]).strip(),
                            "capability": str(item.get("capability") or "").strip()
                            or None,
                            "produces_artifact": item.get("produces_artifact") or None,
                            "consumes_artifacts": list(
                                item.get("consumes_artifacts") or []
                            ),
                            "output_schema": list(item.get("output_schema") or []),
                        }
                        for item in parsed
                        if str(item.get("question", "")).strip()
                    ]
                # Legacy flat format: list of strings
                if all(isinstance(item, str) for item in parsed):
                    return [
                        {
                            "question": q.strip(),
                            "produces_artifact": None,
                            "consumes_artifacts": [],
                        }
                        for q in parsed
                        if q.strip()
                    ]
                # Mixed or partial: extract what we can
                result = []
                for item in parsed:
                    if isinstance(item, dict) and item.get("question"):
                        result.append(
                            {
                                "question": str(item["question"]).strip(),
                                "capability": str(item.get("capability") or "").strip()
                                or None,
                                "produces_artifact": item.get("produces_artifact")
                                or None,
                                "consumes_artifacts": list(
                                    item.get("consumes_artifacts") or []
                                ),
                                "output_schema": list(item.get("output_schema") or []),
                            }
                        )
                    elif isinstance(item, str) and item.strip():
                        result.append(
                            {
                                "question": item.strip(),
                                "capability": None,
                                "produces_artifact": None,
                                "consumes_artifacts": [],
                                "output_schema": [],
                            }
                        )
                if result:
                    return result
        except json.JSONDecodeError:
            pass

    # Fallback: split on numbered lines (same as legacy _parse_json_list)
    lines = []
    for line in raw.splitlines():
        line = re.sub(r"^\d+[\.)\-]\s*", "", line.strip())
        if line and not line.startswith("{") and not line.startswith("["):
            lines.append(
                {
                    "question": line,
                    "capability": None,
                    "produces_artifact": None,
                    "consumes_artifacts": [],
                    "output_schema": [],
                }
            )
    if lines:
        return lines
    return [
        {
            "question": raw.strip(),
            "capability": None,
            "produces_artifact": None,
            "consumes_artifacts": [],
            "output_schema": [],
        }
    ]


def _parse_json_list(raw: str) -> list[str]:
    """
    Parse a JSON array of strings from LLM output.

    Handles extra prose or markdown fences around the JSON.
    Falls back to splitting on newlines if JSON parsing fails.

    Args:
        raw: Raw LLM response string.

    Returns:
        List of question strings (non-empty).
    """
    # Try to extract a JSON array
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, list) and all(isinstance(q, str) for q in result):
                return [q.strip() for q in result if q.strip()]
        except json.JSONDecodeError:
            pass

    # Fallback: numbered lines
    lines = []
    for line in raw.splitlines():
        line = re.sub(r"^\d+[\.)\]\s*", "", line.strip())
        if line:
            lines.append(line)
    return lines or [raw.strip()]


def _parse_critique_json(raw: str) -> dict[str, Any]:
    """
    Parse the critique JSON response defensively.

    Returns a safe default (revise_code) if parsing fails so the loop
    never silently accepts a result due to a malformed critique response.

    Args:
        raw: Raw LLM response string expected to contain a JSON object.

    Returns:
        Dict with ``trust``, ``reason``, ``action`` keys.
    """
    _safe_default: dict[str, Any] = {
        "trust": "low",
        "reason": f"Critique response could not be parsed: {raw[:200]}",
        "action": "revise_code",
    }

    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()

    # Extract first JSON object
    match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if not match:
        return _safe_default

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return _safe_default

    # Validate required fields and allowed values
    trust = parsed.get("trust", "low")
    action = parsed.get("action", "revise_code")
    reason = parsed.get("reason", "")

    if trust not in ("high", "low"):
        trust = "low"
    if action not in ("accept", "revise_code", "revise_plan"):
        action = "revise_code"

    return {"trust": trust, "reason": str(reason), "action": action}


def _strip_markdown_fences(code: str) -> str:
    """
    Remove markdown code fences that LLMs frequently wrap output in.

    Handles all variants::

        ```python\ncode\n```
        ```\ncode\n```
        ` ` `python\ncode\n` ` `   (no spaces in real output)

    This is applied as a post-processing step after every CODE-GEN call
    because prompt instructions alone are not reliable enough to prevent
    models from adding fences, especially on the first attempt.

    Args:
        code: Raw LLM response string.

    Returns:
        Code string with leading/trailing markdown fences removed.
    """
    stripped = code.strip()
    # Remove leading ```python or ``` fence
    stripped = re.sub(r"^```(?:python|py)?\s*\n?", "", stripped)
    # Remove trailing ``` fence
    stripped = re.sub(r"\n?```\s*$", "", stripped)
    return stripped.strip()


def _log(state: AgentRun, entry: LogEntry) -> None:
    """Write a log entry, catching I/O errors so they don't crash the run."""
    try:
        write_log_entry(entry, state.log_dir)
    except OSError as exc:
        logger.error("Failed to write log entry for run_id=%s: %s", state.run_id, exc)


def _print_step_header(step_idx: int, total: int, sub_question: str) -> None:
    """Print a visible step header for console output."""
    print(f"┌─ Step {step_idx + 1}/{total} {'─' * 50}")
    print(f"│  {sub_question}")
    print(f"└{'─' * 58}")


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
    print(f"\nRun JSON exported: {json_path}")
