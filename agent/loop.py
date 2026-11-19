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
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from agent import config, reporting
from agent.artifact_store import ArtifactStore
from agent.dry_run import dry_run_llm
from agent.errors import PipelineAbortedError
from agent.environment import EnvironmentManifest
from agent.execution_engine import ExecutionEngine
from agent.execution_graph import ExecutionGraph
from agent.findings import (
    build_working_memory_prompt,
    coerce_trust,
    finding_from_attempt,
)
from agent.llm_client import call_llm
from agent.logger import LogEntry, new_run_id
from agent.planning import apply_replan, do_plan, do_replan, task_order
from agent.run_export import finalize_run, log_entry
from agent.stages import (
    do_summary,
    generate_code,
    observe_and_critique,
    record_attempt,
    record_constraints,
)
from agent.sandbox import run_sandboxed
from agent.scheduling import (
    ensure_dependencies_available,
    mark_failed_and_skipped,
    next_ready_task,
)
from agent.schema import build_schema_summary
from agent.state import (
    AttemptRecord,
    Finding,
    TaskMemory,
    TaskNode,
    WorkingMemory,
)
from agent.validator import CapabilityValidator

logger = logging.getLogger(__name__)


MAX_PLAN_REVISIONS: int = 2


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


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
    # LLM callable for this run; set by run() (counting wrapper), None until then.
    llm: Callable[..., str] | None = None
    environment_manifest: EnvironmentManifest = field(
        default_factory=EnvironmentManifest
    )
    task_memories: dict[str, TaskMemory] = field(default_factory=dict)
    artifact_store: ArtifactStore = field(default_factory=ArtifactStore)
    execution_engine: ExecutionEngine = field(default_factory=ExecutionEngine)
    execution_graph: ExecutionGraph = field(default_factory=ExecutionGraph)


def _counting_llm(state: AgentRun, llm: Callable[..., str]) -> Callable[..., str]:
    """Wrap *llm* so every call increments ``state.llm_call_count``."""

    def _call(prompt: str, role: str, *args: Any, **kwargs: Any) -> str:
        state.llm_call_count += 1
        return llm(prompt, role, *args, **kwargs)

    return _call


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(
    goal: str,
    df: pd.DataFrame,
    log_dir: str = config.LOG_DIR,
    max_retries_per_step: int = config.MAX_PIPELINE_RETRIES,
    dry_run: bool = False,
    llm: Callable[..., str] | None = None,
    dataset_label: str = "clean",
) -> AgentRun:
    """
    Execute the full five-stage loop for *goal* against *df*.

    At the end of every run (including aborted runs), writes a structured
    ``runs/<run_id>.json`` summary via :func:`agent.run_export.finalize_run`.

    Args:
        goal:               High-level analytical question.
        df:                 The pandas DataFrame to analyse.
        log_dir:            Directory for JSONL and JSON run logs.
        max_retries_per_step: Hard cap on CODE-GEN retries per sub-question.
        dry_run:            When True, uses canned LLM responses so the
                            pipeline runs without API keys.  Defaults to False.
        dataset_label:      Label recorded in the run JSON (e.g. "clean" or
                            "corrupted"); " (aborted)" is appended on abort.
        llm:                Callable ``(prompt, role) -> str`` used for every LLM
                            call. Defaults to :func:`agent.llm_client.call_llm`
                            (or :func:`agent.dry_run.dry_run_llm` when
                            ``dry_run`` is True). An explicit *llm* wins.

    Returns:
        Completed :class:`AgentRun` with full history and summary populated.
    """
    t_start = time.monotonic()
    state = _init_run(goal, df, log_dir, max_retries_per_step, dry_run, llm)

    do_plan(state, df, state.llm, log_entry)  # sets state.plan / state.tasks

    try:
        _execute_plan(state, df)
    except PipelineAbortedError as exc:
        _handle_abort(state, exc)
        finalize_run(state, t_start, dataset_label)
        return state

    state.summary = do_summary(state)
    logger.info("=== AgentRun complete: run_id=%s ===", state.run_id)
    reporting.final_summary(state.summary)
    finalize_run(state, t_start, dataset_label)
    return state


def _init_run(
    goal: str,
    df: pd.DataFrame,
    log_dir: str,
    max_retries_per_step: int,
    dry_run: bool,
    llm: Callable[..., str] | None,
) -> AgentRun:
    """Create the run state and select the LLM callable (no monkeypatching)."""
    state = AgentRun(
        goal=goal,
        schema_summary=build_schema_summary(df),
        log_dir=log_dir,
        max_retries_per_step=max_retries_per_step,
    )
    logger.info("=== AgentRun start: run_id=%s ===", state.run_id)
    reporting.run_banner(state.run_id, goal)

    if llm is None:
        llm = dry_run_llm if dry_run else call_llm
    if dry_run:
        logger.info("Dry-run mode enabled — using canned LLM responses")
        reporting.dry_run_notice()
    state.llm = _counting_llm(state, llm)
    return state


def _execute_plan(state: AgentRun, df: pd.DataFrame) -> None:
    """Run ready tasks in graph order until all complete (or the run aborts)."""
    while not state.execution_graph.all_completed():
        task = next_ready_task(state)
        state.current_step = state.execution_graph.task_index(task.task_id)
        reporting.step_header(state.current_step, len(state.plan), task.question)
        ensure_dependencies_available(state, task)
        task.status = "running"

        if _run_question(state, task, df):
            _replan(state, df, task)
        else:
            state.current_step += 1


def _replan(state: AgentRun, df: pd.DataFrame, task: TaskNode) -> None:
    """Handle a revise_plan request: abort at the cap, otherwise rebuild the tail."""
    if state.plan_revisions >= MAX_PLAN_REVISIONS:
        # Never advance past a task that did not complete: abort.
        raise PipelineAbortedError(
            sub_question=task.question,
            last_reason=(
                f"Plan revision cap ({MAX_PLAN_REVISIONS}) reached and the critique "
                f"still requested a plan revision: {task.failure_reason}"
            ),
            run_id=state.run_id,
            kind="plan revision cap reached",
        )
    new_specs = do_replan(state, df, state.llm, build_working_memory_prompt(state))
    new_tasks = apply_replan(state, new_specs)
    state.plan_revisions += 1
    logger.info(
        "Plan revised (revision %d/%d): %d new tasks",
        state.plan_revisions,
        MAX_PLAN_REVISIONS,
        len(new_tasks),
    )


def _handle_abort(state: AgentRun, exc: PipelineAbortedError) -> None:
    """Record an aborted run: statuses, log entry and console banner."""
    state.aborted = True
    state.abort_reason = str(exc)
    mark_failed_and_skipped(state, exc.sub_question, exc.last_reason)
    logger.error("Pipeline aborted: %s", exc)
    log_entry(
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
        exc.kind,
        state.current_step + 1,
        len(state.plan),
        state.max_retries_per_step,
        exc.sub_question,
        exc.last_reason,
        state.run_id,
    )


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------


def _run_question(
    state: AgentRun,
    task: TaskNode,
    df: pd.DataFrame,
) -> bool:
    """
    Run one sub-question: deterministic capability first, CODE-GEN loop otherwise.

    Returns:
        True if critique returned "revise_plan", False otherwise.
    """
    if task.task_id not in state.task_memories:
        state.task_memories[task.task_id] = TaskMemory(task_id=task.task_id)

    if _try_deterministic(state, task, df):
        return False
    return _run_codegen_loop(state, task, df)


def _accept(
    state: AgentRun,
    task: TaskNode,
    attempt: AttemptRecord,
    sandbox_result: dict[str, Any],
    trust: str,
    reason: str,
    attempt_count: int,
    had_execution_errors: bool,
    artifact_key: str | None = None,
    provenance: list[str] | None = None,
) -> Finding:
    """
    The single accept path for both the deterministic and the CODE-GEN branch.

    Sets the task status, promotes the attempt into working memory, refreshes
    ``final_results`` and registers the produced artifact.

    Artifact ownership: ``_accept`` is the ONLY place a task's artifact is
    registered in ``state.artifact_store``. ``ExecutionEngine.execute`` merely
    executes and validates, and never writes to the store, so an artifact is
    registered exactly once (bug D2).

    Args:
        artifact_key: Store key override (the deterministic executor's default
            key); falls back to ``task.produces_artifact``.
        provenance:   Optional provenance list recorded on the artifact.
    """
    task.status = "accepted"
    finding = finding_from_attempt(
        attempt=attempt,
        task=task,
        attempt_count=attempt_count,
        had_execution_errors=had_execution_errors,
        reason=reason,
    )
    state.working_memory.add_finding(finding)
    state.final_results = state.working_memory.accepted_findings(
        task_order(state.tasks)
    )

    key = task.produces_artifact or artifact_key
    extracted_df = sandbox_result.get("artifact") if key else None
    if key and isinstance(extracted_df, pd.DataFrame):
        registered = state.artifact_store.register(
            key=key,
            df=extracted_df,
            producing_task_id=task.task_id,
            consumers=state.execution_graph.consumers_for(task.produces_artifact)
            if task.produces_artifact
            else [],
            confidence=trust,
            provenance=provenance,
        )
        if registered:
            finding.produced_artifact_name = key
            logger.info(
                "Artifact %r registered from %s (%d rows)",
                key,
                task.task_id,
                len(extracted_df),
            )
        else:
            logger.warning(
                "Task %s declared produces_artifact=%r but registration failed",
                task.task_id,
                key,
            )
    return finding


def _try_deterministic(state: AgentRun, task: TaskNode, df: pd.DataFrame) -> bool:
    """
    Try to resolve *task* with a registered deterministic capability.

    Returns True when the task was executed, validated and accepted (the engine
    does not register artifacts; :func:`_accept` does), False when the caller
    should fall back to the CODE-GEN loop.
    """
    sandbox_result = state.execution_engine.execute(task, df, state.artifact_store)
    if sandbox_result is None:
        return False

    reporting.deterministic_execute_start()
    reporting.deterministic_execute_ok()
    sub_question = task.question
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
        trust=coerce_trust(trust),
        error_category="none",
    )
    state.debug_history.append(attempt)
    log_entry(
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
    _accept(
        state,
        task,
        attempt,
        sandbox_result,
        trust,
        reason,
        attempt_count=1,
        had_execution_errors=False,
        artifact_key=sandbox_result.get("artifact_key"),
        provenance=[sandbox_result["capability_id"], task.task_id],
    )
    return True


def _run_codegen_loop(state: AgentRun, task: TaskNode, df: pd.DataFrame) -> bool:
    """
    Run the CODE-GEN → EXECUTE → CRITIQUE loop for one sub-question.

    Before CODE-GEN, builds an artifact namespace from the ArtifactStore for
    all keys listed in ``task.consumes_artifacts``. This namespace is injected
    into the sandbox so generated code can read pre-validated DataFrames without
    re-deriving them.

    After CRITIQUE accepts a result, :func:`_accept` registers the task's
    declared artifact in ``state.artifact_store`` for downstream steps.

    Returns:
        True if critique returned "revise_plan", False otherwise.
    """
    previous_code: str | None = None
    previous_reason: str | None = None
    had_execution_errors = False
    task_memory = state.task_memories[task.task_id]
    artifact_namespace = _artifact_namespace(state, task)

    for iteration in range(state.max_retries_per_step):
        code = generate_code(
            state, task, task_memory, iteration, previous_code, previous_reason
        )
        sandbox_result, error_category = _execute_and_validate(
            state, task, df, code, artifact_namespace
        )
        had_execution_errors = had_execution_errors or not sandbox_result["success"]
        critique_raw, trust, action, reason, error_category = observe_and_critique(
            state, task, df, code, sandbox_result, error_category
        )
        record_constraints(state, task_memory, error_category, reason)
        attempt = record_attempt(
            state,
            task,
            iteration,
            code,
            sandbox_result,
            critique_raw,
            trust,
            action,
            error_category,
        )

        if action == "accept":
            _accept(
                state,
                task,
                attempt,
                sandbox_result,
                trust,
                reason,
                attempt_count=iteration + 1,
                had_execution_errors=had_execution_errors,
            )
            return False  # move to next sub-question
        if action == "revise_plan":
            task.status = "failed"
            task.failure_reason = reason
            return True  # signal caller to replan
        # action == "revise_code" → continue loop
        previous_code = code
        previous_reason = reason

    _abort_retries_exhausted(state, task, previous_reason)
    return False  # unreachable: the call above always raises


def _artifact_namespace(state: AgentRun, task: TaskNode) -> dict[str, pd.DataFrame]:
    """Artifacts injected into the sandbox: the task's declared inputs, else all."""
    if task.consumes_artifacts:
        return state.artifact_store.as_namespace_for(task.consumes_artifacts)
    return state.artifact_store.as_namespace()


def _execute_and_validate(
    state: AgentRun,
    task: TaskNode,
    df: pd.DataFrame,
    code: str,
    artifact_namespace: dict[str, pd.DataFrame],
) -> tuple[dict[str, Any], str]:
    """Run *code* in the sandbox, then apply the task's semantic contract.

    Returns the sandbox result (marked failed on a contract violation) and its
    error category.
    """
    reporting.execute_start()
    sandbox_result = run_sandboxed(
        code,
        df,
        artifacts=artifact_namespace,
        extract_artifact=task.produces_artifact,
        timeout_seconds=config.SANDBOX_TIMEOUT_SECONDS,
    )
    status = (
        "✓ OK"
        if sandbox_result["success"]
        else f"✗ {sandbox_result.get('error_type', 'ERROR')}"
    )
    reporting.execute_status(status, sandbox_result.get("stdout"))

    error_category = sandbox_result.get("error_category", "none")
    if sandbox_result["success"]:
        semantic = CapabilityValidator().validate_semantic_contract(
            task, sandbox_result.get("artifact")
        )
        if not semantic.success:
            sandbox_result["success"] = False
            sandbox_result["error"] = "Semantic Validation Failed: " + "; ".join(
                semantic.reasons
            )
            sandbox_result["error_category"] = "logic_error"
            error_category = "logic_error"
            reporting.semantic_error(sandbox_result["error"])
    return sandbox_result, error_category


def _abort_retries_exhausted(
    state: AgentRun, task: TaskNode, last_reason: str | None
) -> None:
    """Abort the pipeline once retries are exhausted.

    Do NOT force-accept: a foundational step that can't produce a trustworthy
    answer would corrupt every downstream step that depends on it.
    """
    logger.error(
        "Max retries (%d) exhausted for sub_question=%r — aborting pipeline",
        state.max_retries_per_step,
        task.question[:60],
    )
    reporting.max_retries_exceeded(state.max_retries_per_step)
    raise PipelineAbortedError(
        sub_question=task.question,
        last_reason=last_reason or "(unknown)",
        run_id=state.run_id,
    )
