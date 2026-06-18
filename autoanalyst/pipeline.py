"""
Pipeline orchestrator — runs the full PLAN→CODEGEN→EXECUTE→OBSERVE→CRITIQUE loop.

Design decisions:
- Each sub-question gets its own retry loop (up to config.max_retries).
- After CRITIQUE, if the verdict is FAIL and retries remain, the previous
  error/issues are passed back to CODE-GEN so the LLM can fix them.
- Every stage outcome is persisted to the run log immediately (not batched)
  so partial results survive a crash.
- The pipeline returns a summary dict rather than bare strings so callers
  can act on individual question outcomes without re-parsing the log.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoanalyst.config import PipelineConfig
from autoanalyst.logger import LogEntry, write_log_entry
from autoanalyst.stages.codegen import generate_code
from autoanalyst.stages.critic import CritiqueResult, critique
from autoanalyst.stages.executor import execute, format_execution_result
from autoanalyst.stages.observer import observe
from autoanalyst.stages.planner import plan

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class QuestionResult:
    """
    Outcome for a single sub-question after all retry iterations.

    Attributes:
        sub_question: The question that was answered.
        final_answer: The observation text from the last accepted (PASS) or
                      last attempted (exhausted retries) iteration.
        critique:     The final :class:`~autoanalyst.stages.critic.CritiqueResult`.
        iterations:   Number of CODEGEN→EXECUTE→OBSERVE→CRITIQUE cycles attempted.
        passed:       True if the critique passed on the final iteration.
    """

    sub_question: str
    final_answer: str
    critique: CritiqueResult | None
    iterations: int
    passed: bool


@dataclass
class PipelineRunResult:
    """
    Aggregated outcome of a full pipeline run.

    Attributes:
        run_id:          Unique run identifier (also the log file stem).
        goal:            The original high-level goal.
        sub_questions:   All sub-questions produced by the PLAN stage.
        question_results: Per-question outcomes.
        log_path:        Path to the JSONL run log.
    """

    run_id: str
    goal: str
    sub_questions: list[str]
    question_results: list[QuestionResult]
    log_path: Path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_pipeline(
    goal: str,
    dataset_schema: str,
    df: Any,  # pandas.DataFrame — typed as Any to avoid hard pandas import here
    config: PipelineConfig,
    *,
    run_id: str | None = None,
) -> PipelineRunResult:
    """
    Execute the full PLAN→CODEGEN→EXECUTE→OBSERVE→CRITIQUE pipeline.

    The pipeline:
    1. Calls the PLAN stage to decompose *goal* into sub-questions.
    2. For each sub-question, runs the CODEGEN→EXECUTE→OBSERVE→CRITIQUE
       loop up to ``config.max_retries`` times.
    3. Logs every stage result to a JSONL file immediately.
    4. Returns a :class:`PipelineRunResult` summarising all outcomes.

    Args:
        goal:           High-level analytical question from the user.
        dataset_schema: Text description of the dataset (columns, dtypes,
                        sample values) used to ground LLM prompts.
        df:             The loaded pandas DataFrame to analyse.
        config:         Active :class:`~autoanalyst.config.PipelineConfig`.
        run_id:         Optional explicit run ID; auto-generated if omitted.

    Returns:
        :class:`PipelineRunResult` with per-question outcomes and log path.
    """
    run_id = run_id or _new_run_id()
    logger.info("Pipeline run starting: run_id=%s goal=%r", run_id, goal[:80])

    # ------------------------------------------------------------------
    # PLAN — decompose goal into sub-questions
    # ------------------------------------------------------------------
    sub_questions = plan(goal, dataset_schema, config)
    _log(run_id, LogEntry(
        run_id=run_id,
        sub_question=goal,
        stage="plan",
        iteration=0,
        result="\n".join(f"{i+1}. {q}" for i, q in enumerate(sub_questions)),
    ), config)

    # ------------------------------------------------------------------
    # Per-question loop
    # ------------------------------------------------------------------
    question_results: list[QuestionResult] = []

    for sub_question in sub_questions:
        result = _run_question_loop(
            run_id=run_id,
            sub_question=sub_question,
            dataset_schema=dataset_schema,
            df=df,
            config=config,
        )
        question_results.append(result)

    # ------------------------------------------------------------------
    # Determine log path (the first entry's log gives us the path)
    # ------------------------------------------------------------------
    log_path = config.log_dir / f"{run_id}.jsonl"
    logger.info(
        "Pipeline run complete: run_id=%s passed=%d/%d",
        run_id,
        sum(r.passed for r in question_results),
        len(question_results),
    )

    return PipelineRunResult(
        run_id=run_id,
        goal=goal,
        sub_questions=sub_questions,
        question_results=question_results,
        log_path=log_path,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_question_loop(
    run_id: str,
    sub_question: str,
    dataset_schema: str,
    df: Any,
    config: PipelineConfig,
) -> QuestionResult:
    """
    Run the CODEGEN→EXECUTE→OBSERVE→CRITIQUE loop for one sub-question.

    Retries up to ``config.max_retries`` times. On failure, passes the
    critique issues back to CODE-GEN as ``previous_error`` so the model
    can fix its own mistakes.

    Args:
        run_id:         The parent run's identifier.
        sub_question:   The specific question to answer.
        dataset_schema: Dataset schema for prompt grounding.
        df:             The DataFrame to analyse.
        config:         Active pipeline configuration.

    Returns:
        :class:`QuestionResult` with the final answer, critique, and
        iteration count.
    """
    previous_error: str | None = None
    final_observation = ""
    final_critique: CritiqueResult | None = None

    for iteration in range(config.max_retries):
        logger.info(
            "Sub-question loop: sub_question=%r iteration=%d/%d",
            sub_question[:60],
            iteration,
            config.max_retries,
        )

        # ── CODE-GEN ──────────────────────────────────────────────────
        code = generate_code(
            sub_question,
            dataset_schema,
            config,
            previous_error=previous_error,
        )
        _log(run_id, LogEntry(
            run_id=run_id,
            sub_question=sub_question,
            stage="codegen",
            iteration=iteration,
            code=code,
        ), config)

        # ── EXECUTE ───────────────────────────────────────────────────
        sandbox_result = execute(code, local_vars={"df": df}, config=config)
        exec_output = format_execution_result(sandbox_result)
        _log(run_id, LogEntry(
            run_id=run_id,
            sub_question=sub_question,
            stage="execute",
            iteration=iteration,
            code=code,
            result=exec_output,
            extra={"timed_out": sandbox_result.timed_out},
        ), config)

        if not sandbox_result.success:
            # Code crashed — feed the error back to CODE-GEN on next iteration
            previous_error = sandbox_result.error or "Unknown execution error"
            continue

        # ── OBSERVE ───────────────────────────────────────────────────
        observation = observe(sub_question, code, exec_output, config)
        _log(run_id, LogEntry(
            run_id=run_id,
            sub_question=sub_question,
            stage="observe",
            iteration=iteration,
            code=code,
            result=observation,
        ), config)

        # ── CRITIQUE ──────────────────────────────────────────────────
        critique_result = critique(sub_question, code, observation, config)
        final_observation = observation
        final_critique = critique_result
        _log(run_id, LogEntry(
            run_id=run_id,
            sub_question=sub_question,
            stage="critique",
            iteration=iteration,
            code=code,
            result=observation,
            critique=critique_result.feedback,
            passed=critique_result.passed,
            extra={"issues": critique_result.issues},
        ), config)

        if critique_result.passed:
            logger.info(
                "Sub-question PASSED critique on iteration %d: %r",
                iteration,
                sub_question[:60],
            )
            return QuestionResult(
                sub_question=sub_question,
                final_answer=final_observation,
                critique=final_critique,
                iterations=iteration + 1,
                passed=True,
            )

        # Build error context for the next CODE-GEN iteration
        issues_text = "\n".join(f"- {i}" for i in critique_result.issues)
        previous_error = (
            f"Critique FAILED. Issues identified:\n{issues_text}\n\n"
            f"Critique feedback:\n{critique_result.feedback}"
        )

    # Retries exhausted
    logger.warning(
        "Sub-question exhausted %d retries without passing critique: %r",
        config.max_retries,
        sub_question[:60],
    )
    return QuestionResult(
        sub_question=sub_question,
        final_answer=final_observation,
        critique=final_critique,
        iterations=config.max_retries,
        passed=False,
    )


def _log(run_id: str, entry: LogEntry, config: PipelineConfig) -> None:
    """
    Write *entry* to the run log, catching and logging any I/O errors.

    Args:
        run_id: Run identifier (for error messages).
        entry:  The :class:`~autoanalyst.logger.LogEntry` to persist.
        config: Config providing the log directory.
    """
    try:
        write_log_entry(entry, config.log_dir)
    except OSError as exc:
        logger.error("Failed to write log entry for run_id=%s: %s", run_id, exc)


def _new_run_id() -> str:
    """
    Generate a unique run ID combining a short UUID and ISO timestamp.

    Returns:
        String of the form ``"run_<8-char-hex>"``.
    """
    return f"run_{uuid.uuid4().hex[:8]}"
