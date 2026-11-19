"""
Run-level logging and export: JSONL log entries and the structured run JSON.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from agent import reporting
from agent.config import GROQ_CODEGEN_MODEL, GROQ_CRITIQUE_MODEL, GROQ_DEFAULT_MODEL
from agent.logger import LogEntry, export_run_json, write_log_entry

if TYPE_CHECKING:
    from agent.loop import AgentRun

logger = logging.getLogger(__name__)


def log_entry(state: "AgentRun", entry: LogEntry) -> None:
    """Write a log entry, catching I/O errors so they don't crash the run."""
    try:
        write_log_entry(entry, state.log_dir)
    except OSError as exc:
        logger.error("Failed to write log entry for run_id=%s: %s", state.run_id, exc)


def finalize_run(
    state: "AgentRun", t_start: float, dataset_label: str = "clean"
) -> None:
    """
    Write the structured run JSON summary at the end of a run.

    Called from ``run()`` on both successful completion and abort.
    Aggregates metadata (goal, model versions, LLM call count, wall clock)
    + the full debug history into ``runs/<run_id>.json``.
    """
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
        dataset_label=dataset_label
        if not state.aborted
        else f"{dataset_label} (aborted)",
        model_versions=model_versions,
        debug_entries=debug_entries,
        final_summary=state.summary if not state.aborted else "",
        aborted=state.aborted,
        wall_clock_seconds=wall_clock,
        total_llm_calls=state.llm_call_count,
        log_dir=state.log_dir,
    )
    reporting.run_json_exported(json_path)
