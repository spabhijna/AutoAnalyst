"""
Structured JSONL run-log writer + run-level JSON export.

Every history entry produced by the loop is appended as one JSON line.
At the end of each run, a consolidated run JSON is written with metadata
(goal, dataset info, model versions, wall clock, total LLM calls).

Files:
- ``runs/<run_id>.jsonl`` — streaming JSONL (one LogEntry per line)
- ``runs/<run_id>.json``  — structured run summary (written at run end)

The log is the primary deliverable — it's what the blog post will be built from.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class LogEntry:
    """
    One record in the pipeline run log.

    Written after every loop iteration (one per sub-question × iteration).

    Attributes:
        run_id:         Unique identifier for the full pipeline run.
        sub_question:   The analytical question being answered.
        iteration:      0-based retry index for this sub-question.
        code:           Agent-generated Python code (None for plan/summary steps).
        sandbox_result: Full sandbox output dict (success, result, stdout, error).
        critique:       Raw critique JSON string from the LLM.
        action:         Critique action taken: "accept" | "revise_code" |
                        "revise_plan" | "max_retries_exceeded".
        trust:          "high" | "low" | None (from critique).
        stage:          "plan" | "codegen_execute_critique" | "summary".
        timestamp:      UTC ISO 8601 (auto-set).
        extra:          Optional metadata dict.
    """

    run_id: str
    sub_question: str
    iteration: int
    stage: str
    action: str
    code: str | None = None
    sandbox_result: dict[str, Any] | None = None
    critique: str | None = None
    trust: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API — JSONL (streaming, one entry at a time)
# ---------------------------------------------------------------------------


def new_run_id() -> str:
    """Generate a short unique run identifier."""
    return f"run_{uuid.uuid4().hex[:8]}"


def write_log_entry(entry: LogEntry, log_dir: str = "runs") -> Path:
    """
    Append *entry* as a single JSON line to ``<log_dir>/<run_id>.jsonl``.

    The file is created if it doesn't exist; subsequent entries for the same
    run_id are appended so the file contains the full run history.

    Args:
        entry:   The :class:`LogEntry` to persist.
        log_dir: Directory for run logs (created if missing).

    Returns:
        Path to the log file written.

    Raises:
        OSError: If the file cannot be opened or written.
    """
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    log_path = path / f"{entry.run_id}.jsonl"

    row = asdict(entry)
    line = json.dumps(row, default=str)

    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")

    logger.debug(
        "log entry: run=%s stage=%s sub_question=%.50s iter=%d action=%s",
        entry.run_id,
        entry.stage,
        entry.sub_question,
        entry.iteration,
        entry.action,
    )
    return log_path


def read_run_log(run_id: str, log_dir: str = "runs") -> list[LogEntry]:
    """
    Read all :class:`LogEntry` records for *run_id* from *log_dir*.

    Args:
        run_id:  The run identifier to look up.
        log_dir: Directory containing run log files.

    Returns:
        List of :class:`LogEntry` instances in write order.

    Raises:
        FileNotFoundError: If the log file does not exist.
    """
    log_path = Path(log_dir) / f"{run_id}.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(
            f"No run log found for run_id={run_id!r} in {log_dir!r}"
        )

    entries: list[LogEntry] = []
    with log_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                entries.append(LogEntry(**data))
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("Skipping malformed log line %d: %s", lineno, exc)

    return entries


# ---------------------------------------------------------------------------
# Public API — run-level structured JSON export
# ---------------------------------------------------------------------------


def export_run_json(
    run_id: str,
    goal: str,
    dataset_label: str,
    model_versions: dict[str, str],
    debug_entries: list[dict[str, Any]],
    final_summary: str,
    aborted: bool,
    wall_clock_seconds: float,
    total_llm_calls: int,
    log_dir: str = "runs",
) -> Path:
    """
    Write a structured run summary JSON file at the end of a run.

    Produces ``<log_dir>/<run_id>.json`` with metadata at the top and
    the full debug history as an array.  This is the file the log viewer
    and blog post scripts consume.

    Args:
        run_id:            Unique run identifier.
        goal:              Original analytical goal text.
        dataset_label:     Description of the dataset (e.g. "clean", "corrupted").
        model_versions:    Dict mapping stage → model name used.
        debug_entries:     List of serialized AttemptRecord dicts (all iterations).
        final_summary:     The LLM-generated summary text.
        aborted:           Whether the run aborted before completing all steps.
        wall_clock_seconds: Elapsed wall-clock time for the full run.
        total_llm_calls:   Total number of call_llm invocations during the run.
        log_dir:           Directory for run logs.

    Returns:
        Path to the written JSON file.
    """
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "goal": goal,
            "dataset": dataset_label,
            "model_versions": model_versions,
            "total_iterations": len(debug_entries),
            "total_llm_calls": total_llm_calls,
            "wall_clock_seconds": round(wall_clock_seconds, 2),
            "aborted": aborted,
            "final_summary": final_summary,
        },
        "history": debug_entries,
    }

    json_path = path / f"{run_id}.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    logger.info("Run JSON exported: %s (%d entries)", json_path, len(debug_entries))
    return json_path
