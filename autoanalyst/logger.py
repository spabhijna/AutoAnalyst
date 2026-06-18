"""
Structured JSON Lines run-log writer.

Every agent cycle emits one :class:`LogEntry` which is serialized to a
``.jsonl`` file in the configured log directory. The log is the project's
primary deliverable — it's what the blog post analysis is based on.

File naming: ``runs/<run_id>.jsonl``
Each line:   one JSON object (a :class:`LogEntry` as a dict).
"""

from __future__ import annotations

import json
import logging
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
    A single record in the pipeline run log.

    One entry is written per (sub_question × iteration) cycle.
    All fields are populated before :func:`write_log_entry` is called;
    fields not yet available at write time should be left as ``None``
    and backfilled in a subsequent entry.

    Attributes:
        run_id:       Unique identifier for the entire pipeline run (UUID or
                      ISO timestamp string).
        sub_question: The analytical question being answered in this cycle.
        stage:        Which stage produced this entry
                      (``"plan"``, ``"codegen"``, ``"execute"``,
                       ``"observe"``, ``"critique"``).
        iteration:    Retry index, 0-based.
        code:         Agent-generated Python code (if applicable).
        result:       Execution output or observation text.
        critique:     Critique LLM's assessment of the result.
        passed:       Whether the critique deemed the result acceptable.
        timestamp:    ISO 8601 UTC timestamp (set automatically).
        extra:        Optional dict for any stage-specific metadata.
    """

    run_id: str
    sub_question: str
    stage: str
    iteration: int
    code: str | None = None
    result: str | None = None
    critique: str | None = None
    passed: bool | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_log_entry(entry: LogEntry, log_dir: Path) -> Path:
    """
    Append *entry* as a single JSON line to the run log file.

    The log file is ``<log_dir>/<run_id>.jsonl``. It is created if it
    does not already exist; subsequent entries for the same run are
    appended so the file always represents the full run history.

    Args:
        entry:   The :class:`LogEntry` to persist.
        log_dir: Directory in which run logs are stored.

    Returns:
        The :class:`~pathlib.Path` to the log file that was written to.

    Raises:
        OSError: If the log file cannot be opened or written to.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{entry.run_id}.jsonl"

    row = asdict(entry)
    line = json.dumps(row, default=str)  # default=str handles non-serializable extras

    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")

    logger.debug("Wrote log entry run_id=%s stage=%s iter=%d", entry.run_id, entry.stage, entry.iteration)
    return log_path


def read_run_log(run_id: str, log_dir: Path) -> list[LogEntry]:
    """
    Read all :class:`LogEntry` records for *run_id* from *log_dir*.

    Useful for post-run analysis and testing.

    Args:
        run_id:  The run identifier to look up.
        log_dir: Directory containing run log files.

    Returns:
        List of :class:`LogEntry` instances in write order.

    Raises:
        FileNotFoundError: If the log file does not exist.
    """
    log_path = log_dir / f"{run_id}.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"No run log found for run_id={run_id!r} in {log_dir}")

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
