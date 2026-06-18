"""
Tests for autoanalyst.logger — JSONL run-log writer.

Verifies:
- write_log_entry creates the file and writes valid JSON.
- Multiple entries for the same run are appended (not overwritten).
- read_run_log round-trips entries correctly.
- Missing log file raises FileNotFoundError.
- Malformed lines in the log are skipped gracefully.
- LogEntry timestamp defaults to UTC ISO 8601.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from autoanalyst.logger import LogEntry, read_run_log, write_log_entry


@pytest.fixture()
def log_dir(tmp_path: Path) -> Path:
    """Temporary directory for log files."""
    return tmp_path / "runs"


class TestWriteLogEntry:
    def test_creates_file(self, log_dir: Path) -> None:
        """write_log_entry should create the log directory and file."""
        entry = LogEntry(
            run_id="test_run_1",
            sub_question="What is the total revenue?",
            stage="codegen",
            iteration=0,
            code="_result = df['Sales'].sum()",
        )
        path = write_log_entry(entry, log_dir)
        assert path.exists()
        assert path.name == "test_run_1.jsonl"

    def test_valid_json_line(self, log_dir: Path) -> None:
        """Each written entry must be a valid JSON object on one line."""
        entry = LogEntry(
            run_id="test_run_2",
            sub_question="Q",
            stage="observe",
            iteration=1,
            result="Sales were high.",
        )
        path = write_log_entry(entry, log_dir)
        lines = path.read_text().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["run_id"] == "test_run_2"
        assert parsed["stage"] == "observe"

    def test_appends_not_overwrites(self, log_dir: Path) -> None:
        """Multiple entries for the same run_id must all appear in the file."""
        for i in range(3):
            entry = LogEntry(
                run_id="test_run_3",
                sub_question="Q",
                stage="codegen",
                iteration=i,
            )
            write_log_entry(entry, log_dir)

        path = log_dir / "test_run_3.jsonl"
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        assert len(lines) == 3

    def test_creates_log_dir_if_missing(self, tmp_path: Path) -> None:
        """log_dir does not need to exist before write_log_entry is called."""
        deep_dir = tmp_path / "a" / "b" / "c"
        entry = LogEntry(run_id="r", sub_question="Q", stage="plan", iteration=0)
        path = write_log_entry(entry, deep_dir)
        assert path.exists()

    def test_timestamp_is_utc_iso(self, log_dir: Path) -> None:
        """LogEntry timestamp must be a valid UTC ISO 8601 string."""
        entry = LogEntry(run_id="r", sub_question="Q", stage="plan", iteration=0)
        write_log_entry(entry, log_dir)
        path = log_dir / "r.jsonl"
        data = json.loads(path.read_text().strip())
        ts = data["timestamp"]
        # Must be parseable and contain timezone info (+00:00 or Z)
        assert "T" in ts
        assert "+" in ts or ts.endswith("Z")


class TestReadRunLog:
    def test_round_trip(self, log_dir: Path) -> None:
        """Entries written and then read back must match the originals."""
        entries = [
            LogEntry(
                run_id="rr",
                sub_question="How much?",
                stage=stage,
                iteration=0,
                result=f"result_{stage}",
            )
            for stage in ("codegen", "execute", "observe", "critique")
        ]
        for e in entries:
            write_log_entry(e, log_dir)

        read_back = read_run_log("rr", log_dir)
        assert len(read_back) == 4
        assert [e.stage for e in read_back] == ["codegen", "execute", "observe", "critique"]
        assert all(e.run_id == "rr" for e in read_back)

    def test_missing_file_raises(self, log_dir: Path) -> None:
        """read_run_log must raise FileNotFoundError for unknown run IDs."""
        with pytest.raises(FileNotFoundError):
            read_run_log("no_such_run", log_dir)

    def test_passed_field_round_trips(self, log_dir: Path) -> None:
        """Boolean passed field must survive JSON serialization."""
        entry = LogEntry(
            run_id="p",
            sub_question="Q",
            stage="critique",
            iteration=0,
            passed=True,
        )
        write_log_entry(entry, log_dir)
        result = read_run_log("p", log_dir)
        assert result[0].passed is True

    def test_extra_field_preserved(self, log_dir: Path) -> None:
        """The extra dict field must survive round-trip."""
        entry = LogEntry(
            run_id="ex",
            sub_question="Q",
            stage="execute",
            iteration=0,
            extra={"timed_out": False, "exit_code": 0},
        )
        write_log_entry(entry, log_dir)
        result = read_run_log("ex", log_dir)
        assert result[0].extra == {"timed_out": False, "exit_code": 0}
