"""
Deterministic dataframe executors.

This module provides small, reusable execution primitives for common
dataframe operations so the loop can bypass LLM code generation when a task
matches a known analytical shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from agent.state import TaskNode


@dataclass
class DeterministicExecutionResult:
    success: bool
    result: str | None
    stdout: str
    error: str | None
    error_type: str | None
    error_category: str
    artifact: pd.DataFrame | None
    artifact_key: str | None

    def to_sandbox_result(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "result": self.result,
            "stdout": self.stdout,
            "error": self.error,
            "error_type": self.error_type,
            "error_category": self.error_category,
            "artifact": self.artifact,
            "artifact_key": self.artifact_key,
        }


def _last_quarter_window(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if "date" not in df.columns:
        return None
    date_series = pd.to_datetime(df["date"], errors="coerce").dropna()
    if date_series.empty:
        return None
    end_ts = date_series.max()
    start_ts = (end_ts - pd.DateOffset(months=2)).replace(day=1)
    return start_ts, end_ts


def _filter_last_quarter(df: pd.DataFrame) -> pd.DataFrame:
    window = _last_quarter_window(df)
    if window is None:
        return df.copy(deep=False)
    start_ts, end_ts = window
    date_series = pd.to_datetime(df["date"], errors="coerce")
    mask = (date_series >= start_ts) & (date_series <= end_ts)
    return df.loc[mask].copy(deep=False)


def _region_revenue(df: pd.DataFrame) -> pd.DataFrame:
    if "region" not in df.columns or "revenue" not in df.columns:
        return pd.DataFrame()
    grouped = (
        df.groupby("region", as_index=False)["revenue"]
        .sum()
        .rename(columns={"revenue": "total_revenue"})
        .sort_values("total_revenue", ascending=False)
        .reset_index(drop=True)
    )
    return grouped


def try_execute(
    task: TaskNode, df: pd.DataFrame, artifacts: dict[str, pd.DataFrame]
) -> DeterministicExecutionResult | None:
    """Return a deterministic execution result when a task matches a known pattern."""
    q = task.question.lower()

    if task.produces_artifact == "quarter_df" and "last quarter" in q:
        quarter_df = _filter_last_quarter(df)
        return DeterministicExecutionResult(
            success=True,
            result=f"quarter_df: {len(quarter_df):,} rows",
            stdout="",
            error=None,
            error_type=None,
            error_category="none",
            artifact=quarter_df,
            artifact_key=task.produces_artifact,
        )

    if task.produces_artifact == "region_revenue" and "revenue" in q and "region" in q:
        base = artifacts.get("quarter_df")
        if base is None:
            base = _filter_last_quarter(df)
        grouped = _region_revenue(base)
        if grouped.empty:
            return None
        return DeterministicExecutionResult(
            success=True,
            result=grouped.head().to_string(index=False),
            stdout="",
            error=None,
            error_type=None,
            error_category="none",
            artifact=grouped,
            artifact_key=task.produces_artifact,
        )

    return None
