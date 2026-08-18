"""
Tests for deterministic dataframe executors.
"""

from __future__ import annotations

import pandas as pd

from agent.executors import try_execute
from agent.state import TaskNode


def test_try_execute_filters_last_quarter() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "region": ["North", "South", "East"],
            "revenue": [10.0, 20.0, 30.0],
        }
    )
    task = TaskNode(
        task_id="step_1",
        question="Filter df to the last quarter and return quarter_df",
        produces_artifact="quarter_df",
    )

    result = try_execute(task, df, {})

    assert result is not None
    payload = result.to_sandbox_result()
    assert payload["success"] is True
    assert payload["artifact_key"] == "quarter_df"
    assert isinstance(payload["artifact"], pd.DataFrame)
    assert len(payload["artifact"]) == 3


def test_try_execute_uses_existing_quarter_artifact_for_region_revenue() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "region": ["North", "South", "North"],
            "revenue": [10.0, 20.0, 30.0],
        }
    )
    quarter_df = df.copy(deep=False)
    task = TaskNode(
        task_id="step_2",
        question="What is the total revenue by region in the last quarter?",
        produces_artifact="region_revenue",
        consumes_artifacts=["quarter_df"],
    )

    result = try_execute(task, df, {"quarter_df": quarter_df})

    assert result is not None
    payload = result.to_sandbox_result()
    assert payload["success"] is True
    assert payload["artifact_key"] == "region_revenue"
    assert list(payload["artifact"].columns) == ["region", "total_revenue"]
    assert payload["artifact"].iloc[0]["region"] == "North"
