"""
Tests for the capability registry.
"""

from __future__ import annotations

import pandas as pd

from agent.capabilities import build_default_capability_registry
from agent.state import TaskNode


def test_registry_exposes_core_capabilities() -> None:
    registry = build_default_capability_registry()
    ids = {capability.capability_id for capability in registry.list()}

    assert {
        "FilterByDateRange",
        "GroupRevenueByRegion",
        "FindMaximumRevenueRegion",
        "MonthlyRevenueTrend",
        "RevenueShare",
        "CategoryAggregation",
        "AverageUnitPrice",
        "TopSellingProductCategory",
    }.issubset(ids)


def test_registry_resolves_task_by_explicit_capability() -> None:
    registry = build_default_capability_registry()
    task = TaskNode(
        task_id="step_1",
        question="What is the total revenue by region?",
        capability="GroupRevenueByRegion",
    )

    capability = registry.resolve(task)

    assert capability is not None
    assert capability.capability_id == "GroupRevenueByRegion"


def test_registry_resolves_task_by_question_terms() -> None:
    registry = build_default_capability_registry()
    task = TaskNode(
        task_id="step_1",
        question="Filter df to the last quarter and return quarter_df",
        produces_artifact="quarter_df",
    )

    capability = registry.resolve(task)

    assert capability is not None
    assert capability.capability_id == "FilterByDateRange"


def test_execute_average_unit_price() -> None:
    from agent.capabilities import CapabilityExecutionContext, execute_average_unit_price
    from agent.state import TaskNode

    df = pd.DataFrame(
        {
            "product_category": ["A", "A", "B"],
            "unit_price": [10.0, 20.0, 25.0],
        }
    )
    context = CapabilityExecutionContext(
        task=TaskNode(task_id="t1", question="avg price", produces_artifact="avg_prices"),
        df=df,
        artifacts={"quarter_df": df},
    )
    result = execute_average_unit_price(context)
    assert result.success
    assert result.artifact is not None
    assert result.artifact_key == "avg_prices"
    assert len(result.artifact) == 2
    assert result.artifact.iloc[0]["product_category"] == "B"  # sorted desc
    assert result.artifact.iloc[0]["average_price"] == 25.0


def test_execute_top_selling_product_category() -> None:
    from agent.capabilities import CapabilityExecutionContext, execute_top_selling_product_category
    from agent.state import TaskNode

    df = pd.DataFrame(
        {
            "product_category": ["A", "A", "B"],
            "units_sold": [2, 3, 10],
        }
    )
    context = CapabilityExecutionContext(
        task=TaskNode(task_id="t1", question="top selling", produces_artifact="top_cat"),
        df=df,
        artifacts={"quarter_df": df},
    )
    result = execute_top_selling_product_category(context)
    assert result.success
    assert result.artifact is not None
    assert result.artifact_key == "top_cat"
    assert len(result.artifact) == 2
    assert result.artifact.iloc[0]["product_category"] == "B"
    assert result.artifact.iloc[0]["total_units_sold"] == 10


def test_deterministic_execution_result_to_sandbox_result() -> None:
    import pandas as pd

    from agent.capabilities import DeterministicExecutionResult

    frame = pd.DataFrame({"a": [1]})
    result = DeterministicExecutionResult(
        success=True,
        result="ok",
        stdout="",
        error=None,
        error_type=None,
        error_category="none",
        artifact=frame,
        artifact_key="k",
    )
    payload = result.to_sandbox_result()
    assert payload["success"] is True
    assert payload["artifact_key"] == "k"
    assert payload["artifact"] is frame
    assert payload["error_category"] == "none"
