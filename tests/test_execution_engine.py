"""
Tests for the capability-driven execution engine.
"""

from __future__ import annotations

import pandas as pd

from agent.artifact_store import ArtifactStore
from agent.capabilities import Capability, CapabilityRegistry
from agent.execution_engine import ExecutionEngine
from agent.state import TaskNode


def test_engine_executes_registered_capability_without_registering() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "region": ["North", "South", "North"],
            "revenue": [10.0, 20.0, 30.0],
            "product_category": ["A", "B", "A"],
        }
    )
    engine = ExecutionEngine()
    store = ArtifactStore()
    task = TaskNode(
        task_id="step_1",
        question="Filter df to the last quarter and return quarter_df",
        capability="FilterByDateRange",
        produces_artifact="quarter_df",
    )

    result = engine.execute(task, df, store)

    assert result is not None
    assert result["success"] is True
    assert isinstance(result["artifact"], pd.DataFrame)
    assert result["artifact_key"] == "quarter_df"
    # The engine only executes and validates; the loop's _accept() registers.
    assert store.get_artifact("quarter_df") is None


def test_engine_reuses_region_artifact() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "region": ["North", "South", "North"],
            "revenue": [10.0, 20.0, 30.0],
            "product_category": ["A", "B", "A"],
        }
    )
    engine = ExecutionEngine()
    store = ArtifactStore()
    store.register(
        "quarter_df",
        df,
        producing_task_id="step_0",
        confidence="high",
        provenance=["seed"],
    )
    task = TaskNode(
        task_id="mock_step",
        question="What is the total revenue by region in the last quarter?",
        capability="GroupRevenueByRegion",
        produces_artifact="region_revenue",
        consumes_artifacts=["quarter_df"],
    )

    result = engine.execute(task, df, store)

    assert result is not None
    assert result["success"] is True
    assert list(result["artifact"].columns) == ["region", "total_revenue"]
    assert store.get_artifact("region_revenue") is None  # registered by _accept()


def test_engine_rejects_missing_dependency_for_capability() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "region": ["North"],
            "revenue": [10.0],
        }
    )
    engine = ExecutionEngine()
    store = ArtifactStore()
    task = TaskNode(
        task_id="step_3",
        question="Which region had highest revenue?",
        capability="FindMaximumRevenueRegion",
        produces_artifact="top_region",
    )

    result = engine.execute(task, df, store)

    assert result is None


def test_engine_rejects_output_schema_mismatch() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "region": ["North", "South"],
            "revenue": [10.0, 20.0],
            "product_category": ["A", "B"],
        }
    )
    registry = CapabilityRegistry()
    engine = ExecutionEngine(registry=registry)
    store = ArtifactStore()
    store.register(
        "quarter_df",
        df,
        producing_task_id="seed",
        confidence="high",
        provenance=["seed"],
    )

    registry.register(
        Capability(
            capability_id="BadCapability",
            description="Returns the wrong output schema.",
            required_artifacts=("quarter_df",),
            produced_artifacts=("region_revenue",),
            input_schema=("region", "revenue"),
            output_schema=("region", "total_revenue"),
            executor=lambda context: type(
                "BadResult",
                (),
                {
                    "success": True,
                    "stdout": "",
                    "error": None,
                    "error_type": None,
                    "error_category": "none",
                    "artifact": pd.DataFrame({"wrong": [1]}),
                    "artifact_key": "region_revenue",
                    "to_sandbox_result": lambda self: {
                        "success": True,
                        "result": "bad",
                        "stdout": "",
                        "error": None,
                        "error_type": None,
                        "error_category": "none",
                        "artifact": self.artifact,
                        "artifact_key": self.artifact_key,
                    },
                },
            )(),
        )
    )
    task = TaskNode(
        task_id="mock_step_2",
        question="What is the total revenue by region in the last quarter?",
        capability="BadCapability",
        produces_artifact="region_revenue",
        consumes_artifacts=["quarter_df"],
    )

    result = engine.execute(task, df, store)

    assert result is None
