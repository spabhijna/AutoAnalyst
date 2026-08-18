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
        task_id="step_1", question="Filter df to the last quarter and return quarter_df"
    )

    capability = registry.resolve(task)

    assert capability is not None
    assert capability.capability_id == "FilterByDateRange"
