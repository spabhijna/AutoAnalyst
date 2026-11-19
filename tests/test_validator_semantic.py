"""D6: the semantic contract is generic: no checks keyed on task IDs / plan order."""

from __future__ import annotations

import pandas as pd
import pytest

from agent.state import TaskNode
from agent.validator import CapabilityValidator

V = CapabilityValidator()


def _task(task_id="step_2", produces="out", schema=None) -> TaskNode:
    return TaskNode(
        task_id=task_id,
        question="q",
        produces_artifact=produces,
        output_schema=schema or [],
    )


@pytest.mark.parametrize("task_id", ["step_2", "step_4", "step_5", "step_6", "x", "step_99"])
def test_any_task_id_accepts_a_valid_dataframe(task_id) -> None:
    df = pd.DataFrame({"region": ["N"], "total_revenue": [1.0]})
    assert V.validate_semantic_contract(_task(task_id), df).success


def test_no_artifact_declared_means_nothing_to_check() -> None:
    assert V.validate_semantic_contract(_task(produces=None), None).success
    assert V.validate_semantic_contract(_task(produces=None), "True").success


def test_declared_artifact_must_be_a_dataframe() -> None:
    result = V.validate_semantic_contract(_task(), 3.14)
    assert not result.success
    assert "no DataFrame artifact" in result.reasons[0]
    assert not V.validate_semantic_contract(_task(), None).success


def test_declared_artifact_must_be_non_empty() -> None:
    result = V.validate_semantic_contract(_task(), pd.DataFrame({"a": []}))
    assert not result.success
    assert "empty" in result.reasons[0]


def test_columns_must_match_declared_schema() -> None:
    task = _task(schema=["region", "total_revenue"])
    good = pd.DataFrame({"region": ["N"], "total_revenue": [1.0]})
    bad = pd.DataFrame({"region": ["N"], "revenue": [1.0]})
    assert V.validate_semantic_contract(task, good).success
    result = V.validate_semantic_contract(task, bad)
    assert not result.success
    assert "output_schema" in result.reasons[0]


@pytest.mark.parametrize("placeholder", [["same_as_input"], ["<scalar>"]])
def test_schema_placeholders_are_skipped(placeholder) -> None:
    df = pd.DataFrame({"anything": [1]})
    assert V.validate_semantic_contract(_task(schema=placeholder), df).success


def test_column_order_is_ignored_but_missing_and_extra_are_reported() -> None:
    task = _task(schema=["region", "avg_units"])
    reordered = pd.DataFrame({"avg_units": [1.0], "region": ["N"]})
    assert V.validate_semantic_contract(task, reordered).success

    wrong = pd.DataFrame({"region": ["N"], "units_sold": [1.0]})
    reason = V.validate_semantic_contract(task, wrong).reasons[0]
    assert "missing: ['avg_units']" in reason
    assert "extra: ['units_sold']" in reason
