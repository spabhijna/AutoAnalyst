"""
Tests for agent.evaluator — evaluation against corruption manifest.

Tests cover:
- Clean data (no corruption) → precision=1.0, recall=1.0 (no defects either way)
- Fully corrupted data → all 5 categories detected
- Category-specific matching: duplicates, nulls, negative revenue, zero-units,
  outliers
- EvalReport.summary() formatting
- Pipeline resilience metrics (abort / attempts)
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from agent.evaluator import (
    CategoryResult,
    EvalReport,
    evaluate,
    evaluate_from_corrupt_run,
)
from agent.loop import AgentRun
from agent.observer import ObservationReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def clean_df() -> pd.DataFrame:
    """Clean 10-row DataFrame with no corruption."""
    return pd.DataFrame({
        "transaction_id": [f"A{i:03d}" for i in range(1, 11)],
        "region": ["North", "South", "East", "West"] * 2 + ["North", "South"],
        "product_category": [
            "Electronics", "Clothing", "Food", "Home", "Sports",
            "Beauty", "Electronics", "Clothing", "Food", "Home",
        ],
        "units_sold": [2, 3, 5, 1, 4, 2, 3, 1, 2, 4],
        "unit_price": [99.99, 29.99, 9.99, 149.99, 79.99,
                        49.99, 199.99, 19.99, 7.99, 89.99],
        "revenue": [199.98, 89.97, 49.95, 149.99, 319.96,
                     99.98, 599.97, 19.99, 15.98, 359.96],
    })


@pytest.fixture()
def manifest_corrupted() -> dict[str, Any]:
    """Simulated manifest for a fully-corrupted dataset."""
    return {
        "duplicated_transaction_ids": {
            "A001": [0, 10],
            "A005": [4, 11],
        },
        "null_revenue_indices": [3, 7],
        "impossible_rows": [
            {
                "index": 12,
                "type": "negative_revenue",
                "injected_values": {"revenue": -999.99},
            },
            {
                "index": 13,
                "type": "revenue_outlier_50x",
                "injected_values": {"revenue": 15000.0},
            },
            {
                "index": 14,
                "type": "zero_units_positive_revenue",
                "injected_values": {"units_sold": 0, "revenue": 59.99},
            },
        ],
        "total_defects": 7,
    }


@pytest.fixture()
def corrupted_df(clean_df: pd.DataFrame, manifest_corrupted: dict[str, Any]) -> pd.DataFrame:
    """Corrupted DataFrame matching the manifest fixture."""
    df = clean_df.copy()

    # Add duplicates (simulating the manifest)
    dupe_rows = df.iloc[[0, 4]].copy()
    dupe_rows.index = [10, 11]
    df = pd.concat([df, dupe_rows], ignore_index=False)
    df = df.reset_index(drop=True)

    # Add nulls in revenue
    df.loc[3, "revenue"] = None
    df.loc[7, "revenue"] = None

    # Add impossible rows (beyond the original 10 + 2 duplicates = 12 rows,
    # these become rows 12, 13, 14)
    extra_rows = pd.DataFrame([
        {
            "transaction_id": "CORRUPT_NEG001",
            "region": "North",
            "product_category": "Electronics",
            "units_sold": 5,
            "unit_price": 199.99,
            "revenue": -999.99,
        },
        {
            "transaction_id": "CORRUPT_OUT001",
            "region": "South",
            "product_category": "Food",
            "units_sold": 1,
            "unit_price": 15000.0,
            "revenue": 15000.0,
        },
        {
            "transaction_id": "CORRUPT_LOG001",
            "region": "East",
            "product_category": "Clothing",
            "units_sold": 0,
            "unit_price": 59.99,
            "revenue": 59.99,
        },
    ])
    df = pd.concat([df, extra_rows], ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# CategoryResult tests
# ---------------------------------------------------------------------------


class TestCategoryResult:
    """Tests for CategoryResult property logic."""

    def test_true_positive(self) -> None:
        cat = CategoryResult(name="test", present_in_manifest=True, detected=True)
        assert cat.is_true_positive is True
        assert cat.is_false_positive is False
        assert cat.is_false_negative is False

    def test_false_positive(self) -> None:
        cat = CategoryResult(name="test", present_in_manifest=False, detected=True)
        assert cat.is_true_positive is False
        assert cat.is_false_positive is True
        assert cat.is_false_negative is False

    def test_false_negative(self) -> None:
        cat = CategoryResult(name="test", present_in_manifest=True, detected=False)
        assert cat.is_true_positive is False
        assert cat.is_false_positive is False
        assert cat.is_false_negative is True

    def test_true_negative(self) -> None:
        cat = CategoryResult(name="test", present_in_manifest=False, detected=False)
        assert cat.is_true_positive is False
        assert cat.is_false_positive is False
        assert cat.is_false_negative is False


# ---------------------------------------------------------------------------
# EvalReport tests
# ---------------------------------------------------------------------------


class TestEvalReport:
    """Tests for EvalReport dataclass and summary()."""

    def test_perfect_scores(self) -> None:
        """Perfect precision and recall when all 5 categories match."""
        cats = {
            "a": CategoryResult(name="a", present_in_manifest=True, detected=True, manifest_count=2, observer_count=2),
            "b": CategoryResult(name="b", present_in_manifest=True, detected=True, manifest_count=1, observer_count=1),
            "c": CategoryResult(name="c", present_in_manifest=True, detected=True, manifest_count=3, observer_count=3),
        }
        report = EvalReport(
            categories=cats,
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            true_positives=3,
            false_positives=0,
            false_negatives=0,
            total_defects_in_manifest=6,
            total_defects_detected=6,
        )
        assert report.precision == 1.0
        assert report.recall == 1.0
        assert report.f1_score == 1.0

    def test_no_defects(self) -> None:
        """No defects in manifest and no detection scores default to 1.0."""
        cats = {
            "a": CategoryResult(name="a", present_in_manifest=False, detected=False),
        }
        report = EvalReport(
            categories=cats,
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            true_positives=0,
            false_positives=0,
            false_negatives=0,
        )
        assert report.precision == 1.0
        assert report.recall == 1.0

    def test_summary_contains_key_metrics(self) -> None:
        """Summary string includes precision, recall, F1, and TP/FP/FN."""
        cats = {
            "test_cat": CategoryResult(
                name="test_cat",
                present_in_manifest=True,
                detected=True,
                manifest_count=2,
                observer_count=2,
            ),
        }
        report = EvalReport(
            categories=cats,
            precision=1.0,
            recall=0.5,
            f1_score=0.6667,
            true_positives=1,
            false_positives=0,
            false_negatives=1,
            total_defects_in_manifest=5,
            total_defects_detected=2,
        )
        summary = report.summary()
        assert "precision" in summary.lower()
        assert "recall" in summary.lower()
        assert "f1" in summary.lower()
        assert "true positives" in summary.lower()
        assert "false positives" in summary.lower()
        assert "false negatives" in summary.lower()
        assert "100.00%" in summary  # precision
        assert "50.00%" in summary   # recall
        assert "66.67%" in summary   # F1

    def test_summary_pipeline_not_aborted(self) -> None:
        """Summary shows "Pipeline completed" when not aborted."""
        report = EvalReport()
        summary = report.summary()
        assert "Pipeline completed without abort" in summary
        assert "ABORTED" not in summary

    def test_summary_pipeline_aborted(self) -> None:
        """Summary shows "PIPELINE ABORTED" when aborted."""
        report = EvalReport(
            pipeline_aborted=True,
            abort_reason="Max retries exceeded on step_1",
        )
        summary = report.summary()
        assert "PIPELINE ABORTED" in summary
        assert "Max retries exceeded" in summary


# ---------------------------------------------------------------------------
# Evaluate — clean data
# ---------------------------------------------------------------------------


class TestEvaluateCleanData:
    """evaluate() on clean (uncorrupted) data."""

    def test_clean_data_returns_perfect_scores(self, clean_df: pd.DataFrame) -> None:
        """Clean data with an empty manifest → all scores perfect."""
        manifest: dict[str, Any] = {
            "duplicated_transaction_ids": {},
            "null_revenue_indices": [],
            "impossible_rows": [],
            "total_defects": 0,
        }
        report = evaluate(clean_df, manifest)
        assert report.precision == 1.0
        assert report.recall == 1.0
        assert report.f1_score == 1.0
        assert report.true_positives == 0
        assert report.false_positives == 0
        assert report.false_negatives == 0

    def test_no_categories_flagged(self, clean_df: pd.DataFrame) -> None:
        """No observer signals triggered for clean data."""
        # Build an empty manifest
        manifest: dict[str, Any] = {
            "duplicated_transaction_ids": {},
            "null_revenue_indices": [],
            "impossible_rows": [],
            "total_defects": 0,
        }
        report = evaluate(clean_df, manifest)
        for cat in report.categories.values():
            assert cat.detected is False, f"{cat.name} should not be detected in clean data"
            assert cat.present_in_manifest is False


# ---------------------------------------------------------------------------
# Evaluate — corrupted data
# ---------------------------------------------------------------------------


class TestEvaluateCorruptedData:
    """evaluate() on fully-corrupted data — all 5 categories."""

    def test_all_categories_detected(self, corrupted_df: pd.DataFrame, manifest_corrupted: dict[str, Any]) -> None:
        """All 5 defect categories are detected by the observer."""
        report = evaluate(corrupted_df, manifest_corrupted)
        # All categories should be true positives
        for name, cat in report.categories.items():
            assert cat.is_true_positive, f"{name}: TP expected but got present={cat.present_in_manifest} detected={cat.detected}"
        assert report.true_positives == 5
        assert report.false_positives == 0
        assert report.false_negatives == 0
        assert report.precision == 1.0
        assert report.recall == 1.0

    def test_duplicate_count(self, corrupted_df: pd.DataFrame, manifest_corrupted: dict[str, Any]) -> None:
        """Duplicate detection count matches injection."""
        report = evaluate(corrupted_df, manifest_corrupted)
        cat = report.categories["duplicate_transactions"]
        assert cat.detected is True
        assert cat.present_in_manifest is True
        assert cat.manifest_count == 2
        assert cat.observer_count == 2

    def test_null_count(self, corrupted_df: pd.DataFrame, manifest_corrupted: dict[str, Any]) -> None:
        """Null detection count matches injection."""
        report = evaluate(corrupted_df, manifest_corrupted)
        cat = report.categories["null_revenue"]
        assert cat.detected is True
        assert cat.manifest_count == 2
        assert cat.observer_count == 2

    def test_negative_revenue_count(self, corrupted_df: pd.DataFrame, manifest_corrupted: dict[str, Any]) -> None:
        """Negative revenue detection matches injection."""
        report = evaluate(corrupted_df, manifest_corrupted)
        cat = report.categories["negative_revenue"]
        assert cat.detected is True
        assert cat.manifest_count == 1
        assert cat.observer_count == 1

    def test_zero_units_count(self, corrupted_df: pd.DataFrame, manifest_corrupted: dict[str, Any]) -> None:
        """Zero-units-positive-revenue detection matches injection."""
        report = evaluate(corrupted_df, manifest_corrupted)
        cat = report.categories["zero_units_with_revenue"]
        assert cat.detected is True
        assert cat.manifest_count == 1
        assert cat.observer_count == 1

    def test_outlier_count(self, corrupted_df: pd.DataFrame, manifest_corrupted: dict[str, Any]) -> None:
        """Revenue outlier detection matches injection."""
        report = evaluate(corrupted_df, manifest_corrupted)
        cat = report.categories["revenue_outlier"]
        assert cat.detected is True
        assert cat.manifest_count == 1
        assert cat.observer_count >= 1  # IQR may catch >1; at minimum the 50x outlier


# ---------------------------------------------------------------------------
# Evaluate — partial corruption
# ---------------------------------------------------------------------------


class TestEvaluatePartialCorruption:
    """evaluate() with only some defects injected."""

    def test_only_duplicates(self, clean_df: pd.DataFrame) -> None:
        """Only duplicates injected — only that category should be TP."""
        df = clean_df.copy()
        # Use uniform revenue to avoid natural IQR outliers
        df["revenue"] = [100.0, 110.0, 120.0, 130.0, 140.0,
                          150.0, 160.0, 170.0, 180.0, 190.0]
        # Add one duplicate (row 0 with revenue 100)
        dup = df.iloc[[0]].copy()
        dup.index = [len(df)]
        df = pd.concat([df, dup], ignore_index=False)
        df = df.reset_index(drop=True)

        manifest: dict[str, Any] = {
            "duplicated_transaction_ids": {df.iloc[0]["transaction_id"]: [0, len(df) - 1]},
            "null_revenue_indices": [],
            "impossible_rows": [],
            "total_defects": 1,
        }

        report = evaluate(df, manifest)
        assert report.categories["duplicate_transactions"].is_true_positive
        assert report.categories["null_revenue"].is_true_negative
        assert report.categories["negative_revenue"].is_true_negative
        assert report.categories["zero_units_with_revenue"].is_true_negative
        assert report.categories["revenue_outlier"].is_true_negative

    def test_only_nulls(self, clean_df: pd.DataFrame) -> None:
        """Only nulls injected — only that category should be TP."""
        df = clean_df.copy()
        df.loc[0, "revenue"] = None
        df.loc[1, "revenue"] = None

        manifest: dict[str, Any] = {
            "duplicated_transaction_ids": {},
            "null_revenue_indices": [0, 1],
            "impossible_rows": [],
            "total_defects": 2,
        }

        report = evaluate(df, manifest)
        assert report.categories["duplicate_transactions"].is_true_negative
        assert report.categories["null_revenue"].is_true_positive
        assert report.categories["negative_revenue"].is_true_negative
        assert report.categories["zero_units_with_revenue"].is_true_negative
        assert report.categories["revenue_outlier"].is_true_negative
        assert report.true_positives == 1
        assert report.false_negatives == 0
        assert report.false_positives == 0

    def test_null_detected_without_manifest(self, clean_df: pd.DataFrame) -> None:
        """Observer detects nulls but manifest has none → false positive."""
        df = clean_df.copy()
        df.loc[0, "revenue"] = None  # null exists but not in manifest

        manifest: dict[str, Any] = {
            "duplicated_transaction_ids": {},
            "null_revenue_indices": [],  # empty manifest — no nulls recorded
            "impossible_rows": [],
            "total_defects": 0,
        }

        report = evaluate(df, manifest)
        assert report.categories["null_revenue"].detected is True
        assert report.categories["null_revenue"].present_in_manifest is False
        assert report.categories["null_revenue"].is_false_positive is True
        assert report.false_positives == 1
        assert report.precision < 1.0

    def test_manifest_defect_undetected(self, clean_df: pd.DataFrame) -> None:
        """Manifest says there's a defect but observer doesn't detect it → false negative.

        This can happen if synthetic data naturally has characteristics that
        mask the injected defect (e.g., the injected outlier is within IQR
        because the data already has extreme spread).
        """
        manifest: dict[str, Any] = {
            "duplicated_transaction_ids": {},
            "null_revenue_indices": [],
            "impossible_rows": [
                {
                    "index": 0,
                    "type": "negative_revenue",
                    "injected_values": {"revenue": -999.99},
                },
            ],
            "total_defects": 1,
        }
        # Clean data has no negative revenue, so observer won't detect it
        report = evaluate(clean_df, manifest)
        assert report.categories["negative_revenue"].present_in_manifest is True
        assert report.categories["negative_revenue"].detected is False
        assert report.categories["negative_revenue"].is_false_negative is True
        assert report.false_negatives == 1
        assert report.recall < 1.0


# ---------------------------------------------------------------------------
# Pipeline resilience
# ---------------------------------------------------------------------------


class TestPipelineResilience:
    """evaluate() with AgentRun state for resilience metrics."""

    def test_aborted_pipeline(self, clean_df: pd.DataFrame) -> None:
        """Aborted pipeline is reflected in the report."""
        state = AgentRun(
            goal="test",
            schema_summary="",
            aborted=True,
            abort_reason="Max retries exceeded on step_1: critique returned revise_code 5 times.",
            debug_history=[],  # empty but present
        )
        manifest: dict[str, Any] = {
            "duplicated_transaction_ids": {},
            "null_revenue_indices": [],
            "impossible_rows": [],
            "total_defects": 0,
        }
        report = evaluate(clean_df, manifest, state=state)
        assert report.pipeline_aborted is True
        assert "Max retries exceeded" in report.abort_reason

    def test_completed_pipeline(self, clean_df: pd.DataFrame) -> None:
        """Completed pipeline shows no abort."""
        state = AgentRun(
            goal="test",
            schema_summary="",
            aborted=False,
            debug_history=[],
        )
        manifest: dict[str, Any] = {
            "duplicated_transaction_ids": {},
            "null_revenue_indices": [],
            "impossible_rows": [],
            "total_defects": 0,
        }
        report = evaluate(clean_df, manifest, state=state)
        assert report.pipeline_aborted is False

    def test_attempts_counted(self, clean_df: pd.DataFrame) -> None:
        """Total attempts from debug_history length."""
        state = AgentRun(
            goal="test",
            schema_summary="",
            debug_history=[],  # empty, will set directly
        )
        # Simulate 3 attempts
        from agent.state import AttemptRecord
        state.debug_history = [
            AttemptRecord(
                task_id="step_1", sub_question="Q1", iteration=0,
                code="", sandbox_result={}, critique="", action="revise_code",
                trust="low",
            )
            for _ in range(3)
        ]
        manifest: dict[str, Any] = {
            "duplicated_transaction_ids": {},
            "null_revenue_indices": [],
            "impossible_rows": [],
            "total_defects": 0,
        }
        report = evaluate(clean_df, manifest, state=state)
        assert report.total_attempts == 3


# ---------------------------------------------------------------------------
# evaluate_from_corrupt_run convenience wrapper
# ---------------------------------------------------------------------------


class TestEvaluateFromCorruptRun:
    """Tests for the evaluate_from_corrupt_run convenience wrapper."""

    def test_wrapper_passes_state(self, clean_df: pd.DataFrame) -> None:
        """Wrapper passes AgentRun through to evaluate()."""
        state = AgentRun(
            goal="test",
            schema_summary="",
            aborted=False,
            debug_history=[],
        )
        manifest: dict[str, Any] = {
            "duplicated_transaction_ids": {},
            "null_revenue_indices": [],
            "impossible_rows": [],
            "total_defects": 0,
        }
        report = evaluate_from_corrupt_run(clean_df, manifest, state)
        assert isinstance(report, EvalReport)
        assert report.pipeline_aborted is False
        assert report.total_attempts == 0
