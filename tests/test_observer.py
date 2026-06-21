"""
Unit tests for the deterministic data-quality observer.

Tests cover:
- ObservationReport.as_critique_text formatting (clean vs corrupted data)
- observe() null counting, duplicate detection, negative revenue detection
- Edge cases: empty DataFrame, missing columns, single row
- IQR outlier detection correctness
"""
from __future__ import annotations

import pandas as pd
import pytest

from agent.observer import ObservationReport, observe


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def clean_df() -> pd.DataFrame:
    """Clean 5-row DataFrame with no corruption."""
    return pd.DataFrame({
        "transaction_id": ["A001", "A002", "A003", "A004", "A005"],
        "region": ["North", "South", "East", "West", "North"],
        "product_category": ["Electronics", "Clothing", "Food", "Home", "Sports"],
        "units_sold": [2, 3, 5, 1, 4],
        "unit_price": [99.99, 29.99, 9.99, 149.99, 79.99],
        "revenue": [199.98, 89.97, 49.95, 149.99, 319.96],
    })


@pytest.fixture()
def corrupted_df() -> pd.DataFrame:
    """DataFrame with known corruption: nulls, duplicates, negative revenue."""
    return pd.DataFrame({
        "transaction_id": ["A001", "A001", "A003", "A004", "A005"],  # A001 duplicated
        "region": ["North", "South", "East", "West", "North"],
        "product_category": ["Electronics", "Clothing", "Food", "Home", "Sports"],
        "units_sold": [2, 3, 5, 0, 4],
        "unit_price": [99.99, 29.99, 9.99, 149.99, 79.99],
        "revenue": [199.98, None, 49.95, 59.99, -319.96],  # null, zero_units+revenue, negative
    })


@pytest.fixture()
def empty_sandbox_result() -> dict:
    """Minimal sandbox result for a failed execution."""
    return {"success": False, "result": None, "stdout": "", "error": "SyntaxError", "error_type": "SyntaxError"}


@pytest.fixture()
def sandbox_result_with_stdout() -> dict:
    """Sandbox result with stdout (simulating printed DataFrames)."""
    return {
        "success": True,
        "result": "150.0",
        "stdout": "  region  revenue\n0  North     200\n1  South     300\n2   East     150\n",
        "error": None,
    }


# ── ObservationReport.as_critique_text ───────────────────────────────────────


class TestObservationReportText:
    """Tests for ObservationReport.as_critique_text formatting."""

    def test_clean_data_shows_clean(self) -> None:
        """Clean report shows 'none detected' for all checks."""
        report = ObservationReport(
            null_counts={"revenue": 0, "region": 0},
            duplicate_txn_ids=0,
            negative_revenue_rows=0,
            zero_units_positive_revenue=0,
            revenue_outliers_iqr=[],
            total_rows=100,
        )
        text = report.as_critique_text()
        assert "Nulls: none detected" in text
        assert "DUPLICATE TRANSACTION IDs" not in text
        assert "Duplicate transaction IDs: none detected" in text
        assert "NEGATIVE REVENUE" not in text
        assert "Input rows: 100" in text

    def test_null_report_shows_counts(self) -> None:
        """Report with nulls shows per-column counts and percentages."""
        report = ObservationReport(
            null_counts={"revenue": 5, "units_sold": 2, "region": 0},
            null_pct={"revenue": 5.0, "units_sold": 2.0, "region": 0.0},
            total_rows=100,
        )
        text = report.as_critique_text()
        assert "NULLS FOUND:" in text
        assert "revenue: 5 nulls (5.0% of rows)" in text
        assert "units_sold: 2 nulls (2.0% of rows)" in text

    def test_duplicate_txn_ids_shown(self) -> None:
        """Duplicate transaction IDs are listed with examples."""
        report = ObservationReport(
            duplicate_txn_ids=3,
            duplicate_txn_examples=["A001", "A002", "A003"],
            total_rows=100,
        )
        text = report.as_critique_text()
        assert "DUPLICATE TRANSACTION IDs: 3" in text
        assert "A001" in text

    def test_negative_revenue_shown(self) -> None:
        """Negative revenue rows are flagged."""
        report = ObservationReport(
            negative_revenue_rows=2,
            total_rows=100,
        )
        text = report.as_critique_text()
        assert "NEGATIVE REVENUE: 2 row(s)" in text

    def test_zero_units_positive_revenue_shown(self) -> None:
        """Logical inconsistency is flagged."""
        report = ObservationReport(
            zero_units_positive_revenue=1,
            total_rows=100,
        )
        text = report.as_critique_text()
        assert "LOGICAL INCONSISTENCY" in text
        assert "units_sold=0 AND revenue>0" in text

    def test_outliers_shown(self) -> None:
        """IQR outliers are reported with indices."""
        report = ObservationReport(
            revenue_outliers_iqr=[42, 99],
            total_rows=100,
        )
        text = report.as_critique_text()
        assert "REVENUE OUTLIERS" in text
        assert "42" in text
        assert "99" in text

    def test_outliers_truncated_at_10(self) -> None:
        """More than 10 outliers use ellipsis."""
        report = ObservationReport(
            revenue_outliers_iqr=list(range(15)),
            total_rows=100,
        )
        text = report.as_critique_text()
        assert "…" in text

    def test_notes_included(self) -> None:
        """Free-text notes are included."""
        report = ObservationReport(
            notes=["Note 1", "Note 2"],
            total_rows=100,
        )
        text = report.as_critique_text()
        assert "ADDITIONAL OBSERVATIONS:" in text
        assert "Note 1" in text
        assert "Note 2" in text


# ── observe() function ───────────────────────────────────────────────────────


class TestObserve:
    """Integration tests for the observe() function."""

    def test_clean_data_all_zeros(self, clean_df: pd.DataFrame, empty_sandbox_result: dict) -> None:
        """Clean DataFrame produces a clean report with no flags."""
        report = observe(clean_df, empty_sandbox_result)
        assert report.total_rows == 5
        for null_count in report.null_counts.values():
            assert null_count == 0
        assert report.duplicate_txn_ids == 0
        assert report.negative_revenue_rows == 0
        assert report.zero_units_positive_revenue == 0

    def test_detects_duplicate_txn_ids(self, corrupted_df: pd.DataFrame, empty_sandbox_result: dict) -> None:
        """Detects duplicate transaction_id values."""
        report = observe(corrupted_df, empty_sandbox_result)
        assert report.duplicate_txn_ids == 1  # A001 appears twice
        assert "A001" in report.duplicate_txn_examples

    def test_detects_null_revenue(self, corrupted_df: pd.DataFrame, empty_sandbox_result: dict) -> None:
        """Detects null values in the revenue column."""
        report = observe(corrupted_df, empty_sandbox_result)
        assert report.null_counts["revenue"] == 1

    def test_detects_negative_revenue(self, corrupted_df: pd.DataFrame, empty_sandbox_result: dict) -> None:
        """Detects negative revenue rows."""
        report = observe(corrupted_df, empty_sandbox_result)
        assert report.negative_revenue_rows == 1

    def test_detects_zero_units_positive_revenue(self, corrupted_df: pd.DataFrame, empty_sandbox_result: dict) -> None:
        """Detects zero units_sold with positive revenue."""
        report = observe(corrupted_df, empty_sandbox_result)
        assert report.zero_units_positive_revenue == 1

    def test_iqr_outliers_on_uniform_data(self, clean_df: pd.DataFrame, empty_sandbox_result: dict) -> None:
        """Uniform-ish data produces no IQR outliers."""
        report = observe(clean_df, empty_sandbox_result)
        assert len(report.revenue_outliers_iqr) == 0

    def test_stdout_lines_counted(self, clean_df: pd.DataFrame, sandbox_result_with_stdout: dict) -> None:
        """Stdout lines are counted as a row-filtering heuristic."""
        report = observe(clean_df, sandbox_result_with_stdout)
        assert report.rows_after_code_filter > 0

    def test_missing_transaction_id_column(self, clean_df: pd.DataFrame, empty_sandbox_result: dict) -> None:
        """DataFrame without transaction_id column doesn't crash."""
        df = clean_df.drop(columns=["transaction_id"])
        report = observe(df, empty_sandbox_result)
        assert report.duplicate_txn_ids == 0  # no column → no duplicates detected

    def test_empty_dataframe(self, empty_sandbox_result: dict) -> None:
        """Empty DataFrame doesn't crash — null_counts are zero for all columns."""
        df = pd.DataFrame(columns=["revenue", "units_sold"])
        report = observe(df, empty_sandbox_result)
        assert report.total_rows == 0
        # Iterates over df.columns, so null_counts has entries (all zero)
        assert report.null_counts == {"revenue": 0, "units_sold": 0}

    def test_single_row_iqr(self, empty_sandbox_result: dict) -> None:
        """Single row DataFrame — IQR requires >= 4 data points, returns no outliers."""
        df = pd.DataFrame({"revenue": [100.0], "transaction_id": ["A001"]})
        report = observe(df, empty_sandbox_result)
        assert len(report.revenue_outliers_iqr) == 0

    def test_artifact_keys_in_notes(self, clean_df: pd.DataFrame, empty_sandbox_result: dict) -> None:
        """Available artifact keys appear in notes."""
        report = observe(clean_df, empty_sandbox_result, artifact_store_available_keys=["quarter_df", "region_revenue"])
        assert any("quarter_df" in n for n in report.notes)
        assert any("region_revenue" in n for n in report.notes)

    def test_no_artifact_keys(self, clean_df: pd.DataFrame, empty_sandbox_result: dict) -> None:
        """No artifact keys means no artifact note."""
        report = observe(clean_df, empty_sandbox_result)
        assert not any("ArtifactStore" in n for n in report.notes)


# ── Report completeness ──────────────────────────────────────────────────────


class TestReportCompleteness:
    """Ensure all fields are populated by observe()."""

    def test_all_fields_populated(self, clean_df: pd.DataFrame, empty_sandbox_result: dict) -> None:
        """Every ObservationReport field is set after observe() returns."""
        report = observe(clean_df, empty_sandbox_result)
        assert isinstance(report.null_counts, dict)
        assert isinstance(report.null_pct, dict)
        assert isinstance(report.duplicate_txn_ids, int)
        assert isinstance(report.duplicate_txn_examples, list)
        assert isinstance(report.negative_revenue_rows, int)
        assert isinstance(report.zero_units_positive_revenue, int)
        assert isinstance(report.revenue_outliers_iqr, list)
        assert isinstance(report.total_rows, int)
        assert isinstance(report.rows_after_code_filter, int)
        assert isinstance(report.notes, list)
        # as_critique_text must be a non-empty string
        text = report.as_critique_text()
        assert isinstance(text, str)
        assert len(text) > 0