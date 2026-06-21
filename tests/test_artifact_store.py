"""
Unit tests for agent.artifact_store.ArtifactStore.

Tests cover:
- Registration of valid DataFrames
- Rejection of invalid inputs (non-DataFrame, empty, too large)
- Key lookup (get, __contains__, available_keys)
- Namespace export (as_namespace, as_namespace_for)
- Self-description for prompt injection (describe)
- Metadata accuracy after registration
"""

from __future__ import annotations

import pickle
from unittest.mock import patch

import pandas as pd
import pytest

from agent.artifact_store import ArtifactStore, MAX_ARTIFACT_BYTES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store() -> ArtifactStore:
    return ArtifactStore()


@pytest.fixture()
def small_df() -> pd.DataFrame:
    return pd.DataFrame({
        "region": ["North", "South", "East"],
        "revenue": [100.0, 200.0, 150.0],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    })


@pytest.fixture()
def region_df() -> pd.DataFrame:
    return pd.DataFrame({
        "region": ["North", "South"],
        "total": [500.0, 750.0],
    })


# ---------------------------------------------------------------------------
# Registration — happy path
# ---------------------------------------------------------------------------

class TestRegister:
    def test_register_returns_true(self, store: ArtifactStore, small_df: pd.DataFrame) -> None:
        assert store.register("quarter_df", small_df) is True

    def test_registered_key_is_available(self, store: ArtifactStore, small_df: pd.DataFrame) -> None:
        store.register("quarter_df", small_df)
        assert "quarter_df" in store.available_keys()

    def test_metadata_rows_and_columns_correct(self, store: ArtifactStore, small_df: pd.DataFrame) -> None:
        store.register("quarter_df", small_df, producing_task_id="step_1")
        meta = store._meta["quarter_df"]
        assert meta.rows == 3
        assert set(meta.columns) == {"region", "revenue", "date"}
        assert meta.producing_task_id == "step_1"

    def test_dtype_metadata_recorded(self, store: ArtifactStore, small_df: pd.DataFrame) -> None:
        store.register("quarter_df", small_df)
        meta = store._meta["quarter_df"]
        assert "float64" in meta.dtypes["revenue"]

    def test_register_overwrites_existing_key(self, store: ArtifactStore, small_df: pd.DataFrame) -> None:
        """Registering the same key twice replaces the old entry."""
        store.register("key", small_df)
        new_df = pd.DataFrame({"x": [1, 2, 3, 4]})
        store.register("key", new_df)
        assert store._meta["key"].rows == 4

    def test_shallow_copy_semantics(self, store: ArtifactStore, small_df: pd.DataFrame) -> None:
        """The caller mutating the original DataFrame's index does not affect the stored copy."""
        store.register("df", small_df)
        small_df.index = [10, 20, 30]
        stored = store.get("df")
        assert list(stored.index) != [10, 20, 30]


# ---------------------------------------------------------------------------
# Registration — rejection cases
# ---------------------------------------------------------------------------

class TestRegisterRejection:
    def test_none_is_rejected(self, store: ArtifactStore) -> None:
        assert store.register("x", None) is False  # type: ignore[arg-type]

    def test_non_dataframe_is_rejected(self, store: ArtifactStore) -> None:
        assert store.register("x", [1, 2, 3]) is False  # type: ignore[arg-type]

    def test_empty_dataframe_is_rejected(self, store: ArtifactStore) -> None:
        assert store.register("x", pd.DataFrame()) is False

    def test_empty_dataframe_not_stored(self, store: ArtifactStore) -> None:
        store.register("x", pd.DataFrame())
        assert "x" not in store.available_keys()

    def test_oversized_dataframe_rejected(self, store: ArtifactStore, small_df: pd.DataFrame) -> None:
        """DataFrames whose pickled size exceeds MAX_ARTIFACT_BYTES are rejected."""
        oversized_bytes = MAX_ARTIFACT_BYTES + 1
        with patch("agent.artifact_store.pickle.dumps", return_value=b"x" * oversized_bytes):
            result = store.register("big_df", small_df)
        assert result is False
        assert "big_df" not in store.available_keys()


# ---------------------------------------------------------------------------
# Read API
# ---------------------------------------------------------------------------

class TestGet:
    def test_get_returns_registered_dataframe(self, store: ArtifactStore, small_df: pd.DataFrame) -> None:
        store.register("df", small_df)
        retrieved = store.get("df")
        assert retrieved is not None
        assert list(retrieved.columns) == list(small_df.columns)
        assert len(retrieved) == len(small_df)

    def test_get_missing_key_returns_none(self, store: ArtifactStore) -> None:
        assert store.get("nonexistent") is None

    def test_contains_true_after_register(self, store: ArtifactStore, small_df: pd.DataFrame) -> None:
        store.register("df", small_df)
        assert "df" in store

    def test_contains_false_for_missing(self, store: ArtifactStore) -> None:
        assert "not_there" not in store

    def test_available_keys_empty_initially(self, store: ArtifactStore) -> None:
        assert store.available_keys() == []

    def test_available_keys_after_multiple_registrations(
        self, store: ArtifactStore, small_df: pd.DataFrame, region_df: pd.DataFrame
    ) -> None:
        store.register("quarter_df", small_df)
        store.register("region_revenue", region_df)
        keys = store.available_keys()
        assert set(keys) == {"quarter_df", "region_revenue"}


# ---------------------------------------------------------------------------
# Namespace export
# ---------------------------------------------------------------------------

class TestNamespace:
    def test_as_namespace_returns_all_keys(
        self, store: ArtifactStore, small_df: pd.DataFrame, region_df: pd.DataFrame
    ) -> None:
        store.register("quarter_df", small_df)
        store.register("region_revenue", region_df)
        ns = store.as_namespace()
        assert set(ns.keys()) == {"quarter_df", "region_revenue"}

    def test_as_namespace_is_independent_copy(
        self, store: ArtifactStore, small_df: pd.DataFrame
    ) -> None:
        """Mutating the namespace dict does not affect the store's internal registry."""
        store.register("df", small_df)
        ns = store.as_namespace()
        ns["injected"] = pd.DataFrame()
        assert "injected" not in store.available_keys()

    def test_as_namespace_for_returns_subset(
        self, store: ArtifactStore, small_df: pd.DataFrame, region_df: pd.DataFrame
    ) -> None:
        store.register("quarter_df", small_df)
        store.register("region_revenue", region_df)
        ns = store.as_namespace_for(["quarter_df"])
        assert "quarter_df" in ns
        assert "region_revenue" not in ns

    def test_as_namespace_for_handles_missing_keys_gracefully(
        self, store: ArtifactStore, small_df: pd.DataFrame
    ) -> None:
        store.register("quarter_df", small_df)
        # "nonexistent" is silently omitted
        ns = store.as_namespace_for(["quarter_df", "nonexistent"])
        assert "quarter_df" in ns
        assert "nonexistent" not in ns

    def test_as_namespace_for_empty_list_returns_empty(self, store: ArtifactStore, small_df: pd.DataFrame) -> None:
        store.register("df", small_df)
        assert store.as_namespace_for([]) == {}


# ---------------------------------------------------------------------------
# describe() for prompt injection
# ---------------------------------------------------------------------------

class TestDescribe:
    def test_describe_empty_store(self, store: ArtifactStore) -> None:
        assert store.describe() == "(no pre-computed artifacts available yet)"

    def test_describe_mentions_key(self, store: ArtifactStore, small_df: pd.DataFrame) -> None:
        store.register("quarter_df", small_df, producing_task_id="step_1")
        desc = store.describe()
        assert "quarter_df" in desc

    def test_describe_mentions_row_count(self, store: ArtifactStore, small_df: pd.DataFrame) -> None:
        store.register("quarter_df", small_df)
        desc = store.describe()
        assert "3" in desc  # 3 rows

    def test_describe_mentions_columns(self, store: ArtifactStore, small_df: pd.DataFrame) -> None:
        store.register("quarter_df", small_df)
        desc = store.describe()
        assert "region" in desc
        assert "revenue" in desc

    def test_describe_mentions_producer(self, store: ArtifactStore, small_df: pd.DataFrame) -> None:
        store.register("quarter_df", small_df, producing_task_id="step_1")
        desc = store.describe()
        assert "step_1" in desc

    def test_describe_contains_usage_example(self, store: ArtifactStore, small_df: pd.DataFrame) -> None:
        store.register("quarter_df", small_df)
        desc = store.describe()
        assert 'artifacts["quarter_df"]' in desc

    def test_describe_multiple_artifacts(
        self, store: ArtifactStore, small_df: pd.DataFrame, region_df: pd.DataFrame
    ) -> None:
        store.register("quarter_df", small_df)
        store.register("region_revenue", region_df)
        desc = store.describe()
        assert "quarter_df" in desc
        assert "region_revenue" in desc

    def test_describe_column_truncation(self, store: ArtifactStore) -> None:
        """Wide DataFrames should have columns truncated in the description."""
        wide_df = pd.DataFrame({f"col_{i}": [1, 2] for i in range(20)})
        store.register("wide", wide_df)
        desc = store.describe()
        assert "…" in desc or "more" in desc
