"""
Unit tests for agent.artifact_store.ArtifactStore.

Tests cover:
- Registration of valid DataFrames
- Rejection of invalid inputs (non-DataFrame, empty, too large)
- Key lookup (get, __contains__, available_keys)
- Namespace export (as_namespace, as_namespace_for)
- Self-description for prompt injection (describe) with typed artifacts
- Artifact classification (DataFrame vs Series vs scalar)
- Artifact first-class object (name, type, schema, producer, value)
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from agent.artifact_store import (
    Artifact,
    ArtifactStore,
    classify_artifact,
    MAX_ARTIFACT_BYTES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store() -> ArtifactStore:
    return ArtifactStore()


@pytest.fixture()
def small_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["North", "South", "East"],
            "revenue": [100.0, 200.0, 150.0],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        }
    )


@pytest.fixture()
def region_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["North", "South"],
            "total": [500.0, 750.0],
        }
    )


# ---------------------------------------------------------------------------
# Registration — happy path
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_returns_true(
        self, store: ArtifactStore, small_df: pd.DataFrame
    ) -> None:
        assert store.register("quarter_df", small_df) is True

    def test_registered_key_is_available(
        self, store: ArtifactStore, small_df: pd.DataFrame
    ) -> None:
        store.register("quarter_df", small_df)
        assert "quarter_df" in store.available_keys()

    def test_artifact_name_and_type(
        self, store: ArtifactStore, small_df: pd.DataFrame
    ) -> None:
        store.register("quarter_df", small_df, producing_task_id="step_1")
        a = store._artifacts["quarter_df"]
        assert a.name == "quarter_df"
        assert a.type == "DataFrame"

    def test_artifact_rows_and_columns(
        self, store: ArtifactStore, small_df: pd.DataFrame
    ) -> None:
        store.register(
            "quarter_df",
            small_df,
            producing_task_id="step_1",
            consumers=["step_2"],
            confidence="high",
        )
        a = store._artifacts["quarter_df"]
        assert a.rows == 3
        assert list(a.value.columns) == ["region", "revenue", "date"]
        assert a.producer == "step_1"
        assert a.consumers == ["step_2"]
        assert a.confidence == "high"
        assert a.is_frozen is True

    def test_artifact_is_immutable_after_registration(
        self, store: ArtifactStore, small_df: pd.DataFrame
    ) -> None:
        store.register("quarter_df", small_df, producing_task_id="step_1")
        artifact = store._artifacts["quarter_df"]
        with pytest.raises(AttributeError):
            artifact.confidence = "low"

    def test_dtypes_preserved(
        self, store: ArtifactStore, small_df: pd.DataFrame
    ) -> None:
        store.register("quarter_df", small_df)
        a = store._artifacts["quarter_df"]
        assert "float64" in str(a.value["revenue"].dtype)

    def test_register_overwrites_existing_key(
        self, store: ArtifactStore, small_df: pd.DataFrame
    ) -> None:
        """Registering the same key twice replaces the old entry."""
        store.register("key", small_df)
        new_df = pd.DataFrame({"x": [1, 2, 3, 4]})
        store.register("key", new_df)
        assert store._artifacts["key"].rows == 4
        assert store._artifacts["key"].name == "key"

    def test_shallow_copy_semantics(
        self, store: ArtifactStore, small_df: pd.DataFrame
    ) -> None:
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

    def test_oversized_dataframe_rejected(
        self, store: ArtifactStore, small_df: pd.DataFrame
    ) -> None:
        """DataFrames whose pickled size exceeds MAX_ARTIFACT_BYTES are rejected."""
        oversized_bytes = MAX_ARTIFACT_BYTES + 1
        with patch(
            "agent.artifact_store.pickle.dumps", return_value=b"x" * oversized_bytes
        ):
            result = store.register("big_df", small_df)
        assert result is False
        assert "big_df" not in store.available_keys()


# ---------------------------------------------------------------------------
# Read API
# ---------------------------------------------------------------------------


class TestGet:
    def test_get_returns_registered_dataframe(
        self, store: ArtifactStore, small_df: pd.DataFrame
    ) -> None:
        store.register("df", small_df)
        retrieved = store.get("df")
        assert retrieved is not None
        assert list(retrieved.columns) == list(small_df.columns)
        assert len(retrieved) == len(small_df)

    def test_get_missing_key_returns_none(self, store: ArtifactStore) -> None:
        assert store.get("nonexistent") is None

    def test_get_artifact_returns_full_object(
        self, store: ArtifactStore, small_df: pd.DataFrame
    ) -> None:
        store.register("df", small_df, producing_task_id="step_1")
        a = store.get_artifact("df")
        assert a is not None
        assert isinstance(a, Artifact)
        assert a.name == "df"
        assert a.type == "DataFrame"
        assert a.producer == "step_1"

    def test_get_artifact_missing_returns_none(self, store: ArtifactStore) -> None:
        assert store.get_artifact("x") is None

    def test_contains_true_after_register(
        self, store: ArtifactStore, small_df: pd.DataFrame
    ) -> None:
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
        ns = store.as_namespace_for(["quarter_df", "nonexistent"])
        assert "quarter_df" in ns
        assert "nonexistent" not in ns

    def test_as_namespace_for_empty_list_returns_empty(
        self, store: ArtifactStore, small_df: pd.DataFrame
    ) -> None:
        store.register("df", small_df)
        assert store.as_namespace_for([]) == {}


# ---------------------------------------------------------------------------
# describe() for prompt injection
# ---------------------------------------------------------------------------


class TestDescribe:
    def test_describe_empty_store(self, store: ArtifactStore) -> None:
        assert store.describe() == "(no pre-computed artifacts available yet)"

    def test_describe_mentions_key(
        self, store: ArtifactStore, small_df: pd.DataFrame
    ) -> None:
        store.register("quarter_df", small_df, producing_task_id="step_1")
        desc = store.describe()
        assert "quarter_df" in desc

    def test_describe_mentions_row_count(
        self, store: ArtifactStore, small_df: pd.DataFrame
    ) -> None:
        store.register("quarter_df", small_df)
        desc = store.describe()
        assert "3" in desc  # 3 rows

    def test_describe_mentions_columns(
        self, store: ArtifactStore, small_df: pd.DataFrame
    ) -> None:
        store.register("quarter_df", small_df)
        desc = store.describe()
        assert "region" in desc
        assert "revenue" in desc

    def test_describe_does_not_mention_producer(
        self, store: ArtifactStore, small_df: pd.DataFrame
    ) -> None:
        """Typed describe() is self-contained — producer ID is internal metadata, not prompt content."""
        store.register(
            "quarter_df", small_df, producing_task_id="step_1", confidence="high"
        )
        desc = store.describe()
        assert "step_1" not in desc
        assert "quarter_df" in desc
        assert "type=DataFrame" in desc
        assert "confidence=high" in desc

    def test_describe_contains_usage_example(
        self, store: ArtifactStore, small_df: pd.DataFrame
    ) -> None:
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

    def test_describe_shows_type(
        self, store: ArtifactStore, small_df: pd.DataFrame
    ) -> None:
        store.register("quarter_df", small_df)
        desc = store.describe()
        assert "type=DataFrame" in desc

    def test_describe_shows_schema(
        self, store: ArtifactStore, small_df: pd.DataFrame
    ) -> None:
        store.register("quarter_df", small_df)
        desc = store.describe()
        assert "schema=[" in desc

    def test_describe_scalar_hides_schema(self, store: ArtifactStore) -> None:
        scalar_df = pd.DataFrame({"result": [42.5]})
        store.register("answer", scalar_df)
        desc = store.describe()
        assert "type=scalar" in desc
        assert "schema=[<scalar>]" not in desc


# ---------------------------------------------------------------------------
# Artifact classification (DataFrame vs Series vs scalar)
# ---------------------------------------------------------------------------


class TestArtifactClassification:
    """Tests for classify_artifact, exercised through register()."""

    def test_multi_column_df_is_dataframe(self, store: ArtifactStore) -> None:
        store.register("t", pd.DataFrame({"a": [1], "b": [2], "c": [3]}))
        assert store._artifacts["t"].type == "DataFrame"

    def test_single_column_df_is_series(self, store: ArtifactStore) -> None:
        store.register("rev", pd.DataFrame({"total": [100, 200, 300]}))
        assert store._artifacts["rev"].type == "Series"

    def test_single_cell_df_is_scalar(self, store: ArtifactStore) -> None:
        store.register("ans", pd.DataFrame({"result": [42]}))
        assert store._artifacts["ans"].type == "scalar"

    def test_single_column_single_row_is_scalar(self, store: ArtifactStore) -> None:
        store.register("val", pd.DataFrame({"x": [99]}))
        assert store._artifacts["val"].type == "scalar"

    def test_single_column_multi_row_is_series(self, store: ArtifactStore) -> None:
        store.register("ser", pd.DataFrame({"col": [1, 2, 3, 4, 5]}))
        assert store._artifacts["ser"].type == "Series"

    def test_schema_for_dataframe(self, store: ArtifactStore) -> None:
        store.register("df", pd.DataFrame({"region": ["N"], "revenue": [100]}))
        assert store._artifacts["df"].schema == ["region", "revenue"]

    def test_schema_for_series(self, store: ArtifactStore) -> None:
        store.register("ser", pd.DataFrame({"total": [100, 200]}))
        assert store._artifacts["ser"].schema == ["total"]

    def test_schema_for_scalar(self, store: ArtifactStore) -> None:
        store.register("s", pd.DataFrame({"result": [99]}))
        assert store._artifacts["s"].schema == ["<scalar>"]


# ---------------------------------------------------------------------------
# classify_artifact() as a pure function
# ---------------------------------------------------------------------------


class TestClassifyArtifact:
    """Direct tests for the public classify_artifact() function."""

    def test_dataframe(self) -> None:
        t, s = classify_artifact(pd.DataFrame({"a": [1], "b": [2]}))
        assert t == "DataFrame"
        assert s == ["a", "b"]

    def test_series(self) -> None:
        t, s = classify_artifact(pd.DataFrame({"total": [100, 200]}))
        assert t == "Series"
        assert s == ["total"]

    def test_scalar(self) -> None:
        t, s = classify_artifact(pd.DataFrame({"result": [42]}))
        assert t == "scalar"
        assert s == ["<scalar>"]


# ---------------------------------------------------------------------------
# Artifact first-class object completeness
# ---------------------------------------------------------------------------


class TestArtifactCompleteness:
    """Verify the Artifact dataclass bundles name, type, schema, producer, and value."""

    def test_all_fields_set(self, store: ArtifactStore, small_df: pd.DataFrame) -> None:
        store.register("df", small_df, producing_task_id="step_1")
        a = store._artifacts["df"]
        assert a.name == "df"
        assert a.type == "DataFrame"
        assert a.schema == list(small_df.columns)
        assert a.rows == len(small_df)
        assert a.producer == "step_1"
        assert isinstance(a.value, pd.DataFrame)
        assert isinstance(a.column_summary(), str)

    def test_series_artifact(self, store: ArtifactStore) -> None:
        store.register("totals", pd.DataFrame({"revenue": [100, 200]}))
        a = store._artifacts["totals"]
        assert a.type == "Series"
        assert a.schema == ["revenue"]
        assert a.rows == 2

    def test_scalar_artifact(self, store: ArtifactStore) -> None:
        store.register("answer", pd.DataFrame({"result": [3.14]}))
        a = store._artifacts["answer"]
        assert a.type == "scalar"
        assert a.schema == ["<scalar>"]
        assert a.rows == 1
