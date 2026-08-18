"""
Unit tests for agent.runtime — Phase 1 AnalysisRuntime and ArtifactRegistry.

Test matrix
-----------
Runtime initialisation:
  - df exists and references the input DataFrame (no copy)
  - pd and np are injected into namespace
  - registry starts empty

Artifact registration:
  - register a DataFrame and verify it can be retrieved via get()
  - register a non-DataFrame (bool, scalar, string) and retrieve it

Artifact retrieval:
  - exists() returns False for unknown IDs
  - exists() returns True for registered IDs
  - get() returns None for unknown IDs
  - get() returns the Artifact for registered IDs
  - get_value() returns the raw Python value

Multiple artifact types:
  - DataFrame artifact captures shape and columns
  - bool artifact has type "bool"
  - number artifact has type "number"
  - string artifact has type "string"
  - dict artifact has type "dict"
  - list artifact has type "list"

Run isolation:
  - two AnalysisRuntime instances do not share artifacts
  - an artifact in runtime A is not visible in runtime B

Artifact metadata:
  - DataFrame shape captured correctly
  - DataFrame columns captured correctly
  - DataFrame dtypes captured correctly
  - creation_order increments across registrations

describe():
  - includes df shape info
  - includes registered artifact info
  - does NOT include raw DataFrame contents

capture_result():
  - stable artifact_id derived from task_id
  - value is injected into namespace under stable ID
  - variable_name alias is also injected into namespace if provided

namespace_snapshot():
  - returns a copy (mutations don't affect runtime namespace)
  - contains df, pd, np

update_namespace():
  - new variables are merged
  - protected names (df, pd, np) are not overwritten
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from agent.runtime import AnalysisRuntime, ArtifactRegistry, Artifact


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Small 5-row DataFrame for runtime tests."""
    return pd.DataFrame(
        {
            "region": ["North", "South", "East", "North", "West"],
            "revenue": [100.0, 200.0, 150.0, 300.0, 250.0],
        }
    )


@pytest.fixture()
def runtime(sample_df: pd.DataFrame) -> AnalysisRuntime:
    return AnalysisRuntime(sample_df)


# ---------------------------------------------------------------------------
# Runtime initialisation
# ---------------------------------------------------------------------------


class TestRuntimeInitialisation:
    def test_df_is_accessible(self, runtime: AnalysisRuntime, sample_df: pd.DataFrame) -> None:
        """df property must return the original DataFrame (not a copy)."""
        assert runtime.df is sample_df

    def test_df_in_namespace(self, runtime: AnalysisRuntime, sample_df: pd.DataFrame) -> None:
        """namespace must contain 'df' pointing to the input DataFrame."""
        assert runtime.namespace["df"] is sample_df

    def test_pd_in_namespace(self, runtime: AnalysisRuntime) -> None:
        assert runtime.namespace["pd"] is pd

    def test_np_in_namespace(self, runtime: AnalysisRuntime) -> None:
        assert runtime.namespace["np"] is np

    def test_registry_starts_empty(self, runtime: AnalysisRuntime) -> None:
        assert len(runtime.registry) == 0
        assert runtime.registry.list() == []

    def test_requires_dataframe(self) -> None:
        with pytest.raises(TypeError):
            AnalysisRuntime("not a dataframe")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Artifact registration and retrieval
# ---------------------------------------------------------------------------


class TestArtifactRegistration:
    def test_register_and_get_dataframe(self, runtime: AnalysisRuntime, sample_df: pd.DataFrame) -> None:
        artifact = runtime.capture_result(
            task_id="step_1",
            value=sample_df,
            variable_name="quarter_df",
        )
        assert artifact is not None
        assert artifact.artifact_id == "artifact_step_1"
        assert artifact.task_id == "step_1"
        assert artifact.type == "DataFrame"

    def test_artifact_retrievable_via_registry(self, runtime: AnalysisRuntime, sample_df: pd.DataFrame) -> None:
        runtime.capture_result(task_id="step_1", value=sample_df)
        a = runtime.registry.get("artifact_step_1")
        assert a is not None
        assert a.value is sample_df

    def test_exists_true_after_registration(self, runtime: AnalysisRuntime, sample_df: pd.DataFrame) -> None:
        runtime.capture_result(task_id="step_2", value=sample_df)
        assert runtime.registry.exists("artifact_step_2") is True

    def test_exists_false_for_unknown(self, runtime: AnalysisRuntime) -> None:
        assert runtime.registry.exists("artifact_ghost") is False

    def test_get_returns_none_for_unknown(self, runtime: AnalysisRuntime) -> None:
        assert runtime.registry.get("artifact_ghost") is None

    def test_get_value_returns_raw(self, runtime: AnalysisRuntime, sample_df: pd.DataFrame) -> None:
        runtime.capture_result(task_id="step_3", value=sample_df)
        val = runtime.registry.get_value("artifact_step_3")
        assert val is sample_df

    def test_stable_id_in_namespace(self, runtime: AnalysisRuntime, sample_df: pd.DataFrame) -> None:
        """After capture_result, stable ID must be in namespace."""
        runtime.capture_result(task_id="step_4", value=sample_df)
        assert "artifact_step_4" in runtime.namespace
        assert runtime.namespace["artifact_step_4"] is sample_df

    def test_variable_name_alias_in_namespace(self, runtime: AnalysisRuntime, sample_df: pd.DataFrame) -> None:
        """variable_name alias is also inserted into namespace for backward compat."""
        runtime.capture_result(task_id="step_5", value=sample_df, variable_name="quarter_df")
        assert "quarter_df" in runtime.namespace

    def test_empty_dataframe_rejected(self, runtime: AnalysisRuntime) -> None:
        empty = pd.DataFrame()
        artifact = runtime.capture_result(task_id="step_bad", value=empty)
        assert artifact is None
        assert not runtime.registry.exists("artifact_step_bad")


# ---------------------------------------------------------------------------
# Multiple artifact types
# ---------------------------------------------------------------------------


class TestMultipleArtifactTypes:
    def test_bool_artifact(self, runtime: AnalysisRuntime) -> None:
        a = runtime.capture_result(task_id="step_bool", value=True)
        assert a is not None
        assert a.type == "bool"
        assert runtime.registry.get_value("artifact_step_bool") is True

    def test_false_bool_artifact(self, runtime: AnalysisRuntime) -> None:
        a = runtime.capture_result(task_id="step_bool_f", value=False)
        assert a is not None
        assert a.type == "bool"

    def test_number_artifact_int(self, runtime: AnalysisRuntime) -> None:
        a = runtime.capture_result(task_id="step_num", value=42)
        assert a is not None
        assert a.type == "number"

    def test_number_artifact_float(self, runtime: AnalysisRuntime) -> None:
        a = runtime.capture_result(task_id="step_flt", value=3.14)
        assert a is not None
        assert a.type == "number"

    def test_string_artifact(self, runtime: AnalysisRuntime) -> None:
        a = runtime.capture_result(task_id="step_str", value="West")
        assert a is not None
        assert a.type == "string"

    def test_dict_artifact(self, runtime: AnalysisRuntime) -> None:
        a = runtime.capture_result(task_id="step_dict", value={"a": 1})
        assert a is not None
        assert a.type == "dict"

    def test_list_artifact(self, runtime: AnalysisRuntime) -> None:
        a = runtime.capture_result(task_id="step_list", value=[1, 2, 3])
        assert a is not None
        assert a.type == "list"

    def test_dataframe_artifact_type(self, runtime: AnalysisRuntime, sample_df: pd.DataFrame) -> None:
        a = runtime.capture_result(task_id="step_df", value=sample_df)
        assert a is not None
        assert a.type == "DataFrame"


# ---------------------------------------------------------------------------
# Run isolation
# ---------------------------------------------------------------------------


class TestRunIsolation:
    def test_two_runtimes_are_independent(self, sample_df: pd.DataFrame) -> None:
        """Artifacts in runtime A must not be visible in runtime B."""
        rt_a = AnalysisRuntime(sample_df)
        rt_b = AnalysisRuntime(sample_df)

        rt_a.capture_result(task_id="step_1", value=sample_df)
        assert rt_a.registry.exists("artifact_step_1") is True
        assert rt_b.registry.exists("artifact_step_1") is False

    def test_namespace_isolation(self, sample_df: pd.DataFrame) -> None:
        rt_a = AnalysisRuntime(sample_df)
        rt_b = AnalysisRuntime(sample_df)

        rt_a.namespace["my_var"] = "hello"
        assert "my_var" not in rt_b.namespace

    def test_separate_registry_objects(self, sample_df: pd.DataFrame) -> None:
        rt_a = AnalysisRuntime(sample_df)
        rt_b = AnalysisRuntime(sample_df)
        assert rt_a.registry is not rt_b.registry


# ---------------------------------------------------------------------------
# Artifact metadata
# ---------------------------------------------------------------------------


class TestArtifactMetadata:
    def test_dataframe_shape_captured(self, runtime: AnalysisRuntime, sample_df: pd.DataFrame) -> None:
        runtime.capture_result(task_id="step_meta", value=sample_df)
        a = runtime.registry.get("artifact_step_meta")
        assert a is not None
        assert a.shape == (len(sample_df), len(sample_df.columns))

    def test_dataframe_columns_captured(self, runtime: AnalysisRuntime, sample_df: pd.DataFrame) -> None:
        runtime.capture_result(task_id="step_cols", value=sample_df)
        a = runtime.registry.get("artifact_step_cols")
        assert a is not None
        assert a.columns == list(sample_df.columns)

    def test_dataframe_dtypes_captured(self, runtime: AnalysisRuntime, sample_df: pd.DataFrame) -> None:
        runtime.capture_result(task_id="step_dtypes", value=sample_df)
        a = runtime.registry.get("artifact_step_dtypes")
        assert a is not None
        assert a.dtypes is not None
        assert "revenue" in a.dtypes

    def test_non_dataframe_has_no_shape(self, runtime: AnalysisRuntime) -> None:
        runtime.capture_result(task_id="step_scalar", value=42)
        a = runtime.registry.get("artifact_step_scalar")
        assert a is not None
        assert a.shape is None
        assert a.columns is None

    def test_creation_order_increments(self, runtime: AnalysisRuntime, sample_df: pd.DataFrame) -> None:
        runtime.capture_result(task_id="step_a", value=sample_df)
        runtime.capture_result(task_id="step_b", value=True)
        a = runtime.registry.get("artifact_step_a")
        b = runtime.registry.get("artifact_step_b")
        assert a is not None and b is not None
        assert b.creation_order > a.creation_order

    def test_variable_name_stored(self, runtime: AnalysisRuntime, sample_df: pd.DataFrame) -> None:
        runtime.capture_result(task_id="step_named", value=sample_df, variable_name="quarter_df")
        a = runtime.registry.get("artifact_step_named")
        assert a is not None
        assert a.variable_name == "quarter_df"

    def test_parent_ids_stored(self, runtime: AnalysisRuntime, sample_df: pd.DataFrame) -> None:
        runtime.capture_result(task_id="step_parent", value=sample_df)
        runtime.capture_result(
            task_id="step_child",
            value=sample_df,
            parent_ids=["artifact_step_parent"],
        )
        child = runtime.registry.get("artifact_step_child")
        assert child is not None
        assert "artifact_step_parent" in child.parent_ids


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------


class TestDescribe:
    def test_describe_contains_df_shape(self, runtime: AnalysisRuntime, sample_df: pd.DataFrame) -> None:
        desc = runtime.describe()
        assert "df" in desc
        assert str(sample_df.shape) in desc

    def test_describe_contains_artifact_info(self, runtime: AnalysisRuntime, sample_df: pd.DataFrame) -> None:
        runtime.capture_result(task_id="step_1", value=sample_df)
        desc = runtime.describe()
        assert "artifact_step_1" in desc

    def test_describe_does_not_dump_full_data(self, runtime: AnalysisRuntime, sample_df: pd.DataFrame) -> None:
        """describe() must not include raw cell values."""
        runtime.capture_result(task_id="step_1", value=sample_df)
        desc = runtime.describe()
        # The string 'North' is a cell value — it must NOT appear in describe()
        assert "North" not in desc
        assert "South" not in desc


# ---------------------------------------------------------------------------
# namespace_snapshot() and update_namespace()
# ---------------------------------------------------------------------------


class TestNamespace:
    def test_snapshot_is_a_copy(self, runtime: AnalysisRuntime) -> None:
        snap = runtime.namespace_snapshot()
        snap["injected"] = "value"
        assert "injected" not in runtime.namespace

    def test_snapshot_contains_fundamentals(self, runtime: AnalysisRuntime, sample_df: pd.DataFrame) -> None:
        snap = runtime.namespace_snapshot()
        assert snap["df"] is sample_df
        assert snap["pd"] is pd
        assert snap["np"] is np

    def test_update_namespace_merges(self, runtime: AnalysisRuntime) -> None:
        runtime.update_namespace({"my_result": 42})
        assert runtime.namespace["my_result"] == 42

    def test_update_namespace_protects_df(self, runtime: AnalysisRuntime, sample_df: pd.DataFrame) -> None:
        other_df = pd.DataFrame({"x": [1, 2]})
        runtime.update_namespace({"df": other_df})
        # Protected name — must not be overwritten
        assert runtime.namespace["df"] is sample_df

    def test_update_namespace_protects_pd(self, runtime: AnalysisRuntime) -> None:
        runtime.update_namespace({"pd": "fake"})
        assert runtime.namespace["pd"] is pd

    def test_update_namespace_skips_private(self, runtime: AnalysisRuntime) -> None:
        runtime.update_namespace({"_hidden": "secret"})
        assert "_hidden" not in runtime.namespace


# ---------------------------------------------------------------------------
# ArtifactRegistry direct tests
# ---------------------------------------------------------------------------


class TestArtifactRegistryDirect:
    def test_list_in_creation_order(self) -> None:
        reg = ArtifactRegistry()
        df = pd.DataFrame({"x": [1]})
        reg.register("a1", "task_1", df)
        reg.register("a2", "task_2", True)
        ids = reg.list()
        assert ids == ["a1", "a2"]

    def test_remove_existing(self) -> None:
        reg = ArtifactRegistry()
        reg.register("a1", "task_1", 42)
        assert reg.remove("a1") is True
        assert not reg.exists("a1")

    def test_remove_nonexistent(self) -> None:
        reg = ArtifactRegistry()
        assert reg.remove("ghost") is False

    def test_len(self) -> None:
        reg = ArtifactRegistry()
        assert len(reg) == 0
        reg.register("a1", "t1", 1)
        assert len(reg) == 1
        reg.register("a2", "t2", 2)
        assert len(reg) == 2

    def test_contains(self) -> None:
        reg = ArtifactRegistry()
        reg.register("a1", "t1", "hello")
        assert "a1" in reg
        assert "ghost" not in reg
