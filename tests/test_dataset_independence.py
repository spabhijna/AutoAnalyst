"""
Regression tests for running the pipeline on data that isn't the sales dataset.

- A capability whose input columns are missing, or whose executor raises, must
  fall back to CODE-GEN instead of crashing the run.
- The critique prompt must not contain figures from the default dataset.
- LAST_QUARTER_START / LAST_QUARTER_END, which the prompts tell the model to use,
  must be defined in the sandbox.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from agent.artifact_store import ArtifactStore
from agent.capabilities import Capability, CapabilityRegistry
from agent.dates import last_quarter_constants
from agent.execution_engine import ExecutionEngine
from agent.loop import run
from agent.prompts import critique_prompt
from agent.sandbox import run_sandboxed
from agent.state import TaskNode

ACCEPT = json.dumps({"trust": "high", "action": "accept", "reason": "ok"})


@pytest.fixture()
def store_df() -> pd.DataFrame:
    """A non-sales dataset: no region, revenue or product_category columns."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=200, freq="D"),
            "store": rng.choice(["A", "B", "C"], 200),
            "sales": rng.uniform(10, 100, 200),
        }
    )


# ---------------------------------------------------------------------------
# Capabilities on data without their columns
# ---------------------------------------------------------------------------


def test_engine_skips_capability_when_input_columns_missing(store_df) -> None:
    store = ArtifactStore()
    store.register("quarter_df", store_df, producing_task_id="step_1")
    task = TaskNode(
        task_id="step_2",
        question="Total sales by store",
        capability="GroupRevenueByRegion",
        produces_artifact="region_revenue",
        consumes_artifacts=["quarter_df"],
    )

    assert ExecutionEngine().execute(task, store_df, store) is None


def test_engine_falls_back_when_executor_raises(store_df) -> None:
    def boom(context):
        raise KeyError("region")

    registry = CapabilityRegistry()
    registry.register(
        Capability(
            capability_id="Boom",
            description="always fails",
            required_artifacts=(),
            produced_artifacts=("out",),
            input_schema=(),
            output_schema=(),
            executor=boom,
        )
    )
    task = TaskNode(task_id="step_1", question="q", capability="Boom", produces_artifact="out")

    assert ExecutionEngine(registry=registry).execute(task, store_df, ArtifactStore()) is None


def test_run_on_non_sales_data_falls_back_to_codegen(store_df, tmp_path) -> None:
    plan = json.dumps(
        [
            {
                "question": "Filter to last quarter",
                "capability": "FilterByDateRange",
                "produces_artifact": "quarter_df",
                "consumes_artifacts": [],
                "output_schema": [],
            },
            {
                "question": "Total sales by store in the last quarter",
                "capability": "GroupRevenueByRegion",
                "produces_artifact": "store_sales",
                "consumes_artifacts": ["quarter_df"],
                "output_schema": ["store", "sales"],
            },
        ]
    )
    code = (
        "store_sales = artifacts['quarter_df'].groupby('store', as_index=False)['sales'].sum()\n"
        "result = store_sales"
    )

    def llm(prompt: str, role: str, *a, **k) -> str:
        return {"plan": plan, "codegen": code, "critique": ACCEPT}.get(role, "summary")

    state = run("goal", store_df, llm=llm, log_dir=str(tmp_path))

    assert not state.aborted
    assert state.debug_history[0].code == "(deterministic executor)"
    assert state.debug_history[1].code == code  # step 2 fell back to codegen
    assert "store_sales" in state.artifact_store.available_keys()


# ---------------------------------------------------------------------------
# No default-dataset figures in the critique prompt
# ---------------------------------------------------------------------------


def test_critique_prompt_has_no_default_dataset_figures() -> None:
    prompt = critique_prompt(
        "goal", "q", "result = 1", {"success": True, "result": "1"}, "(none)"
    )
    for figure in ("46384", "45317", "West"):
        assert figure not in prompt


# ---------------------------------------------------------------------------
# LAST_QUARTER constants in the sandbox
# ---------------------------------------------------------------------------


def test_last_quarter_constants_keep_the_last_timestamp() -> None:
    """START is midnight of the first day; END keeps the time of the latest row,
    so `<= LAST_QUARTER_END` still includes the last day's rows."""
    df = pd.DataFrame(
        {"date": [pd.Timestamp("2024-02-10"), pd.Timestamp("2024-04-15 13:45")]}
    )
    constants = last_quarter_constants(df)

    assert constants == {
        "LAST_QUARTER_START": pd.Timestamp("2024-02-01"),
        "LAST_QUARTER_END": pd.Timestamp("2024-04-15 13:45"),
    }
    assert (df["date"] <= constants["LAST_QUARTER_END"]).all()


def test_last_quarter_constants_empty_without_date_column() -> None:
    assert last_quarter_constants(pd.DataFrame({"x": [1, 2]})) == {}


def test_sandbox_defines_constants_but_never_overrides_core_globals(store_df) -> None:
    code = (
        "mask = (df['date'] >= LAST_QUARTER_START) & (df['date'] <= LAST_QUARTER_END)\n"
        "result = int(mask.sum())"
    )
    constants = {**last_quarter_constants(store_df), "df": "not a dataframe"}

    out = run_sandboxed(code, store_df, constants=constants, timeout_seconds=30)

    assert out["success"], out["error"]
    assert int(out["result"]) > 0  # df was not replaced by the bogus constant


def test_codegen_using_last_quarter_names_succeeds_first_try(store_df, tmp_path) -> None:
    plan = json.dumps(
        [{"question": "Rows in the last quarter", "capability": None,
          "produces_artifact": None, "consumes_artifacts": [], "output_schema": []}]
    )
    code = (
        "mask = (pd.to_datetime(df['date']) >= LAST_QUARTER_START) & "
        "(pd.to_datetime(df['date']) <= LAST_QUARTER_END)\n"
        "result = int(mask.sum())"
    )

    def llm(prompt: str, role: str, *a, **k) -> str:
        return {"plan": plan, "codegen": code, "critique": ACCEPT}.get(role, "summary")

    state = run("goal", store_df, llm=llm, log_dir=str(tmp_path))

    assert not state.aborted
    assert len(state.debug_history) == 1
    assert state.debug_history[0].sandbox_result["success"]
