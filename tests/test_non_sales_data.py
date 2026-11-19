"""
The pipeline must not assume the synthetic sales dataset.

- No "last quarter" machinery unless the goal asks for a quarter AND a datetime
  column exists (detected by dtype/name, not only a column called ``date``).
- Planner examples come from the capabilities that apply to the columns, or
  from the dataset's own columns.
- An optional data dictionary reaches the plan, code-gen, repair, critique and
  summary prompts.
- The observer runs generic checks on any dataset.
- ``main.py --data/--describe`` loads CSV/Parquet and a dictionary.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from agent.artifact_store import ArtifactStore
from agent.capabilities import build_default_capability_registry
from agent.dates import (
    TimeScope,
    detect_date_column,
    goal_mentions_quarter,
    goal_mentions_time_window,
    last_quarter_window,
)
from agent.execution_engine import ExecutionEngine
from agent.inputs import (
    MAX_DICTIONARY_CHARS,
    format_data_dictionary,
    load_data_dictionary,
    load_dataset,
)
from agent.loop import run
from agent.observer import observe, observe_dataset
from agent.planning import plan_guidance
from agent.prompts import (
    codegen_prompt,
    critique_prompt,
    plan_prompt,
    repair_prompt,
    replan_prompt,
    summary_prompt,
)
from agent.schema import build_schema_summary
from agent.state import TaskNode
from data.generate_data import generate_sales_data

ACCEPT = json.dumps({"trust": "high", "action": "accept", "reason": "ok"})
REVISE = json.dumps({"trust": "low", "action": "revise_code", "reason": "fix it"})
SALES_GOAL = "Explain the revenue trend by region over the last quarter."
TRIP_GOAL = "Which payment types have the highest average tip?"
DICTIONARY = {"payment_type": "1 = Credit card, 2 = Cash", "tip_amount": "Cash tips are not recorded."}


@pytest.fixture()
def trips() -> pd.DataFrame:
    """A small taxi-like dataset: datetime columns not called 'date', coded ints."""
    rng = np.random.default_rng(1)
    n = 300
    pickup = pd.Timestamp("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 31 * 24 * 60, n), unit="min"
    )
    df = pd.DataFrame(
        {
            "VendorID": rng.choice([1, 2], n),
            "tpep_dropoff_datetime": pickup + pd.Timedelta(minutes=15),
            "tpep_pickup_datetime": pickup,
            "payment_type": rng.choice([0, 1, 2, 3], n),
            "trip_distance": rng.gamma(2.0, 1.5, n).round(2),
            "fare_amount": rng.uniform(5, 60, n).round(2),
            "tip_amount": rng.uniform(0, 10, n).round(2),
            "store_and_fwd_flag": rng.choice(["Y", "N"], n),
        }
    )
    return df


@pytest.fixture()
def sales() -> pd.DataFrame:
    return generate_sales_data(n_rows=200, seed=7)


# ---------------------------------------------------------------------------
# Date detection and goal time scope
# ---------------------------------------------------------------------------


def test_detect_date_column_prefers_date_then_start_datetime(trips, sales) -> None:
    assert detect_date_column(sales) == "date"
    # Pickup beats dropoff even though dropoff comes first.
    assert detect_date_column(trips) == "tpep_pickup_datetime"


def test_detect_date_column_parses_date_like_strings() -> None:
    df = pd.DataFrame({"x": [1, 2], "created_at": ["2024-01-01", "2024-02-03"]})
    assert detect_date_column(df) == "created_at"
    # A string column without a date-like name is not guessed at.
    assert detect_date_column(pd.DataFrame({"code": ["2024-01-01"]})) is None
    assert detect_date_column(pd.DataFrame({"x": [1]})) is None


def test_last_quarter_window_uses_detected_column(trips) -> None:
    start, end = last_quarter_window(trips)
    assert start == pd.Timestamp("2023-11-01")
    assert end == trips["tpep_pickup_datetime"].max()


@pytest.mark.parametrize(
    "goal,quarter,window",
    [
        ("Explain revenue over the last quarter", True, True),
        ("Compare Q3 with Q4", True, True),
        ("How did tips change month-over-month?", False, True),
        ("What happened in 2023?", False, True),
        ("Which payment types tip the most?", False, False),
        ("Which factors may drive tips?", False, False),  # "may" is not a month
    ],
)
def test_goal_time_detection(goal, quarter, window) -> None:
    assert goal_mentions_quarter(goal) is quarter
    assert goal_mentions_time_window(goal) is window


def test_time_scope_requires_a_date_column() -> None:
    no_dates = pd.DataFrame({"x": [1, 2]})
    assert TimeScope.from_goal("last quarter", no_dates) == TimeScope()


# ---------------------------------------------------------------------------
# Schema summary
# ---------------------------------------------------------------------------


def test_schema_has_no_quarter_block_unless_goal_asks(trips) -> None:
    plain = build_schema_summary(trips, TimeScope.from_goal(TRIP_GOAL, trips))
    assert "LAST_QUARTER" not in plain
    assert "last quarter" not in plain
    assert "Main datetime column: 'tpep_pickup_datetime'" in plain

    assert "LAST_QUARTER" not in build_schema_summary(trips)  # no scope at all


def test_schema_quarter_block_uses_detected_column(trips) -> None:
    text = build_schema_summary(trips, TimeScope.from_goal("tips last quarter", trips))
    assert "LAST_QUARTER_START = pd.Timestamp('2023-11-01')" in text
    assert "df['tpep_pickup_datetime']" in text


def test_schema_window_note_for_other_time_windows(trips) -> None:
    text = build_schema_summary(trips, TimeScope.from_goal("tips by week", trips))
    assert "LAST_QUARTER" not in text
    assert "Time window note" in text


def test_sales_schema_unchanged_for_quarter_goal(sales) -> None:
    text = build_schema_summary(sales, TimeScope.from_goal(SALES_GOAL, sales))
    assert "PRE-COMPUTED DATE WINDOW" in text
    assert "(pd.to_datetime(df['date']) >= LAST_QUARTER_START)" in text


# ---------------------------------------------------------------------------
# Planner guidance
# ---------------------------------------------------------------------------


def test_sales_quarter_goal_keeps_capability_chain_example(sales) -> None:
    scope = TimeScope.from_goal(SALES_GOAL, sales)
    caps, examples = plan_guidance(sales, build_default_capability_registry(), scope)
    tasks = json.loads(examples)
    assert [t["capability"] for t in tasks] == [
        "FilterByDateRange",
        "GroupRevenueByRegion",
        "FindMaximumRevenueRegion",
    ]
    assert tasks[0]["produces_artifact"] == "quarter_df"
    assert tasks[0]["output_schema"] == list(sales.columns)
    assert "FilterByDateRange" in caps


def test_sales_goal_without_quarter_lists_no_date_filter(sales) -> None:
    scope = TimeScope.from_goal("Which region sells the most units?", sales)
    caps, examples = plan_guidance(sales, build_default_capability_registry(), scope)
    assert "FilterByDateRange" not in caps
    assert "quarter" not in examples


def test_non_sales_plan_prompt_has_no_sales_or_quarter_assumptions(trips) -> None:
    scope = TimeScope.from_goal(TRIP_GOAL, trips)
    caps, examples = plan_guidance(trips, build_default_capability_registry(), scope)
    assert caps == ""  # no sales capability applies to these columns
    tasks = json.loads(examples)
    assert all(t["capability"] is None for t in tasks)
    used = {c for t in tasks for c in t["output_schema"]}
    assert used & set(trips.columns)  # built from the dataset's own columns

    prompt = plan_prompt(
        TRIP_GOAL,
        build_schema_summary(trips, scope),
        capabilities=caps,
        examples=examples,
        time_scoped=scope.quarter,
    )
    for banned in ("quarter", "revenue", "region", "FilterByDateRange"):
        assert banned not in prompt


def test_prompts_include_quarter_rules_only_when_time_scoped() -> None:
    def all_prompts(scoped: bool) -> list[str]:
        return [
            plan_prompt("g", "s", time_scoped=scoped),
            codegen_prompt("g", "q", "s", time_scoped=scoped),
            repair_prompt("g", "q", "s", "", "", "code", "err", time_scoped=scoped),
            critique_prompt("g", "q", "code", {"success": True}, "", time_scoped=scoped),
        ]

    for prompt in all_prompts(False):
        assert "LAST_QUARTER" not in prompt
        assert "quarter_df" not in prompt
    for prompt in all_prompts(True)[1:]:
        assert "LAST_QUARTER" in prompt
    assert "last-quarter window" in all_prompts(True)[0]
    # Default plan examples carry no dataset-specific scope either.
    assert "quarter" not in replan_prompt("g", "s", "(none)", "reason")


# ---------------------------------------------------------------------------
# Data dictionary
# ---------------------------------------------------------------------------


def test_format_data_dictionary() -> None:
    assert format_data_dictionary(None) == ""
    assert format_data_dictionary("  text  ") == "text"
    text = format_data_dictionary({"a": "first", "b": {"1": "x"}})
    assert text == '- a: first\n- b: {"1": "x"}'
    long = format_data_dictionary("x" * (MAX_DICTIONARY_CHARS + 50))
    assert long.endswith("(data dictionary truncated)")


def test_load_data_dictionary_json_and_text(tmp_path) -> None:
    json_path = tmp_path / "dict.json"
    json_path.write_text(json.dumps(DICTIONARY))
    assert "- payment_type: 1 = Credit card, 2 = Cash" in load_data_dictionary(json_path)

    md_path = tmp_path / "dict.md"
    md_path.write_text("payment_type: 2 = cash\n")
    assert load_data_dictionary(md_path) == "payment_type: 2 = cash"

    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]")
    with pytest.raises(ValueError):
        load_data_dictionary(bad)


def test_prompts_without_dictionary_forbid_guessing_codes() -> None:
    for prompt in (plan_prompt("g", "s"), summary_prompt("g", [])):
        assert "do not assign meanings to integer codes" in prompt
        assert "DATA DICTIONARY" not in prompt


def test_data_dictionary_reaches_every_llm_stage(trips, tmp_path) -> None:
    plan = json.dumps(
        [{"question": "Average tip by payment_type", "capability": None,
          "produces_artifact": None, "consumes_artifacts": [], "output_schema": ["<scalar>"]}]
    )
    prompts: dict[str, list[str]] = {}
    critiques = iter([REVISE, ACCEPT])  # one revise so REPAIR runs too

    def llm(prompt: str, role: str, *a, **k) -> str:
        prompts.setdefault(role, []).append(prompt)
        if role == "plan":
            return plan
        if role == "codegen":
            return "result = df.groupby('payment_type')['tip_amount'].mean()"
        if role == "critique":
            return next(critiques)
        return "Summary."

    state = run(TRIP_GOAL, trips, llm=llm, log_dir=str(tmp_path), data_dictionary=DICTIONARY)

    assert not state.aborted
    assert len(prompts["codegen"]) == 2  # CODE-GEN then REPAIR
    for role in ("plan", "codegen", "critique", "observe"):
        for prompt in prompts[role]:
            assert "1 = Credit card, 2 = Cash" in prompt, role
            assert "Cash tips are not recorded." in prompt, role


# ---------------------------------------------------------------------------
# A full run on non-sales data
# ---------------------------------------------------------------------------


def test_run_without_time_goal_has_no_quarter_anywhere(trips, tmp_path) -> None:
    plan = json.dumps(
        [{"question": "Filter out trips with zero distance", "capability": None,
          "produces_artifact": "valid_trips", "consumes_artifacts": [],
          "output_schema": ["same_as_input"]}]
    )
    prompts: list[str] = []
    code = "valid_trips = df[df['trip_distance'] > 0]\nresult = valid_trips"

    def llm(prompt: str, role: str, *a, **k) -> str:
        prompts.append(prompt)
        return {"plan": plan, "codegen": code, "critique": ACCEPT}.get(role, "Summary.")

    state = run(TRIP_GOAL, trips, llm=llm, log_dir=str(tmp_path))

    assert not state.aborted
    assert not state.scope.quarter
    # The "filter" task went to CODE-GEN, not the last-quarter capability.
    assert state.debug_history[0].code == code
    for prompt in prompts:
        assert "LAST_QUARTER" not in prompt
        assert "quarter_df" not in prompt


def test_filter_capability_refuses_tasks_that_are_not_last_quarter(trips) -> None:
    task = TaskNode(
        task_id="step_1",
        question="Filter out trips with zero distance",
        capability="FilterByDateRange",
        produces_artifact="valid_trips",
    )
    assert ExecutionEngine().execute(task, trips, ArtifactStore()) is None

    quarter_task = TaskNode(
        task_id="step_1",
        question="Filter to the last quarter",
        capability="FilterByDateRange",
        produces_artifact="quarter_df",
    )
    result = ExecutionEngine().execute(quarter_task, trips, ArtifactStore())
    assert result is not None  # works with a datetime column not named 'date'
    assert len(result["artifact"]) == len(trips)


# ---------------------------------------------------------------------------
# Generic observer checks
# ---------------------------------------------------------------------------


def test_observer_generic_checks(trips) -> None:
    df = trips.copy()
    df.loc[0, "fare_amount"] = -12.5
    df.loc[1, "tpep_pickup_datetime"] = pd.Timestamp("2002-12-31 23:00")
    df.loc[2, "trip_distance"] = 5000.0
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)  # one duplicate row

    report = observe(df, {"stdout": ""})

    assert report.duplicate_rows == 1
    assert report.negative_values == {"fare_amount": 1}
    assert report.datetime_out_of_range["tpep_pickup_datetime"]["count"] == 1
    assert "tpep_dropoff_datetime" not in report.datetime_out_of_range
    assert report.numeric_outliers_iqr.get("trip_distance", 0) >= 1
    # Code-like columns are not IQR-checked.
    assert "payment_type" not in report.numeric_outliers_iqr
    assert "VendorID" not in report.numeric_outliers_iqr

    text = report.as_critique_text()
    assert "DUPLICATE ROWS: 1" in text
    assert "fare_amount: 1 row(s)" in text
    assert "DATETIMES OUTSIDE THE DATA'S MAIN RANGE" in text
    assert "NUMERIC OUTLIERS" in text


def test_observer_sales_checks_still_work_alongside_generic(sales) -> None:
    df = sales.copy()
    df.loc[0, "revenue"] = -5.0
    report = observe(df, {"stdout": ""})
    assert report.negative_revenue_rows == 1
    assert report.negative_values["revenue"] == 1
    # Revenue is reported once, under its sales-specific heading.
    text = report.as_critique_text()
    assert "NEGATIVE REVENUE: 1" in text
    assert "  revenue: 1 row(s)" not in text


def test_observer_reuses_baseline_without_mutating_it(trips) -> None:
    baseline = observe_dataset(trips)
    first = observe(trips, {"stdout": "a\nb"}, ["x"], baseline=baseline)
    observe(trips, {"stdout": "c"}, ["y"], baseline=baseline)
    assert baseline.notes == []
    assert first.rows_after_code_filter == 2
    assert first.null_counts == baseline.null_counts


# ---------------------------------------------------------------------------
# Loading data and the CLI
# ---------------------------------------------------------------------------


def test_load_dataset_csv_parses_date_like_columns(tmp_path, trips) -> None:
    path = tmp_path / "trips.csv"
    trips.to_csv(path, index=False)
    loaded = load_dataset(path)
    assert pd.api.types.is_datetime64_any_dtype(loaded["tpep_pickup_datetime"])
    assert not pd.api.types.is_datetime64_any_dtype(loaded["store_and_fwd_flag"])
    with pytest.raises(ValueError):
        load_dataset(tmp_path / "trips.xlsx")


def test_load_dataset_parquet(tmp_path, trips) -> None:
    pytest.importorskip("pyarrow")
    path = tmp_path / "trips.parquet"
    trips.to_parquet(path)
    pd.testing.assert_frame_equal(load_dataset(path), trips)


def test_main_data_and_describe_flags(tmp_path, trips, monkeypatch) -> None:
    import main

    data = tmp_path / "trips.csv"
    trips.to_csv(data, index=False)
    describe = tmp_path / "dict.json"
    describe.write_text(json.dumps(DICTIONARY))
    seen = {}

    def fake_run(goal, df, dry_run=False, **kwargs):
        seen.update(goal=goal, df=df, dry_run=dry_run, **kwargs)

        class State:
            run_id = "test"
            debug_history: list = []

        return State()

    monkeypatch.setattr(main, "run", fake_run)
    monkeypatch.setattr(
        sys, "argv", ["main.py", "--data", str(data), "--describe", str(describe), "--dry-run"]
    )
    main.main()

    assert seen["goal"] == main.DATA_GOAL
    assert "quarter" not in seen["goal"].lower()
    assert len(seen["df"]) == len(trips)
    assert "1 = Credit card" in seen["data_dictionary"]
    assert seen["dataset_label"] == "trips"

    monkeypatch.setattr(sys, "argv", ["main.py", "--data", str(data), "--corrupt"])
    with pytest.raises(SystemExit):
        main.main()
