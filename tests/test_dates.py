"""Tests for agent.dates.last_quarter_window."""

from __future__ import annotations

import pandas as pd

from agent.dates import last_quarter_window


def test_window_spans_three_calendar_months() -> None:
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-05", "2024-06-15"])})
    start, end = last_quarter_window(df)
    assert start == pd.Timestamp("2024-04-01")
    assert end == pd.Timestamp("2024-06-15")


def test_window_none_without_date_column() -> None:
    assert last_quarter_window(pd.DataFrame({"x": [1]})) is None


def test_window_none_when_dates_unparseable_or_empty() -> None:
    assert last_quarter_window(pd.DataFrame({"date": ["nope", None]})) is None
    assert last_quarter_window(pd.DataFrame({"date": []})) is None


def test_window_accepts_string_dates() -> None:
    df = pd.DataFrame({"date": ["2024-03-31", "2024-03-01"]})
    start, end = last_quarter_window(df)
    assert start == pd.Timestamp("2024-01-01")
    assert end == pd.Timestamp("2024-03-31")
