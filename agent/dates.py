"""Date-window helpers shared by capabilities and the schema summary."""

from __future__ import annotations

import pandas as pd


def last_quarter_window(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """
    Return the inclusive ``(start, end)`` of the "last quarter" in *df*.

    The window is the 3 most recent calendar months in the data: ``end`` is the
    latest date, ``start`` is the first day of the month two months earlier.
    Returns ``None`` when there is no usable ``date`` column.
    """
    if "date" not in df.columns:
        return None
    date_series = pd.to_datetime(df["date"], errors="coerce").dropna()
    if date_series.empty:
        return None
    end_ts = date_series.max()
    start_ts = (end_ts - pd.DateOffset(months=2)).replace(day=1)
    return start_ts, end_ts


def last_quarter_constants(df: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """
    Return ``LAST_QUARTER_START`` / ``LAST_QUARTER_END`` for the sandbox namespace.

    The values are day-precision, matching the ``YYYY-MM-DD`` strings the schema
    summary shows the model, so code that uses the names and code that copies
    the printed literals filter the same rows. Empty when there is no window.
    """
    window = last_quarter_window(df)
    if window is None:
        return {}
    start_ts, end_ts = window
    return {
        "LAST_QUARTER_START": start_ts.normalize(),
        "LAST_QUARTER_END": end_ts.normalize(),
    }
