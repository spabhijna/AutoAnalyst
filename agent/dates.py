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
