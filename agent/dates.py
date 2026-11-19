"""Date helpers: datetime-column detection, goal time-scope detection, and the
"last quarter" window shared by capabilities and the schema summary."""

from __future__ import annotations

import re
from dataclasses import dataclass
import warnings

import pandas as pd

# Token used in a capability's input_schema for "the detected datetime column".
DATE_COLUMN_TOKEN = "date"

# Name fragments that suggest a column holds dates/timestamps.
_DATE_NAME_HINTS = ("date", "time", "timestamp", "datetime", "day", "_at", "_dt")
# Fraction of sampled non-null values that must parse for an object column to count.
_PARSE_THRESHOLD = 0.9
_SAMPLE_SIZE = 1000

# The goal asks for a calendar quarter (the LAST_QUARTER_* constants apply).
_QUARTER_RE = re.compile(r"\bquarter(ly|s)?\b|\bq[1-4]\b", re.IGNORECASE)

# The goal names some other time window or time dimension.
_TIME_WINDOW_RE = re.compile(
    r"\b("
    r"quarter(ly|s)?|q[1-4]|month(ly|s)?|week(ly|s)?|year(ly|s)?|annual(ly)?|"
    r"daily|days?|hour(ly|s)?|season(al|ality)?|ytd|mtd|"
    r"over time|time window|time period|date range|"
    r"month-over-month|year-over-year|week-over-week|yoy|"
    # "May"/"March" are left out as words: too often a verb.
    r"january|february|april|june|july|august|september|october|"
    r"november|december|jan|feb|apr|jun|jul|aug|sep|sept|oct|nov|dec|"
    r"(19|20)\d{2}"
    r")\b",
    re.IGNORECASE,
)


def goal_mentions_quarter(goal: str | None) -> bool:
    """True when *goal* asks about a quarter ("last quarter", "Q3", "quarterly")."""
    return bool(goal and _QUARTER_RE.search(goal))


def goal_mentions_time_window(goal: str | None) -> bool:
    """True when *goal* names a time window or time dimension (quarter, month,
    a year, "over time", ...)."""
    return bool(goal and _TIME_WINDOW_RE.search(goal))


def looks_like_date_name(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in _DATE_NAME_HINTS)


def parses_as_dates(series: pd.Series) -> bool:
    sample = series.dropna()
    if sample.empty:
        return False
    sample = sample.head(_SAMPLE_SIZE)
    if not (pd.api.types.is_object_dtype(sample) or pd.api.types.is_string_dtype(sample)):
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() >= _PARSE_THRESHOLD


def detect_date_column(df: pd.DataFrame) -> str | None:
    """
    Return the name of the main datetime column in *df*, or ``None``.

    Order of preference:
    1. A column literally named ``date`` that parses as dates (the sales data).
    2. A datetime64 column; among several, one whose name looks date-like
       (e.g. ``tpep_pickup_datetime`` before ``dropoff``), else the first.
    3. A string column with a date-like name whose values parse as dates.
    """
    if "date" in df.columns and (
        pd.api.types.is_datetime64_any_dtype(df["date"]) or parses_as_dates(df["date"])
    ):
        return "date"

    datetime_cols = [
        c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    if datetime_cols:
        named = [c for c in datetime_cols if looks_like_date_name(str(c))]
        # Prefer "start" style columns (pickup, start, created) over "end" ones.
        for col in named:
            lowered = str(col).lower()
            if not any(t in lowered for t in ("dropoff", "end", "finish", "closed")):
                return col
        return (named or datetime_cols)[0]

    for col in df.columns:
        if looks_like_date_name(str(col)) and parses_as_dates(df[col]):
            return col
    return None


def date_range(df: pd.DataFrame, date_col: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Return ``(min, max)`` of *date_col*, or ``None`` when nothing parses."""
    series = df[date_col]
    if not pd.api.types.is_datetime64_any_dtype(series):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            series = pd.to_datetime(series, errors="coerce")
    series = series.dropna()
    if series.empty:
        return None
    return series.min(), series.max()


def last_quarter_window(
    df: pd.DataFrame, date_col: str | None = None
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """
    Return the inclusive ``(start, end)`` of the "last quarter" in *df*.

    The window is the 3 most recent calendar months in the data: ``end`` is the
    latest date, ``start`` is the first day of the month two months earlier.
    *date_col* defaults to :func:`detect_date_column`. Returns ``None`` when
    there is no usable datetime column.
    """
    if date_col is None:
        date_col = detect_date_column(df)
    if date_col is None or date_col not in df.columns:
        return None
    bounds = date_range(df, date_col)
    if bounds is None:
        return None
    end_ts = bounds[1]
    start_ts = (end_ts - pd.DateOffset(months=2)).replace(day=1)
    start_ts = start_ts.normalize()
    return start_ts, end_ts


@dataclass(frozen=True)
class TimeScope:
    """
    What time machinery a run needs, decided once from the goal and the data.

    Attributes:
        date_column: Detected datetime column, or ``None``.
        quarter:     The goal asks about a quarter AND a datetime column exists:
                     the LAST_QUARTER_* constants and ``quarter_df`` rules apply.
        window:      The goal names any time window or time dimension AND a
                     datetime column exists (``quarter`` implies ``window``).
    """

    date_column: str | None = None
    quarter: bool = False
    window: bool = False

    @classmethod
    def from_goal(cls, goal: str | None, df: pd.DataFrame) -> "TimeScope":
        date_col = detect_date_column(df)
        if date_col is None:
            return cls()
        quarter = goal_mentions_quarter(goal)
        return cls(
            date_column=date_col,
            quarter=quarter,
            window=quarter or goal_mentions_time_window(goal),
        )


def last_quarter_constants(
    df: pd.DataFrame, date_col: str | None = None
) -> dict[str, pd.Timestamp]:
    """
    Return ``LAST_QUARTER_START`` / ``LAST_QUARTER_END`` for the sandbox namespace.

    The values match the literals the schema summary shows the model (see
    :func:`agent.schema.build_schema_summary`), so code that uses the names and
    code that copies the printed literals filter the same rows: ``START`` is
    midnight of the first day; ``END`` is the latest timestamp in the data
    (midnight for date-only data, the full time when the data has times, so
    the last day's rows stay in). Empty when there is no window.
    """
    window = last_quarter_window(df, date_col)
    if window is None:
        return {}
    start_ts, end_ts = window
    return {"LAST_QUARTER_START": start_ts, "LAST_QUARTER_END": end_ts}
