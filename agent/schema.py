"""Schema summary builder for LLM prompts."""

from __future__ import annotations

import pandas as pd

from agent.dates import TimeScope, date_range, last_quarter_window


def build_schema_summary(df: pd.DataFrame, scope: TimeScope | None = None) -> str:
    """
    Build a concise schema description for use in LLM prompts.

    Includes column names, dtypes, null percentages, and 3 sample values.
    Does NOT include full row data.

    Time information depends on *scope* (see :class:`agent.dates.TimeScope`):

    - a detected datetime column is always named, with its min/max;
    - only when ``scope.quarter`` is set (the goal asks about a quarter and a
      datetime column exists) is the "last quarter" window computed here in
      Python and injected as concrete constants (LAST_QUARTER_START / END), so
      the LLM never derives the boundaries itself;
    - when ``scope.window`` is set for another window ("last month", "2023"),
      a short note says to anchor it on the data's max date, not today.

    With no *scope* (or an empty one) no window machinery is added.

    Args:
        df:    The DataFrame to describe.
        scope: Time scope of the run; ``None`` means no time window requested.

    Returns:
        Multi-line schema string.
    """
    scope = scope or TimeScope()
    lines = [f"Rows: {len(df):,}  |  Columns: {len(df.columns)}"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_pct = df[col].isna().mean() * 100
        samples = df[col].dropna().head(3).tolist()
        sample_str = ", ".join(repr(v) for v in samples)
        lines.append(f"  • {col!r} ({dtype}) nulls={null_pct:.1f}% — e.g. {sample_str}")

    date_col = scope.date_column
    if date_col is not None and date_col in df.columns:
        bounds = date_range(df, date_col)
        if bounds is not None:
            lines.append("")
            lines.append(
                f"Main datetime column: {date_col!r} — spans "
                f"{bounds[0]:%Y-%m-%d} to {bounds[1]:%Y-%m-%d}."
            )

    if scope.quarter and date_col is not None:
        lines.append("")
        lines.append(_quarter_block(df, date_col))
    elif scope.window and date_col is not None:
        lines.append("")
        lines.append(
            "Time window note: if the goal names a time window, compute it from\n"
            f"df[{date_col!r}].max() (the latest date in the data), never from\n"
            "today's date."
        )
    return "\n".join(lines)


def _quarter_block(df: pd.DataFrame, date_col: str) -> str:
    """The PRE-COMPUTED DATE WINDOW block (or a fallback note if nothing parses)."""
    window = last_quarter_window(df, date_col)
    if window is None:
        return (
            'Time range note: "last quarter" means the most recent 3 calendar\n'
            f"months in the data. Use df[{date_col!r}].max() as reference, filter back\n"
            "2 full months to the first of that month, and do not use today's date."
        )
    start = window[0].strftime("%Y-%m-%d")
    # Keep the full timestamp when the data has a time of day, so the window
    # still includes the last day's rows.
    end = (
        window[1].strftime("%Y-%m-%d")
        if window[1] == window[1].normalize()
        else window[1].strftime("%Y-%m-%d %H:%M:%S")
    )
    return (
        "PRE-COMPUTED DATE WINDOW (do NOT re-derive these; copy them verbatim):\n"
        f"  LAST_QUARTER_START = pd.Timestamp('{start}')  # inclusive\n"
        f"  LAST_QUARTER_END   = pd.Timestamp('{end}')    # inclusive\n"
        "\n"
        "Use EXACTLY this filter when the goal mentions 'last quarter':\n"
        f"  mask = (pd.to_datetime(df['{date_col}']) >= LAST_QUARTER_START) & \\\n"
        f"         (pd.to_datetime(df['{date_col}']) <= LAST_QUARTER_END)\n"
        "  quarter_df = df[mask]\n"
        "\n"
        "Do NOT use pd.Timestamp('today'), datetime.date objects, or\n"
        "pd.DateOffset arithmetic in your code — the boundaries are already\n"
        "computed above. Mismatching these constants is a logic error."
    )
