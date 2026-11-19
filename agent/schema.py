"""Schema summary builder for LLM prompts."""

from __future__ import annotations

import pandas as pd

from agent.dates import last_quarter_window


def build_schema_summary(df: pd.DataFrame) -> str:
    """
    Build a concise schema description for use in LLM prompts.

    Includes column names, dtypes, null percentages, and 3 sample values.
    Does NOT include full row data.

    The "last quarter" date window is computed here in Python and injected as
    concrete constants (LAST_QUARTER_START / LAST_QUARTER_END) so the LLM
    never has to derive date boundaries itself — eliminating the entire class
    of off-by-one critique loops that arise when the model re-derives the
    window independently on every attempt.

    Args:
        df: The DataFrame to describe.

    Returns:
        Multi-line schema string.
    """
    lines = [f"Rows: {len(df):,}  |  Columns: {len(df.columns)}"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_pct = df[col].isna().mean() * 100
        samples = df[col].dropna().head(3).tolist()
        sample_str = ", ".join(repr(v) for v in samples)
        lines.append(f"  • {col!r} ({dtype}) nulls={null_pct:.1f}% — e.g. {sample_str}")

    # ── Pre-compute the "last quarter" window in Python so LLMs never have to ──
    # "Last quarter" = the 3 most-recent complete calendar months in the data.
    # Example: if max date is 2024-06-15, the window is 2024-04-01 → 2024-06-15.
    window = last_quarter_window(df)
    last_quarter_start = window[0].strftime("%Y-%m-%d") if window else None
    last_quarter_end = window[1].strftime("%Y-%m-%d") if window else None

    lines.append("")
    if last_quarter_start and last_quarter_end:
        lines.append(
            "PRE-COMPUTED DATE WINDOW (do NOT re-derive these; copy them verbatim):\n"
            f"  LAST_QUARTER_START = pd.Timestamp('{last_quarter_start}')  # inclusive\n"
            f"  LAST_QUARTER_END   = pd.Timestamp('{last_quarter_end}')    # inclusive\n"
            "\n"
            "Use EXACTLY this filter when the goal mentions 'last quarter':\n"
            "  mask = (pd.to_datetime(df['date']) >= LAST_QUARTER_START) & \\\n"
            "         (pd.to_datetime(df['date']) <= LAST_QUARTER_END)\n"
            "  quarter_df = df[mask]\n"
            "\n"
            "Do NOT use pd.Timestamp('today'), datetime.date objects, or\n"
            "pd.DateOffset arithmetic in your code — the boundaries are already\n"
            "computed above. Mismatching these constants is a logic error."
        )
    else:
        lines.append(
            'Time range note: "last quarter" means the most recent 3 calendar\n'
            "months in the data. Use df['date'].max() as reference, filter back\n"
            "2 full months to the first of that month, and do not use today's date."
        )
    return "\n".join(lines)
