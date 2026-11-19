"""
Deterministic data-quality observer that runs between EXECUTE and CRITIQUE.

The observer computes actual null counts, duplicate rates, outlier indices,
and logical-consistency checks on the live DataFrame — producing a structured
:class:`ObservationReport` that the critique prompt receives as hard facts
instead of relying on LLM speculation.

Design decisions
----------------
1. **Deterministic, not LLM-driven.**  Every check is a pure function of the
   DataFrame.  The critic receives computed numbers, not guesses.
2. **Operates on the original ``df``.**  The sandboxed code may have filtered
   or grouped the data, but the observer inspects the *input* DataFrame to
   catch quality issues the code might have silently propagated.
3. **Separate concern.**  The observer does not judge correctness — it only
   surfaces data-quality signals.  The critique LLM decides whether those
   signals imply a logic error.
4. **Generic checks plus sales checks.**  Duplicate rows, negative values,
   out-of-range datetimes and IQR outliers run on any dataset. The sales
   checks (``transaction_id`` duplicates, negative ``revenue``, zero
   ``units_sold`` with revenue, revenue outliers) stay as separate fields,
   because :mod:`agent.evaluator` scores them against the corruption manifest.
5. **Computed once per run.**  The dataset-level checks do not depend on the
   attempt: :func:`observe_dataset` computes them once and the loop passes
   that baseline to every :func:`observe` call, so retries stay cheap on a
   multi-million-row dataset.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# Numeric columns with at most this many distinct values are treated as codes
# (e.g. payment_type 1-6): IQR outliers mean nothing for them.
_CODE_MAX_UNIQUE = 10
# Rows sampled to decide whether a numeric column looks like a code.
_SAMPLE_ROWS = 100_000
# Datetimes before this are almost always placeholders / parse errors.
_MIN_PLAUSIBLE_DATE = pd.Timestamp("1900-01-01")
# Columns listed per generic check in the critique text.
_MAX_COLUMNS_SHOWN = 6


# ---------------------------------------------------------------------------
# ObservationReport dataclass
# ---------------------------------------------------------------------------


@dataclass
class ObservationReport:
    """
    Structured data-quality report for one sandbox execution.

    Every field is computed deterministically from the input DataFrame.
    The critique prompt receives this report as structured facts.
    """

    # Per-column null counts (absolute and percentage)
    null_counts: dict[str, int] = field(default_factory=dict)
    null_pct: dict[str, float] = field(default_factory=dict)

    # Transaction-ID duplicates in the *input* DataFrame
    duplicate_txn_ids: int = 0
    duplicate_txn_examples: list[str] = field(default_factory=list)

    # Impossible / suspicious values in the input DataFrame
    negative_revenue_rows: int = 0
    zero_units_positive_revenue: int = 0

    # Outlier indices detected via IQR on revenue
    revenue_outliers_iqr: list[int] = field(default_factory=list)

    # ── Generic checks (any dataset) ─────────────────────────────────────────
    # Fully duplicated rows (all columns equal), counting extra copies only.
    duplicate_rows: int = 0
    # Numeric column -> rows with a value < 0 (non-zero counts only).
    negative_values: dict[str, int] = field(default_factory=dict)
    # Datetime column -> {"count", "min", "max", "expected_start", "expected_end"}
    # for values outside the column's plausible range (non-zero counts only).
    datetime_out_of_range: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Numeric column -> rows outside 1.5 × IQR (code-like columns skipped).
    numeric_outliers_iqr: dict[str, int] = field(default_factory=dict)

    # Row counts for scope awareness
    total_rows: int = 0
    rows_after_code_filter: int = 0  # from sandbox stdout (head() prints)

    # Free-text observations (for non-quantifiable issues)
    notes: list[str] = field(default_factory=list)

    def as_critique_text(self) -> str:
        """
        Return a compact structured block for embedding in the critique prompt.

        Designed to be token-efficient: only non-trivial signals are included.
        When everything is clean the block is a single line confirming no issues.
        """
        lines: list[str] = []

        # ── Null summary ──────────────────────────────────────────────────────
        null_cols = {c: n for c, n in self.null_counts.items() if n > 0}
        if null_cols:
            lines.append("NULLS FOUND:")
            for col, count in sorted(null_cols.items(), key=lambda x: -x[1]):
                pct = self.null_pct.get(col, 0.0)
                lines.append(f"  {col}: {count} nulls ({pct:.1f}% of rows)")
        else:
            lines.append("Nulls: none detected in input DataFrame.")

        # ── Duplicate transaction IDs ─────────────────────────────────────────
        if self.duplicate_txn_ids > 0:
            examples = ", ".join(self.duplicate_txn_examples[:3])
            lines.append(
                f"DUPLICATE TRANSACTION IDs: {self.duplicate_txn_ids} "
                f"duplicate values found (e.g. {examples})."
            )
        else:
            lines.append("Duplicate transaction IDs: none detected.")

        # ── Negative revenue ──────────────────────────────────────────────────
        if self.negative_revenue_rows > 0:
            lines.append(
                f"NEGATIVE REVENUE: {self.negative_revenue_rows} row(s) "
                f"with revenue < 0."
            )

        # ── Zero units / positive revenue ────────────────────────────────────
        if self.zero_units_positive_revenue > 0:
            lines.append(
                f"LOGICAL INCONSISTENCY: {self.zero_units_positive_revenue} "
                f"row(s) with units_sold=0 AND revenue>0."
            )

        # ── Revenue outliers ──────────────────────────────────────────────────
        if self.revenue_outliers_iqr:
            lines.append(
                f"REVENUE OUTLIERS (IQR method): {len(self.revenue_outliers_iqr)} "
                f"row(s) at indices {self.revenue_outliers_iqr[:10]}{'…' if len(self.revenue_outliers_iqr) > 10 else ''}."
            )

        # ── Generic checks ────────────────────────────────────────────────────
        if self.duplicate_rows > 0:
            lines.append(
                f"DUPLICATE ROWS: {self.duplicate_rows:,} row(s) are exact copies "
                f"of an earlier row."
            )
        negatives = {c: n for c, n in self.negative_values.items() if c != "revenue"}
        if negatives:
            lines.append("NEGATIVE VALUES (check whether the column can be negative):")
            lines.extend(_top_counts(negatives))
        if self.datetime_out_of_range:
            lines.append("DATETIMES OUTSIDE THE DATA'S MAIN RANGE:")
            for col, info in list(self.datetime_out_of_range.items())[:_MAX_COLUMNS_SHOWN]:
                lines.append(
                    f"  {col}: {info['count']:,} row(s) outside "
                    f"{info['expected_start']} → {info['expected_end']} "
                    f"(min {info['min']}, max {info['max']})"
                )
        outliers = {c: n for c, n in self.numeric_outliers_iqr.items() if c != "revenue"}
        if outliers:
            lines.append("NUMERIC OUTLIERS (IQR method, 1.5 × IQR):")
            lines.extend(_top_counts(outliers, self.total_rows))

        # ── Scope ─────────────────────────────────────────────────────────────
        lines.append(f"Input rows: {self.total_rows:,}")
        if self.rows_after_code_filter:
            lines.append(f"Rows after code filter: {self.rows_after_code_filter:,}")

        # ── Notes ─────────────────────────────────────────────────────────────
        if self.notes:
            lines.append("ADDITIONAL OBSERVATIONS:")
            for note in self.notes:
                lines.append(f"  • {note}")

        return "\n".join(lines)


def _top_counts(counts: dict[str, int], total: int = 0) -> list[str]:
    """Format the largest per-column counts, one indented line each."""
    ordered = sorted(counts.items(), key=lambda item: -item[1])
    lines = []
    for col, n in ordered[:_MAX_COLUMNS_SHOWN]:
        pct = f" ({n / total * 100:.1f}%)" if total else ""
        lines.append(f"  {col}: {n:,} row(s){pct}")
    if len(ordered) > _MAX_COLUMNS_SHOWN:
        lines.append(f"  … and {len(ordered) - _MAX_COLUMNS_SHOWN} more column(s)")
    return lines


# ---------------------------------------------------------------------------
# Generic dataset-level checks
# ---------------------------------------------------------------------------


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    """Numeric, non-boolean columns."""
    return [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]


def count_duplicate_rows(df: pd.DataFrame) -> int:
    """Rows that exactly repeat an earlier row (0 if rows are unhashable)."""
    try:
        return int(df.duplicated().sum())
    except TypeError:  # unhashable cells (lists, dicts)
        return 0


def negative_value_counts(df: pd.DataFrame) -> dict[str, int]:
    """Numeric column -> number of values below zero (zero counts omitted)."""
    counts = {}
    for col in _numeric_columns(df):
        n = int((df[col] < 0).sum())
        if n:
            counts[str(col)] = n
    return counts


def numeric_outlier_counts(df: pd.DataFrame) -> dict[str, int]:
    """
    Numeric column -> rows outside ``[Q1 - 1.5·IQR, Q3 + 1.5·IQR]``.

    Skips code-like columns (at most ``_CODE_MAX_UNIQUE`` distinct values in a
    sample), columns with fewer than 4 values, and columns with zero IQR.
    """
    counts = {}
    sample = df.head(_SAMPLE_ROWS)
    for col in _numeric_columns(df):
        if sample[col].nunique() <= _CODE_MAX_UNIQUE:
            continue
        values = df[col].dropna()
        if len(values) < 4:
            continue
        q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            continue
        n = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
        if n:
            counts[str(col)] = n
    return counts


def datetime_out_of_range(
    df: pd.DataFrame, now: pd.Timestamp | None = None
) -> dict[str, dict[str, Any]]:
    """
    Datetime column -> summary of values outside the column's plausible range.

    The expected range is the 1%–99% quantile span widened by half its
    length (at least one day) on each side, clipped to 1900-01-01 and to
    *now* + 1 day. So a month of taxi trips flags the stray 2002 / 2009
    timestamps but not trips that end just after midnight on the 1st.
    """
    now = now or pd.Timestamp.now()
    out: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        series = df[col]
        if not pd.api.types.is_datetime64_any_dtype(series):
            continue
        if getattr(series.dt, "tz", None) is not None:
            series = series.dt.tz_localize(None)
        values = series.dropna()
        if len(values) < 4:
            continue
        lo, hi = values.quantile(0.01), values.quantile(0.99)
        pad = max((hi - lo) / 2, pd.Timedelta(days=1))
        start = max(lo - pad, _MIN_PLAUSIBLE_DATE)
        end = min(hi + pad, now + pd.Timedelta(days=1))
        mask = (values < start) | (values > end)
        n = int(mask.sum())
        if n:
            out[str(col)] = {
                "count": n,
                "min": f"{values.min():%Y-%m-%d %H:%M}",
                "max": f"{values.max():%Y-%m-%d %H:%M}",
                "expected_start": f"{start:%Y-%m-%d}",
                "expected_end": f"{end:%Y-%m-%d}",
            }
    return out


def observe_dataset(df: pd.DataFrame) -> "ObservationReport":
    """
    Run every check that depends only on *df* (nulls, duplicates, negative
    values, out-of-range datetimes, outliers, and the sales checks).

    Compute it once per run and pass it to :func:`observe` as ``baseline``.
    """
    report = ObservationReport()
    report.total_rows = len(df)

    # ── 1. Per-column null counts ──────────────────────────────────────────────
    for col in df.columns:
        n_null = int(df[col].isna().sum())
        report.null_counts[col] = n_null
        report.null_pct[col] = round(n_null / len(df) * 100, 2) if len(df) > 0 else 0.0

    # ── 2. Duplicate transaction IDs ───────────────────────────────────────────
    if "transaction_id" in df.columns:
        dupes = df["transaction_id"].value_counts()
        dupes = dupes[dupes > 1]
        report.duplicate_txn_ids = len(dupes)
        report.duplicate_txn_examples = dupes.head(5).index.tolist()

    # ── 3. Negative revenue ────────────────────────────────────────────────────
    if "revenue" in df.columns:
        report.negative_revenue_rows = int((df["revenue"] < 0).sum())

    # ── 4. Zero units sold with positive revenue ───────────────────────────────
    if "units_sold" in df.columns and "revenue" in df.columns:
        mask = (df["units_sold"] == 0) & (df["revenue"] > 0)
        report.zero_units_positive_revenue = int(mask.sum())

    # ── 5. Revenue outliers (IQR method) ───────────────────────────────────────
    report.revenue_outliers_iqr = _revenue_outliers_iqr(df)

    # ── 6. Generic checks ──────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report.duplicate_rows = count_duplicate_rows(df)
        report.negative_values = negative_value_counts(df)
        report.datetime_out_of_range = datetime_out_of_range(df)
        report.numeric_outliers_iqr = numeric_outlier_counts(df)

    return report


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _revenue_outliers_iqr(df: pd.DataFrame) -> list[int]:
    """Row indices whose revenue lies outside 1.5 × IQR (needs ≥ 4 values)."""
    outliers: list[int] = []
    if "revenue" in df.columns:
        rev = df["revenue"].dropna()
        if len(rev) >= 4:
            q1 = float(rev.quantile(0.25))
            q3 = float(rev.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_mask = (rev < lower) | (rev > upper)
            outlier_indices = rev[outlier_mask].index.tolist()
            outliers = [int(i) for i in outlier_indices]
    return outliers


def observe(
    df: pd.DataFrame,
    sandbox_result: dict[str, Any],
    artifact_store_available_keys: list[str] | None = None,
    baseline: ObservationReport | None = None,
) -> ObservationReport:
    """
    Run deterministic data-quality checks on *df* and return a structured report.

    The observer is called after every EXECUTE stage, before CRITIQUE.  It
    inspects the *original input DataFrame* (not the sandbox result string)
    so the critic receives actual computed counts, not LLM speculation.

    Args:
        df:                              The original full DataFrame (not filtered).
        sandbox_result:                  The full sandbox output dict
                                         (used to extract row-count hints).
        artifact_store_available_keys:   Keys currently in the ArtifactStore
                                         (for contextual notes only).
        baseline:                        Result of :func:`observe_dataset` for
                                         *df*; computed here when omitted.

    Returns:
        A populated :class:`ObservationReport`.
    """
    report = copy.deepcopy(baseline) if baseline is not None else observe_dataset(df)

    # ── Rows after code filter (heuristic from stdout head() prints) ────────
    stdout = sandbox_result.get("stdout", "")
    if stdout:
        # Heuristic: count non-empty lines in stdout blocks that look like
        # DataFrame head prints (tabular output)
        lines = [ln for ln in stdout.strip().splitlines() if ln.strip()]
        if lines:
            report.rows_after_code_filter = len(lines)
            report.notes.append(
                f"Stdout had {len(lines)} lines (may include printed DataFrames, "
                f"i.e., rows the code operated on)."
            )

    # ── Schema-level notes ──────────────────────────────────────────────────
    if artifact_store_available_keys:
        report.notes.append(
            f"ArtifactStore has {len(artifact_store_available_keys)} "
            f"pre-validated DataFrames: {', '.join(artifact_store_available_keys)}"
        )

    return report
