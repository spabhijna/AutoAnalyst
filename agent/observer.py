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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


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

    Returns:
        A populated :class:`ObservationReport`.
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

    # ── 6. Rows after code filter (heuristic from stdout head() prints) ────────
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

    # ── 7. Schema-level notes ──────────────────────────────────────────────────
    if artifact_store_available_keys:
        report.notes.append(
            f"ArtifactStore has {len(artifact_store_available_keys)} "
            f"pre-validated DataFrames: {', '.join(artifact_store_available_keys)}"
        )

    return report
