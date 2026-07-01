"""
Phase 5: Evaluation against corruption manifest.

Computes precision, recall, and F1 by comparing observer-detected
signals against the ground-truth manifest from ``inject_corruption()``.

Usage
-----
    from agent.evaluator import evaluate
    from data.generate_data import generate_sales_data, inject_corruption

    df = generate_sales_data(n_rows=700, seed=42)
    corrupt_df, manifest = inject_corruption(df, seed=99)
    report = evaluate(corrupt_df, manifest)
    print(report.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from agent.loop import AgentRun
from agent.observer import observe


# ---------------------------------------------------------------------------
# Category-level result
# ---------------------------------------------------------------------------


@dataclass
class CategoryResult:
    """
    Evaluation result for one defect category.

    Attributes:
        name:               Human-readable category name.
        present_in_manifest: Whether the manifest contains defects of this type.
        detected:            Whether the observer detected signals of this type.
        manifest_count:      Number of injected defects in this category.
        observer_count:      Magnitude of the observer signal (rows affected).
    """

    name: str
    present_in_manifest: bool = False
    detected: bool = False
    manifest_count: int = 0
    observer_count: int = 0

    @property
    def is_true_positive(self) -> bool:
        """Defect exists AND was detected."""
        return self.present_in_manifest and self.detected

    @property
    def is_false_positive(self) -> bool:
        """Detected but no matching defect in manifest."""
        return self.detected and not self.present_in_manifest

    @property
    def is_false_negative(self) -> bool:
        """Defect exists but was not detected."""
        return self.present_in_manifest and not self.detected

    @property
    def is_true_negative(self) -> bool:
        """No defect and no detection — correct negative."""
        return not self.present_in_manifest and not self.detected


# ---------------------------------------------------------------------------
# EvalReport
# ---------------------------------------------------------------------------


@dataclass
class EvalReport:
    """
    Complete evaluation report for a corruption run.

    Attributes:
        categories:              Per-category results, keyed by category name.
        precision:               TP / (TP + FP) — fraction of flagged issues
                                 that were real defects.
        recall:                  TP / (TP + FN) — fraction of injected defects
                                 that were detected.
        f1_score:                2 * precision * recall / (precision + recall).
        true_positives:          Number of categories where TP = True.
        false_positives:         Number of categories where FP = True.
        false_negatives:         Number of categories where FN = True.
        total_defects_in_manifest: Sum of all manifest defect counts.
        total_defects_detected:    Sum of all observer signal magnitudes.
        pipeline_aborted:        Whether the pipeline aborted (from AgentRun).
        abort_reason:            Reason string if aborted.
        total_attempts:          Total codegen-execute-critique iterations.
    """

    categories: dict[str, CategoryResult] = field(default_factory=dict)
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total_defects_in_manifest: int = 0
    total_defects_detected: int = 0
    pipeline_aborted: bool = False
    abort_reason: str = ""
    total_attempts: int = 0

    def summary(self) -> str:
        """
        Return a human-readable summary of the evaluation report.
        """
        lines = [
            "=" * 56,
            "  EVALUATION REPORT — Precision / Recall vs Manifest",
            "=" * 56,
            "",
            f"  precision : {self.precision:.2%}",
            f"  recall    : {self.recall:.2%}",
            f"  F1 score  : {self.f1_score:.2%}",
            "",
            f"  true positives : {self.true_positives}",
            f"  false positives: {self.false_positives}",
            f"  false negatives: {self.false_negatives}",
            "",
            f"  defects in manifest: {self.total_defects_in_manifest}",
            f"  defects detected   : {self.total_defects_detected}",
            "",
        ]

        if self.pipeline_aborted:
            lines.append("  ❌ PIPELINE ABORTED")
            lines.append(f"     reason: {self.abort_reason}")
        else:
            lines.append("  ✅ Pipeline completed without abort")

        if self.total_attempts > 0:
            lines.append(f"  total codegen attempts: {self.total_attempts}")

        lines.append("")
        lines.append("  ── Per-Category Details ──")
        for name, cat in sorted(self.categories.items()):
            status = "✅" if cat.is_true_positive else (
                "⚠️ FP" if cat.is_false_positive else
                "❌ FN" if cat.is_false_negative else "—"
            )
            lines.append(
                f"  {status} {name}: "
                f"manifest={cat.manifest_count}, "
                f"observer={cat.observer_count}"
            )
        lines.append("")
        lines.append("=" * 56)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------


def _check_duplicates(
    report, manifest: dict[str, Any]
) -> CategoryResult:
    """Check duplicate transaction ID detection against manifest."""
    dupes_manifest = manifest.get("duplicated_transaction_ids", {})
    n_injected = len(dupes_manifest)
    n_detected = report.duplicate_txn_ids
    return CategoryResult(
        name="duplicate_transactions",
        present_in_manifest=n_injected > 0,
        detected=n_detected > 0,
        manifest_count=n_injected,
        observer_count=n_detected,
    )


def _check_nulls(
    report, manifest: dict[str, Any]
) -> CategoryResult:
    """Check null revenue detection against manifest."""
    null_manifest = manifest.get("null_revenue_indices", [])
    n_injected = len(null_manifest)
    n_detected = report.null_counts.get("revenue", 0)
    return CategoryResult(
        name="null_revenue",
        present_in_manifest=n_injected > 0,
        detected=n_detected > 0,
        manifest_count=n_injected,
        observer_count=n_detected,
    )


def _check_negative_revenue(
    report, manifest: dict[str, Any]
) -> CategoryResult:
    """Check negative revenue detection against manifest."""
    impossible = manifest.get("impossible_rows", [])
    n_injected = sum(
        1 for r in impossible if r.get("type") == "negative_revenue"
    )
    n_detected = report.negative_revenue_rows
    return CategoryResult(
        name="negative_revenue",
        present_in_manifest=n_injected > 0,
        detected=n_detected > 0,
        manifest_count=n_injected,
        observer_count=n_detected,
    )


def _check_zero_units_positive_revenue(
    report, manifest: dict[str, Any]
) -> CategoryResult:
    """Check zero-units-positive-revenue detection against manifest."""
    impossible = manifest.get("impossible_rows", [])
    n_injected = sum(
        1 for r in impossible if r.get("type") == "zero_units_positive_revenue"
    )
    n_detected = report.zero_units_positive_revenue
    return CategoryResult(
        name="zero_units_with_revenue",
        present_in_manifest=n_injected > 0,
        detected=n_detected > 0,
        manifest_count=n_injected,
        observer_count=n_detected,
    )


def _check_outliers(
    report, manifest: dict[str, Any]
) -> CategoryResult:
    """Check IQR outlier detection against manifest."""
    impossible = manifest.get("impossible_rows", [])
    n_injected = sum(
        1 for r in impossible if r.get("type") == "revenue_outlier_50x"
    )
    n_detected = len(report.revenue_outliers_iqr)
    return CategoryResult(
        name="revenue_outlier",
        present_in_manifest=n_injected > 0,
        detected=n_detected > 0,
        manifest_count=n_injected,
        observer_count=n_detected,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate(
    df: pd.DataFrame,
    manifest: dict[str, Any],
    state: AgentRun | None = None,
) -> EvalReport:
    """
    Evaluate observer detection accuracy against a corruption manifest.

    Runs the deterministic observer on *df* and compares its signals
    against the ground-truth *manifest* from ``inject_corruption()``.

    When *state* (an :class:`AgentRun`) is provided, pipeline resilience
    metrics (abort status, total attempts) are also populated.

    Args:
        df:       The (possibly corrupted) DataFrame the pipeline ran on.
        manifest: Ground-truth manifest from ``inject_corruption()``.
        state:    Optional AgentRun state for pipeline behaviour metrics.

    Returns:
        An :class:`EvalReport` with precision, recall, and per-category
        results.
    """
    # Run the observer deterministically on the DataFrame
    sandbox_stub: dict[str, Any] = {
        "success": True,
        "result": "",
        "stdout": "",
        "error": None,
    }
    report = observe(df, sandbox_stub)

    # Check each defect category
    categories: dict[str, CategoryResult] = {
        "duplicate_transactions": _check_duplicates(report, manifest),
        "null_revenue": _check_nulls(report, manifest),
        "negative_revenue": _check_negative_revenue(report, manifest),
        "zero_units_with_revenue": _check_zero_units_positive_revenue(report, manifest),
        "revenue_outlier": _check_outliers(report, manifest),
    }

    # Aggregate counts
    tps = sum(1 for c in categories.values() if c.is_true_positive)
    fps = sum(1 for c in categories.values() if c.is_false_positive)
    fns = sum(1 for c in categories.values() if c.is_false_negative)

    precision = tps / (tps + fps) if (tps + fps) > 0 else 1.0
    recall = tps / (tps + fns) if (tps + fns) > 0 else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    total_manifest = sum(
        c.manifest_count for c in categories.values()
    )
    total_detected = sum(
        c.observer_count for c in categories.values()
    )

    # Pipeline resilience from AgentRun state
    pipeline_aborted = False
    abort_reason = ""
    total_attempts = 0
    if state is not None:
        pipeline_aborted = state.aborted
        abort_reason = state.abort_reason
        total_attempts = len(state.debug_history)

    return EvalReport(
        categories=categories,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
        true_positives=tps,
        false_positives=fps,
        false_negatives=fns,
        total_defects_in_manifest=total_manifest,
        total_defects_detected=total_detected,
        pipeline_aborted=pipeline_aborted,
        abort_reason=abort_reason,
        total_attempts=total_attempts,
    )


def evaluate_from_corrupt_run(
    df: pd.DataFrame,
    manifest: dict[str, Any],
    state: AgentRun,
) -> EvalReport:
    """
    Convenience wrapper: runs ``evaluate()`` with the AgentRun state
    for pipeline resilience metrics.

    Args:
        df:       The corrupted DataFrame.
        manifest: Ground-truth manifest from ``inject_corruption()``.
        state:    The :class:`AgentRun` returned by ``agent.loop.run()``.

    Returns:
        An :class:`EvalReport`.
    """
    return evaluate(df, manifest, state=state)
