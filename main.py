"""
AutoAnalyst CLI entry point.

Usage examples:
  # Dry run (no LLM calls, no API keys needed)
  python main.py --dry-run

  # Live run against the default dataset
  python main.py --goal "What drove the revenue decline in Q3?" \\
                 --data data/raw/superstore_sales.csv

  # Live run with custom retry limit
  python main.py --goal "Which product category has highest margin?" \\
                 --data data/raw/superstore_sales.csv \\
                 --max-retries 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _configure_logging(verbose: bool) -> None:
    """Configure root logger — INFO by default, DEBUG with --verbose."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="autoanalyst",
        description="Self-Correcting Data Pipeline Agent",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run without live LLM calls (uses stub responses). No API keys required.",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="Provide a comprehensive sales analysis of this dataset.",
        help="High-level analytical goal (default: general sales analysis).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/superstore_sales.csv"),
        help="Path to the input CSV dataset (default: data/raw/superstore_sales.csv).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Override MAX_RETRIES from .env (default: use env value, typically 3).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable debug logging.",
    )
    return parser


def _load_dataset(data_path: Path) -> "pd.DataFrame":  # type: ignore[name-defined]
    """
    Load the dataset CSV into a pandas DataFrame.

    Args:
        data_path: Path to the CSV file.

    Returns:
        Loaded :class:`pandas.DataFrame`.

    Raises:
        SystemExit: If the file does not exist or cannot be parsed.
    """
    import pandas as pd

    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}", file=sys.stderr)
        print(
            "Run: python scripts/download_data.py  to fetch the sample dataset.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        df = pd.read_csv(data_path)
        print(f"Loaded dataset: {len(df):,} rows × {len(df.columns)} columns")
        return df
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Failed to read {data_path}: {exc}", file=sys.stderr)
        sys.exit(1)


def _build_schema_description(df: "pd.DataFrame") -> str:  # type: ignore[name-defined]
    """
    Build a concise schema string for use in LLM prompts.

    Args:
        df: The loaded DataFrame.

    Returns:
        Multi-line string listing each column with its dtype and sample values.
    """
    lines: list[str] = [f"Rows: {len(df):,}  Columns: {len(df.columns)}"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample_vals = df[col].dropna().head(3).tolist()
        sample_str = ", ".join(repr(v) for v in sample_vals)
        null_pct = df[col].isna().mean() * 100
        lines.append(
            f"  {col!r} ({dtype}): nulls={null_pct:.1f}% — e.g. {sample_str}"
        )
    return "\n".join(lines)


def main() -> None:
    """CLI entry point — parse args, run pipeline, print summary."""
    parser = _build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)

    log = logging.getLogger(__name__)

    # ── Config ────────────────────────────────────────────────────────────────
    import os
    from autoanalyst.config import PipelineConfig

    # Allow --max-retries to override env without modifying the env var
    if args.max_retries is not None:
        os.environ["MAX_RETRIES"] = str(args.max_retries)

    config = PipelineConfig.from_env(dry_run=args.dry_run)
    log.info("Config: model=%s, retries=%d, dry_run=%s", config.groq_model, config.max_retries, config.dry_run)

    # ── Dataset ───────────────────────────────────────────────────────────────
    df = _load_dataset(args.data)
    schema = _build_schema_description(df)

    if args.verbose:
        print("\n── Dataset Schema ──")
        print(schema)
        print()

    # ── Run pipeline ──────────────────────────────────────────────────────────
    from autoanalyst.pipeline import run_pipeline

    print(f"\nGoal: {args.goal}")
    print(f"Dry-run: {config.dry_run}\n")

    result = run_pipeline(
        goal=args.goal,
        dataset_schema=schema,
        df=df,
        config=config,
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(f"Run complete  ·  run_id={result.run_id}")
    print(f"Log written → {result.log_path}")
    print("═" * 60)

    passed = sum(r.passed for r in result.question_results)
    print(f"\nResults: {passed}/{len(result.question_results)} sub-questions passed critique\n")

    for i, qr in enumerate(result.question_results, 1):
        status = "✓ PASS" if qr.passed else "✗ FAIL"
        print(f"  {i}. [{status}] {qr.sub_question[:80]}")
        print(f"       Iterations: {qr.iterations}")
        if qr.final_answer:
            preview = qr.final_answer[:160].replace("\n", " ")
            print(f"       Answer: {preview}…")
        print()


if __name__ == "__main__":
    main()
