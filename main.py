"""
AutoAnalyst CLI entry point.

Usage:
    python main.py                  # run on default synthetic data
    python main.py --goal "..."     # custom analytical goal
    python main.py --rows 1000      # larger dataset
    python main.py --corrupt        # inject known corruption
    python main.py --seed 123       # reproducible dataset
    python main.py --retries 3      # override max retries per step
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    stream=sys.stdout,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)

from data.generate_data import generate_sales_data, inject_corruption  # noqa: E402
from agent.loop import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoAnalyst — Self-Correcting Data Pipeline Agent",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=(
            "Explain the revenue trend by region over the last quarter. "
            "Identify which region performed best, whether performance was "
            "consistent month-over-month, and which product category drove "
            "the most revenue in the top-performing region."
        ),
        help="High-level analytical goal.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=700,
        help="Number of synthetic transaction rows to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible data generation.",
    )
    parser.add_argument(
        "--corrupt",
        action="store_true",
        help="Inject known corruption (duplicates, nulls, outliers).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=None,
        help="Override max CODE-GEN retries per sub-question.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without LLM API keys (uses canned responses).",
    )
    args = parser.parse_args()

    print("Generating synthetic sales data …")
    df = generate_sales_data(n_rows=args.rows, seed=args.seed)

    if args.corrupt:
        corrupt_df, manifest = inject_corruption(df, seed=args.seed + 57)
        defect_total = manifest.get("total_defects", 0)
        print(f"  Injected {defect_total} known defects (ground truth in manifest).")
        df = corrupt_df

    print(f"  {len(df):,} rows, {len(df.columns)} columns\n")

    kwargs = {}
    if args.retries is not None:
        kwargs["max_retries_per_step"] = args.retries

    state = run(args.goal, df, dry_run=args.dry_run, **kwargs)

    print(f"\nLog written to: runs/{state.run_id}.jsonl")
    print(f"Total steps executed: {len(state.history)}")


if __name__ == "__main__":
    main()
