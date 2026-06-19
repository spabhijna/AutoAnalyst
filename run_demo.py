"""
Demo script — runs the full five-stage agent loop on the clean synthetic dataset.

Usage:
    source .venv/bin/activate
    python run_demo.py

Requires:
    GROQ_API_KEY in .env (or environment)
    GEMINI_API_KEY in .env (fallback)

Output:
    Console: step-by-step log
    File:    runs/<run_id>.jsonl
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    stream=sys.stdout,
)

# Silence noisy library loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)

from data.generate_data import generate_sales_data  # noqa: E402
from agent.loop import run  # noqa: E402


def main() -> None:
    print("Generating synthetic sales data …")
    df = generate_sales_data(n_rows=700, seed=42)
    print(f"  {len(df)} rows, {len(df.columns)} columns\n")

    goal = (
        "Explain the revenue trend by region over the last quarter. "
        "Identify which region performed best, whether performance was "
        "consistent month-over-month, and which product category drove "
        "the most revenue in the top-performing region."
    )

    state = run(goal, df)

    print(f"\nLog written to: runs/{state.run_id}.jsonl")
    print(f"Total steps executed: {len(state.history)}")


if __name__ == "__main__":
    main()
