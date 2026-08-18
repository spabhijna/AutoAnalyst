"""
Human-readable run log viewer.

Reads a ``runs/<run_id>.json`` file produced by :func:`agent.logger.export_run_json`
and prints a clean trace of every step: the question asked, the code generated,
the sandbox result, the critique verdict, and the action taken.

Usage
-----
    python -m agent.log_viewer runs/run_abc12345.json

Or from code::

    from agent.log_viewer import print_run_trace
    print_run_trace("runs/run_abc12345.json")
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def print_run_trace(json_path: str | Path) -> None:
    """
    Print a human-readable trace of the run stored in *json_path*.

    Each step shows: the question, code, result, critique verdict, and action.
    Metadata (goal, dataset, model versions, timing) is printed in a header.

    Args:
        json_path: Path to a ``runs/<run_id>.json`` file.
    """
    path = Path(json_path)
    if not path.exists():
        print(f"Error: file not found — {path}", file=sys.stderr)
        sys.exit(1)

    with path.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)

    meta: dict[str, Any] = data.get("metadata", {})
    history: list[dict[str, Any]] = data.get("history", [])

    # ── Header ────────────────────────────────────────────────────────────────
    print("=" * 72)
    print(f"  RUN TRACE — {data.get('run_id', 'unknown')}")
    print("=" * 72)
    print()
    print("📋 METADATA")
    print("-" * 72)
    print(f"  Goal:       {meta.get('goal', '(none)')[:120]}")
    print(f"  Dataset:    {meta.get('dataset', 'unknown')}")
    print(f"  Models:     {json.dumps(meta.get('model_versions', {}))}")
    print(f"  Iterations: {meta.get('total_iterations', 0)}")
    print(f"  LLM calls:  {meta.get('total_llm_calls', 0)}")
    print(f"  Wall clock: {meta.get('wall_clock_seconds', 0):.1f}s")
    aborted = meta.get("aborted", False)
    print(f"  Status:     {'❌ ABORTED' if aborted else '✅ Complete'}")
    print()

    # ── Step trace ────────────────────────────────────────────────────────────
    step_number = 0
    last_question = None

    for entry in history:
        stage = entry.get("stage", "?")
        question = entry.get("sub_question", "")

        # Group entries by sub_question — print a step header when it changes
        if question != last_question and stage != "plan" and stage != "summary":
            step_number += 1
            print(f"━━━ STEP {step_number} ━━━{'━' * 60}")
            print(f"  ❓ {question[:120]}")
            print()

        if stage == "plan":
            print("📝 PLAN")
            plan_items = entry.get("extra", {}).get("plan", [])
            for i, q in enumerate(plan_items, 1):
                print(f"  {i}. {q}")
            print()

        elif stage == "codegen_execute_critique":
            iteration = entry.get("iteration", 0)
            if iteration == 0:
                action = entry.get("action", "")
                status = "✅ ACCEPTED" if action == "accept" else "🔄 retry →"
                print(f"  Iteration {iteration}: {status}")
            else:
                print(f"  Iteration {iteration}: 🔄 retry")

            code = entry.get("code", "")
            if code:
                print("  ── Code ──")
                for line in code.strip().splitlines()[:10]:
                    print(f"    {line}")
                if len(code.strip().splitlines()) > 10:
                    print(f"    … (+{len(code.strip().splitlines()) - 10} more lines)")

            sandbox = entry.get("sandbox_result") or {}
            if sandbox.get("stdout"):
                stdout_lines = sandbox["stdout"].strip().splitlines()
                print(f"  📤 stdout ({len(stdout_lines)} lines):")
                for line in stdout_lines[:5]:
                    print(f"    | {line}")

            result_repr = sandbox.get("result")
            if result_repr:
                print(f"  📊 result: {result_repr[:200]}")

            critique = entry.get("critique", "")
            trust = entry.get("trust", "?")
            action = entry.get("action", "?")
            if critique:
                try:
                    crit_data = json.loads(critique)
                    reason = crit_data.get("reason", critique)[:300]
                except (json.JSONDecodeError, TypeError):
                    reason = critique[:300]
                print(f"  🔍 critique [{trust} trust, {action}]: {reason}")

            print()

        elif stage == "summary":
            extra = entry.get("extra", {})
            summary_text = extra.get("summary", "")
            print("📝 FINAL SUMMARY")
            print("-" * 72)
            print(summary_text)
            print()

        last_question = question

    # ── Final summary from metadata ───────────────────────────────────────────
    final = meta.get("final_summary", "")
    if final:
        print("=" * 72)
        print("  FINAL SUMMARY")
        print("=" * 72)
        print(final)
        print()

    print(f"Log file: {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m agent.log_viewer <run.json>", file=sys.stderr)
        sys.exit(2)
    print_run_trace(sys.argv[1])
