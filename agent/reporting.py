"""
Console output for a run.

Every ``print`` the pipeline emits lives here; ``agent.loop`` only calls these
functions. Output text is unchanged from the pre-split loop.
"""

from __future__ import annotations

from typing import Any

_RULE = "=" * 60


def run_banner(run_id: str, goal: str) -> None:
    print(f"\n{_RULE}")
    print(f"  run_id : {run_id}")
    print(f"  goal   : {goal}")
    print(f"{_RULE}\n")


def dry_run_notice() -> None:
    print("  [dry-run] LLM calls return canned responses")


def step_header(step_idx: int, total: int, sub_question: str) -> None:
    """Print a visible step header for console output."""
    print(f"┌─ Step {step_idx + 1}/{total} {'─' * 50}")
    print(f"│  {sub_question}")
    print(f"└{'─' * 58}")


def plan_stage_start() -> None:
    print("► PLAN stage …")


def plan_generated(questions: list[str]) -> None:
    print(f"  → {len(questions)} sub-questions generated")
    for i, q in enumerate(questions, 1):
        print(f"     {i}. {q}")
    print()


def replan_stage_start() -> None:
    print("► RE-PLAN stage …")


def replan_generated(questions: list[str]) -> None:
    print(f"  → {len(questions)} new sub-questions")
    for i, q in enumerate(questions, 1):
        print(f"     {i}. {q}")
    print()


def deterministic_execute_start() -> None:
    print("  ► DETERMINISTIC EXECUTE …")


def deterministic_execute_ok() -> None:
    print("    ✓ OK (deterministic)")


def codegen_start(iteration: int) -> None:
    print(f"  ► CODE-GEN (iter {iteration}) …")


def codegen_done(n_chars: int) -> None:
    print(f"    {n_chars} chars generated")


def execute_start() -> None:
    print("  ► EXECUTE …")


def execute_status(status: str, stdout: str | None) -> None:
    print(f"    {status}")
    if stdout:
        # Print first 3 lines of stdout for visibility
        lines = stdout.strip().splitlines()[:3]
        for ln in lines:
            print(f"    | {ln}")


def semantic_error(error: str) -> None:
    print(f"    ✗ Semantic error: {error}")


def observe_start() -> None:
    print("  ► OBSERVE …")


def observe_result(obs_report: Any) -> None:
    print(
        f"    ✓ nulls={obs_report.null_counts}, dupes={obs_report.duplicate_txn_ids}, "
        f"neg_rev={obs_report.negative_revenue_rows}, "
        f"outliers={len(obs_report.revenue_outliers_iqr)}"
    )


def critique_start() -> None:
    print("  ► CRITIQUE …")


def critique_result(trust: str, action: str, reason: str) -> None:
    trust_icon = "✓" if trust == "high" else "⚠"
    print(f"    {trust_icon} trust={trust} action={action}")
    print(f"    reason: {reason[:400]}")
    print()


def max_retries_exceeded(max_retries: int) -> None:
    print(f"  ✖ max_retries_exceeded after {max_retries} attempts — aborting\n")


def abort_banner(
    step_number: int,
    total_steps: int,
    max_retries: int,
    sub_question: str,
    last_reason: str,
    run_id: str,
) -> None:
    print(f"\n{_RULE}")
    print("  ✖  PIPELINE ABORTED")
    print(f"{_RULE}")
    print(f"  Step {step_number}/{total_steps} could not produce a")
    print(f"  trustworthy answer after {max_retries} attempts.")
    print(f"  Sub-question: {sub_question}")
    print(f"  Last critique: {last_reason}")
    print()
    print("  Completed steps are in the JSONL log. Fix the data or")
    print("  increase max_retries_per_step and re-run.")
    print(f"  Log: runs/{run_id}.jsonl")


def summary_stage_start() -> None:
    print("► SUMMARY stage …")


def final_summary(summary: str) -> None:
    print(f"\n{_RULE}")
    print("  FINAL SUMMARY")
    print(f"{_RULE}")
    print(summary)


def run_json_exported(json_path: Any) -> None:
    print(f"\nRun JSON exported: {json_path}")
