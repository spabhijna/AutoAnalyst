# Refactor Notes

## Phase 0 — Safety net

### 1. What changed
- Created branch `refactor/cleanup` from `main`.
- Added `tests/test_regressions.py` (only file touched, plus this notes file):
  - `test_dry_run_smoke` — passes today. `run(dry_run=True, log_dir=tmp_path)` must not abort and must leave 5 findings / 5 final results and a summary.
  - `test_parse_json_list_numbered_fallback` — `xfail(strict)` bug D1.
  - `test_deterministic_artifacts_registered_once` — `xfail(strict)` bug D2 (spies on `ArtifactStore.register`).
  - `test_dry_run_does_not_leak` — `xfail(strict)` bug D3.
  - `test_plan_revision_cap_aborts` — `xfail(strict)` bug D5, parametrized over 1- and 3-question plans.
  - `test_dry_run_is_fully_deterministic` — `xfail(strict)` bug D6 (added after review).
- Tests that run `dry_run=True` use `monkeypatch.setattr(loop, "call_llm", loop.call_llm)` so teardown undoes the D3 leak and other tests aren't polluted.

### 2. Deviations from the plan
- `REFACTOR_PLAN.md` was untracked on `main` and is deliberately **not** committed here.
- None otherwise.

### 3. Test assertions changed
None. No existing test was modified.

### 4. Results
- `pytest tests/ -q`: `250 passed, 4 xfailed in 55.70s` at first commit; after review feedback `250 passed, 6 xfailed` (baseline was 249 passed; +1 is the smoke test).
- `python main.py --dry-run`: completes, 5 steps.
- `agent/loop.py`: 1524 lines (unchanged).

### 5. Noticed, left alone (confirmed with `--runxfail`)
- D1: `re.error: unterminated character set at position 4`, as the plan said.
- D2: only `quarter_df` is double-registered, because step_1 is the only step that actually takes the deterministic path in the dry run (steps 2-5 are rejected by the step-specific semantic checks and fall back to canned codegen; root cause D6).
- D3: confirmed, `loop.call_llm` stays replaced by `_dry_call` after `run()`.
- D5: the loop does **not** hang. With an always-`revise_plan` critique and a 1-task plan the run "succeeds" with zero findings (`aborted == False`); with a longer plan it aborts with the misleading reason "(scheduler)" because later tasks can never become ready. The test is parametrized over both plans and asserts abort plus "revision" in the reason, the Phase 4 target.
- D6 is covered by `test_dry_run_is_fully_deterministic` (xfail, removed in Phase 4).
- The dry-run summary text claiming all five steps were resolved deterministically is currently false. Left alone until Phase 3 moves dry-run into `agent/dry_run.py`.
- `runs/` is gitignored, so dry-run output is not committed.
