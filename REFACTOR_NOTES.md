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

## Phase 1 — Delete dead code

### 1. What changed
- Deleted `agent/runtime.py`, `agent/executors.py`, `tests/test_runtime.py` (47 tests), `tests/test_executors.py` (2 tests, covering the removed `try_execute` duplicate).
- `agent/loop.py` (1524 → 1446 lines):
  - removed `AgentRun.runtime`, `AgentRun.capability_registry`, every `state.runtime` call (including the `update_namespace(new_vars)` merge), `_sync_artifact_consumers` and its 4 calls, `_build_completed_findings` (caller now uses `_build_working_memory_prompt`), and the unused `obs_report` in the deterministic branch.
  - `_artifact_key` kept: `_finding_from_attempt` uses it.
  - the schema summary now calls `dates.last_quarter_window`.
- `DeterministicExecutionResult` moved into `agent/capabilities.py` (no re-export).
- New `agent/dates.py` (`last_quarter_window`), used by `capabilities.py` and the schema summary; new `tests/test_dates.py` (4 tests). `tests/test_capabilities.py` gained one test for `DeterministicExecutionResult.to_sandbox_result`.
- Unused imports removed (`ruff check agent --select F401 --fix`): `loop.py` (`AnalysisRuntime`, `CapabilityRegistry`, `TaskStatus`), `observer.py` (`numpy`), `llm_client.py` (`Any`, `CEREBRAS_BASE_URL`; import lines only, no provider logic touched).

### 2. Deviations from the plan
- `AgentRun.llm_call_count` is NOT removed here: it is still read by `_finalize_run`. Phase 3 makes it real, per the plan.
- The deterministic branch previously registered the artifact only inside `if state.runtime is not None`; the guard is gone but the `artifact_store.register` call is kept as-is, so the D2 double registration is unchanged (xfail still strict).
- Schema summary now uses `pd.to_datetime(..., errors="coerce")` via `last_quarter_window`; the old inline code used no `coerce` inside try/except. Only differs for unparseable date values (old: fell back to the generic text; new: window from the parseable rows).

### 3. Test assertions changed
None changed. Removed whole files only: `test_runtime.py` (the runtime is gone) and `test_executors.py` (the duplicate `try_execute`). The capabilities' own executors stay covered by `test_capabilities.py` / `test_execution_engine.py`.

### 4. Results
- `pytest tests/ -q`: `206 passed, 6 xfailed` (250 − 49 deleted + 5 added).
- `python main.py --dry-run`: completes, 5 steps.
- `agent/loop.py`: 1446 lines.

### 5. Noticed, left alone
- `ruff` F841 in `agent/prompts.py` (`obs_block` unused, in `critique_prompt`) and `agent/sandbox.py` (`last_line`): out of scope (prompts/sandbox).
- `README.md`, `ARCHITECTURE.md` and `PROGRESS.md` still mention `runtime.py` / `executors.py`; Phase 6 rewrites them.
- `_accepted_findings` and the `AgentRun.history` shim remain (tests still use them).
