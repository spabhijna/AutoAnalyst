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
  - `test_critique_prompt_includes_observation_text` — `xfail(strict)` bug D8 (added in the Phase 1 review; Phase 4 fixes it).
- Tests that run `dry_run=True` use `monkeypatch.setattr(loop, "call_llm", loop.call_llm)` so teardown undoes the D3 leak and other tests aren't polluted.

### 2. Deviations from the plan
- `REFACTOR_PLAN.md` was untracked on `main` and is deliberately **not** committed here.
- None otherwise.

### 3. Test assertions changed
None. No existing test was modified.

**Correction (Phase 1 review):** the `fix(phase-0)` commit rewrote `test_plan_revision_cap_aborts` and silently dropped `assert state.plan_revisions == MAX_PLAN_REVISIONS` (a new Phase 0 test, not a pre-existing one). It is restored in `fix(phase-1)`. It holds for both parametrized plans (verified with `--runxfail`: the failure comes later, at `state.aborted` / the "revision" reason check).

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

### Phase 1 review follow-ups
- Restored the `plan_revisions` assertion (see Phase 0 §3 correction) and added the D8 xfail test.
- Reminder for Phase 4: the D5 fix must update `tests/test_loop.py::TestRevisePlan::test_plan_revision_cap` ("After MAX_PLAN_REVISIONS, revise_plan requests are ignored"), whose behaviour D5 deliberately changes.

## Phase 2 — Split `loop.py` (moves only)

### 1. What changed
- New `agent/parsing.py`: `parse_plan_json`, `parse_json_list`, `parse_critique_json`, `strip_markdown_fences` (underscore dropped, bodies unchanged except the task-dict collapse below), plus private `_task_spec` / `_normalise_task_spec`.
- New `agent/schema.py`: `build_schema_summary`.
- New `agent/planning.py`: `task_id`, `task_order`, `build_tasks`, `sync_task_capabilities`, `do_plan`, `do_replan`.
- New `agent/reporting.py`: every console `print` (run banner, dry-run notice, step header, plan/replan lines, each stage line, abort banner, final summary, retries exceeded, run-JSON exported). `loop.py` has no `print` left. Output verified byte-identical: normalised stdout of `python main.py --dry-run` is identical before and after (diff, log lines and run ids excluded).
- `agent/loop.py`: 1446 → 1009 lines; calls the new modules.
- Tests: `tests/test_loop.py` and `tests/test_regressions.py` now import from the new modules (no aliases in `loop.py`, none in the tests).

### 2. Deviations from the plan
- **Moved functions take their dependencies as arguments.** Tests patch `agent.loop.call_llm` and the dry-run replaces `loop.call_llm` at run time, so `planning.py` calling its own imported `call_llm` would silently bypass both. `loop.py` therefore passes `call_llm` (looked up at call time) and `_log`:
  - `do_plan(state, df, llm, log)`
  - `do_replan(state, df, llm, completed)`, where `completed` is `_build_working_memory_prompt(state)` computed by the caller.
  
  This also avoids a `planning` ↔ `loop` import cycle, and Phase 3's "pass the llm callable through the stage functions" builds on it. The `df` parameter is unused in both (as before); kept to keep the move faithful.
- `task_id`, `task_order`, `build_tasks`, `sync_task_capabilities` are public (no underscore) because they are now cross-module. The plan only named the parsers.
- Public names follow the Phase 2 text (`parse_plan_json`, …), not the Target-shape list (`parse_plan`, `strip_fences`).
- Task-dict collapse (`_normalise_task_spec`) preserves the plan's three return paths. Two edge differences:
  - legacy flat items now carry `capability` / `output_schema` keys (None / `[]`); `build_tasks` read them via `.get`, so behaviour is the same;
  - a *mixed* list item `{"question": "   "}` (whitespace-only) is now dropped; before it produced a spec with an empty question.

### 3. Test assertions changed
None. Only import lines and the names used in call sites changed in `test_loop.py` (`_parse_json_list` → `parse_json_list`, etc.) and in `test_regressions.py`.

### 4. Results
- `pytest tests/ -q`: `206 passed, 7 xfailed`.
- `python main.py --dry-run` and `--dry-run --corrupt` both finish (5 steps).
- `agent/loop.py`: 1009 lines (target is ≤450 after Phase 3; `_run_question` is still ~340 lines).

### 5. Noticed, left alone
- `_do_summary`, `_do_codegen`, `_do_repair`, `_do_critique` still call `call_llm` directly in `loop.py`; Phase 3 threads the callable through.
- `_log`, `_finalize_run` (with its hard-coded `"clean"` label, D7) and `_finalize_run`'s Groq model names stay in `loop.py`.
- `ruff --select F` also flags unused imports in tests (`MagicMock`, `call`, `ObservationReport`, numpy in `test_sandbox.py`) and F841 in `prompts.py` / `sandbox.py`; untouched (out of scope).

## Phase 2 review follow-up
- `99bfbb6 fix(phase-2)`: `planning.py` types `state` as `"AgentRun"` via `TYPE_CHECKING` (no runtime import, no cycle). Names stay `parse_plan_json` / `parse_json_list` (the plan's `parse_plan` / `parse_string_list` mean these).

## Phase 3 — One accept path, injectable LLM (fixes D2, D3)

### 1. What changed
- `agent/loop.py`:
  - New `_accept(...)`: the single accept path (task status, Finding, working memory, `final_results`, artifact registration once, `produced_artifact_name`). Both branches call it.
  - `_run_question` is now a dispatcher over `_try_deterministic(state, task, df) -> bool` and `_run_codegen_loop(state, task, df) -> bool`.
  - `run(..., llm=None)`: `llm` defaults to `call_llm`, or `dry_run_llm` when `dry_run=True` (an explicit `llm` wins). It is wrapped by `_counting_llm`, stored on the new `AgentRun.llm` field, and used by every stage (`do_plan`, `do_replan`, codegen, repair, critique, summary). The dry-run monkeypatch is gone.
- New `agent/dry_run.py` with `dry_run_llm` (canned responses moved verbatim).
- `agent/execution_engine.py`: **owner of artifact registration is `_accept`**; the engine only executes and validates. Dropped the now-meaningless `consumers` argument of `execute()`.
- `tests/test_regressions.py`: D2 and D3 xfail markers removed (both pass); the Phase 0 monkeypatch workarounds (restoring `loop.call_llm` after dry-run) removed since the leak is fixed; new test `test_explicit_llm_callable_is_used_and_counted`.
- `tests/test_execution_engine.py`: see §3.

### 2. Deviations from the plan
- `_accept` has two extra optional parameters, `artifact_key` and `provenance`. The engine used to register under `task.produces_artifact or result.artifact_key`, with provenance `[capability, task_id]`; `_try_deterministic` passes the executor's `artifact_key` and `[task.capability, task.task_id]` so artifacts of tasks without a declared `produces_artifact` are still registered, and the artifact keeps the engine's provenance (previously the loop's second registration overwrote it with `[task_id]`).
- Behaviour that changes beyond D2/D3 (all intended by the plan, or tiny side effects):
  - `state.llm_call_count` (and `total_llm_calls` in the run JSON) is now real: a dry run reports 10 (plan + 4 codegen + 4 critique + summary, because D6 is still open). Before, always 0.
  - Deterministic artifacts now carry the engine's provenance and `confidence=trust` (`"high"`); `finding.produced_artifact_name` is set whenever registration succeeds, including executor-default keys.
  - `TaskMemory` is still created for every task (kept in `_run_question`).
- `_run_codegen_loop` is still ~190 lines and `run()` ~140; `loop.py` is 1040 lines (up from 1009: new docstrings and `_accept`). Shrinking to the plan's ≤450 / ≤80-line targets needs the later moves; not attempted here (Phase 3 scope is the split, not more extraction).

### 3. Test assertions changed
- `tests/test_execution_engine.py::test_engine_executes_registered_capability_and_registers_artifact` → renamed `..._without_registering`. Was: after `execute(..., consumers=["step_2"])` the store holds `quarter_df` with producer `step_1`, consumers `["step_2"]`, frozen, provenance `["FilterByDateRange", "step_1"]`, `validated_at` set. Now: result carries a DataFrame artifact with `artifact_key == "quarter_df"` and the store is still empty. Why: registration moved to `_accept` (D2 owner decision); the `consumers` argument no longer exists. Coverage of the registered artifact's metadata (producer, consumers, freeze, provenance) now comes from the dry-run path through `_accept`; I did not add a dedicated unit test for it.
- `tests/test_execution_engine.py::test_engine_reuses_region_artifact`: was `store.get_artifact("region_revenue")` has columns `[region, total_revenue]`; now `result["artifact"]` has those columns and the store has no `region_revenue`. Same reason.
- `tests/test_regressions.py`: D2/D3 xfail removal (intended); `test_dry_run_does_not_leak` assertion unchanged.

### 4. Results
- `pytest tests/ -q`: `209 passed, 5 xfailed` (remaining xfails: D1, D5 ×2, D6, D8).
- `python main.py --dry-run` and `--dry-run --corrupt`: 5 steps each. Normalised dry-run stdout is identical to the previous commit.
- `agent/loop.py`: 1040 lines.

### 5. Noticed, left alone
- The dry-run summary text still claims five deterministic steps (false until D6 is fixed in Phase 4); the text was only moved to `agent/dry_run.py`.
- `ExecutionEngine.execute` still runs `validate_semantic_contract` (the hard-coded step_N checks, D6).
- `_do_replan` still sets `state.tasks = build_tasks(state.plan)` from the flat list (D4).
