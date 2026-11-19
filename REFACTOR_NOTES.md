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

## Phase 3 review follow-up
- New `tests/test_accept.py`: `_accept` registration metadata (producer, consumers `["step_2"]`, frozen, confidence, provenance, `validated_at`, `produced_artifact_name`, status, working memory), no-artifact case, and provenance for `_try_deterministic` with a declared and an undeclared `task.capability`. This replaces the metadata coverage removed from `test_execution_engine.py`.
- `ExecutionEngine.execute` now returns `capability_id` (the resolved capability) in its result dict; `_try_deterministic` builds provenance `[capability_id, task_id]` from it, so replanned tasks (no `task.capability`) keep provenance. `test_engine_executes_registered_capability_without_registering` also asserts `capability_id`.
- Assertion changes: none (only additions).

## Phase 3 review follow-up #2
- Commit `fix(phase-3)` (05d7c48) added `tests/test_accept.py` and the resolved `capability_id` in the engine result (see "Phase 3 review follow-up" above). Phase 5 scope per the updated plan (shims, `findings.py`, `run_export`/`logger` move, splitting `_run_codegen_loop` and `run()`, ≤600 lines, no function >80 lines) was re-read; nothing done early.

## Phase 4 — Remaining behaviour fixes (D1, D4, D5, D6, D7, D8)

### 1. What changed
- **D1** `agent/parsing.py`: regex `r"^\d+[\.)\]]\s*"` (was an unterminated character set).
- **D8** `agent/prompts.py`: `{obs_block}` is now interpolated in `critique_prompt`, right after the "PREVIOUSLY ACCEPTED FINDINGS" block and before the audit steps, where STEP 3 refers to it ("a DATA-QUALITY OBSERVATION block is present above"). No other wording touched.
- **D4** replan keeps structure:
  - `replan_prompt` now asks for the same JSON task shape as `plan_prompt` (new optional `available_artifacts` argument, listing the registered artifacts). This is the one allowed prompt rewrite.
  - `planning.do_replan` returns structured specs (via `parse_plan_json`); new `planning.apply_replan(state, specs)` keeps the completed tasks as the same `TaskNode` objects, drops the failed task and everything after it, and appends new tasks whose IDs continue after the highest existing `step_N` (`next_task_index`) and which depend on every kept task. `build_tasks` gained keyword-only `first_index` / `depends_on_prefix`. `sync_task_capabilities` runs after a replan, so new tasks get capabilities too.
- **D5** `loop.run`: when a revise_plan request arrives with the cap reached, it raises `PipelineAbortedError(kind="plan revision cap reached")` with a clear reason instead of advancing past the unfinished task. `PipelineAbortedError` gained a `kind` (default `"max retries exhausted"`, so existing messages are unchanged); the scheduler abort now says `"scheduling failure"`; `reporting.abort_banner` takes `kind` and only prints the "after N attempts" text for retry exhaustion.
- **D6** `CapabilityValidator.validate_semantic_contract(task, artifact)` is now generic: only when `task.produces_artifact` is declared, the artifact must be a non-empty DataFrame whose columns equal `task.output_schema` (placeholders `["same_as_input"]` / `["<scalar>"]` skipped). All `step_2/4/5/6` branches, the `quarter_df` top-revenue computation and the numpy import are gone. Callers (`ExecutionEngine.execute`, the CODE-GEN path in the loop) updated. `grep -rn 'step_[0-9]' agent/` finds nothing.
- **D7** `config.py` reads `MAX_RETRIES`, `SANDBOX_TIMEOUT_SECONDS`, `LOG_DIR` from the environment (after `load_dotenv()`; blank = default; an invalid number raises `ValueError` naming the variable). `loop.SANDBOX_TIMEOUT` is deleted; the loop passes `config.SANDBOX_TIMEOUT_SECONDS`; `run_sandboxed`'s default timeout is `config.SANDBOX_TIMEOUT_SECONDS`. `run(..., dataset_label="clean")` feeds `_finalize_run` (`"<label> (aborted)"` on abort); `main.py` passes `"corrupted"` with `--corrupt`.
- Tests: all four Phase 0 xfail markers (D1, D5 ×2, D6, D8) removed from `tests/test_regressions.py`, now plain passing tests. New: `tests/test_replan.py` (D4), `tests/test_validator_semantic.py` (D6), `tests/test_config.py` (D7 and the dataset label).

### 2. Deviations from the plan
- `validate_semantic_contract` lost its `task_id`, `result_str` and `artifacts` arguments (all only fed the step checks); it takes the `TaskNode`.
- The effective sandbox timeout in the loop goes 10 s → 30 s (`config.SANDBOX_TIMEOUT_SECONDS`), and the bare `run_sandboxed` default 5 s → 30 s; both are what D7 prescribes. Every sandbox test passes an explicit timeout, so none depends on the old defaults.
- On the CODE-GEN path the new semantic check can now fail a task that declares `produces_artifact` but whose code produced no DataFrame. As before, a semantic failure only marks the sandbox result failed and the critique still decides the action.
- D5's abort message uses the existing exception class with a `kind` instead of a new exception type.

### 3. Test assertions changed
- `tests/test_loop.py::TestRevisePlan::test_plan_revision_cap`: docstring "revise_plan requests are ignored" → "another revise_plan request aborts the run". Assertion was `state.plan_revisions <= MAX_PLAN_REVISIONS`; now `== MAX_PLAN_REVISIONS`, plus `state.aborted` and `"revision cap" in state.abort_reason`. Reason: D5 deliberately changes the old "ignore and carry on" behaviour.
- `tests/test_regressions.py`: only the four `xfail` decorators (and the module docstring describing them) removed; no assertion changed. (`test_plan_revision_cap_aborts` keeps `plan_revisions == MAX_PLAN_REVISIONS`, `aborted`, and "revision" in the reason for both plans.)
- No existing test depended on the step-specific semantic checks (they were only exercised through the xfail test), so none had to change for D6.
- `test_revise_plan_replaces_tail` (flat `["Q2_new?"]` replan response) passes unchanged: `parse_plan_json` still accepts the legacy flat format.

### D9 — regression found in the Phase 4 review, fixed in `fix(phase-4)`
- **Bug:** the new generic `validate_semantic_contract` fails a CODE-GEN attempt when `task.produces_artifact` is declared but no DataFrame with that variable name exists (or its columns differ from `output_schema`), yet `codegen_prompt` / `repair_prompt` never told the model the variable name or the columns. The dry run no longer exercises codegen, so nothing caught it. Reviewer probe: `produces_artifact="units_by_region"`, `output_schema=["region","avg_units"]`, answer assigned to `out`: accepted on attempt 1 before Phase 4, aborted after max retries after it.
- **Fix:**
  - `codegen_prompt` and `repair_prompt` take an optional `artifact_contract` text, rendered as its own block right after the sub-question only when non-empty (the prompts are byte-identical otherwise, tested). This is the only prompt edit.
  - `loop._artifact_contract_text(task)` builds it ("OUTPUT CONTRACT: Assign your final DataFrame to a variable named `x` with columns: a, b (any order)."); empty without `produces_artifact`; no columns line for an empty or placeholder schema. `_do_codegen` / `_do_repair` now take the `TaskNode` instead of the question string.
  - `validate_semantic_contract` compares columns as a set (order-insensitive); the reason lists `missing` and `extra` columns. `validate_output` (capability outputs, exact order) is untouched.
- **Tests:** new `tests/test_codegen_contract.py` (reviewer scenario with reordered columns is accepted on attempt 1 with the artifact registered and the codegen prompt naming `units_by_region` / `avg_units`; variable never assigned is still rejected as `revise_code` with "no DataFrame artifact was produced" and the repair prompt carries the contract; wrong columns still rejected with missing/extra; `_artifact_contract_text` unit tests; prompts unchanged without a contract). `tests/test_validator_semantic.py` gained an order-insensitivity test.
- **Assertions changed:** none (additions only). The earlier Phase 4 test `test_columns_must_match_declared_schema` still passes (it uses a genuinely different column set).
- **Results after the fix:** `pytest tests/ -q`: `253 passed`; both dry-runs finish with 5 deterministic steps.
- Note: §1.D item 9 of `REFACTOR_PLAN.md` was not present in the copy I could read on disk, so this entry follows the review message.

- **D9 follow-up (`fix(phase-4)`, "clarify contract vs result, share schema placeholders"):** the contract block now ends with "Then also set `result = <name>`." (so a model that only assigns `result` still satisfies the contract), and the placeholder list lives once in `agent/state.py` as `SCHEMA_PLACEHOLDERS`, used by `validate_semantic_contract` and `_artifact_contract_text`. Test changes: `tests/test_codegen_contract.py` asserts the new last line (contract-text unit tests and the reviewer scenario); additions only, no assertion removed. (`validate_output`'s `same_as_input` handling and `artifact_store`'s `<scalar>` schema are different concerns and were left alone.)

### 4. Results
- `pytest tests/ -q`: `243 passed` (no xfail left).
- `python main.py --dry-run` and `--dry-run --corrupt`: 5 steps, all deterministic. Run JSON: `dataset` is `clean` / `corrupted`, `total_llm_calls` 2 (plan + summary). The dry-run summary text claiming five deterministic steps is now true.
- `agent/loop.py`: 1061 lines.

### 5. Noticed, left alone
- `parsing.parse_json_list` is now unused outside its tests (the replan was its only caller). Kept because D1 is a fix to it; Phase 5/6 can drop it.
- `ruff` F841 for `last_line` in `agent/sandbox.py` (out of scope: sandbox).
- `.env.example` already documents the three variables; no change needed. `README`/`ARCHITECTURE` text is Phase 6.
- A replan that still returns the same questions will loop until the cap and then abort cleanly (covered by the D5 test).
