# AutoAnalyst Cleanup Plan

Goal: make the codebase coherent without changing what the agent does. Every phase
leaves `pytest tests/ -q` green and `python main.py --dry-run` working, and lands as
its own commit so each one can be reviewed (and reverted) on its own.

Baseline (2026-11-19): 249 tests pass, `main` is clean.

**Workflow: one phase at a time.** The implementing session does a single phase,
commits it, appends that phase's section to `REFACTOR_NOTES.md`, and stops. A separate
review happens before the next phase starts. Review feedback is fixed in an extra commit
on the same branch (`fix(phase-N): …`), never by rewriting the earlier commits.

---

## 1. What is wrong today

### A. Two artifact systems, one of them dead
- `agent/artifact_store.py` (`ArtifactStore`, `Artifact`) is the real store: the engine,
  validator, sandbox injection, observer and prompts all read from it.
- `agent/runtime.py` (`AnalysisRuntime`, `ArtifactRegistry`, a *second* class also named
  `Artifact`) is **write-only**. `loop.py` writes to `state.runtime` in three places, but
  nothing ever reads `runtime.namespace`, `runtime.registry` or `namespace_snapshot()`.
  The sandbox still gets its inputs from `artifact_store`.
- The loop writes every accepted DataFrame to both, with "keep in sync for backward
  compat" comments.

### B. Duplicated and dead code
- `agent/executors.py`: `try_execute`, `_filter_last_quarter` and `_region_revenue` copy
  logic that already lives in `capabilities.py`. Only `tests/test_executors.py` calls them.
  The one live piece is the `DeterministicExecutionResult` dataclass.
- The "last quarter" date window is computed in three places: `executors._last_quarter_window`,
  `capabilities._last_quarter_window` and inline in `loop._build_schema_summary`.
- `capabilities.py` repeats the same 9-line `DeterministicExecutionResult(...)` literal in
  each of its 8 executors.
- `_parse_plan_json` builds the same normalised task dict three times.
- Code with no effect:
  - `_sync_artifact_consumers` only writes a debug log.
  - `_build_completed_findings` just calls another function.
  - `obs_report` is computed in the deterministic path and never used.
  - `AgentRun.capability_registry` is a second registry instance that nothing uses.
  - `AgentRun.llm_call_count` is never incremented.
  - `_artifact_key` looks unused, but check before you delete it.
- Shims kept only for tests: the `AgentRun.history` getter/setter, which rebuilds the
  whole state from legacy dicts, and `_accepted_findings`.
- `main.py` and `run_demo.py` are near-duplicate entry points.

### C. `loop.py` is a 1,524-line god module
It holds the orchestrator, plan/replan parsing, critique parsing, schema summary
building, prompt-context builders, console printing, run-JSON export and the dry-run
monkeypatch. `_run_question` is about 340 lines, and its deterministic branch duplicates
the accept/register logic of the LLM branch.

### D. Bugs found during the audit (verify each with a failing test before fixing)
1. **`_parse_json_list` fallback crashes.** The regex `r"^\d+[\.)\]\s*"` has an
   unterminated character set and raises `re.error` whenever the replan response isn't
   valid JSON. (Confirmed: `re.compile` fails.) The intended pattern is `r"^\d+[\.)\]]\s*"`.
2. **Deterministic artifacts are registered twice.** `ExecutionEngine.execute` registers
   the artifact into the store, then `_run_question` registers it again. The second call
   probably returns `False` silently.
3. **Dry-run leaks.** `run(dry_run=True)` replaces `agent.loop.call_llm` on the module
   permanently, so any later `run()` in the same process also gets canned answers.
4. **Replan loses structure.** The replan uses a flat string list and rebuilds
   `state.tasks = _build_tasks(state.plan)` for *all* tasks. Completed tasks lose their
   capability, artifact and schema metadata, and new tasks get no artifact declarations.
5. **The plan-revision cap may hang or abort incorrectly.** When the cap is hit, the task
   stays `failed` and never enters working memory, so `ExecutionGraph.all_completed()` can
   never become true. Write a test to confirm what actually happens.
6. **The semantic validator is hard-coded to one demo goal.**
   `CapabilityValidator.validate_semantic_contract` branches on `task_id == "step_2"`,
   `"step_4"`, `"step_5"` and `"step_6"`, and asserts things like "step 4 must contain only
   the top region". Any other goal or plan order gets wrong rejections.
7. **The config is ignored.** `config.SANDBOX_TIMEOUT_SECONDS = 30` is never used: the loop
   uses its own `SANDBOX_TIMEOUT = 10` and the sandbox defaults to 5. The `MAX_RETRIES`,
   `SANDBOX_TIMEOUT_SECONDS` and `LOG_DIR` env vars in `.env.example` are never read.
   `_finalize_run` hard-codes `dataset_label="clean"`.

8. **The observer's facts never reach the critic.** (Found in the Phase 1 review.)
   `critique_prompt` builds `obs_block` from `observation_text`, but never puts it in the
   returned f-string. The OBSERVE stage runs on every attempt, yet its null, duplicate and
   outlier facts are thrown away, and the critic guesses instead. The fix is to put
   `{obs_block}` into the template, in the slot the docstring's STEP 3 describes. This is
   the one prompt change allowed besides D4.

9. **Codegen is never told the artifact contract.** (Introduced in Phase 4, found in its
   review.) The generic D6 check fails a codegen attempt when the declared
   `produces_artifact` DataFrame is missing, or its columns don't match `output_schema`.
   But `codegen_prompt` and `repair_prompt` never mention either one, so these tasks abort
   after every retry. Fix: pass both into the two prompts, and compare columns ignoring
   order on the codegen path. The user approved this prompt edit on 2026-11-19.

### E. Packaging and docs drift
- `pyproject.toml` and `requirements.txt` disagree:
  - The pyproject is missing `numpy` and `cerebras-cloud-sdk`.
  - It lists `pydantic-settings`, which is unused, and has `pytest` and `pytest-mock` as
    runtime dependencies.
  - `requirements.txt` lists `mistralai`, which is unused.
- README, ARCHITECTURE and PROGRESS describe models, files and flow that no longer match.
  - README says Groq llama + Gemini; the config uses `gpt-oss-120b` with a
    Cerebras → OpenRouter → Gemini chain.
  - PROGRESS says "run_demo is dry-run by default"; it isn't.
- PROGRESS.md is a 774-line historical log, not a status page.

---

## 2. Target shape

```
agent/
  config.py          constants + env overrides that are actually read
  llm_client.py      unchanged (already clean)
  state.py           TaskNode, Finding, WorkingMemory, AttemptRecord, TaskMemory
  artifact_store.py  the ONLY artifact store (Artifact, ArtifactStore)
  capabilities.py    Capability, registry, executors, DeterministicExecutionResult,
                     shared result helper
  dates.py           last_quarter_window(df): the single implementation
  execution_engine.py
  execution_graph.py
  validator.py       generic contract checks only (no task-id special cases)
  sandbox.py         unchanged except timeout default from config
  observer.py, evaluator.py, environment.py, logger.py, log_viewer.py, prompts.py
  parsing.py         NEW: parse_plan, parse_string_list, parse_critique, strip_fences
  planning.py        NEW: build_tasks, plan, replan (structured)
  schema.py          NEW: build_schema_summary (uses dates.py)
  reporting.py       NEW: all console print() output for a run
  loop.py            orchestration only: AgentRun, run(), run_task(), accept(), summary
main.py              single CLI entry point
run_demo.py          thin wrapper: main() with the demo defaults, or deleted
```

Removed: `agent/runtime.py`, `agent/executors.py`, `tests/test_runtime.py`,
`tests/test_executors.py`. Coverage for `DeterministicExecutionResult` and the date
window moves to `test_capabilities.py` and a new `test_dates.py`.

Target: `loop.py` under 450 lines, and no function in it over ~80 lines.

---

## 3. Phases (one commit each, tests green after each)

### Phase 0: Safety net
- Create the branch `refactor/cleanup`.
- Add **failing tests first** for bugs D1, D2, D3 and D5, and mark them
  `@pytest.mark.xfail(strict=True, reason="bug D#")` so the suite stays green. Later
  phases remove each marker when they fix that bug.
- Add a dry-run smoke test: `run(goal, df, dry_run=True, log_dir=tmp_path)` must finish,
  not abort, and leave 5 accepted findings.

### Phase 1: Delete dead code (no behaviour change)
- Delete `agent/runtime.py` and `tests/test_runtime.py`. Remove `AgentRun.runtime` and all
  `state.runtime` calls.
  *(Decision point: see §4.1. Default is delete.)*
- Move `DeterministicExecutionResult` into `capabilities.py`. Delete `executors.py` and
  `tests/test_executors.py`. Re-export nothing.
- Add `agent/dates.py` with `last_quarter_window(df)` and use it from capabilities and the
  schema summary. Add `tests/test_dates.py`.
- Remove these:
  - `AgentRun.capability_registry`
  - `_sync_artifact_consumers` and its calls
  - `_build_completed_findings` (call `_build_working_memory_prompt` directly)
  - the unused `obs_report` in the deterministic branch
  - `_artifact_key` only if it really is unused; `_finding_from_attempt` uses it, so
    check first
- Remove unused imports across `agent/`. Check with `python -m pyflakes agent` if it's
  available, otherwise by grep.

### Phase 2: Split `loop.py` (pure moves, no logic edits)
- Move `_parse_plan_json`, `_parse_json_list`, `_parse_critique_json` and
  `_strip_markdown_fences` → `agent/parsing.py`. Make them public with the leading
  underscore dropped. Collapse the three copies of the task-dict normalisation into one
  `_normalise_task_spec(item)` helper.
- Move `_build_schema_summary` → `agent/schema.py`.
- Move `_build_tasks`, `_task_id`, `_task_order`, `_do_plan` and `_do_replan` →
  `agent/planning.py`.
- Move every `print(...)` block (run banner, step header, stage lines, abort banner,
  summary banner) → `agent/reporting.py` as small functions. The loop calls those
  functions and does no printing itself.
- Update test imports. Add no new aliases in `loop.py`, so tests import from the new
  modules.

### Phase 3: One accept path, injectable LLM (fixes D2, D3)
- Extract `_accept(state, task, attempt, sandbox_result, trust, reason, attempt_count,
  had_execution_errors)`. It must:
  - set the task status
  - create the Finding
  - update working memory and `final_results`
  - register the artifact once
  - set `produced_artifact_name`

  Both the deterministic branch and the LLM branch call it.
- The engine already registers the artifact, so the deterministic branch must not register
  it again. Pick one owner: the engine only executes and validates, and `_accept`
  registers. Document the choice in a docstring. **Fixes D2.**
- Split `_run_question` into `_try_deterministic(state, task, df) -> bool` and
  `_run_codegen_loop(state, task, df) -> bool`.
- Replace the dry-run monkeypatch with a parameter: `run(..., llm: Callable[[str, str], str]
  | None = None)`. `dry_run=True` builds `llm = dry_run_llm` (moved to
  `agent/dry_run.py`). Pass the callable through the `_do_*` stage functions, or store it
  on `AgentRun`. Keep the `dry_run` kwarg for the CLI. **Fixes D3.**
- Increment `state.llm_call_count` inside that callable wrapper.

### Phase 4: Remaining behaviour fixes (each with its test going from xfail to pass)
- **D1:** fix the regex in `parse_string_list`.
- **D4:** replan uses the structured plan format. Add a `replan_prompt` that asks for the
  same JSON shape as `plan_prompt`, parse it with `parse_plan`, and keep the completed
  tasks' TaskNodes unchanged when rebuilding: only the tail is new. New task IDs continue
  numbering after the current max, so they never reuse an ID that's already in working
  memory.
- **D5:** when the revision cap is hit, abort with `PipelineAbortedError` and a clear
  reason, instead of advancing past a task that never completed. This matches the project
  rule "do not force-accept broken results". The existing
  `tests/test_loop.py::test_revise_plan_cap…` ("revise_plan requests are ignored") asserts
  the old behaviour, so update it and record the change in the notes.
- **D8:** put `{obs_block}` into `critique_prompt`'s returned template.
- **D6:** strip every `task_id == "step_N"` branch out of `validate_semantic_contract`. Keep
  only generic checks:
  - the artifact is a non-empty DataFrame when the task declares `produces_artifact`
  - the columns match `task.output_schema` when one is declared

  If tests depended on the step-specific checks, update them and list each change in the
  notes. *(Decision point: see §4.2.)*
- **D7:** `config.py` reads `MAX_RETRIES`, `SANDBOX_TIMEOUT_SECONDS` and `LOG_DIR` from the
  environment, with the current values as defaults. Delete `loop.SANDBOX_TIMEOUT` and use
  the config value. The sandbox's default timeout comes from config. `_finalize_run` takes
  `dataset_label` from `run()`, and `main.py --corrupt` passes `"corrupted"`.

### Phase 5: Capabilities tidy and shrinking `loop.py`
(Added in the Phase 3 review. After Phase 3, `loop.py` was 1040 lines, `_run_codegen_loop`
was 188 lines and `run()` was 132.)
- Remove the legacy test shims from §1.B: the `AgentRun.history` getter/setter and
  `_accepted_findings`. Move the tests that use them over to `debug_history` and
  `working_memory`, and record each assertion change.
- Move the finding helpers (`_finding_from_attempt`, `_attempt_for_finding`,
  `_artifact_key`, `_coerce_trust`, `_build_history_summary`,
  `_build_working_memory_prompt`, `_final_results`) into a new `agent/findings.py`.
- Move `_log` and `_finalize_run` next to the writers in `agent/logger.py`, or into a small
  `agent/run_export.py`.
- Split `_run_codegen_loop` into one function per stage: generate, execute+validate,
  observe+critique, record constraints. The loop body then just calls them in order.
- Move `run()`'s replan branch and abort handling into helpers.
- Revised target: `loop.py` 600 lines or fewer, and no function in `agent/` over 80 lines.
  This replaces the 450-line target in §2 and §5.

Capabilities:
- Add `_df_result(df, artifact_key, result_text=None)` and use it in all 8 executors, so
  each executor is just its pandas expression.
- `RevenueShare` uses `iloc[:, 1]`. Use the named `total_revenue` column instead.

### Phase 6: Entry points, packaging, docs
- Make `main.py` the only CLI. Reduce `run_demo.py` to `from main import main; main()`, or
  delete it and update the README.
- `pyproject.toml` becomes the single source of dependencies:
  - add `numpy`
  - move `cerebras-cloud-sdk` (and an OpenAI-compatible client, if OpenRouter needs one)
    under runtime or an optional `providers` extra
  - remove `pydantic-settings`
  - move pytest to `[dependency-groups] dev` / optional `dev`

  Regenerate `requirements.txt` from it, or delete it and point the README at `uv sync` /
  `pip install -e .[dev]`.
- Rewrite `README.md`, `ARCHITECTURE.md` and the top of `PROGRESS.md` to match the code.
  Move the historical phase log to `docs/HISTORY.md`, and keep PROGRESS.md as a short
  status + next steps page of 100 lines or fewer.
- Update the `agent/__init__.py` docstring.

### Explicitly out of scope
- No changes to prompt wording in `prompts.py`, except adding the structured replan
  prompt (D4).
- No changes to `llm_client.py` provider logic, `sandbox.py` isolation, `observer.py` or
  `evaluator.py` maths.
- No new capabilities, and no new features.

---

## 4. Decisions for review (defaults given)

1. **`AnalysisRuntime` (`runtime.py`)** was the last feature commit, but nothing reads it.
   *Default: delete.* The alternative is to make it the single store and delete
   `ArtifactStore`, which is a much bigger change.
2. **Semantic validator step checks.** *Default: delete the task-id checks, keep only
   generic checks.* The alternative is to move them onto specific capabilities as declared
   invariants.
3. **`run_demo.py`.** *Default: keep it as a one-line wrapper.*

## 5. Done when
- `pytest tests/ -q` passes with no `xfail` markers left from Phase 0.
- `python main.py --dry-run` and `python main.py --dry-run --corrupt` both finish.
- `loop.py` is 600 lines or fewer, and no function in `agent/` is over 80 lines.
  The exception is `prompts.py` and `sandbox.py`, which are out of scope: their long
  functions are prompt templates and the process-isolation worker. (This exemption was
  agreed in the Phase 5 review.) `runtime.py` and `executors.py` are gone.
- `grep -rn 'step_[0-9]' agent/` finds only `_task_id`-style generation, with no
  special-casing.
- `REFACTOR_NOTES.md` lists every deviation from this plan and every test whose
  *assertion* (not just its import) changed, with the reason.
