# AutoAnalyst Architecture Guide for Coding Agents

> **Notice to AI Agents**: this is the architectural reference for AutoAnalyst. Read it
> before changing the codebase. It describes the code as it is now; the old phase-by-phase
> log lives in `docs/HISTORY.md` and is not a reference.

## 1. System overview

AutoAnalyst is a self-correcting data-analysis agent. It turns a goal ("explain the revenue
trend by region over the last quarter ...") into a dependency graph of tasks, executes each
task **deterministically through a registered capability whenever one matches** and through
sandboxed LLM-written pandas code otherwise, has an independent LLM critique every
code-generated result, and finishes with a summary built only from accepted findings.

```text
User goal
 → PLAN         (LLM → structured task specs → TaskNodes → ExecutionGraph)
 → for each ready task:
     DETERMINISTIC   capability registry + execution engine + validator
       └─ no match / fails →
     CODE-GEN LOOP   generate → sandbox + semantic contract → observe → critique
 → ACCEPT        (single accept path: finding, working memory, artifact registration)
 → SUMMARY       (LLM, from accepted findings only)
 → run JSON + JSONL log
```

## 2. Module map (`agent/`)

| Module | Responsibility |
|---|---|
| `loop.py` | Orchestration only: `AgentRun` state, `run()`, task dispatch, the single `_accept()` path, `_try_deterministic()`, `_run_codegen_loop()`, replan/abort handling. Re-exports `PipelineAbortedError` (defined in `errors.py`). |
| `stages.py` | Stage implementations used by the loop: `do_codegen`, `do_repair`, `do_critique`, `do_summary`, `artifact_contract_text`, and the per-attempt steps `generate_code`, `observe_and_critique`, `record_constraints`, `record_attempt`. |
| `planning.py` | `build_tasks`, `do_plan`, `do_replan`, `apply_replan`, `sync_task_capabilities`, task-ID helpers, and `plan_guidance` (the capabilities and example plan the planner sees, built from the data's columns). |
| `parsing.py` | Defensive parsers for LLM output: `parse_plan_json`, `parse_critique_json`, `strip_markdown_fences`. |
| `schema.py` | `build_schema_summary(df, scope)`: the dataset description that goes into prompts; adds the pre-computed last-quarter window only when `scope.quarter`. |
| `dates.py` | `detect_date_column`, `goal_mentions_quarter` / `goal_mentions_time_window`, `TimeScope` (decided once per run), and `last_quarter_window` / `last_quarter_constants`: the one implementation of the "last quarter" window. |
| `inputs.py` | `load_dataset` (CSV/Parquet, date-like string columns parsed), `load_data_dictionary`, `format_data_dictionary`. |
| `findings.py` | Promotes accepted attempts to `Finding`s and builds the prompt-facing views of memory (`build_history_summary`, `build_working_memory_prompt`, `final_results`). |
| `scheduling.py` | `next_ready_task`, `ensure_dependencies_available`, `mark_failed_and_skipped`. |
| `errors.py` | `PipelineAbortedError` (with a `kind`: retries exhausted, plan-revision cap, scheduling failure). |
| `reporting.py` | **All** console output of a run (banners, stage lines, abort banner). Nothing else prints. |
| `run_export.py` | `log_entry` (JSONL line, I/O errors swallowed) and `finalize_run` (writes `runs/<run_id>.json`). |
| `logger.py` | `LogEntry`, `write_log_entry`, `read_run_log`, `export_run_json`, `new_run_id`. |
| `log_viewer.py` | Pretty-prints a run JSON (`python -m agent.log_viewer runs/<id>.json`). |
| `state.py` | Typed state: `TaskNode`, `AttemptRecord`, `Finding`, `WorkingMemory`, `TaskMemory`, `ExecutionDifficulty`, `SCHEMA_PLACEHOLDERS`. |
| `execution_graph.py` | `ExecutionGraph`: ready-task selection and artifact lineage over `TaskNode`s. |
| `capabilities.py` | `Capability`, `CapabilityRegistry`, `DeterministicExecutionResult`, the 8 deterministic executors (each ends in `_df_result`) and the default registry. |
| `execution_engine.py` | `ExecutionEngine.execute`: resolve capability → check inputs → run executor → validate output and semantic contract. **Never writes to the artifact store.** |
| `validator.py` | `CapabilityValidator`: `validate_context`, `validate_output` (exact schema, capability outputs) and the generic `validate_semantic_contract`. |
| `artifact_store.py` | `ArtifactStore` / `Artifact`: the only artifact store; frozen, typed artifacts with producer, consumers, confidence, provenance. |
| `sandbox.py` | `run_sandboxed`: process-isolated execution of generated code with a hard timeout. |
| `observer.py` | Deterministic data-quality facts given to the critic: generic checks (nulls, duplicate rows, negative values, out-of-range datetimes, IQR outliers) plus the sales checks the evaluator scores. `observe_dataset` runs once per run; `observe` adds per-attempt notes. |
| `environment.py` | `EnvironmentManifest`: what the sandbox allows, passed to codegen. |
| `prompts.py` | **All** prompt templates. |
| `llm_client.py` | `call_llm(prompt, role)`: the only gateway to LLM providers (see §5). |
| `config.py` | Model names, retry/timeout/log-dir constants and their environment overrides. |
| `dry_run.py` | `dry_run_llm`: canned responses so the pipeline runs without API keys. |
| `evaluator.py` | Scores the observer against the corruption manifest (precision/recall/F1). |

Entry points: `main.py` (the CLI: `--data PATH` for a CSV/Parquet file, `--describe PATH` for
a data dictionary), `run_demo.py` (thin wrapper around `main.main()`).
Synthetic data: `data/generate_data.py` (`generate_sales_data`, `inject_corruption`).

## 3. Execution pipeline in detail

0. **Inputs** (`loop._init_run`). Besides the goal and `df`, `run()` takes an optional
   `data_dictionary` (free text or `{column: description}`), normalised by
   `inputs.format_data_dictionary` and stored on `AgentRun.data_dictionary`. It is passed to
   the PLAN, RE-PLAN, CODE-GEN, REPAIR, CRITIQUE and SUMMARY prompts; without one, those
   prompts forbid guessing what coded values mean. `TimeScope.from_goal(goal, df)` decides
   once whether the "last quarter" machinery applies (see *Time scope* below).
1. **PLAN** (`planning.do_plan`). The planner returns a JSON array of task objects
   (`question`, `capability`, `produces_artifact`, `consumes_artifacts`, `output_schema`),
   parsed by `parse_plan_json` (legacy flat string arrays and numbered lines are still
   accepted). Tasks chain `depends_on` to every earlier task; `ExecutionGraph` schedules them.
2. **Deterministic first** (`loop._try_deterministic`). `ExecutionEngine.execute` resolves a
   capability (the task's declared one, else term matching), checks its required artifacts,
   runs the executor, then runs `validate_output` and the semantic contract. On success the
   result is accepted without any LLM call; on any failure it returns `None` and the task
   falls through to code generation.
3. **CODE-GEN loop** (`loop._run_codegen_loop`), up to `max_retries_per_step` attempts, each:
   `generate_code` (CODE-GEN prompt, REPAIR prompt after the first attempt) →
   `_execute_and_validate` (sandbox, then the semantic contract) → `observe_and_critique`
   (observer facts → independent critique) → `record_constraints` → `record_attempt`, then
   route on the critique's action: `accept`, `revise_code` (next attempt) or `revise_plan`.
4. **Replan** (`loop._replan`). Allowed `MAX_PLAN_REVISIONS` (2) times. The replan prompt asks
   for the same structured task shape; `apply_replan` keeps completed tasks as the same
   `TaskNode` objects and appends new ones whose IDs continue after the highest existing
   `step_N`. A further request aborts the run (`PipelineAbortedError`).
5. **Abort.** Retries exhausted, plan-revision cap hit, or a stuck scheduler raises
   `PipelineAbortedError`; the run records `aborted`, the reason, and still writes its
   logs and run JSON.
6. **SUMMARY** (`stages.do_summary`). The summary prompt sees only `final_results`
   (question / answer / trust / artifact key), never code, retries or critiques.

### Time scope (no dataset assumptions)

`AgentRun.scope` is a `TimeScope(date_column, quarter, window)`:

- `date_column`: `detect_date_column(df)`: a column named `date`, else a datetime64 column
  (pickup/start preferred over dropoff/end), else a date-like string column.
- `quarter`: the goal mentions a quarter (`last quarter`, `Q3`, `quarterly`) **and** a date
  column exists. Only then does the schema summary carry `LAST_QUARTER_START/END`, and only
  then do the PLAN / CODE-GEN / REPAIR / CRITIQUE prompts include the `quarter_df` rules
  (`time_scoped=True`). The `FilterByDateRange` capability (`requires_time_scope=True`) is
  only offered to the planner then, and its executor refuses tasks that do not ask for the
  last quarter, so a generic "filter ..." task goes to code generation.
- `window`: any other time window in the goal (month, week, a year, "over time"): the
  schema gets a one-line note to anchor it on the data's latest date, not today.

The sandbox always defines `LAST_QUARTER_START/END` when there is a date column (harmless
when unused). Capability `input_schema` may use the token `"date"` for "the detected datetime
column".

`planning.plan_guidance` builds the planner's capability list and example from
`CapabilityRegistry.applicable(columns, time_scoped)`: capabilities whose input columns and
required artifacts can be satisfied. When at least two chain together (sales data with a
quarter goal: FilterByDateRange → GroupRevenueByRegion → FindMaximumRevenueRegion) the
example is that chain; otherwise it is a two-task example built from one numeric and one
categorical column of the data. When adding a capability, give it an `example_question`.

### The artifact contract

A task that declares `produces_artifact` must leave a DataFrame under that variable name in
the sandbox. `validate_semantic_contract` (generic, never keyed on task IDs) requires it to
be a non-empty DataFrame whose columns match `output_schema` as a *set* (placeholders
`["same_as_input"]` / `["<scalar>"]` are skipped). The CODE-GEN and REPAIR prompts are told
the contract (`stages.artifact_contract_text`): the variable name, the columns, and to also
set `result = <name>`. Capability outputs are additionally checked exactly (column order
included) by `validate_output`.

### One accept path, one registration owner

`loop._accept` is the only place that marks a task accepted, creates the `Finding`, updates
working memory and `final_results`, and registers the artifact (once). The engine only
executes and validates. Provenance is `[capability_id, task_id]` for deterministic results.

### Injectable LLM

`run(goal, df, ..., llm=None, dry_run=False, dataset_label="clean", data_dictionary=None)`. `llm` is any callable
`(prompt, role) -> str`; by default `llm_client.call_llm`, or `dry_run.dry_run_llm` when
`dry_run=True`. It is wrapped to count calls (`AgentRun.llm_call_count`, recorded as
`total_llm_calls` in the run JSON) and stored on `AgentRun.llm`; every stage calls
`state.llm`. Nothing is monkeypatched, so tests pass `llm=` or patch `agent.loop.call_llm`.

### Observer

`observe_dataset(df)` computes the dataset-level facts once per run (cached on
`AgentRun.data_quality`); `observe(df, sandbox_result, keys, baseline=...)` copies it and
adds per-attempt notes. Generic checks: duplicate rows, negative values per numeric column,
datetimes outside the 1%–99% span widened by half its length (and before 1900 or in the
future), and IQR outliers per numeric column (columns with ≤10 distinct values are treated
as codes and skipped). The sales fields (`duplicate_txn_ids`, `negative_revenue_rows`,
`zero_units_positive_revenue`, `revenue_outliers_iqr`) are kept because `evaluator.py`
scores them against the corruption manifest.

## 4. State management (crucial constraints)

State is strictly separated; do not mix these views.

- **`debug_history` (`list[AttemptRecord]`)**: every attempt, including failures and
  retries. Used for the audit log only. **Never** pass it to the summary or planning prompts.
- **`working_memory` (`WorkingMemory`)**: accepted `Finding`s only. `final_results` is the
  same list in plan order.
- **`tasks` / `execution_graph`**: the dependency graph. A task whose dependencies are not
  accepted is skipped and the run aborts predictably.
- **`task_memories`**: per-task constraints accumulated from failed attempts.
- **`artifact_store`**: validated DataFrames, immutable once registered (50 MB cap each).

## 5. LLM providers (`llm_client.py`, `config.py`)

`call_llm(prompt, role)` picks a provider chain per role; each level is retried once with
backoff and the next level is tried on 429 / 5xx / timeout / SDK errors (400 and 401 surface
immediately). If every level fails, `LLMUnavailableError` is raised.

| Role | Chain (primary → fallbacks) |
|---|---|
| `plan` | Groq → OpenRouter |
| `codegen` | Groq → Cerebras → OpenRouter |
| `critique` | Groq → Cerebras → Gemini |
| `observe` (summary) | Groq → OpenRouter |

Models (all in `config.py`): `openai/gpt-oss-120b` on Groq and Cerebras;
OpenRouter `meta-llama/llama-3.3-70b-instruct:free` (plan) and
`mistralai/codestral-2501:free` (codegen); Gemini `gemini-2.5-flash`. OpenRouter is called
over HTTP with `urllib` (no SDK).

## 6. Configuration (environment / `.env`)

| Variable | Default | Used for |
|---|---|---|
| `GROQ_API_KEY`, `CEREBRAS_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY` | none | provider keys; a missing key makes that level fail over |
| `MAX_RETRIES` | `5` | CODE-GEN attempts per task (`config.MAX_PIPELINE_RETRIES`) |
| `SANDBOX_TIMEOUT_SECONDS` | `30` | wall-clock limit per sandbox execution (also `run_sandboxed`'s default) |
| `LOG_DIR` | `runs` | where JSONL logs and run JSON are written |

Blank values mean "use the default"; an unparsable number raises `ValueError` naming the variable.

## 7. Implementing new features (workflow rules)

1. **Prefer capabilities over prompts.** Add a deterministic executor in `capabilities.py`
   (return `_df_result(...)`), add a `Capability` entry to `_DEFAULT_CAPABILITIES`, and let the
   registry resolve it. Do not fix analytical gaps by tweaking the CODE-GEN prompt.
2. **Use the execution graph.** Tasks declare `produces_artifact` / `consumes_artifacts` /
   `output_schema` so the engine and validator can check them.
3. **Enforce contracts in `validator.py`**, generically. No checks keyed on task IDs or plan
   position.
4. **Keep console output in `reporting.py`** and prompts in `prompts.py`.

## 8. Hard engineering constraints

- **Never call LLM SDKs directly.** Everything goes through `llm_client.call_llm(prompt, role)`
  (inside a run: through `state.llm`).
- **No unbounded loops.** `max_retries_per_step` and `MAX_PLAN_REVISIONS` are strictly
  enforced; when exhausted, raise `PipelineAbortedError`. Do not force-accept broken results.
- **Test-driven.** `pytest tests/ -q` must pass before you finish. Update tests when you
  change the loop, validator or capabilities. LLM calls are mocked in tests; no network.
- **Log everything.** New loop stages write to the structured run log (`run_export.log_entry`
  / `logger.py`); that trace is the primary deliverable.
