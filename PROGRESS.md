# PROGRESS.md — AutoAnalyst: Self-Correcting Data Pipeline Agent

> **Source of truth** for project state. Update this file at the end of every
> session before stopping. The "NEXT STEPS" section is the handoff to the next
> agent or session.

---

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | LLM client wrapper + synthetic data | ✅ Complete |
| 2 | Safe code execution sandbox | ✅ Complete |
| 3 | Five-stage PLAN→CODEGEN→EXECUTE→CRITIQUE loop | ✅ Complete |
| 3.5 | Architectural Redesign (ArtifactStore) | ✅ **Complete** |
| 4 | Data quality checks in OBSERVE stage | 🔲 Next |
| 5 | Evaluation against corruption manifest | 🔲 Planned |
| 6 | Blog post run log + write-up | 🔲 Planned |

---

## Current Repo Structure

```
AutoAnalyst/
├── agent/
│   ├── __init__.py
│   ├── config.py             # model names, retry caps, role constants
│   ├── llm_client.py         # call_llm() — Groq-first, Gemini-fallback
│   ├── sandbox.py            # run_sandboxed() — process-isolated exec
│   ├── state.py              # typed debug history, working memory, findings, tasks
│   ├── artifact_store.py     # in-process registry for pre-validated DataFrames
│   ├── logger.py             # LogEntry dataclass + JSONL writer
│   ├── prompts.py            # All prompt-template functions (no inline strings)
│   └── loop.py               # AgentRun + run() — five-stage orchestration
├── data/
│   └── generate_data.py      # generate_sales_data() + inject_corruption()
├── tests/
│   ├── __init__.py
│   ├── test_llm_client.py    # 15 tests ✅
│   ├── test_sandbox.py       # 25 tests ✅
│   ├── test_artifact_store.py# 28 tests ✅
│   └── test_loop.py          # 47 tests ✅
├── run_demo.py               # Demo script — runs loop on clean synthetic data
├── .env.example              # Key template — copy to .env and populate
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

---

## ⚠️ Hard Dependencies Carried Forward

### Phase 2 → Phase 3: Sandbox result convention
> Agent-generated code MUST assign its final answer to `result`.
> The CODE-GEN prompt in `agent/prompts.py` enforces this explicitly.

### Phase 3 → Phase 4: Schema summary format
> `_build_schema_summary()` in `agent/loop.py` produces the schema string
> fed to all LLM prompts. It shows column names, dtypes, null%, and 3 sample
> values. Any changes to this format must be reflected in prompt templates.

---

## Phase 3 — What Was Built

### Files

| File | Notes |
|------|-------|
| `agent/logger.py` | `LogEntry` dataclass + `write_log_entry()` + `read_run_log()` |
| `agent/prompts.py` | `plan_prompt`, `replan_prompt`, `codegen_prompt`, `critique_prompt`, `summary_prompt` |
| `agent/loop.py` | `AgentRun` dataclass + `run()` orchestrator + all stage functions |
| `agent/state.py` | `AttemptRecord`, `Finding`, `TaskNode`, `WorkingMemory`, `ExecutionDifficulty` |
| `tests/test_loop.py` | 36 tests — all passing |
| `run_demo.py` | Demo script for live runs |

### Test Results

```
tests/test_llm_client.py     — 15 passed  ✅
tests/test_loop.py           — 47 passed  ✅
tests/test_sandbox.py        — 25 passed  ✅
tests/test_artifact_store.py — 28 passed  ✅
Total: 115 passed, 0 failed
```

---

## Phase 3.5 — Architectural Redesign (ArtifactStore)

Driven by excessive recomputation (e.g. repeatedly applying date filters) and the observer making unsupported claims, this phase redesigned data flow and state accumulation.

### What Was Built

| File | Change |
|------|--------|
| `agent/artifact_store.py` | In-process registry: `artifact_key → pd.DataFrame`. Safe pickling checks. |
| `agent/state.py` | `Finding.produced_artifact_name`, `TaskNode.produces_artifact` + `consumes_artifacts`. |
| `agent/sandbox.py` | `artifacts` namespace injection and `extract_artifact` extraction logic. |
| `agent/prompts.py` | Structured plan JSON format with artifact definitions. Explicit None-checking for artifacts in codegen/repair. Strict grounding constraints for summary reporting. |
| `agent/loop.py` | Structured task parsing (`_parse_plan_json`), artifact wiring around CODE-GEN and EXECUTE. |
| `tests/test_artifact_store.py`| Comprehensive tests for artifact lifecycle and metadata. |

### Problems Solved

1. **Repeated recomputation**: Step 1 registers `quarter_df`; steps 2-N read from `artifacts["quarter_df"]` directly in the sandbox instead of filtering `df` from scratch.
2. **Disconnected dependencies**: `TaskNode` now carries `consumes_artifacts` which drives the sandbox namespace injection.
3. **Flat plans**: The planner outputs structured JSON `[{"question": ..., "produces_artifact": ..., "consumes_artifacts": [...]}]` rather than string arrays.
4. **Unsupported summary claims**: The observer is bound to a strict reporting contract where each claim must cite a specific finding's `artifact_key`.

### Loop Architecture (current)

```
run(goal, df)
  │
  ├── PLAN: call_llm(plan_prompt, role="plan")
  │         → parse JSON list of 3-6 sub-questions
  │         → build TaskNode graph with explicit dependencies
  │
  └── for each sub_question in plan:
        │
        ├── DEPENDENCY CHECK: all parent TaskNode findings must exist in WorkingMemory
        │
        └── for iteration in range(max_retries_per_step):
              │
              ├── CODEGEN:  call_llm(codegen_prompt, role="codegen")
              │             → Receives `AVAILABLE ARTIFACTS` description
              │             → _strip_markdown_fences() post-processing
              │             (includes previous_code + critique_reason on retry)
              │             (receives WorkingMemory only, never failed attempts)
              ├── EXECUTE:  run_sandboxed(code, df, artifacts=namespace, extract_artifact=...)
              ├── CRITIQUE: call_llm(critique_prompt, role="critique")
              │             (receives accepted findings/code previews only)
              │             → parse {"trust", "reason", "action"}
              │
              ├── every attempt        → append AttemptRecord to debug_history + JSONL
              ├── action="accept"      → promote to Finding in WorkingMemory + final_results
              ├── action="revise_code" → continue loop (up to max_retries)
              │                         on exhaustion → PipelineAbortedError
              │                         → run() catches, sets state.aborted=True, returns
              └── action="revise_plan" → replan remaining questions (cap=2)
                                         carry forward accepted findings
  │
  └── SUMMARY: _final_results(state)      ← question/answer/trust/artifact_key only
               call_llm(summary_prompt, role="observe")
               → reporter-style conclusion grounded in explicit findings
```

### State Separation (current)

| State object | Contains | Explicitly excludes |
|--------------|----------|---------------------|
| `debug_history: list[AttemptRecord]` | every attempt, code, sandbox result, critique, retry action | summary evidence |
| `working_memory: WorkingMemory` | accepted `Finding` objects only | failed attempts, exceptions, rejected critiques |
| `final_results: list[Finding]` | accepted findings in plan order | iteration/retry details |
| `tasks: list[TaskNode]` | task IDs, questions, dependencies, status | code/debug traces |
| `ExecutionDifficulty` | attempts, execution-error metadata | answer confidence |

### Critique Prompt — STEP 0 First (code verification before memory)

The critique follows a strict ordered process:

| Step | Check |
|------|-------|
| **STEP 0** | **CODE VERIFICATION** — read the code line by line, determine what it *should* produce before looking at the result |
| STEP 1 | Execution failure — if FAILED or result=None, must revise_code |
| STEP 2 | Code logic errors — wrong column, aggregating before filtering, bad denominator |
| STEP 3 | Data quality: null handling, duplicate rows, sample size, outlier dominance — all verified from the code, not guesswork |
| STEP 4 | Contradiction check — **read prior code snippets in history** to compare filters; scope differences are only acceptable if visible in the prior code |
| STEP 5 | Logical consistency — impossible values (avg > total, negative revenue, etc.) |

Key rules baked into the prompt:
- Do NOT flag `revise_code` solely because a number differs from a prior finding — verify the code's scope first
- History now includes a 200-char code preview per prior step so the critic can actually compare filter operations

---

## Post-Phase-3 Live Demo Fixes (2026-06-19, same session)

### Run 1 — run_1acd99f8 (Phase 3 baseline)

```
6 sub-questions | 15 history entries | 3 SyntaxErrors | 4 LOW TRUST force-accepts
Issues: markdown fences, contradiction over-flagging, force-accept cascade
```

### Fixes applied after Run 1

| # | Fix | File | Impact |
|---|-----|------|--------|
| F1 | `_strip_markdown_fences()` post-processing on every CODE-GEN output | `loop.py` | Eliminated markdown-fence SyntaxErrors (3 of 4 failures in run_1acd99f8) |
| F2 | Critique restructured: STEP 0 code verification before memory comparison | `prompts.py` | Critic now reads code before comparing numbers to history |
| F3 | History summary includes 200-char code preview per prior step | `loop.py` `_build_history_summary` | Critic can compare actual filter operations across steps |
| F4 | STEP 4 contradiction check: reads prior code snippets to distinguish scope-diff vs logic-error | `prompts.py` | No more false contradictions from different time windows |
| F5 | Critique reason print extended: 120 → 400 chars | `loop.py` | Full critique verdict visible on console |
| F6 | `PipelineAbortedError` — max retries exhausted → abort, not force-accept | `loop.py` | Prevents cascading low-trust results into downstream steps |
| F7 | `_accepted_findings()` — summary receives one entry per question (accepted only) | `loop.py` | Summary LLM never sees intermediate failures as numbered "findings" |
| F8 | `summary_prompt` findings block: dropped `[ACCEPT]` action label, added `[accepted after N attempts]` retry note | `prompts.py` | Clean narrative — retries noted, intermediate failures invisible |
| F9 | `_final_results()` sends SUMMARY only `{question, answer, trust}` | `loop.py` | Summary can no longer see retry counts, code, exceptions, or raw critiques |
| F10 | Removed "accepted after N attempts" from summary prompt | `prompts.py` | Retries no longer imply lower answer confidence |
| F11 | CODE-GEN prompt now receives original goal + accepted findings | `prompts.py`, `loop.py` | Dependent steps have global scope/context, e.g. "last quarter" |
| F12 | CRITIQUE prompt now checks goal-scope and parent-finding consistency | `prompts.py`, `loop.py` | Critic should reject "South" when accepted last-quarter totals imply "West" |
| F13 | `_build_history_summary()` includes accepted findings only | `loop.py` | Critic memory excludes failed retry noise while retaining code previews |
| F14 | Added typed state model: `AttemptRecord`, `Finding`, `TaskNode`, `WorkingMemory`, `ExecutionDifficulty` | `state.py`, `loop.py` | Debug history, operational memory, dependencies, and final evidence are structurally separate |
| F15 | Accepted attempts are promoted to `Finding`; rejected attempts remain only in `debug_history` | `loop.py` | Intermediate failures cannot enter summary/working memory through shared dict history |
| F16 | `TaskNode` dependency graph built from the plan; missing dependencies abort predictably | `loop.py` | Downstream tasks cannot run without accepted parent findings |
| F17 | Failure propagation marks failed task and downstream tasks as `failed`/`skipped` | `loop.py` | Pipeline state is predictable after abort |
| F18 | `ExecutionDifficulty` records retries separately from `answer_confidence` | `state.py`, `loop.py` | Retry count can no longer be represented as lower answer trust |

### Run 2 — run_27f116f0 (after F1–F5)

```
6 sub-questions | 6 accepted | 0 contradiction loops | All HIGH TRUST
Issue identified: summary still referenced "South vs West" confusion from
  Step 1 (quarter filter) vs Step 2 (full data) — summary saw both narratives
```

### Run 3 — run_db627269 (after F1–F5)

```
6 sub-questions | Step 1 exhausted retries → force-accepted (old behaviour)
→ Steps 2–6 cascaded on broken Step 1 result
→ Summary said "Findings 6 and 7 are LOW TRUST" (these were intermediate iterations)
Triggered: F6 (abort) + F7 (summary working memory)
```

### Run 4 — run_e3e7ca61 (after F1–F6, before F7)

```
6 sub-questions | Step 1 hit Empty DataFrame edge-case → accepted by critic
  (correct: data had no rows matching the quarter filter)
Summary still numbered intermediate failures as findings 6 and 7
Triggered: F7 + F8
```

### Run 5 — run_0dd8a2b7 (after F1–F8)

```
6 sub-questions | 6 accepted | Step 1 accepted after 3 CODE-GEN attempts
Issues identified:
  - SUMMARY inferred LOW TRUST from np.float64(8003.595000000001), despite critique trust=high
  - SUMMARY treated retry count as answer uncertainty
  - Step 2 recomputed full-dataset top region ("South") even though Step 1
    last-quarter totals showed West highest (46384.76 > 45317.93)
Triggered: F9–F13
```

### Architecture Hardening — after run_0dd8a2b7

```
Tests: 79 passed
Changes:
  - Replaced loose shared history as operational memory with typed state views
  - Kept every retry in debug_history for audit/logging
  - Promoted only accepted attempts into WorkingMemory and final_results
  - Added TaskNode dependency checks and failure/skipped status propagation
  - Split ExecutionDifficulty from answer_confidence
```

### Phase 3.5 — Progressive Constraint-Satisfaction Loop

Implemented a robust error-recovery loop that accumulates context over execution failures to steer the LLM away from repeat mistakes and capability violations.

```
Tests: 79 passed
Changes:
  - Created `EnvironmentManifest` to track supported packages, globals, and dynamically discovered constraints.
  - Added `TaskMemory` to `state.py` for tracking per-task constraints across retries.
  - Refactored `sandbox.py` to identify capability constraints (ImportError, NameError, PermissionError) vs execution errors.
  - Created `repair_prompt` in `prompts.py` for targeted code fixing that preserves working logic.
  - `_run_question` now intercepts execution errors before calling the LLM critic (saving tokens) and aggregates feedback into the loop.
  - Verified via `run_demo.py` that the pipeline safely traps capability constraints and passes them to the next iteration.
```

### 💡 Key Architectural Insight for the Blog: Infrastructure vs. LLM Errors

During the progressive constraint implementation, an interesting pattern emerged that is a massive gap in agentic systems today: **The agent correctly identified failure states but had no way to distinguish "I wrote bad code" from "the infrastructure is broken."**

In one run, the sandbox was incorrectly extracting the `result` variable due to a namespace bug. The Critique model correctly flagged "no result variable set", but the Code-gen model—assuming its code was to blame—spiraled into increasingly complex attempts to fix a problem that didn't exist in the code. Both look identical from the critique's perspective. 

*Insight:* Agent architectures need a way to surface "this might be an infrastructure problem" versus a reasoning problem, otherwise the agent will be blamed for system flaws and waste retries trying to fix the unfixable. The progressive constraint loop surfaced this gap beautifully.

---

## Decisions Made

### Phase 3 Decisions

#### 10. All prompts in `agent/prompts.py` (not inline)
**Why:** Prompt iteration is the highest-leverage activity in Phase 4–5.
Isolating prompts in one module means you can change a prompt, re-run the
demo, and see the effect without hunting through loop logic.

#### 11. Critique JSON parsed defensively — malformed → revise_code, never accept
**Why:** `_parse_critique_json()` extracts the first `{...}` block with regex,
validates `trust` and `action` against allowed values, and defaults to
`{"trust": "low", "action": "revise_code"}` on any parse failure. A broken
critique can never silently accept a bad result.

#### 12. `revise_plan` replaces only the *tail* of the plan (not the whole plan)
**Why:** Re-investigating already-accepted findings wastes LLM calls and
contradicts established results. `replan_prompt` explicitly feeds back the
high-trust accepted findings.

#### 13. `LogEntry` written after every iteration (not batched)
**Why:** If the run crashes mid-way, the JSONL log up to that point is still
valid. Post-run analysis can pick up where it left off.

#### 14. `_strip_markdown_fences` applied after every CODE-GEN call
**Why:** Demo run_1acd99f8 showed 3 of 6 steps failing with SyntaxError
because llama-3.3-70b-versatile wraps output in ` ```python ``` ` fences
despite prompt instructions. Post-processing is more reliable than prompt
pressure alone. Handles ` ```python`, ` ```py`, and plain ` ``` ` variants.

#### 15. Critique is code-first, not memory-first
**Why:** The critic's primary job is to verify the CODE LOGIC, not compare
result strings against memory. History now includes code previews. The
contradiction check explicitly requires reading prior code snippets to
determine if scope differs before flagging.

#### 16. `PipelineAbortedError` — hard stop on max retries exhausted
**Why:** Force-accepting a step that can't produce a trustworthy result
(old behaviour) cascades errors into all downstream steps. A broken
foundational step (e.g. "regional totals") makes every subsequent step that
builds on it unreliable. A clean abort with a clear message is safer.
`AgentRun` now exposes `state.aborted` and `state.abort_reason`.

#### 17. `_accepted_findings()` — summary working memory is one entry per sub-question
**Why:** `state.history` contains every iteration of every step. With a
3-iteration step, the history has 3 entries for that question. The summary LLM
was numbering them `Findings 6, 7, 8` and treating the intermediate failures
as unresolved inconsistencies in the narrative. `_accepted_findings()` filters
to exactly one entry per sub-question (the final accepted one), in plan order.
Intermediate iterations are internal loop state, not analytical findings.

#### 18. `_run_with_responses` helper passes `max_retries_per_step` explicitly
**Why:** Tests were hardcoding `max_r = 3` but calling `run()` without
overriding `config.MAX_PIPELINE_RETRIES` (which is 5). The mock side_effect
lists would run out mid-loop, causing `StopIteration`. The helper now accepts
`max_retries_per_step` and passes it through to `run()`.

#### 19. SUMMARY is a reporter, not a critic
**Why:** Demo run_0dd8a2b7 showed the summary stage inventing caveats from
debug artifacts: it treated retry count as uncertainty and interpreted
`np.float64(8003.595000000001)` as a LOW TRUST rounding problem even though
the accepted critique was `trust=high`. The summary prompt now receives only
structured final results: `question`, `answer`, and `trust`. It is explicitly
forbidden from inferring new trust levels or interpreting missing retry/debug
data.

#### 20. Debug history and final results are separate views of state
**Why:** `state.history` remains the full audit/debug log with every attempt,
code snippet, sandbox result, and critique. `_final_results()` creates the
summary-facing view and intentionally strips code, retries, exceptions, raw
critiques, and iteration counts. This preserves the blog-worthy audit log
without letting the reporting stage reason over failed attempts.

#### 21. Goal scope is carried into CODE-GEN and CRITIQUE
**Why:** run_0dd8a2b7 exposed scope drift: Step 1 answered "last quarter"
regional totals, but Step 2 recomputed top region over the full dataset and
the critic accepted it. `codegen_prompt()` now receives the original goal and
accepted findings; `critique_prompt()` now has an explicit goal-scope and
dependency-consistency check. If the original goal says "last quarter",
dependent revenue/top-region/trend/category computations must keep that scope
unless the sub-question explicitly says otherwise.

#### 22. Debug history is no longer operational memory
**Why:** A shared `history` list made it too easy for failed attempts,
exceptions, and rejected critiques to leak into prompts and summaries.
`state.debug_history` now stores every attempt for audit/debugging, while
`state.working_memory` stores only accepted `Finding` objects. The old
`state.history` name remains as a compatibility view over debug attempts, but
new logic must not use it as analytical memory.

#### 23. Findings are first-class state, not filtered log rows
**Why:** Filtering debug logs at summary time still leaves every downstream
stage one helper call away from reading rejected attempts. Accepted attempts
are now explicitly promoted to `Finding` objects with answer, confidence,
evidence, artifact key, dependencies, and execution difficulty metadata.
Downstream prompts and summary payloads read from these findings.

#### 24. Task dependencies are explicit
**Why:** The prior loop relied on plan order by convention but did not model
prerequisites. `TaskNode` now records dependencies, and the loop checks that
all parent findings exist in `WorkingMemory` before executing a task. If a
dependency is unavailable, the pipeline aborts with a clear reason instead of
continuing with missing state.

#### 25. Execution difficulty is not answer confidence
**Why:** Retries, exceptions, and code-generation friction describe how hard it
was to produce an accepted answer. They do not lower trust in an answer that
the critic accepted as high confidence. `ExecutionDifficulty` records attempts
and execution errors separately from `Finding.answer_confidence`; summaries see
only answer confidence.

---

## NEXT STEPS — Phase 4: Data Quality Checks in OBSERVE Stage

### Problem
Currently the loop has no OBSERVE stage — it goes CODE-GEN → EXECUTE → CRITIQUE.
The critique has the full checklist but operates on the `result` repr (a string).
It cannot actually *compute* null counts, duplicate rates, or outlier statistics.

### Goal
Add an OBSERVE stage between EXECUTE and CRITIQUE that:
1. Runs deterministic data-quality checks on the **live DataFrame** (not the repr string).
2. Feeds a structured quality report into the CRITIQUE prompt.
3. Makes critique actionable with actual numbers ("14 null revenue rows = 2% of data")
   instead of model speculation.

### Files to create / modify

| File | Change |
|------|--------|
| `agent/observer.py` | `observe(df, sandbox_result)` → `ObservationReport` dataclass |
| `agent/prompts.py` | Update `critique_prompt` to accept `ObservationReport` |
| `agent/loop.py` | Insert observer call between EXECUTE and CRITIQUE |
| `tests/test_observer.py` | Unit tests for each check against corrupted dataset |

### ObservationReport fields (proposed)
```python
@dataclass
class ObservationReport:
    null_counts: dict[str, int]          # per-column null count
    null_pct: dict[str, float]           # per-column null %
    duplicate_txn_ids: int               # count of duplicate transaction_id values
    negative_revenue_rows: int           # count of revenue < 0
    zero_units_positive_revenue: int     # count of units_sold=0 with revenue>0
    revenue_outliers_iqr: list[int]      # row indices flagged by IQR method
    sample_size_after_filter: int        # rows the code operated on
    notes: list[str]                     # free-text observations
```

### Verification target
Run `python run_demo.py` with a **real GROQ_API_KEY** in `.env` and capture a
full `runs/<run_id>.jsonl` with at least 3 accepted steps. Paste a sample log
into PROGRESS.md under "Phase 4 Demo Output" before moving to Phase 5.

---

## Known Issues / Risks

| Issue | Severity | Mitigation |
|-------|----------|------------|
| `google.generativeai` deprecated → FutureWarning | Medium | Switch to `google.genai` package in Phase 4 (update `agent/llm_client.py` `_call_gemini`) |
| Gemini free-tier quota exhausted | Medium | Use paid tier or rate-limit; for demo Groq alone is sufficient |
| Sandbox `spawn` overhead (~1-2s per call) | Low | Acceptable; pool reuse is Phase 6 |
| Step 1 "last quarter" filter returns empty DF if data is fresh | Low | Critique correctly accepts empty result as "data reflects no rows in window" |
