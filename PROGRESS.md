# PROGRESS.md — AutoAnalyst: Self-Correcting Data Pipeline Agent

> **Source of truth** for project state. Update this file at the end of every
> session before stopping. The "NEXT STEPS" section is the handoff to the next
> agent or session.

---

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| **0** | Project skeleton (exploratory scaffold) | ✅ Complete |
| **1** | LLM client wrapper + synthetic data | ✅ **Complete** |
| 2 | Sandbox + pipeline loop | 🔲 Next |
| 3 | Real prompts + live integration | 🔲 Planned |
| 4 | Data quality checks in OBSERVE stage | 🔲 Planned |
| 5 | Evaluation against corruption manifest | 🔲 Planned |
| 6 | Blog post run log + write-up | 🔲 Planned |

---

## Phase 1 — What Was Completed

### Files Created / Modified

| File | Status | Notes |
|------|--------|-------|
| `agent/__init__.py` | ✅ Created | Package init with `__version__` |
| `agent/config.py` | ✅ Created | All constants: model names, retry caps, role strings |
| `agent/llm_client.py` | ✅ Created | `call_llm()` — Groq-first + retry + Gemini fallback |
| `data/generate_data.py` | ✅ Created | `generate_sales_data()` + `inject_corruption()` with manifest |
| `tests/test_llm_client.py` | ✅ Replaced | 19 tests, all passing — covers full retry/fallback contract |
| `.env.example` | ✅ Updated | Two keys only: `GROQ_API_KEY`, `GEMINI_API_KEY` — no real values |
| `requirements.txt` | ✅ Updated | Added `numpy` for data generator |
| `README.md` | ✅ Written | Quick-start, structure, LLM architecture table, phase roadmap |

### Test Results

```
tests/test_llm_client.py — 19 passed in 0.12s  ✅
```

All 19 tests pass with zero live API calls (Groq and Gemini both mocked).

### Data Generator Output (verified)

```
Clean:    700 rows × 7 cols | Jan–Apr 2024 | 4 regions | 6 categories | 0 nulls | 0 dupes
Corrupt:  703 rows
  - 21 duplicated transaction IDs  (~3% of 700)
  - 14 null revenue rows           (~2% of 700)
  - 3  impossible value rows       (indices 700–702)
    [700] negative_revenue:              revenue = -999.99
    [701] revenue_outlier_50x:           revenue = 4960.0 (50× Food & Beverage median)
    [702] zero_units_positive_revenue:   units_sold = 0, revenue = 59.99
Total defects in manifest: 38
```

---

## Decisions Made

### Phase 1 Decisions

#### 1. All roles use Groq (not split-provider per role)
**Why:** Phase 0 sent `critique` to Gemini as a separate provider for "independence."
Phase 1 spec overrides this: all roles use Groq by default, with Gemini as an
availability fallback (not a design choice for independence). The critique model
(`openai/gpt-oss-120b` on Groq) is larger/stronger than the plan/codegen model
(`llama-3.3-70b-versatile`) — that model gap provides the independence, not the
provider gap.

#### 2. Role parameter is logging-only in `call_llm`
**Why:** Keeps routing logic centralized in `_default_model_for_role()` rather than
spread across a role → provider dispatch table. Any caller can override with
`model=` if needed. This also means the fallback logic is identical for all roles.

#### 3. Groq retry once before fallback (not immediate fallback)
**Why:** Transient 429s and timeouts are common on free-tier Groq. Falling back to
Gemini on every transient error would make Groq the "primary" in name only.
One retry with 1.5s backoff handles the vast majority of transient failures
without hitting rate limits on the retry itself.

#### 4. `inject_corruption` returns a manifest dict (not just a corrupted DF)
**Why:** The manifest is the ground truth for Phase 5 scoring. Without it, you'd have
to re-derive what was injected, which defeats the point of a controlled evaluation.
The manifest records exact indices, corruption types, and injected values.

#### 5. Three impossible-value types (not just one)
**Why:** Each type exercises a different detection mechanism:
- **Negative revenue** → caught by `(df['revenue'] < 0).any()` — trivial range check
- **50× outlier** → caught by z-score or IQR outlier detection — statistical check
- **Zero units + positive revenue** → caught by `df[df['units_sold'] == 0]['revenue'] > 0` — logic check
Phase 5 evaluation can score the agent on each separately.

---

## Architecture Summary

```
call_llm(prompt, role, model=None)
  │
  ├── role → _default_model_for_role() → Groq model ID
  │
  ├── [Attempt 1] _call_groq(prompt, model, groq_api_key)
  │     └── success → return  │  failure → sleep(1.5s)
  │
  ├── [Attempt 2] _call_groq(prompt, model, groq_api_key)  (retry)
  │     └── success → return  │  failure → fall back
  │
  ├── [Fallback]  _call_gemini(prompt, GEMINI_FALLBACK_MODEL, gemini_api_key)
  │     └── success → return  │  failure ↓
  │
  └── raise LLMUnavailableError(groq_error=..., gemini_error=...)
```

```
generate_sales_data(n_rows=700, seed=42) → clean DataFrame
inject_corruption(clean_df, seed=99)    → (corrupted DataFrame, manifest dict)
```

---

## Phase 0 Notes (kept for reference)

Phase 0 created the `autoanalyst/` package as an exploratory scaffold with
Groq→generate / Gemini→critique split-provider routing. **This is superseded
by `agent/` (Phase 1+)**. The `autoanalyst/` directory is kept for reference
but should not be imported by Phase 2+ code. It can be deleted once Phase 3
is complete.

---

## Known Issues / Risks

| Issue | Severity | Mitigation |
|-------|----------|------------|
| `np.str_` appears in manifest printed output (numpy scalar type) | Low | Cast to `str` in manifest dict if needed for JSON serialisation — `json.dump(default=str)` already handles it |
| `autoanalyst/` package conflicts with `agent/` — two separate packages with overlapping purpose | Low | Document as Phase 0 scaffold; delete after Phase 3 |
| `_call_groq` uses temperature=0.2 hardcoded | Low | Move to `config.py` in Phase 3 when prompt tuning begins |

---

## NEXT STEPS — Phase 2: Sandbox + Pipeline Loop

### Goal
Build the execution sandbox and the `PLAN → CODEGEN → EXECUTE → OBSERVE → CRITIQUE`
orchestration loop. The loop should:
- Be importable from `agent/pipeline.py`
- Use `agent/llm_client.call_llm()` for all LLM calls
- Execute agent-generated code only inside `agent/sandbox.py`
- Write a structured `LogEntry` (JSONL) after every stage
- Hard-cap retries at `config.MAX_PIPELINE_RETRIES`

### Files to Create

1. **`agent/sandbox.py`** — Restricted `exec()` jail with:
   - Curated safe globals (pandas, math, statistics, json, re, datetime)
   - Stdout capture via `io.StringIO`
   - Wall-clock timeout via `threading.Event` + daemon thread
   - `SandboxResult(stdout, return_value, error, timed_out)` dataclass

2. **`agent/logger.py`** — Append-only JSONL writer:
   - `LogEntry` dataclass with fields: `run_id, sub_question, stage, iteration, code, result, critique, passed, timestamp, extra`
   - `write_log_entry(entry, log_dir)` — atomic append
   - `read_run_log(run_id, log_dir)` — for tests and post-run analysis

3. **`agent/pipeline.py`** — `run_pipeline(goal, df, config)`:
   - Calls `agent/stages/planner.py`, `codegen.py`, `executor.py`, `observer.py`, `critic.py`
   - Returns `PipelineRunResult` with per-question outcomes and log path

4. **`agent/stages/`** — 5 stage files, each with one public function

5. **`tests/test_sandbox.py`** — isolation, timeout, stdout capture, `_result` return

6. **`tests/test_pipeline.py`** — integration smoke test with mocked `call_llm`

### Entry point
```bash
python main.py --dry-run --data data/sales_corrupted.csv
```
Should produce a JSONL file in `runs/` with entries for every stage.
