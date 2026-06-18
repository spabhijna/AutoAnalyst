# PROGRESS.md — AutoAnalyst: Self-Correcting Data Pipeline Agent

> **Source of truth** for project state. Update this file at the end of every
> session before stopping. The "NEXT STEPS" section is the handoff to the next
> agent or session.

---

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | LLM client wrapper + synthetic data | ✅ Complete |
| 2 | Safe code execution sandbox | ✅ **Complete** |
| 3 | Pipeline loop + prompts | 🔲 Next |
| 4 | Data quality checks in OBSERVE stage | 🔲 Planned |
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
│   └── sandbox.py            # run_sandboxed() — process-isolated exec
├── data/
│   └── generate_data.py      # generate_sales_data() + inject_corruption()
├── tests/
│   ├── __init__.py
│   ├── test_llm_client.py    # 19 tests ✅
│   └── test_sandbox.py       # 24 tests ✅
├── .env.example
├── .gitignore
├── main.py                   # stub — wired to agent/pipeline.py in Phase 3
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

---

## ⚠️ Phase 3 Hard Dependency — Sandbox Result Convention

> **Agent-generated code MUST assign its final answer to a variable named `result`.**

The sandbox reads `local_ns["result"]` after `exec()` and returns its `repr()`
(truncated to 2000 chars). If `result` is not set, `out["result"]` is `None`.

**Do not change this convention without updating the CODE-GEN prompt in Phase 3.**

The `run_sandboxed()` return dict always has exactly these five keys:
```python
{
    "success":    bool,        # True if no exception was raised
    "result":     str | None,  # repr of `result` variable, truncated to 2000 chars
    "stdout":     str,         # everything printed to stdout
    "error":      str | None,  # full traceback if success=False
    "error_type": str | None,  # exception class name (e.g. "ZeroDivisionError")
}
```

On timeout: `success=False`, `error_type="TimeoutError"`.

---

## Phase 2 — What Was Built

### Files

| File | Notes |
|------|-------|
| `agent/sandbox.py` | `run_sandboxed(code, df, timeout_seconds=5)` → dict |
| `tests/test_sandbox.py` | 24 tests — all passing |

### Test Results

```
tests/test_llm_client.py  — 19 passed  ✅
tests/test_sandbox.py     — 24 passed  ✅
Total: 43 passed, 0 failed
```

### Bugs Fixed During Phase 2

1. **Parquet serialization failed** — `pyarrow` not installed. Switched to
   `pickle` for DataFrame inter-process transfer. No extra dependency needed;
   pickle is fine for same-version subprocess communication.

2. **`__builtins__` restriction blocked worker-level imports** — The whitelist
   was set before the worker could import `pandas`/`numpy` at module level.
   Fixed by applying the restriction only inside `exec()`, not at worker scope.

3. **Test assertions used Python `float` repr** — `df['revenue'].sum()` returns
   `np.float64`, whose `repr()` is `'np.float64(1000.0)'` not `'1000.0'`.
   Fixed tests to check `"1000.0" in out["result"]` instead of exact equality.

---

## Decisions Made

### Phase 2 Decisions

#### 6. Process-based sandbox (not thread-based)
**Why:** A thread cannot be hard-killed in CPython. `threading.Event` only
signals cooperative exit — a `while True: pass` loop will never yield to it.
`multiprocessing.Process.kill()` sends SIGKILL, which is unconditional.

#### 7. `result` variable convention (not last-expression capture)
**Why:** AST last-expression capture is fragile:
- Multi-line code has no single "last expression"
- The `compile()` + `eval()` pattern breaks on assignment-only last lines
- Different code styles (trailing comments, blank lines) confuse AST walkers

A simple convention (`result = ...`) is explicit, reliable, and easy to
enforce in the CODE-GEN prompt. The prompt will say:
> "Always assign your final answer to a variable named `result`."

#### 8. DataFrame via `pickle` (not Parquet, not shared memory)
**Why:** Parquet requires `pyarrow` or `fastparquet` (not installed). Shared
memory (`multiprocessing.shared_memory`) requires manual lifecycle management
and array-based serialization incompatible with arbitrary DataFrame schemas.
`pickle` is stdlib, same-version safe, and fast enough for ~1000-row DataFrames.

#### 9. `spawn` start method (not `fork`)
**Why:** `fork` inherits the parent's open file descriptors, sockets, and
random state — a security and stability risk in a sandboxed executor. `spawn`
starts a clean Python interpreter, matching production safety expectations.
On macOS, `spawn` is also the default since Python 3.8, so this is consistent.

---

## NEXT STEPS — Phase 3: Pipeline Loop + Prompts

### Goal
Build the `PLAN → CODEGEN → EXECUTE → OBSERVE → CRITIQUE` orchestration loop
using `call_llm()` and `run_sandboxed()`. Wire `main.py` to run it end-to-end.

### Files to create

| File | Responsibility |
|------|---------------|
| `agent/logger.py` | `LogEntry` dataclass + `write_log_entry()` + `read_run_log()` (JSONL append) |
| `agent/stages/__init__.py` | Package init |
| `agent/stages/planner.py` | `plan(goal, schema, config)` → `list[str]` |
| `agent/stages/codegen.py` | `generate_code(question, schema, previous_error)` → `str` |
| `agent/stages/executor.py` | `execute(code, df)` → sandbox result dict |
| `agent/stages/observer.py` | `observe(question, code, exec_out)` → `str` |
| `agent/stages/critic.py` | `critique(question, code, observation)` → `CritiqueResult` |
| `agent/pipeline.py` | `run_pipeline(goal, df)` → `PipelineRunResult` |
| `tests/test_pipeline.py` | Integration smoke test with mocked `call_llm` |

### CODE-GEN prompt must include (Phase 3 hard dependency from Phase 2)
```
Always assign your final answer to a variable named `result`.
Available: df (DataFrame), pd (pandas), np (numpy).
Do NOT use: import, open, os, sys, subprocess.
```

### Pipeline constraints
- Hard-cap retries at `MAX_PIPELINE_RETRIES` (3) — never unbounded
- `write_log_entry()` called after **every** stage — partial runs are readable
- On critique FAIL: pass issues text back to CODE-GEN as `previous_error`
- Log fields: `run_id, sub_question, stage, iteration, code, result, critique, passed, timestamp`

### CLI target
```bash
source .venv/bin/activate
python main.py --goal "Analyse sales trends" --data data/sales_corrupted.csv
cat runs/<run_id>.jsonl | python -m json.tool
```

---

## Known Issues / Risks

| Issue | Severity | Mitigation |
|-------|----------|------------|
| `scripts/download_data.py` is unused (project uses synthetic data) | Low | Delete at start of Phase 3 |
| Sandbox `spawn` start method adds ~1–2s overhead per execution on macOS | Low | Acceptable for a CLI tool; pool reuse is a Phase 6 optimization |
| `pickle` for DataFrame transfer is not safe across Python versions | Low | Same interpreter guaranteed since child is spawned from parent |
