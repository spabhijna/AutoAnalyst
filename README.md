# AutoAnalyst — Self-Correcting Data Pipeline Agent

An agentic data-analysis loop with **independent LLM critique**, **deterministic
capabilities** and **typed, validated artifacts**. Give it a goal and a dataset (the built-in
synthetic sales data, or your own CSV/Parquet file plus an optional data dictionary); it plans
the analysis as a dependency graph, computes what it can deterministically, has an LLM write
(and a *separate* LLM audit) code for the rest, and summarises only what was accepted.

```
PLAN → for each task: [deterministic capability | CODE-GEN → EXECUTE → OBSERVE → CRITIQUE] → ACCEPT → SUMMARY
```

- Registered **capabilities** (filter to the last quarter, revenue by region, top region,
  monthly trend, revenue share, ...) run without any LLM call when the data has their columns.
  On other data the planner only sees the capabilities that apply, and everything else is code.
- Everything else is generated pandas code, run in a process-isolated sandbox, checked against
  the task's declared output contract, annotated with deterministic data-quality facts
  (nulls, duplicate rows/IDs, negative values, out-of-range datetimes, IQR outliers) and audited by an independent critique before it is accepted.
- Every attempt is logged (`runs/<run_id>.jsonl` and `runs/<run_id>.json`); the run aborts
  rather than force-accepting a result nobody trusts.

For how the pieces fit together read [ARCHITECTURE.md](ARCHITECTURE.md); for current status,
[PROGRESS.md](PROGRESS.md).

---

## Quick start

Requires Python 3.12+.

```bash
git clone <your-repo-url>
cd AutoAnalyst

# Install (pick one)
uv sync                       # creates .venv from pyproject.toml / uv.lock (includes pytest)
# or
pip install -e ".[dev]"
```

Run without any API keys (canned LLM responses; the plan is executed deterministically):

```bash
python main.py --dry-run
```

Run for real (needs at least a Groq key, see below):

```bash
cp .env.example .env          # then fill in your keys
python main.py
```

Run the tests (no network, no credentials: LLM calls are mocked):

```bash
pytest tests/ -q
```

With `uv sync`, prefix commands with `uv run` (for example `uv run python main.py --dry-run`)
or activate `.venv`.

### CLI

`main.py` is the only CLI (`python run_demo.py ...` is a thin wrapper around it).

| Flag | Meaning |
|---|---|
| `--goal "..."` | Analytical goal (default: revenue trend by region over the last quarter; with `--data`, a general profiling goal). |
| `--data PATH` | Analyse a CSV or Parquet file instead of the synthetic sales data. Parquet needs `pyarrow` (`uv sync --extra parquet`). |
| `--describe PATH` | Data dictionary: a JSON object `{column: description}` or a text/Markdown file. |
| `--rows N` | Rows of synthetic data to generate (default 700). |
| `--seed N` | Random seed for the data generator (default 42). |
| `--corrupt` | Inject known defects (duplicates, nulls, outliers, ...) and label the run `corrupted` (sales data only). |
| `--retries N` | CODE-GEN attempts per task (overrides `MAX_RETRIES`). |
| `--dry-run` | Use canned LLM responses instead of calling providers. |

```bash
python main.py --dry-run --corrupt
python main.py --rows 1000 --retries 3 --goal "Which product category drove the most revenue?"
python main.py --data yellow_tripdata_2024-01.parquet \
    --describe examples/nyc_taxi_dictionary.json \
    --goal "How does tipping differ by payment type?"
```

### Your own data

- **Time windows are opt-in.** The "last quarter" machinery (pre-computed
  `LAST_QUARTER_START` / `LAST_QUARTER_END`, the `quarter_df` filter step and the related
  prompt rules) is used only when the goal mentions a quarter *and* the data has a datetime
  column. The datetime column is detected: a column named `date`, else a datetime column
  (preferring pickup/start over dropoff/end), else a date-like string column. For other
  windows ("last month", "2023") the prompts only say to anchor them on the data's latest date.
- **Data dictionary.** Coded columns are easy to mislabel (in the NYC taxi data
  `payment_type` 1 = credit card, 2 = cash, and cash tips are not recorded). Pass
  `--describe` (or `run(..., data_dictionary=...)`) and the descriptions go into the plan,
  code-gen, repair, critique and summary prompts. Without one, the prompts tell the model not
  to guess what codes mean. [`examples/nyc_taxi_dictionary.json`](examples/nyc_taxi_dictionary.json)
  is a complete example.
- **Planner examples** are built from the capabilities that apply to your columns, or from
  your own columns when none do, so no sales-specific step leaks into the plan.

### Reading a run

Each run writes `runs/<run_id>.jsonl` (one line per step) and `runs/<run_id>.json` (summary:
goal, dataset label, models, LLM call count, full history). Pretty-print the JSON:

```bash
python -m agent.log_viewer runs/<run_id>.json
```

### Using it from code

```python
from agent.loop import run
from data.generate_data import generate_sales_data

df = generate_sales_data(n_rows=700, seed=42)
state = run("Explain revenue by region over the last quarter.", df, dry_run=True)

print(state.summary)
for finding in state.final_results:      # accepted findings, in plan order
    print(finding.question, "->", finding.answer)
```

`run(..., llm=my_callable)` swaps the LLM: any callable `(prompt, role) -> str` works.
`run(..., data_dictionary={"payment_type": "1 = credit card, 2 = cash"})` (or a string)
describes columns and codes. `agent.inputs.load_dataset(path)` and
`agent.inputs.load_data_dictionary(path)` load a CSV/Parquet file and a dictionary file.
Evaluate the observer against a corruption manifest with `agent.evaluator.evaluate`.

---

## Configuration

Set in the environment or in `.env` (see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | none | Primary provider for every stage. |
| `CEREBRAS_API_KEY` | none | Fallback for codegen and critique. |
| `OPENROUTER_API_KEY` | none | Fallback for plan, codegen and summary. |
| `GEMINI_API_KEY` | none | Last-resort fallback for critique. |
| `MAX_RETRIES` | `5` | CODE-GEN attempts per task. |
| `SANDBOX_TIMEOUT_SECONDS` | `30` | Timeout per sandbox execution. |
| `LOG_DIR` | `runs` | Where logs are written. |

Provider chain per stage (each level is retried once, then the next is tried; if all fail,
`LLMUnavailableError` is raised, errors are never swallowed):

| Stage | Chain |
|---|---|
| plan | Groq → OpenRouter |
| codegen | Groq → Cerebras → OpenRouter |
| critique | Groq → Cerebras → Gemini |
| summary | Groq → OpenRouter |

Models live in `agent/config.py` (`openai/gpt-oss-120b` on Groq and Cerebras, free OpenRouter
models, `gemini-2.5-flash`).

---

## Project structure

```
AutoAnalyst/
├── main.py                   # the CLI
├── run_demo.py               # thin wrapper around main.main()
├── agent/
│   ├── loop.py               # orchestration: AgentRun, run(), accept path, codegen loop
│   ├── stages.py             # CODE-GEN / REPAIR / CRITIQUE / SUMMARY stages
│   ├── planning.py           # plan / replan, task building
│   ├── parsing.py            # defensive parsers for LLM output
│   ├── schema.py, dates.py   # dataset summary for prompts; datetime detection, time scope, "last quarter" window
│   ├── inputs.py             # load CSV/Parquet data and a data dictionary
│   ├── findings.py           # attempts → findings, memory views for prompts
│   ├── scheduling.py         # ready-task selection, dependency checks, abort marking
│   ├── errors.py             # PipelineAbortedError
│   ├── reporting.py          # all console output
│   ├── run_export.py         # JSONL log entries, run JSON
│   ├── logger.py, log_viewer.py
│   ├── state.py              # TaskNode, AttemptRecord, Finding, WorkingMemory, ...
│   ├── execution_graph.py    # dependency-aware scheduling
│   ├── capabilities.py       # capability registry + deterministic executors
│   ├── execution_engine.py   # capability dispatch + validation
│   ├── validator.py          # output and semantic contracts
│   ├── artifact_store.py     # immutable typed artifacts
│   ├── sandbox.py            # process-isolated code execution
│   ├── observer.py           # deterministic data-quality facts
│   ├── environment.py        # sandbox capability manifest
│   ├── prompts.py            # every prompt template
│   ├── llm_client.py         # call_llm(): provider chains
│   ├── config.py             # models, constants, env overrides
│   ├── dry_run.py            # canned LLM for --dry-run
│   └── evaluator.py          # observer vs. corruption manifest
├── data/generate_data.py     # synthetic sales data + corruption injector
├── examples/                 # example data dictionary (NYC yellow taxi)
├── tests/                    # pytest suite (LLM and sandbox boundaries mocked)
├── docs/HISTORY.md           # historical development log
├── ARCHITECTURE.md           # architecture and engineering rules
├── PROGRESS.md               # current status and next steps
├── .env.example              # key template (no real values)
└── pyproject.toml            # dependencies (single source of truth)
```
