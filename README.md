# AutoAnalyst — Self-Correcting Data Pipeline Agent

A portfolio project demonstrating an agentic data analysis loop with
**independent LLM-based self-critique** and **deterministic semantic validation**.

The agent analyses a sales dataset by running a structured loop:

```
PLAN → CODE-GEN → EXECUTE → OBSERVE → CRITIQUE → (revise or advance)
```

A separate LLM call (using a different model) critiques each result before
the pipeline advances, catching errors that the generator would silently miss
(duplicate-inflated aggregations, null-dropped means, logical inconsistencies).

It also features a robust deterministic execution engine and a resilient runtime (`AnalysisRuntime`) that safely encapsulates state, ensuring robust analysis.

---

## Quick Start

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd AutoAnalyst

# 2. Install dependencies (use your package manager of choice)
pip install -r requirements.txt

# 3. (Optional) Set up API keys for a live run
cp .env.example .env
# Edit .env and add your GROQ_API_KEY and GEMINI_API_KEY

# 4. Run the demo (requires API keys)
python run_demo.py

# 5. Or run the agent in dry-run mode (no API keys needed)
python run_demo.py --dry-run

# 6. Run tests
pytest tests/ -v
```

---

## Project Structure

```
AutoAnalyst/
├── agent/                    # Core agent package
│   ├── config.py             # Model names, retry caps, constants
│   ├── llm_client.py         # call_llm() — single gateway for all LLM calls
│   ├── loop.py               # Main agent loop (PLAN → CODE-GEN → EXECUTE → CRITIQUE)
│   ├── runtime.py            # AnalysisRuntime and ArtifactRegistry for persistent state
│   ├── sandbox.py            # Isolated execution sandbox
│   ├── execution_engine.py   # Deterministic capability-driven execution engine
│   ├── validator.py          # Semantic contract and schema validation
│   └── prompts.py            # LLM prompts
├── data/
│   └── generate_data.py      # Synthetic sales data + corruption injector
├── tests/                    # Comprehensive unit tests
├── runs/                     # JSONL run logs
├── .env.example              # API key template (no real values)
├── run_demo.py               # Main demo script
└── PROGRESS.md               # Source of truth for project state
```

---

## LLM Architecture

| Stage     | Primary Model           | Fallback |
|-----------|-------------------------|----------|
| plan      | Groq llama-3.3-70b      | Gemini   |
| codegen   | Groq llama-3.3-70b      | Gemini   |
| observe   | Groq llama-3.3-70b      | Gemini   |
| critique  | openai/gpt-oss-120b     | Gemini   |

Groq is retried once (with backoff) before falling back. If both fail,
`LLMUnavailableError` is raised — it never silently swallows provider errors.

---

## Required API Keys

| Variable | Where to get it |
|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) — free tier available |
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) — free tier available |

---

## Running Tests

```bash
pytest tests/ -v
```

All tests mock Groq and Gemini — no live API calls, no credentials required. The test suite extensively verifies runtime execution, capabilities, sandbox isolation, semantic validations, and the core analytical loop.
