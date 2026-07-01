# AutoAnalyst — Self-Correcting Data Pipeline Agent

A portfolio project demonstrating an agentic data analysis loop with
**independent LLM-based self-critique**.

The agent analyses a sales dataset by running a structured loop:

```
PLAN → CODE-GEN → EXECUTE → OBSERVE → CRITIQUE → (revise or advance)
```

A separate LLM call (using a different model) critiques each result before
the pipeline advances, catching errors that the generator would silently miss
(duplicate-inflated aggregations, null-dropped means, logical inconsistencies).

---

## Quick Start

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd AutoAnalyst

# 2. Install dependencies (use your package manager of choice)
pip install -r requirements.txt

# 3. Run the demo (requires API keys)
python run_demo.py

# 4. (Optional) Set up API keys for a live run
cp .env.example .env
# Edit .env and add your GROQ_API_KEY and GEMINI_API_KEY

# 5. Run the agent in dry-run mode (no API keys needed)
python main.py --dry-run

# 6. Or use the demo script in dry-run mode
python run_demo.py

# 6. Run tests
uv run pytest tests/ -v
```

---

## Project Structure

```
AutoAnalyst/
├── agent/                    # Core agent package (Phases 1-6)
│   ├── config.py             # Model names, retry caps, constants
│   └── llm_client.py         # call_llm() — single gateway for all LLM calls
├── autoanalyst/              # Phase 0 exploratory scaffold (superseded by agent/)
├── data/
│   └── generate_data.py      # Synthetic sales data + corruption injector
├── tests/
│   └── test_llm_client.py    # LLM client fallback contract tests
├── runs/                     # JSONL run logs (gitignored)
├── .env.example              # API key template (no real values)
├── main.py                   # CLI entry point
└── PROGRESS.md               # Source of truth for project state
```

---

## LLM Architecture

| Stage     | Primary Model           | Fallback |
|-----------|-------------------------|----------|
| plan      | Groq llama-3.3-70b      | Gemini   |
| codegen   | Groq llama-3.3-70b      | Gemini   |
| observe   | Groq llama-3.3-70b      | Gemini   |
| critique  | Groq gpt-oss-120b       | Gemini   |

Groq is retried once (with backoff) before falling back. If both fail,
`LLMUnavailableError` is raised — it never silently swallows provider errors.

---

## Required API Keys

| Variable | Where to get it |
|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) — free tier available |
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) — free tier available |

---

## Phase Roadmap

| Phase | Description | Status |
|---|---|---|
| 0 | Project skeleton (exploratory) | ✅ Complete |
| 1 | LLM client + synthetic data | ✅ Complete |
| 2 | Sandbox + pipeline loop | 🔲 Next |
| 3 | Real prompts + live integration | 🔲 Planned |
| 4 | Data quality checks in OBSERVE | 🔲 Planned |
| 5 | Evaluation against corruption manifest | 🔲 Planned |
| 6 | Blog post run log + write-up | 🔲 Planned |

---

## Running Tests

```bash
uv run pytest tests/ -v
```

All tests mock Groq and Gemini — no live API calls, no credentials required.
