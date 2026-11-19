# PROGRESS — AutoAnalyst

> Short status page. Architecture and rules: [ARCHITECTURE.md](ARCHITECTURE.md). Usage:
> [README.md](README.md). Full development history: [docs/HISTORY.md](docs/HISTORY.md).
> Details of the 2026-11 cleanup (what moved, every deviation, every changed test
> assertion): `docs/REFACTOR_NOTES.md`.

## Status

The pipeline is feature-complete and covered by the test suite.

- `pytest tests/ -q` passes (254 tests; LLM and sandbox boundaries are mocked, no network).
- `python main.py --dry-run` and `python main.py --dry-run --corrupt` finish; in dry-run all
  five planned tasks are resolved by deterministic capabilities (2 LLM calls: plan, summary).
- Live runs need provider keys (`.env.example`); they have not been re-run as part of the
  cleanup because the cleanup was verified offline.

| Area | State |
|---|---|
| Plan → task graph (structured plans, replans keep completed tasks) | done |
| Deterministic capabilities (8) + engine + validator | done |
| CODE-GEN loop with independent critique and observer facts | done |
| Artifact contract (codegen is told variable name and columns) | done |
| Typed, frozen artifacts with producer / consumers / provenance | done |
| Provider chains (Groq / Cerebras / OpenRouter / Gemini) | done |
| Evaluation against the corruption manifest (`agent/evaluator.py`) | done |
| Structured logs (JSONL + run JSON) and viewer | done |

## Layout in one line

`main.py` (CLI) → `agent/loop.py` (orchestration) → `planning`, `stages`, `findings`,
`scheduling`, `execution_engine` + `capabilities` + `validator`, `artifact_store`, `sandbox`,
`observer`, `llm_client`; console output only in `reporting.py`; prompts only in `prompts.py`.
See the module table in ARCHITECTURE.md.

## Known limitations

- **Live behaviour is only lightly exercised.** The CODE-GEN path is tested with mocked LLMs;
  real model quality (for example how reliably a model follows the output contract) has not
  been measured.
- **`--seed` is not fully reproducible.** `data/generate_data.py` builds `transaction_id`
  with `uuid4`, so the IDs differ between runs even with the same seed.
- **Eight capabilities only.** Anything else goes through code generation.
- **Some functions are long by design:** the prompt templates in `agent/prompts.py` and the
  isolation worker in `agent/sandbox.py` exceed 80 lines.
- **Sandbox default timeout is 30 s** (`SANDBOX_TIMEOUT_SECONDS`); it was 10 s inside the
  loop before the cleanup.
- Unused-variable warnings from `ruff` remain in `agent/sandbox.py` (`last_line`).

## Next steps

1. Run a handful of live end-to-end runs (clean and `--corrupt`) and record critique
   accuracy with `agent.evaluator`.
2. Make `data/generate_data.py` reproducible under `--seed` (derive IDs from the seeded RNG).
3. Add capabilities for the analyses that still fall back to code generation in live runs
   (look at which tasks hit CODE-GEN in the run logs).
4. Optional: split the long prompt templates into composable sections once prompt wording is
   allowed to change again.
