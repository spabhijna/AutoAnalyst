"""
agent/ — Self-Correcting Data Pipeline Agent.

Entry point: :func:`agent.loop.run` (the CLI is ``main.py`` at the repo root).

A run plans a goal into a dependency graph of tasks (``planning``, ``execution_graph``),
resolves each task with a deterministic capability when one matches (``capabilities``,
``execution_engine``, ``validator``) and otherwise with sandboxed, LLM-written pandas code
checked by an independent critique (``stages``, ``sandbox``, ``observer``). Accepted results
become findings and typed artifacts (``findings``, ``artifact_store``); the summary is
built from accepted findings only. All LLM traffic goes through ``llm_client.call_llm``
(or the callable passed to ``run(llm=...)``).

See ``ARCHITECTURE.md`` for the module map and engineering rules.
"""

__version__ = "0.1.0"
