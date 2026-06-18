"""
OBSERVE stage — interprets raw execution output into a structured observation.

The observer receives the execution result text and produces a concise
natural-language summary that:
  - States what the code computed.
  - Flags obvious data-quality issues (nulls, duplicates, unexpected dtypes).
  - Is compact enough to fit in the critique stage's context window.

Phase 0: stub that passes execution output through an LLM summariser.
Phase 1: add structured data-quality checks (null %, duplicate row count, etc.)
         before the LLM call so the model has explicit numbers to reason about.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoanalyst.config import PipelineConfig

logger = logging.getLogger(__name__)


def observe(
    sub_question: str,
    code: str,
    execution_output: str,
    config: "PipelineConfig",
) -> str:
    """
    Produce a structured observation from the execution output.

    The observation is a short (≤ 300 word) natural-language summary that
    the CRITIQUE stage will evaluate. It must include:
    - The computed answer / key metrics.
    - Any data-quality warnings visible in the output.
    - Whether the result looks reasonable given the question.

    Args:
        sub_question:     The sub-question being answered.
        code:             The code that was executed (for context).
        execution_output: The formatted output from the EXECUTE stage.
        config:           Active :class:`~autoanalyst.config.PipelineConfig`.

    Returns:
        A natural-language observation string.
    """
    from autoanalyst.llm_client import call_llm

    prompt = _build_prompt(sub_question, code, execution_output)
    logger.info("OBSERVE stage: sub_question=%r", sub_question[:60])

    observation = call_llm(prompt, role="generate", config=config)

    logger.info("OBSERVE stage: observation_len=%d chars", len(observation))
    return observation.strip()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_prompt(sub_question: str, code: str, execution_output: str) -> str:
    """
    Build the observation prompt.

    Args:
        sub_question:     The question being answered.
        code:             The executed Python code.
        execution_output: Formatted execution output (stdout + return_value + errors).

    Returns:
        Full prompt string.
    """
    return (
        f"You are a careful data analyst reviewing the output of a Python script.\n\n"
        f"Sub-question: {sub_question}\n\n"
        f"Code executed:\n```python\n{code}\n```\n\n"
        f"Execution output:\n{execution_output}\n\n"
        f"Write a concise observation (≤ 300 words) that:\n"
        f"1. States the key numeric result or finding.\n"
        f"2. Notes any data-quality issues visible in the output "
        f"   (NaN values, suspicious counts, unexpected data types).\n"
        f"3. States whether the result appears reasonable for the question asked.\n\n"
        f"Be factual. Do not speculate beyond what the output shows."
    )
