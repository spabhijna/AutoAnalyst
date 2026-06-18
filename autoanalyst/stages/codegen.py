"""
CODE-GEN stage — generates a Python snippet to answer a sub-question.

The generated code is always intended to be executed inside
:mod:`autoanalyst.sandbox` and must:
  - Operate on a ``df`` variable (a pandas DataFrame) already in scope.
  - Store its final answer in a ``_result`` variable so the sandbox can
    return it as a structured value.
  - Print any intermediate tables/summaries for the OBSERVE stage.

Phase 0: stub implementation with real prompt scaffolding.
Phase 1: tune the prompt with few-shot examples.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoanalyst.config import PipelineConfig

logger = logging.getLogger(__name__)


def generate_code(
    sub_question: str,
    dataset_schema: str,
    config: "PipelineConfig",
    previous_error: str | None = None,
) -> str:
    """
    Generate a Python code snippet that answers *sub_question*.

    The snippet assumes:
    - A ``pd`` (pandas) import is available in scope.
    - A ``df`` variable holds the full dataset as a :class:`pandas.DataFrame`.
    - The code should store its primary answer in ``_result``.

    Args:
        sub_question:   The specific analytical question to answer.
        dataset_schema: Column names and dtypes of the dataset (for grounding).
        config:         Active :class:`~autoanalyst.config.PipelineConfig`.
        previous_error: If this is a retry, the error message from the last
                        attempt so the LLM can fix its mistake.

    Returns:
        A Python source code string ready to be passed to
        :func:`autoanalyst.sandbox.execute_code`.
    """
    from autoanalyst.llm_client import call_llm

    prompt = _build_prompt(sub_question, dataset_schema, previous_error)
    logger.info("CODEGEN stage: sub_question=%r retry=%s", sub_question[:60], previous_error is not None)

    raw = call_llm(prompt, role="generate", config=config)
    code = _extract_code(raw)

    logger.info("CODEGEN stage: extracted %d chars of code", len(code))
    return code


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_prompt(
    sub_question: str,
    dataset_schema: str,
    previous_error: str | None,
) -> str:
    """
    Construct the code-generation prompt.

    Args:
        sub_question:   The question to answer.
        dataset_schema: Dataset schema description.
        previous_error: Optional error from a prior attempt (for retry).

    Returns:
        Full prompt string.
    """
    retry_block = ""
    if previous_error:
        retry_block = (
            f"\nThe previous attempt raised the following error. Fix it:\n"
            f"```\n{previous_error}\n```\n"
        )

    return (
        f"You are a senior data analyst writing Python/pandas code.\n"
        f"You have access to a pandas DataFrame called `df` with this schema:\n\n"
        f"{dataset_schema}\n\n"
        f"Write a Python snippet that answers the following question:\n"
        f"  {sub_question}\n"
        f"{retry_block}\n"
        f"Rules:\n"
        f"- Use only: pd, df, math, statistics, json, re, datetime, collections, "
        f"  and standard Python builtins.\n"
        f"- Store the final answer (a scalar, list, or dict) in `_result`.\n"
        f"- Print any intermediate DataFrames or tables for human inspection.\n"
        f"- Do NOT use os, sys, subprocess, open, or __import__.\n"
        f"- Return ONLY the code block, no prose.\n\n"
        f"```python\n# your code here\n```"
    )


def _extract_code(raw: str) -> str:
    """
    Extract the Python code block from the LLM's response.

    Handles fenced code blocks (````python ... ``` `` or ` ``` ... ``` ``)
    as well as raw code with no fencing.

    Args:
        raw: Raw LLM response string.

    Returns:
        Cleaned Python source code string.
    """
    # Match ```python ... ``` or ``` ... ```
    pattern = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
    match = pattern.search(raw)
    if match:
        return match.group(1).strip()

    # Fallback: return the raw string stripped of surrounding whitespace
    return raw.strip()
