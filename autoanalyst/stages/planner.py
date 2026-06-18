"""
PLAN stage — decomposes a high-level analytical goal into ordered sub-questions.

Phase 0: stub implementation.
Phase 1: replace _build_prompt() body with a real prompt and uncomment the
         call_llm() invocation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoanalyst.config import PipelineConfig

logger = logging.getLogger(__name__)


def plan(
    goal: str,
    dataset_schema: str,
    config: "PipelineConfig",
) -> list[str]:
    """
    Decompose *goal* into a list of analytical sub-questions.

    Each sub-question should be independently answerable with a short
    Python snippet operating on the dataset described by *dataset_schema*.

    Args:
        goal:           The high-level analytical question from the user
                        (e.g. "What drove the revenue decline in Q3?").
        dataset_schema: String description of the dataset columns and
                        sample values (used to ground the planner).
        config:         Active :class:`~autoanalyst.config.PipelineConfig`.

    Returns:
        Ordered list of sub-question strings. In Phase 0 returns a
        fixed stub list; Phase 1 replaces this with an LLM response.
    """
    from autoanalyst.llm_client import call_llm

    prompt = _build_prompt(goal, dataset_schema)
    logger.info("PLAN stage: goal=%r", goal[:80])

    raw = call_llm(prompt, role="generate", config=config)
    sub_questions = _parse_sub_questions(raw)

    logger.info("PLAN stage: produced %d sub-questions", len(sub_questions))
    return sub_questions


# ---------------------------------------------------------------------------
# Internal helpers (Phase 1 will replace these with real prompt logic)
# ---------------------------------------------------------------------------


def _build_prompt(goal: str, dataset_schema: str) -> str:
    """
    Build the planner prompt.

    Phase 0 stub — returns a minimal prompt.  Phase 1 will expand this
    into a full few-shot prompt with chain-of-thought reasoning.

    Args:
        goal:           The analytical goal.
        dataset_schema: Dataset description.

    Returns:
        Prompt string.
    """
    return (
        f"You are an expert data analyst. Break the following goal into "
        f"3-5 specific, independently answerable sub-questions that can each "
        f"be answered with a short Python/pandas snippet.\n\n"
        f"Dataset schema:\n{dataset_schema}\n\n"
        f"Goal: {goal}\n\n"
        f"Output each sub-question on its own line, prefixed with a number "
        f"and period (e.g. '1. What is ...'). No extra commentary."
    )


def _parse_sub_questions(raw: str) -> list[str]:
    """
    Parse the planner LLM output into a clean list of sub-question strings.

    Handles numbered lists (``1. ...``, ``1) ...``) and plain lines.
    Strips leading numbering so downstream stages receive pure question text.

    Args:
        raw: Raw LLM output from the planner prompt.

    Returns:
        List of sub-question strings (stripped, non-empty).
    """
    import re

    lines = raw.splitlines()
    questions: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip leading "1." or "1)" style numbering
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
        if cleaned:
            questions.append(cleaned)

    # If the LLM returned nothing useful, fall back to a single default
    if not questions:
        questions = [raw.strip() or "Provide a summary of the dataset."]

    return questions
