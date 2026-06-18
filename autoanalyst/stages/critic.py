"""
CRITIQUE stage — independently evaluates the pipeline's own output.

Key design constraint: this stage ALWAYS uses ``role="critique"``, which
routes to Google Gemini — a different LLM provider from the generation
stages (Groq). This ensures the critique cannot simply agree with the
generation due to shared cached reasoning or identical model weights.

The critic looks for:
  - Data-quality problems that invalidated the result (duplicate rows used
    in an aggregation, nulls silently dropped, wrong join type, etc.).
  - Logical errors in the code that produced a plausible-but-wrong answer.
  - Whether the observation actually answers the sub-question.
  - Whether the result passes a basic sanity check (e.g. total revenue ≠ 0).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoanalyst.config import PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class CritiqueResult:
    """
    Structured output from the CRITIQUE stage.

    Attributes:
        passed:   True when the critic deems the result acceptable.
        feedback: Full critique text from the LLM.
        issues:   Extracted list of specific problems (may be empty).
    """

    passed: bool
    feedback: str
    issues: list[str]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def critique(
    sub_question: str,
    code: str,
    observation: str,
    config: "PipelineConfig",
) -> CritiqueResult:
    """
    Independently evaluate the agent's result for *sub_question*.

    This function uses ``role="critique"`` which routes to Google Gemini,
    making it genuinely independent of the Groq-based generation stages.

    Args:
        sub_question: The analytical question that was being answered.
        code:         The Python code the agent generated and executed.
        observation:  The OBSERVE stage's summary of the execution result.
        config:       Active :class:`~autoanalyst.config.PipelineConfig`.

    Returns:
        A :class:`CritiqueResult` with ``passed``, ``feedback``, and
        extracted ``issues``.
    """
    from autoanalyst.llm_client import call_llm

    prompt = _build_prompt(sub_question, code, observation)
    logger.info("CRITIQUE stage: sub_question=%r", sub_question[:60])

    raw = call_llm(prompt, role="critique", config=config)
    result = _parse_critique(raw)

    logger.info(
        "CRITIQUE stage: passed=%s, issues=%d", result.passed, len(result.issues)
    )
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_prompt(sub_question: str, code: str, observation: str) -> str:
    """
    Build the critique prompt sent to the independent LLM.

    Args:
        sub_question: The analytical question.
        code:         Agent-generated code that was executed.
        observation:  Summary of the execution result.

    Returns:
        Full prompt string.
    """
    return (
        f"You are an independent senior data scientist auditing an AI agent's work.\n"
        f"You did NOT write the code below — your job is to find its mistakes.\n\n"
        f"Sub-question: {sub_question}\n\n"
        f"Code written by the agent:\n```python\n{code}\n```\n\n"
        f"Agent's observation:\n{observation}\n\n"
        f"Audit checklist — look for:\n"
        f"1. Data-quality bugs: rows not deduplicated before aggregation, "
        f"   nulls silently dropped or filled incorrectly, wrong date parsing.\n"
        f"2. Logical errors: wrong groupby key, incorrect metric formula, "
        f"   off-by-one in date ranges, wrong join type.\n"
        f"3. Answer validity: does the observation actually answer the question? "
        f"   Does the result pass a basic sanity check?\n\n"
        f"Respond in this exact format:\n"
        f"VERDICT: PASS or FAIL\n"
        f"ISSUES:\n"
        f"- <issue 1> (or 'None' if no issues)\n"
        f"- <issue 2>\n"
        f"FEEDBACK:\n"
        f"<Your full reasoning in 100-200 words.>"
    )


def _parse_critique(raw: str) -> CritiqueResult:
    """
    Parse the structured critique LLM response into a :class:`CritiqueResult`.

    Handles minor formatting variations from the LLM output.

    Args:
        raw: Raw LLM response string.

    Returns:
        :class:`CritiqueResult` with parsed fields.
    """
    # Determine PASS/FAIL
    verdict_match = re.search(r"VERDICT\s*:\s*(PASS|FAIL)", raw, re.IGNORECASE)
    passed = verdict_match.group(1).upper() == "PASS" if verdict_match else False

    # Extract bullet-point issues
    issues_section = re.search(
        r"ISSUES\s*:\s*\n(.*?)(?:FEEDBACK|$)", raw, re.DOTALL | re.IGNORECASE
    )
    issues: list[str] = []
    if issues_section:
        for line in issues_section.group(1).splitlines():
            line = line.strip().lstrip("-• ").strip()
            if line and line.lower() != "none":
                issues.append(line)

    # Extract feedback block
    feedback_match = re.search(
        r"FEEDBACK\s*:\s*\n(.*)", raw, re.DOTALL | re.IGNORECASE
    )
    feedback = feedback_match.group(1).strip() if feedback_match else raw.strip()

    return CritiqueResult(passed=passed, feedback=feedback, issues=issues)
