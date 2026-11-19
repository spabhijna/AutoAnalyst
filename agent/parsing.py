"""
Parsers for LLM output: plan JSON, critique JSON, code fences.

All functions are pure and defensive: malformed LLM output never raises.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _task_spec(question: str, item: dict | None = None) -> dict:
    """Build one normalised task-spec dict (all keys always present)."""
    item = item or {}
    return {
        "question": question,
        "capability": str(item.get("capability") or "").strip() or None,
        "produces_artifact": item.get("produces_artifact") or None,
        "consumes_artifacts": list(item.get("consumes_artifacts") or []),
        "output_schema": list(item.get("output_schema") or []),
    }


def _normalise_task_spec(item: object) -> dict | None:
    """Normalise a dict or plain-string plan item; None when it has no question."""
    if isinstance(item, dict):
        question = str(item.get("question", "")).strip()
        return _task_spec(question, item) if question else None
    if isinstance(item, str) and item.strip():
        return _task_spec(item.strip())
    return None


def parse_plan_json(raw: str) -> list[dict]:
    """
    Parse the plan LLM response into a list of task specification dicts.

    Handles two formats:

    **Structured format (new)** — a JSON array of task objects::

        [
          {"question": "...", "produces_artifact": "quarter_df", "consumes_artifacts": []},
          {"question": "...", "produces_artifact": null,          "consumes_artifacts": ["quarter_df"]}
        ]

    **Legacy flat format (backward-compatible)** — a JSON array of strings::

        ["Q1?", "Q2?", "Q3?"]

    In the legacy case, each string is wrapped in a minimal dict with
    ``produces_artifact=None`` and ``consumes_artifacts=[]`` so the rest
    of the loop can treat both formats identically.

    The function is intentionally defensive: if JSON parsing fails entirely,
    it falls back to line-splitting (same as the old flat-list parser).

    Args:
        raw: Raw LLM response string.

    Returns:
        List of dicts, each with at least ``{"question": str}``.
    """
    # Try to extract the outermost JSON array from the raw string.
    # Use greedy matching so nested [] (e.g. "consumes_artifacts": []) do not
    # cause an early match — we need the full outer array.
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list) and parsed:
                specs = [
                    spec
                    for spec in (_normalise_task_spec(item) for item in parsed)
                    if spec is not None
                ]
                all_structured = all(
                    isinstance(item, dict) and "question" in item for item in parsed
                )
                all_flat = all(isinstance(item, str) for item in parsed)
                # Structured / legacy flat formats return as-is (even if empty);
                # a mixed list only returns when something usable was found.
                if specs or all_structured or all_flat:
                    return specs
        except json.JSONDecodeError:
            pass

    # Fallback: split on numbered lines (same as the legacy flat-list parser)
    lines = []
    for line in raw.splitlines():
        line = re.sub(r"^\d+[\.)\-]\s*", "", line.strip())
        if line and not line.startswith("{") and not line.startswith("["):
            lines.append(_task_spec(line))
    if lines:
        return lines
    return [_task_spec(raw.strip())]


def parse_critique_json(raw: str) -> dict[str, Any]:
    """
    Parse the critique JSON response defensively.

    Returns a safe default (revise_code) if parsing fails so the loop
    never silently accepts a result due to a malformed critique response.

    Args:
        raw: Raw LLM response string expected to contain a JSON object.

    Returns:
        Dict with ``trust``, ``reason``, ``action`` keys.
    """
    _safe_default: dict[str, Any] = {
        "trust": "low",
        "reason": f"Critique response could not be parsed: {raw[:200]}",
        "action": "revise_code",
    }

    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()

    # Extract first JSON object
    match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if not match:
        return _safe_default

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return _safe_default

    # Validate required fields and allowed values
    trust = parsed.get("trust", "low")
    action = parsed.get("action", "revise_code")
    reason = parsed.get("reason", "")

    if trust not in ("high", "low"):
        trust = "low"
    if action not in ("accept", "revise_code", "revise_plan"):
        action = "revise_code"

    return {"trust": trust, "reason": str(reason), "action": action}


def strip_markdown_fences(code: str) -> str:
    """
    Remove markdown code fences that LLMs frequently wrap output in.

    Handles all variants::

        ```python\ncode\n```
        ```\ncode\n```
        ` ` `python\ncode\n` ` `   (no spaces in real output)

    This is applied as a post-processing step after every CODE-GEN call
    because prompt instructions alone are not reliable enough to prevent
    models from adding fences, especially on the first attempt.

    Args:
        code: Raw LLM response string.

    Returns:
        Code string with leading/trailing markdown fences removed.
    """
    stripped = code.strip()
    # Remove leading ```python or ``` fence
    stripped = re.sub(r"^```(?:python|py)?\s*\n?", "", stripped)
    # Remove trailing ``` fence
    stripped = re.sub(r"\n?```\s*$", "", stripped)
    return stripped.strip()
