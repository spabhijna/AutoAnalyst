"""
EXECUTE stage — runs agent-generated code inside the sandbox.

This is a thin adapter between the pipeline loop and
:func:`autoanalyst.sandbox.execute_code`. All exec() calls funnel
through the sandbox; this module adds logging and result formatting.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from autoanalyst.sandbox import SandboxResult, execute_code

if TYPE_CHECKING:
    from autoanalyst.config import PipelineConfig

logger = logging.getLogger(__name__)


def execute(
    code: str,
    local_vars: dict[str, Any] | None = None,
    *,
    config: "PipelineConfig | None" = None,
) -> SandboxResult:
    """
    Execute *code* inside the sandbox and return the result.

    This is the sole entry point for code execution in the pipeline.
    It delegates entirely to :func:`~autoanalyst.sandbox.execute_code`
    so that the sandbox remains the single exec() site.

    Args:
        code:       Python source code string (agent-generated).
        local_vars: Optional dict of variables to inject before execution
                    (typically ``{"df": dataframe}``).
        config:     Active config (used only to derive timeout in future;
                    currently uses sandbox default of 30 s).

    Returns:
        :class:`~autoanalyst.sandbox.SandboxResult` with stdout, return_value,
        error, and timed_out fields populated.
    """
    logger.info("EXECUTE stage: code_len=%d chars", len(code))

    timeout = 30.0  # will be moved to config in Phase 1
    result = execute_code(code, local_vars=local_vars, timeout_seconds=timeout)

    if result.success:
        logger.info("EXECUTE stage: OK — %s", result.summary())
    else:
        logger.warning("EXECUTE stage: FAILED — %s", result.summary())

    return result


def format_execution_result(result: SandboxResult) -> str:
    """
    Format a :class:`~autoanalyst.sandbox.SandboxResult` as a human-readable
    string suitable for passing to the OBSERVE and CRITIQUE stages.

    Args:
        result: The sandbox execution result.

    Returns:
        A multi-line string describing the outcome.
    """
    parts: list[str] = []

    if result.timed_out:
        parts.append("STATUS: TIMED OUT")
    elif result.error:
        parts.append("STATUS: ERROR")
        parts.append(f"Error:\n{result.error}")
    else:
        parts.append("STATUS: OK")

    if result.stdout:
        parts.append(f"Stdout:\n{result.stdout.strip()}")

    if result.return_value is not None:
        parts.append(f"Return value: {result.return_value!r}")

    return "\n\n".join(parts)
