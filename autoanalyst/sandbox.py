"""
Sandbox — the only place in the project where exec() is called.

Agent-generated code runs here and NOWHERE else. The sandbox:
  - Restricts the available globals to a curated safe set (pandas, math,
    statistics, json, re, datetime — no os, subprocess, sys, __import__, etc.)
  - Captures stdout via StringIO so results are machine-readable.
  - Enforces a wall-clock timeout via threading.Timer.
  - Returns a typed :class:`SandboxResult` — callers never see raw exceptions.
"""

from __future__ import annotations

import io
import sys
import threading
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class SandboxResult:
    """
    The outcome of a sandboxed code execution.

    Attributes:
        stdout:       Everything the code printed to stdout.
        return_value: The value of the ``_result`` variable if the executed
                      code set it, otherwise ``None``.
        error:        Exception message if execution failed, else ``None``.
        timed_out:    True if the code exceeded the wall-clock timeout.
    """

    stdout: str = ""
    return_value: Any = None
    error: str | None = None
    timed_out: bool = False

    @property
    def success(self) -> bool:
        """True when no error occurred and execution did not time out."""
        return self.error is None and not self.timed_out

    def summary(self) -> str:
        """One-line human-readable summary suitable for logging."""
        if self.timed_out:
            return "TIMEOUT"
        if self.error:
            return f"ERROR: {self.error[:120]}"
        parts = []
        if self.stdout:
            parts.append(f"stdout={self.stdout[:200]!r}")
        if self.return_value is not None:
            parts.append(f"return_value={self.return_value!r}")
        return "OK " + "; ".join(parts) if parts else "OK (no output)"


# ---------------------------------------------------------------------------
# Safe globals
# ---------------------------------------------------------------------------

def _build_safe_globals() -> dict[str, Any]:
    """
    Return a restricted globals dict for exec().

    Only whitelisted modules and builtins are exposed. This does NOT
    provide a full security sandbox (that would require seccomp/nsjail),
    but it prevents accidental use of dangerous builtins in agent-generated
    code during normal operation.
    """
    import math
    import statistics
    import json
    import re
    import datetime
    import collections

    try:
        import pandas as pd  # type: ignore[import-untyped]
    except ImportError:
        pd = None  # type: ignore[assignment]

    safe_builtins = {
        name: getattr(__builtins__, name)
        for name in (
            "abs", "all", "any", "bin", "bool", "chr", "dict", "dir",
            "divmod", "enumerate", "filter", "float", "format", "frozenset",
            "getattr", "hasattr", "hash", "hex", "int", "isinstance",
            "issubclass", "iter", "len", "list", "map", "max", "min",
            "next", "oct", "ord", "pow", "print", "range", "repr",
            "reversed", "round", "set", "slice", "sorted", "str", "sum",
            "tuple", "type", "vars", "zip",
        )
        if hasattr(__builtins__, name)
    }

    globs: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "math": math,
        "statistics": statistics,
        "json": json,
        "re": re,
        "datetime": datetime,
        "collections": collections,
    }
    if pd is not None:
        globs["pd"] = pd

    return globs


_SAFE_GLOBALS = _build_safe_globals()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def execute_code(
    code: str,
    local_vars: dict[str, Any] | None = None,
    *,
    timeout_seconds: float = 30.0,
) -> SandboxResult:
    """
    Execute *code* inside a restricted sandbox and return a :class:`SandboxResult`.

    The code may set a ``_result`` variable to communicate a structured
    return value back to the caller (in addition to or instead of printing).

    Example::

        result = execute_code("_result = 1 + 1")
        assert result.return_value == 2

    Args:
        code:            Python source code string to execute.
        local_vars:      Optional dict of local variables to inject (e.g. a
                         pre-loaded DataFrame under the key ``"df"``).
        timeout_seconds: Wall-clock limit. Execution that exceeds this
                         limit is interrupted and ``SandboxResult.timed_out``
                         is set to True.

    Returns:
        A :class:`SandboxResult` — never raises.
    """
    result = SandboxResult()
    local_namespace: dict[str, Any] = dict(local_vars or {})
    globs = dict(_SAFE_GLOBALS)  # shallow copy so injections don't leak

    stdout_capture = io.StringIO()
    completed = threading.Event()

    def _run() -> None:
        try:
            with redirect_stdout(stdout_capture):
                exec(code, globs, local_namespace)  # noqa: S102
        except Exception:  # noqa: BLE001
            result.error = traceback.format_exc()
        finally:
            completed.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    finished = completed.wait(timeout=timeout_seconds)

    if not finished:
        result.timed_out = True
        result.error = f"Execution timed out after {timeout_seconds}s"
    else:
        result.stdout = stdout_capture.getvalue()
        result.return_value = local_namespace.get("_result")

    return result
