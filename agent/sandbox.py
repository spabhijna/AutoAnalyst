"""
Safe code execution sandbox.

Design decisions (read before modifying)
-----------------------------------------

1. **Process-based isolation, not thread-based.**
   A daemon thread cannot be hard-killed in CPython — only cooperative
   signalling works, which means a tight ``while True: pass`` loop will
   hang the host process until the GIL eventually yields (which it may
   not). ``multiprocessing`` gives us a real OS process that can be
   terminated with ``Process.kill()`` unconditionally.

2. **DataFrame passed via pickle through multiprocessing.Queue.**
   A raw pandas DataFrame is picklable within the same Python version.
   We pickle it before passing to the child, reconstruct inside the worker,
   and then inject it into the ``exec()`` namespace. pyarrow/fastparquet
   are NOT required — we use stdlib ``pickle`` to avoid optional deps.

3. **Result convention: assign to ``result``.**
   Agent-generated code MUST store its final answer in a variable named
   ``result``. The sandbox reads ``local_ns["result"]`` after execution.
   We chose this over last-expression capture (ast.parse + eval) because:
   - Multi-line code doesn't have a single "last expression"
   - AST inspection is fragile across Python versions and code styles
   - A fixed convention is easy to enforce in the CODE-GEN prompt (Phase 3)
   This convention is a HARD DEPENDENCY for Phase 3. See PROGRESS.md.

4. **Restricted namespace applied inside exec(), not at worker level.**
   The worker module itself imports pandas and numpy normally at module
   load time (required for unpickling and rebuilding the namespace).
   Only the ``__builtins__`` dict passed to ``exec()`` is restricted,
   blocking ``open``, ``__import__``, ``eval``, ``exec``, ``compile``,
   and all OS/sys/subprocess access from *within* the agent-generated code.

5. **stdout capture.**
   The worker redirects ``sys.stdout`` before calling ``exec()`` so any
   ``print()`` calls in the agent code are captured and returned.

Public API
----------
    from agent.sandbox import run_sandboxed
    out = run_sandboxed(code_str, df)
    if out["success"]:
        print(out["result"])
    else:
        print(out["error"])
"""

from __future__ import annotations

import io
import multiprocessing
import pickle
import sys
import traceback
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RESULT_TRUNCATE_CHARS: int = 2000

# Builtins whitelist — injected as __builtins__ inside exec() only.
# The worker itself uses the normal builtins; restriction applies only to
# agent-generated code running under exec().
_SAFE_BUILTINS: dict[str, Any] = {
    name: (
        __builtins__[name]  # type: ignore[index]
        if isinstance(__builtins__, dict)
        else getattr(__builtins__, name)
    )
    for name in (
        # numeric / math
        "abs", "divmod", "max", "min", "pow", "round", "sum",
        # type constructors
        "bool", "bytes", "complex", "dict", "float", "frozenset",
        "int", "list", "set", "str", "tuple",
        # itertools-adjacent
        "all", "any", "enumerate", "filter", "iter", "len",
        "map", "next", "range", "reversed", "sorted", "zip",
        # inspection / formatting
        "bin", "chr", "dir", "format", "getattr", "hasattr",
        "hash", "hex", "id", "isinstance", "issubclass", "oct",
        "ord", "repr", "type", "vars",
        # exceptions
        "Exception", "ValueError", "TypeError", "KeyError",
        "IndexError", "AttributeError", "RuntimeError",
        "StopIteration", "NotImplementedError", "ArithmeticError",
        "ZeroDivisionError", "OverflowError",
        # constants
        "True", "False", "None",
        # printing (output captured)
        "print",
    )
    if (
        (isinstance(__builtins__, dict) and name in __builtins__)  # type: ignore[operator]
        or (not isinstance(__builtins__, dict) and hasattr(__builtins__, name))
    )
}


# ---------------------------------------------------------------------------
# Worker (runs inside the child process)
# ---------------------------------------------------------------------------


def _worker(
    code: str,
    df_pickle: bytes,
    result_queue: "multiprocessing.Queue[dict[str, Any]]",
) -> None:
    """
    Execute *code* in a restricted namespace and push the result dict onto
    *result_queue*.

    Runs inside a child process spawned by :func:`run_sandboxed`. Must never
    raise — all exceptions are caught and serialized into the result dict.

    Args:
        code:         Python source code string to execute.
        df_pickle:    Pickled DataFrame bytes.
        result_queue: Queue to push the result dict onto.
    """
    # Reconstruct DataFrame — normal unpickling, uses standard builtins
    try:
        df = pickle.loads(df_pickle)  # noqa: S301
    except Exception as exc:  # noqa: BLE001
        result_queue.put({
            "success": False,
            "result": None,
            "stdout": "",
            "error": f"DataFrame deserialization failed: {exc}",
            "error_type": type(exc).__name__,
        })
        return

    # Build restricted namespace for exec() — pd and np come from worker imports
    local_ns: dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,  # restricted: no open/import/os/sys
        "df": df,
        "pd": pd,
        "np": np,
    }

    # Capture stdout from agent code
    stdout_buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_buf

    try:
        exec(code, local_ns)  # noqa: S102
        raw_result = local_ns.get("result", None)
        result_repr = _safe_repr(raw_result)
        result_queue.put({
            "success": True,
            "result": result_repr,
            "stdout": stdout_buf.getvalue(),
            "error": None,
            "error_type": None,
        })
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        # Extract the exception class name from the last line of the traceback
        last_line = tb.strip().splitlines()[-1]
        exc_type = last_line.split(":")[0].strip()
        result_queue.put({
            "success": False,
            "result": None,
            "stdout": stdout_buf.getvalue(),
            "error": tb,
            "error_type": exc_type,
        })
    finally:
        sys.stdout = old_stdout


def _safe_repr(value: Any) -> str | None:
    """
    Produce a truncated repr string for *value*.

    Args:
        value: Any Python object returned by the sandboxed code.

    Returns:
        repr string truncated to :data:`_RESULT_TRUNCATE_CHARS` characters,
        or ``None`` if *value* is ``None``.
    """
    if value is None:
        return None
    try:
        r = repr(value)
    except Exception:  # noqa: BLE001
        r = f"<un-repr-able object of type {type(value).__name__}>"
    if len(r) > _RESULT_TRUNCATE_CHARS:
        r = r[:_RESULT_TRUNCATE_CHARS] + f"… [truncated, full length={len(repr(value))}]"
    return r


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_sandboxed(
    code: str,
    df: pd.DataFrame,
    timeout_seconds: int = 5,
) -> dict[str, Any]:
    """
    Execute *code* against *df* in an isolated child process with a hard timeout.

    The child process is hard-killed (``SIGKILL`` on Unix, ``TerminateProcess``
    on Windows) if it does not complete within *timeout_seconds*. No cooperative
    signalling is relied upon — an infinite loop will always be killed.

    **Result convention (Phase 3 hard dependency):**
    Agent-generated code MUST assign its final answer to a variable named
    ``result``. The sandbox reads ``local_ns["result"]`` and returns its repr.
    If the code does not set ``result``, ``out["result"]`` will be ``None``
    even on success. See module docstring § 3 for rationale.

    **Available in the sandbox namespace:**
    - ``df``  — the pandas DataFrame passed as argument
    - ``pd``  — pandas module
    - ``np``  — numpy module
    - ``__builtins__`` — restricted whitelist (no open, no import, no os/sys)

    Args:
        code:            Python source code string to execute.
        df:              The DataFrame to make available as ``df`` inside the code.
        timeout_seconds: Hard wall-clock limit. The child process is killed if
                         execution exceeds this value. Defaults to 5 s.

    Returns:
        A dict with the following keys:

        .. code-block:: python

            {
                "success":    bool,        # True if no exception was raised
                "result":     str | None,  # repr of ``result`` variable (truncated)
                "stdout":     str,         # everything printed to stdout
                "error":      str | None,  # full traceback if success=False
                "error_type": str | None,  # exception class name if success=False
            }

    Notes:
        - Exceptions raised inside the sandboxed code NEVER propagate to the caller.
        - ``success=False`` is returned for: exceptions, import errors, timeouts.
        - On timeout, ``error_type`` is ``"TimeoutError"``.
    """
    # Pickle the DataFrame for inter-process transfer
    try:
        df_pickle = pickle.dumps(df)
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "result": None,
            "stdout": "",
            "error": f"Failed to serialize DataFrame for sandbox: {exc}",
            "error_type": type(exc).__name__,
        }

    # Use 'spawn' context for clean isolation (no inherited file descriptors)
    ctx = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue[dict[str, Any]] = ctx.Queue()

    process = ctx.Process(
        target=_worker,
        args=(code, df_pickle, result_queue),
        daemon=True,
    )
    process.start()
    process.join(timeout=timeout_seconds)

    if process.is_alive():
        # Hard kill — unconditional, no cooperative signalling
        process.kill()
        process.join()
        return {
            "success": False,
            "result": None,
            "stdout": "",
            "error": (
                f"Code execution timed out after {timeout_seconds}s "
                f"and was forcibly terminated."
            ),
            "error_type": "TimeoutError",
        }

    # Retrieve result from the queue
    if not result_queue.empty():
        return result_queue.get_nowait()

    # Process exited without putting anything in the queue (e.g. OOM kill)
    return {
        "success": False,
        "result": None,
        "stdout": "",
        "error": (
            f"Child process exited with code {process.exitcode} "
            f"without returning a result."
        ),
        "error_type": "ProcessError",
    }
