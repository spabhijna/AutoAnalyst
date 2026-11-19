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
import logging
import multiprocessing
import pickle
import sys
import traceback
from typing import Any

import numpy as np
import pandas as pd

from agent import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RESULT_TRUNCATE_CHARS: int = 2000

# Hard limit for DataFrame artifact extraction via sandbox IPC.
# DataFrames larger than this are not extracted (too costly to pickle/transfer).
_ARTIFACT_SIZE_LIMIT_BYTES: int = 50 * 1024 * 1024  # 50 MB

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
        "abs",
        "divmod",
        "max",
        "min",
        "pow",
        "round",
        "sum",
        # type constructors
        "bool",
        "bytes",
        "complex",
        "dict",
        "float",
        "frozenset",
        "int",
        "list",
        "set",
        "str",
        "tuple",
        # itertools-adjacent
        "all",
        "any",
        "enumerate",
        "filter",
        "iter",
        "len",
        "map",
        "next",
        "range",
        "reversed",
        "sorted",
        "zip",
        # inspection / formatting
        "bin",
        "chr",
        "dir",
        "format",
        "getattr",
        "hasattr",
        "hash",
        "hex",
        "id",
        "isinstance",
        "issubclass",
        "oct",
        "ord",
        "repr",
        "type",
        "vars",
        # exceptions
        "Exception",
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "AttributeError",
        "RuntimeError",
        "StopIteration",
        "NotImplementedError",
        "ArithmeticError",
        "ZeroDivisionError",
        "OverflowError",
        # constants
        "True",
        "False",
        "None",
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
    artifacts_pickle: bytes | None = None,
    extract_artifact: str | None = None,
) -> None:
    """
    Execute *code* in a restricted namespace and push the result dict onto
    *result_queue*.

    Runs inside a child process spawned by :func:`run_sandboxed`. Must never
    raise — all exceptions are caught and serialized into the result dict.

    Args:
        code:             Python source code string to execute.
        df_pickle:        Pickled DataFrame bytes.
        result_queue:     Queue to push the result dict onto.
        artifacts_pickle: Pickled dict[str, pd.DataFrame] of pre-computed
                          artifacts from the ArtifactStore. Injected into
                          the sandbox namespace as ``artifacts``.
        extract_artifact: If set, the variable with this name is extracted
                          from the sandbox local namespace after execution
                          and included in the result dict as ``"artifact"``.
                          Must be a pd.DataFrame ≤ _ARTIFACT_SIZE_LIMIT_BYTES.
    """
    # Reconstruct DataFrame — normal unpickling, uses standard builtins
    try:
        df = pickle.loads(df_pickle)  # noqa: S301
    except Exception as exc:  # noqa: BLE001
        result_queue.put(
            {
                "success": False,
                "result": None,
                "stdout": "",
                "error": f"DataFrame deserialization failed: {exc}",
                "error_type": type(exc).__name__,
                "error_category": "execution_error",
                "artifact": None,
                "artifact_key": extract_artifact,
            }
        )
        return

    # Reconstruct artifacts dict — empty dict if not provided
    artifacts: dict[str, Any] = {}
    if artifacts_pickle:
        try:
            artifacts = pickle.loads(artifacts_pickle)  # noqa: S301
        except Exception:  # noqa: BLE001
            artifacts = {}

    # Build restricted namespace for exec() — pd and np come from worker imports
    global_ns: dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,  # restricted: no open/import/os/sys
        "df": df,
        "pd": pd,
        "np": np,
        "artifacts": artifacts,  # pre-validated DataFrames from ArtifactStore
    }
    local_ns: dict[str, Any] = {}

    # Capture stdout from agent code
    stdout_buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_buf

    try:
        exec(code, global_ns, local_ns)  # noqa: S102
        raw_result = local_ns.get("result", global_ns.get("result", None))
        result_repr = _safe_repr(raw_result)

        # ── Artifact extraction ───────────────────────────────────────────────
        # If requested, extract the named DataFrame variable from the sandbox
        # namespace and include it in the result dict for ArtifactStore registration.
        extracted_artifact: pd.DataFrame | None = None
        if extract_artifact:
            # Use explicit None check — `or` would trigger pd.DataFrame.__bool__
            # which raises "ValueError: The truth value of a DataFrame is ambiguous".
            candidate = local_ns.get(extract_artifact)
            if candidate is None:
                candidate = global_ns.get(extract_artifact)
            if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                try:
                    if (
                        len(pickle.dumps(candidate, protocol=4))
                        <= _ARTIFACT_SIZE_LIMIT_BYTES
                    ):
                        extracted_artifact = candidate
                except Exception:  # noqa: BLE001
                    pass  # silently skip oversized or un-picklable artifacts

        # ── New variable capture (Phase-1) ────────────────────────────────────
        # Collect every name that the executed code wrote into local_ns.
        # Protected names (result, _*) and the sandbox globals are excluded.
        # This dict is returned to the caller so the runtime can absorb new
        # variables into its persistent namespace without relying on the LLM
        # to use any particular variable name.
        _protected = {"result", "pd", "np", "df", "artifacts", "__builtins__"}
        new_variables: dict[str, Any] = {
            k: v
            for k, v in local_ns.items()
            if k not in _protected and not k.startswith("_")
        }

        result_queue.put(
            {
                "success": True,
                "result": result_repr,
                "stdout": stdout_buf.getvalue(),
                "error": None,
                "error_type": None,
                "error_category": "none",
                "artifact": extracted_artifact,
                "artifact_key": extract_artifact,
                "new_variables": new_variables,
            }
        )
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        # Extract the exception class name from the last line of the traceback
        last_line = tb.strip().splitlines()[-1]
        exc_type = type(exc).__name__

        is_capability_error = False
        if exc_type in ("ImportError", "ModuleNotFoundError", "PermissionError"):
            is_capability_error = True
        elif exc_type == "NameError":
            is_capability_error = True

        result_queue.put(
            {
                "success": False,
                "result": None,
                "stdout": stdout_buf.getvalue(),
                "error": tb,
                "error_type": exc_type,
                "error_category": "capability_error"
                if is_capability_error
                else "execution_error",
                "artifact": None,
                "artifact_key": extract_artifact,
                "new_variables": {},
            }
        )
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
        r = (
            r[:_RESULT_TRUNCATE_CHARS]
            + f"… [truncated, full length={len(repr(value))}]"
        )
    return r


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_sandboxed(
    code: str,
    df: pd.DataFrame,
    artifacts: dict[str, pd.DataFrame] | None = None,
    extract_artifact: str | None = None,
    timeout_seconds: float = config.SANDBOX_TIMEOUT_SECONDS,
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
    - ``df``        — the pandas DataFrame passed as argument
    - ``pd``        — pandas module
    - ``np``        — numpy module
    - ``artifacts`` — dict[str, pd.DataFrame] from ArtifactStore (may be empty)
    - ``__builtins__`` — restricted whitelist (no open, no import, no os/sys)

    **Artifact extraction:**
    If *extract_artifact* is set (e.g. ``"quarter_df"``), the sandbox extracts
    the variable with that name from the executed code's local namespace and
    returns it as ``out["artifact"]`` (a raw pd.DataFrame). The caller
    (``_run_question`` in loop.py) registers it in the ArtifactStore.
    Only non-empty DataFrames under _ARTIFACT_SIZE_LIMIT_BYTES are extracted.

    Args:
        code:             Python source code string to execute.
        df:               The DataFrame to make available as ``df`` inside the code.
        artifacts:        Pre-computed DataFrames from ArtifactStore to inject as
                          the ``artifacts`` global dict. Defaults to empty dict.
        extract_artifact: Variable name to extract from the sandbox namespace
                          after execution and return as a raw DataFrame.
        timeout_seconds:  Hard wall-clock limit. Defaults to
                          ``config.SANDBOX_TIMEOUT_SECONDS``.

    Returns:
        A dict with the following keys::

            {
                "success":      bool,
                "result":       str | None,   # repr of ``result`` variable
                "stdout":       str,
                "error":        str | None,
                "error_type":   str | None,
                "artifact":     pd.DataFrame | None,  # only if extract_artifact set
                "artifact_key": str | None,           # same as extract_artifact
            }

    Notes:
        - Exceptions inside sandboxed code NEVER propagate to the caller.
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
            "error_category": "execution_error",
            "artifact": None,
            "artifact_key": extract_artifact,
            "new_variables": {},
        }

    # Pickle the artifacts dict for inter-process transfer
    artifacts_pickle: bytes | None = None
    if artifacts:
        try:
            artifacts_pickle = pickle.dumps(artifacts, protocol=4)
        except Exception as exc:  # noqa: BLE001
            # Non-fatal: proceed without artifacts in the sandbox
            logger.warning("Failed to serialize artifacts for sandbox: %s", exc)
            artifacts_pickle = None

    # Use 'spawn' context for clean isolation (no inherited file descriptors)
    ctx = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue[dict[str, Any]] = ctx.Queue()

    process = ctx.Process(
        target=_worker,
        args=(code, df_pickle, result_queue, artifacts_pickle, extract_artifact),
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
            "error_category": "execution_error",
            "artifact": None,
            "artifact_key": extract_artifact,
            "new_variables": {},
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
        "error_category": "execution_error",
        "artifact": None,
        "artifact_key": extract_artifact,
        "new_variables": {},
    }
