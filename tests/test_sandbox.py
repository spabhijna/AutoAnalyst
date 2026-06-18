"""
Tests for agent.sandbox — the safe code execution sandbox.

Test matrix
-----------
Normal operations:
  - pandas aggregation succeeds, result returned via ``result`` variable
  - stdout captured correctly (print statements)
  - ``result`` not set → success=True but result=None
  - multi-line code works correctly
  - numpy available in namespace

Security / isolation:
  - ``import os`` is blocked
  - ``import sys`` is blocked
  - ``import subprocess`` is blocked
  - ``open()`` is blocked
  - ``__import__`` is blocked
  - code cannot reach the parent process filesystem

Timeout / kill:
  - infinite loop is hard-killed within the timeout window
  - timeout sets success=False, error_type="TimeoutError"
  - normal code completes well within timeout

Error handling:
  - ZeroDivisionError caught, returned as error dict
  - NameError for undefined variable caught
  - exception never propagates to the caller (no pytest.raises needed)
  - result dict always has all five expected keys

Result formatting:
  - long repr is truncated to ~2000 chars
  - None result is returned as None (not "None")
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from agent.sandbox import run_sandboxed

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RESULT_KEYS = {"success", "result", "stdout", "error", "error_type"}


@pytest.fixture()
def sales_df() -> pd.DataFrame:
    """Minimal 5-row sales DataFrame for sandbox tests."""
    return pd.DataFrame(
        {
            "region": ["North", "South", "East", "North", "West"],
            "revenue": [100.0, 200.0, 150.0, 300.0, 250.0],
            "units_sold": [1, 2, 3, 4, 5],
        }
    )


def _run(code: str, df: pd.DataFrame, timeout: int = 10) -> dict:
    """Helper to run sandboxed code and assert the result dict has all keys."""
    out = run_sandboxed(code, df, timeout_seconds=timeout)
    assert set(out.keys()) == RESULT_KEYS, f"Missing keys: {RESULT_KEYS - set(out.keys())}"
    return out


# ---------------------------------------------------------------------------
# Normal pandas operations
# ---------------------------------------------------------------------------


class TestNormalOperations:
    def test_simple_aggregation(self, sales_df: pd.DataFrame) -> None:
        """Basic groupby + sum should succeed and return the correct value."""
        code = "result = df.groupby('region')['revenue'].sum().to_dict()"
        out = _run(code, sales_df)
        assert out["success"] is True
        assert out["error"] is None
        assert out["error_type"] is None
        # Result is a repr of a dict — check it's there and mentions regions
        assert out["result"] is not None
        assert "North" in out["result"]

    def test_scalar_result(self, sales_df: pd.DataFrame) -> None:
        """A scalar assigned to result should be returned as its repr.

        pandas .sum() returns np.float64, so we compare against the actual
        repr of that type rather than Python's float repr.
        """
        code = "result = df['revenue'].sum()"
        out = _run(code, sales_df)
        assert out["success"] is True
        # Accept either np.float64 repr or plain float repr — both are correct
        assert out["result"] is not None
        assert "1000.0" in out["result"]

    def test_stdout_captured(self, sales_df: pd.DataFrame) -> None:
        """print() calls in agent code should be captured in stdout."""
        code = "print('hello from sandbox'); result = 42"
        out = _run(code, sales_df)
        assert out["success"] is True
        assert "hello from sandbox" in out["stdout"]
        assert out["result"] == repr(42)

    def test_no_result_variable(self, sales_df: pd.DataFrame) -> None:
        """Code that doesn't set result → success=True, result=None."""
        code = "x = df['revenue'].mean()"
        out = _run(code, sales_df)
        assert out["success"] is True
        assert out["result"] is None

    def test_multiline_code(self, sales_df: pd.DataFrame) -> None:
        """Multi-line code with intermediate variables should work.

        round() on np.float64 returns np.float64, so check for the value
        rather than exact repr type.
        """
        code = """
total = df['revenue'].sum()
count = len(df)
avg = total / count
result = round(avg, 2)
"""
        out = _run(code, sales_df)
        assert out["success"] is True
        assert out["result"] is not None
        assert "200.0" in out["result"]

    def test_numpy_available(self, sales_df: pd.DataFrame) -> None:
        """np (numpy) must be available in the sandbox namespace."""
        code = "result = float(np.mean(df['units_sold'].values))"
        out = _run(code, sales_df)
        assert out["success"] is True
        assert out["result"] == repr(3.0)

    def test_multiple_prints(self, sales_df: pd.DataFrame) -> None:
        """Multiple print calls should all appear in stdout."""
        code = "print('line1'); print('line2'); result = 'done'"
        out = _run(code, sales_df)
        assert "line1" in out["stdout"]
        assert "line2" in out["stdout"]

    def test_dataframe_shape_accessible(self, sales_df: pd.DataFrame) -> None:
        """df should be accessible with correct shape."""
        code = "result = df.shape"
        out = _run(code, sales_df)
        assert out["success"] is True
        assert "(5, 3)" in out["result"]


# ---------------------------------------------------------------------------
# Security / isolation
# ---------------------------------------------------------------------------


class TestSecurityIsolation:
    def test_import_os_blocked(self, sales_df: pd.DataFrame) -> None:
        """import os must be blocked inside the sandbox."""
        code = "import os; result = os.getcwd()"
        out = _run(code, sales_df)
        assert out["success"] is False
        assert out["error"] is not None

    def test_import_sys_blocked(self, sales_df: pd.DataFrame) -> None:
        """import sys must be blocked."""
        code = "import sys; result = sys.version"
        out = _run(code, sales_df)
        assert out["success"] is False

    def test_import_subprocess_blocked(self, sales_df: pd.DataFrame) -> None:
        """import subprocess must be blocked."""
        code = "import subprocess; result = subprocess.run(['ls'], capture_output=True)"
        out = _run(code, sales_df)
        assert out["success"] is False

    def test_open_blocked(self, sales_df: pd.DataFrame) -> None:
        """open() must not be available in the sandbox."""
        code = "result = open('/etc/passwd').read()"
        out = _run(code, sales_df)
        assert out["success"] is False

    def test_dunder_import_blocked(self, sales_df: pd.DataFrame) -> None:
        """__import__ must not be available."""
        code = "os = __import__('os'); result = os.getcwd()"
        out = _run(code, sales_df)
        assert out["success"] is False

    def test_no_state_leak_between_runs(self, sales_df: pd.DataFrame) -> None:
        """Each run_sandboxed call gets a fresh namespace — no state leaks."""
        _run("secret_var = 'should_not_leak'", sales_df)
        out = _run("result = 'secret_var' in dir()", sales_df)
        # secret_var was set in a separate process — cannot be in this run's dir()
        assert out["success"] is True
        assert out["result"] == repr(False)


# ---------------------------------------------------------------------------
# Timeout / hard kill
# ---------------------------------------------------------------------------


class TestTimeout:
    def test_infinite_loop_killed(self, sales_df: pd.DataFrame) -> None:
        """An infinite loop must be hard-killed within the timeout."""
        code = "while True: pass"
        t0 = time.monotonic()
        out = _run(code, sales_df, timeout=2)
        elapsed = time.monotonic() - t0

        assert out["success"] is False
        assert out["error_type"] == "TimeoutError"
        assert "timed out" in out["error"].lower()
        # Should terminate within a reasonable margin above the timeout
        assert elapsed < 5, f"Timeout took {elapsed:.1f}s — too slow"

    def test_timeout_result_structure(self, sales_df: pd.DataFrame) -> None:
        """Timeout result must have all required keys with correct types."""
        out = _run("while True: pass", sales_df, timeout=1)
        assert out["success"] is False
        assert out["result"] is None
        assert isinstance(out["stdout"], str)
        assert isinstance(out["error"], str)
        assert out["error_type"] == "TimeoutError"

    def test_fast_code_completes_before_timeout(self, sales_df: pd.DataFrame) -> None:
        """Fast code should not be killed prematurely."""
        code = "result = df['revenue'].sum()"
        out = _run(code, sales_df, timeout=5)
        assert out["success"] is True


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_zero_division_caught(self, sales_df: pd.DataFrame) -> None:
        """ZeroDivisionError in sandbox must not propagate to caller."""
        code = "result = 1 / 0"
        out = _run(code, sales_df)
        assert out["success"] is False
        assert "ZeroDivisionError" in (out["error"] or "")

    def test_name_error_caught(self, sales_df: pd.DataFrame) -> None:
        """NameError for undefined variable must be caught."""
        code = "result = undefined_variable_xyz"
        out = _run(code, sales_df)
        assert out["success"] is False
        assert out["error"] is not None

    def test_type_error_caught(self, sales_df: pd.DataFrame) -> None:
        """TypeError must be caught and returned."""
        code = "result = 'string' + 42"
        out = _run(code, sales_df)
        assert out["success"] is False
        assert out["error"] is not None

    def test_exception_never_propagates(self, sales_df: pd.DataFrame) -> None:
        """run_sandboxed must NEVER raise — all errors go into the result dict."""
        # This should not raise — pytest would catch it as a test failure
        out = run_sandboxed("raise RuntimeError('boom')", sales_df)
        assert isinstance(out, dict)
        assert out["success"] is False


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------


class TestResultFormatting:
    def test_long_result_truncated(self, sales_df: pd.DataFrame) -> None:
        """Results longer than ~2000 chars must be truncated."""
        # Generate a string longer than 2000 chars
        code = "result = 'x' * 5000"
        out = _run(code, sales_df)
        assert out["success"] is True
        assert len(out["result"]) <= 2100  # repr adds quotes + truncation suffix

    def test_none_result_is_none(self, sales_df: pd.DataFrame) -> None:
        """If result variable is not set, out['result'] is None (not 'None')."""
        code = "x = 1 + 1"
        out = _run(code, sales_df)
        assert out["result"] is None

    def test_dict_result_repr(self, sales_df: pd.DataFrame) -> None:
        """Dict result should return a valid repr string."""
        code = "result = {'a': 1, 'b': 2}"
        out = _run(code, sales_df)
        assert out["success"] is True
        assert "{'a': 1, 'b': 2}" in out["result"]
