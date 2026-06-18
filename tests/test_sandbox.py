"""
Tests for autoanalyst.sandbox — the only exec() site in the project.

Verifies:
- Successful code execution with stdout capture.
- _result variable is returned as return_value.
- Dangerous builtins are blocked.
- Exceptions in the executed code are caught and reported cleanly.
- Timeout enforcement.
- local_vars injection (e.g. a DataFrame).
"""

from __future__ import annotations

import time

import pytest

from autoanalyst.sandbox import SandboxResult, execute_code


class TestExecuteCodeSuccess:
    def test_hello_world(self) -> None:
        """Basic print capture."""
        result = execute_code("print('hello')")
        assert result.success
        assert "hello" in result.stdout
        assert result.error is None

    def test_return_value(self) -> None:
        """Code can communicate a result via _result."""
        result = execute_code("_result = 6 * 7")
        assert result.success
        assert result.return_value == 42

    def test_math_import_available(self) -> None:
        """math module should be available in safe globals."""
        result = execute_code("import math; _result = math.sqrt(9)")
        assert result.success
        assert result.return_value == pytest.approx(3.0)

    def test_local_vars_injection(self) -> None:
        """Variables passed via local_vars should be accessible in the code."""
        result = execute_code("_result = x + 1", local_vars={"x": 41})
        assert result.success
        assert result.return_value == 42

    def test_pandas_available(self) -> None:
        """pd (pandas) should be importable from safe globals."""
        pytest.importorskip("pandas")
        code = "import pandas as pd; _result = len(pd.DataFrame({'a': [1,2,3]}))"
        result = execute_code(code)
        assert result.success
        assert result.return_value == 3

    def test_multiline_code(self) -> None:
        """Multi-line snippets should work correctly."""
        code = """
total = 0
for i in range(10):
    total += i
_result = total
"""
        result = execute_code(code)
        assert result.success
        assert result.return_value == 45


class TestExecuteCodeErrors:
    def test_exception_captured(self) -> None:
        """Exceptions in executed code must NOT propagate — captured in .error."""
        result = execute_code("raise ValueError('oops')")
        assert not result.success
        assert result.error is not None
        assert "ValueError" in result.error
        assert "oops" in result.error

    def test_zero_division(self) -> None:
        """ZeroDivisionError should be caught and reported."""
        result = execute_code("_result = 1 / 0")
        assert not result.success
        assert result.error is not None

    def test_name_error(self) -> None:
        """NameError for undefined variables should be caught."""
        result = execute_code("_result = undefined_variable")
        assert not result.success
        assert result.error is not None


class TestSandboxIsolation:
    def test_os_blocked(self) -> None:
        """os module must NOT be importable from the sandbox."""
        result = execute_code("import os; _result = os.getcwd()")
        # Either NameError (os not in globals) or ImportError — both fine
        assert not result.success

    def test_open_builtin_blocked(self) -> None:
        """open() must not be available — prevents reading arbitrary files."""
        result = execute_code("_result = open('/etc/passwd').read()")
        assert not result.success

    def test_subprocess_blocked(self) -> None:
        """subprocess should not be importable."""
        result = execute_code("import subprocess; subprocess.run(['ls'])")
        assert not result.success

    def test_no_cross_run_state_leak(self) -> None:
        """State from one execute_code call must not leak into the next."""
        execute_code("secret = 'should_not_leak'")
        result = execute_code("_result = 'secret' in dir()")
        # secret was never defined in this call's namespace
        assert result.return_value is False or not result.success


class TestSandboxResult:
    def test_summary_ok(self) -> None:
        """SandboxResult.summary() returns a human-readable string."""
        result = execute_code("print('hi'); _result = 99")
        assert result.success
        summary = result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_summary_error(self) -> None:
        result = execute_code("raise RuntimeError('test')")
        assert not result.success
        assert "ERROR" in result.summary()

    def test_timed_out_flag(self) -> None:
        """Code exceeding timeout must set timed_out=True."""
        result = execute_code(
            "import time; time.sleep(10)",
            timeout_seconds=0.5,
        )
        # time is not in safe globals, so this will likely NameError,
        # but we test with a safe alternative: a tight loop
        result2 = execute_code(
            "i = 0\nwhile True:\n    i += 1",
            timeout_seconds=0.3,
        )
        assert result2.timed_out
        assert not result2.success
