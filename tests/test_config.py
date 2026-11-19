"""D7: config reads its env overrides and the loop/sandbox use them."""

from __future__ import annotations

import importlib
import json

import pandas as pd
import pytest

import agent.config as config


@pytest.fixture()
def reload_config(monkeypatch):
    """Reload agent.config with the current env; restore real values afterwards."""

    def _reload():
        return importlib.reload(config)

    yield _reload
    monkeypatch.undo()
    importlib.reload(config)


def test_defaults_without_env(monkeypatch, reload_config) -> None:
    for name in ("MAX_RETRIES", "SANDBOX_TIMEOUT_SECONDS", "LOG_DIR"):
        monkeypatch.setenv(name, "")
    cfg = reload_config()
    assert cfg.MAX_PIPELINE_RETRIES == 5
    assert cfg.SANDBOX_TIMEOUT_SECONDS == 30.0
    assert cfg.LOG_DIR == "runs"


def test_env_overrides_are_read(monkeypatch, reload_config) -> None:
    monkeypatch.setenv("MAX_RETRIES", "3")
    monkeypatch.setenv("SANDBOX_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("LOG_DIR", "custom_runs")
    cfg = reload_config()
    assert cfg.MAX_PIPELINE_RETRIES == 3
    assert cfg.SANDBOX_TIMEOUT_SECONDS == 12.5
    assert cfg.LOG_DIR == "custom_runs"


def test_invalid_env_value_names_the_variable(monkeypatch, reload_config) -> None:
    monkeypatch.setenv("MAX_RETRIES", "lots")
    with pytest.raises(ValueError, match="MAX_RETRIES"):
        reload_config()


def test_loop_has_no_private_timeout_constant() -> None:
    import agent.loop as loop

    assert not hasattr(loop, "SANDBOX_TIMEOUT")


def test_sandbox_default_timeout_comes_from_config() -> None:
    import inspect

    from agent.sandbox import run_sandboxed

    default = inspect.signature(run_sandboxed).parameters["timeout_seconds"].default
    assert default == config.SANDBOX_TIMEOUT_SECONDS


def test_loop_passes_config_timeout_to_sandbox(tmp_path, monkeypatch) -> None:
    from unittest.mock import patch

    from agent.loop import run

    monkeypatch.setattr(config, "SANDBOX_TIMEOUT_SECONDS", 17.0)
    df = pd.DataFrame(
        {"region": ["N"], "revenue": [1.0], "date": pd.to_datetime(["2024-01-01"])}
    )
    ok = {"success": True, "result": "1", "stdout": "", "error": None, "error_type": None}
    accept = json.dumps({"trust": "high", "reason": "ok", "action": "accept"})
    responses = iter(['["Q1?"]', "result = 1", accept, "Summary."])
    with (
        patch("agent.loop.call_llm", side_effect=lambda *a, **k: next(responses)),
        patch("agent.loop.run_sandboxed", return_value=ok) as sandbox,
        patch("agent.loop.write_log_entry"),
    ):
        run("goal", df, log_dir=str(tmp_path))
    assert sandbox.call_args.kwargs["timeout_seconds"] == 17.0


@pytest.mark.parametrize("label", ["clean", "corrupted"])
def test_run_json_records_dataset_label(tmp_path, label) -> None:
    from agent.loop import run
    from data.generate_data import generate_sales_data

    state = run(
        "goal",
        generate_sales_data(n_rows=100, seed=1),
        dry_run=True,
        log_dir=str(tmp_path),
        dataset_label=label,
    )
    data = json.loads((tmp_path / f"{state.run_id}.json").read_text())
    assert data["metadata"]["dataset"] == label


def test_run_json_abort_suffix(tmp_path) -> None:
    from unittest.mock import patch

    from agent.loop import run

    df = pd.DataFrame({"region": ["N"], "revenue": [1.0], "date": pd.to_datetime(["2024-01-01"])})
    revise = json.dumps({"trust": "low", "reason": "no", "action": "revise_code"})
    ok = {"success": True, "result": "1", "stdout": "", "error": None, "error_type": None}
    with (
        patch("agent.loop.call_llm", side_effect=lambda p, role, *a, **k: '["Q1?"]' if role == "plan" else ("result = 1" if role == "codegen" else revise)),
        patch("agent.loop.run_sandboxed", return_value=ok),
        patch("agent.loop.write_log_entry"),
    ):
        state = run("goal", df, log_dir=str(tmp_path), max_retries_per_step=1, dataset_label="corrupted")
    assert state.aborted
    data = json.loads((tmp_path / f"{state.run_id}.json").read_text())
    assert data["metadata"]["dataset"] == "corrupted (aborted)"
