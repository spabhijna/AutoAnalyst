"""D9: codegen is told the artifact contract, and the contract is still enforced."""

from __future__ import annotations

import json

import pytest

from agent.loop import _artifact_contract_text, run
from agent.state import TaskNode
from data.generate_data import generate_sales_data

PLAN = json.dumps(
    [
        {
            "question": "Average units sold by region",
            "capability": None,
            "produces_artifact": "units_by_region",
            "consumes_artifacts": [],
            "output_schema": ["region", "avg_units"],
        }
    ]
)
ACCEPT = json.dumps({"trust": "high", "reason": "ok", "action": "accept"})
REVISE = json.dumps({"trust": "low", "reason": "fix it", "action": "revise_code"})

ASSIGNS_VARIABLE = (
    "units_by_region = df.groupby('region', as_index=False)['units_sold'].mean()"
    ".rename(columns={'units_sold': 'avg_units'})[['avg_units', 'region']]\n"
    "result = units_by_region"
)
NEVER_ASSIGNS_VARIABLE = (
    "out = df.groupby('region', as_index=False)['units_sold'].mean()\nresult = out"
)


class FakeLLM:
    """Plan → fixed plan; codegen → *code*; critique → revise iff the prompt shows an error."""

    def __init__(self, code: str) -> None:
        self.code = code
        self.prompts: dict[str, list[str]] = {}

    def __call__(self, prompt: str, role: str, *a, **k) -> str:
        self.prompts.setdefault(role, []).append(prompt)
        if role == "plan":
            return PLAN
        if role == "codegen":
            return self.code
        if role == "critique":
            return REVISE if "EXECUTION ERROR" in prompt else ACCEPT
        return "Summary."


@pytest.fixture()
def df():
    return generate_sales_data(n_rows=200, seed=7)


def test_column_order_is_irrelevant_and_prompt_has_contract(df, tmp_path) -> None:
    llm = FakeLLM(ASSIGNS_VARIABLE)
    state = run("goal", df, log_dir=str(tmp_path), llm=llm)

    assert not state.aborted
    assert state.tasks[0].status == "accepted"
    assert len(state.debug_history) == 1  # accepted on attempt 1
    assert state.artifact_store.get("units_by_region") is not None
    assert state.working_memory.findings["step_1"].produced_artifact_name == (
        "units_by_region"
    )
    codegen_prompt_text = llm.prompts["codegen"][0]
    assert "OUTPUT CONTRACT" in codegen_prompt_text
    assert "units_by_region" in codegen_prompt_text
    assert "avg_units" in codegen_prompt_text


def test_missing_variable_is_still_rejected_and_repair_prompt_has_contract(
    df, tmp_path
) -> None:
    llm = FakeLLM(NEVER_ASSIGNS_VARIABLE)
    state = run("goal", df, log_dir=str(tmp_path), llm=llm, max_retries_per_step=2)

    assert state.aborted
    assert state.artifact_store.get("units_by_region") is None
    assert len(state.debug_history) == 2
    for attempt in state.debug_history:
        assert attempt.action == "revise_code"
        assert "no DataFrame artifact was produced" in attempt.sandbox_result["error"]
    # the repair prompt (second codegen call) carries the contract too
    assert len(llm.prompts["codegen"]) == 2
    assert "units_by_region" in llm.prompts["codegen"][1]
    assert "avg_units" in llm.prompts["codegen"][1]


def test_wrong_columns_are_still_rejected_with_missing_and_extra(df, tmp_path) -> None:
    code = (
        "units_by_region = df.groupby('region', as_index=False)['units_sold'].mean()\n"
        "result = units_by_region"  # columns: region, units_sold (avg_units missing)
    )
    state = run(
        "goal", df, log_dir=str(tmp_path), llm=FakeLLM(code), max_retries_per_step=1
    )
    assert state.aborted
    error = state.debug_history[0].sandbox_result["error"]
    assert "missing: ['avg_units']" in error
    assert "extra: ['units_sold']" in error


class TestArtifactContractText:
    def test_no_produces_artifact_gives_empty_string(self) -> None:
        task = TaskNode(task_id="step_1", question="q", output_schema=["a"])
        assert _artifact_contract_text(task) == ""

    def test_names_variable_and_columns(self) -> None:
        task = TaskNode(
            task_id="step_1",
            question="q",
            produces_artifact="units_by_region",
            output_schema=["region", "avg_units"],
        )
        text = _artifact_contract_text(task)
        assert text.startswith("OUTPUT CONTRACT:")
        assert "`units_by_region`" in text
        assert "with columns: region, avg_units (any order)." in text

    @pytest.mark.parametrize("schema", [[], ["same_as_input"], ["<scalar>"]])
    def test_no_columns_line_for_empty_or_placeholder_schema(self, schema) -> None:
        task = TaskNode(
            task_id="step_1",
            question="q",
            produces_artifact="out_df",
            output_schema=schema,
        )
        text = _artifact_contract_text(task)
        assert "`out_df`" in text
        assert "columns" not in text


def test_prompts_unchanged_without_contract() -> None:
    from agent.prompts import codegen_prompt, repair_prompt

    base = codegen_prompt("g", "q", "schema")
    assert "OUTPUT CONTRACT" not in base
    with_contract = codegen_prompt("g", "q", "schema", artifact_contract="OUTPUT CONTRACT:\nX")
    assert "OUTPUT CONTRACT:\nX" in with_contract
    assert with_contract.replace("\nOUTPUT CONTRACT:\nX\n", "") == base
    rbase = repair_prompt("g", "q", "schema", "", "", "code", "err")
    rwith = repair_prompt("g", "q", "schema", "", "", "code", "err", artifact_contract="OUTPUT CONTRACT:\nX")
    assert rwith.replace("\nOUTPUT CONTRACT:\nX\n", "") == rbase
