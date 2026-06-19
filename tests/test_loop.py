"""
Integration tests for the five-stage pipeline loop.

All LLM calls are mocked — no API keys required. Tests verify:
- Loop control: accept, revise_code, max_retries_exceeded, revise_plan
- State mutations: history entries, plan updates, run_id stability
- Logging: write_log_entry called after every iteration
- Prompt functions: schema summary content, critique JSON parsing
- Edge cases: malformed critique JSON, empty plan from LLM
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from agent.loop import (
    AgentRun,
    _accepted_findings,
    _build_schema_summary,
    _final_results,
    _parse_critique_json,
    _parse_json_list,
    _strip_markdown_fences,
    run,
)
from agent.prompts import codegen_prompt, critique_prompt, plan_prompt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_df() -> pd.DataFrame:
    return pd.DataFrame({
        "region": ["North", "South", "East"],
        "revenue": [100.0, 200.0, 150.0],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    })


CALL_LLM = "agent.loop.call_llm"
RUN_SANDBOXED = "agent.loop.run_sandboxed"
WRITE_LOG = "agent.loop.write_log_entry"


def _accept_critique() -> str:
    return json.dumps({"trust": "high", "reason": "Result looks correct.", "action": "accept"})


def _revise_critique() -> str:
    return json.dumps({"trust": "low", "reason": "Nulls not handled.", "action": "revise_code"})


def _ok_sandbox() -> dict:
    return {"success": True, "result": "100.0", "stdout": "", "error": None, "error_type": None}


def _fail_sandbox() -> dict:
    return {"success": False, "result": None, "stdout": "", "error": "NameError: x", "error_type": "NameError"}


# ---------------------------------------------------------------------------
# Helper: run with controlled LLM responses
# ---------------------------------------------------------------------------

def _run_with_responses(
    small_df,
    plan_response: str,
    llm_side_effects: list,
    sandbox_side_effects: list,
    max_retries_per_step: int = 3,
) -> AgentRun:
    """
    Run the loop with fully mocked LLM and sandbox calls.

    llm_side_effects: list of return values for call_llm (after the plan call).
    max_retries_per_step: override so test sequences don't depend on config value.
    """
    all_llm = [plan_response] + llm_side_effects
    with patch(CALL_LLM, side_effect=all_llm), \
         patch(RUN_SANDBOXED, side_effect=sandbox_side_effects), \
         patch(WRITE_LOG):
        return run("test goal", small_df, log_dir="/tmp/test_runs",
                   max_retries_per_step=max_retries_per_step)


# ---------------------------------------------------------------------------
# Loop control — accept path
# ---------------------------------------------------------------------------

class TestAcceptPath:
    def test_single_question_accepted(self, small_df: pd.DataFrame) -> None:
        """One question, accepted on first attempt → history has 1 entry."""
        state = _run_with_responses(
            small_df,
            plan_response='["What is total revenue?"]',
            llm_side_effects=[
                "result = df['revenue'].sum()",  # codegen
                _accept_critique(),              # critique
                "Summary text.",                 # summary
            ],
            sandbox_side_effects=[_ok_sandbox()],
        )
        assert len([h for h in state.history if h["sub_question"] == "What is total revenue?"]) == 1
        assert state.history[0]["action"] == "accept"
        assert state.summary == "Summary text."

    def test_two_questions_both_accepted(self, small_df: pd.DataFrame) -> None:
        """Two questions, each accepted first try → 2 history entries."""
        state = _run_with_responses(
            small_df,
            plan_response='["Q1?", "Q2?"]',
            llm_side_effects=[
                "result = 1",   # codegen Q1
                _accept_critique(),  # critique Q1
                "result = 2",   # codegen Q2
                _accept_critique(),  # critique Q2
                "Summary.",     # summary
            ],
            sandbox_side_effects=[_ok_sandbox(), _ok_sandbox()],
        )
        assert len(state.history) == 2
        assert all(h["action"] == "accept" for h in state.history)


# ---------------------------------------------------------------------------
# Loop control — revise_code path
# ---------------------------------------------------------------------------

class TestReviseCode:
    def test_retry_on_first_failure_then_accept(self, small_df: pd.DataFrame) -> None:
        """Critique says revise_code once, then accept on retry."""
        state = _run_with_responses(
            small_df,
            plan_response='["What is revenue?"]',
            llm_side_effects=[
                "result = bad_code()",   # codegen attempt 0
                _revise_critique(),      # critique → revise
                "result = df['revenue'].sum()",  # codegen attempt 1
                _accept_critique(),      # critique → accept
                "Summary.",
            ],
            sandbox_side_effects=[_ok_sandbox(), _ok_sandbox()],
        )
        q_entries = [h for h in state.history if h["sub_question"] == "What is revenue?"]
        assert len(q_entries) == 2
        assert q_entries[0]["action"] == "revise_code"
        assert q_entries[1]["action"] == "accept"
        assert len(state.debug_history) == 2
        assert len(state.final_results) == 1
        finding = state.final_results[0]
        assert finding.answer_confidence == "high"
        assert finding.execution_difficulty is not None
        assert finding.execution_difficulty.attempts == 2

    def test_max_retries_exceeded_aborts_pipeline(self, small_df: pd.DataFrame) -> None:
        """After max_retries_per_step revise_code critiques → pipeline aborts, not force-accepts."""
        max_r = 3
        # Call order per iteration: codegen then critique (interleaved, not batched)
        interleaved = []
        for _ in range(max_r):
            interleaved.append("result = bad()")
            interleaved.append(_revise_critique())
        state = _run_with_responses(
            small_df,
            plan_response='["Hard question?"]',
            llm_side_effects=interleaved,
            sandbox_side_effects=[_ok_sandbox() for _ in range(max_r)],
            max_retries_per_step=max_r,
        )
        assert state.aborted is True
        assert "Hard question?" in state.abort_reason
        assert state.summary == ""  # summary never generated
        assert state.final_results == []

    def test_abort_stops_downstream_steps(self, small_df: pd.DataFrame) -> None:
        """When step 1 aborts, step 2 must never execute — no downstream codegen calls."""
        max_r = 3
        interleaved = []
        for _ in range(max_r):
            interleaved.append("result = bad()")
            interleaved.append(_revise_critique())
        with patch(CALL_LLM, side_effect=[
            '["Q1 fails?", "Q2 never runs?"]',  # plan
            *interleaved,
            # Q2 and summary must NOT be reached
        ]) as mock_llm, patch(RUN_SANDBOXED, return_value=_ok_sandbox()), \
           patch(WRITE_LOG):
            state = run("goal", small_df, log_dir="/tmp/test_runs", max_retries_per_step=max_r)

        assert state.aborted is True
        assert [task.status for task in state.tasks] == ["failed", "skipped"]
        # Verify Q2 codegen was never invoked
        call_args_list = [str(c) for c in mock_llm.call_args_list]
        assert not any("Q2" in s for s in call_args_list)

    def test_sandbox_failure_forces_revise(self, small_df: pd.DataFrame) -> None:
        """When sandbox fails, critique should return revise_code."""
        state = _run_with_responses(
            small_df,
            plan_response='["Q?"]',
            llm_side_effects=[
                "result = undefined_var",  # codegen
                _revise_critique(),        # critique sees failure
                "result = df.shape[0]",    # codegen retry
                _accept_critique(),
                "Summary.",
            ],
            sandbox_side_effects=[_fail_sandbox(), _ok_sandbox()],
        )
        assert any(h["action"] == "accept" for h in state.history)


# ---------------------------------------------------------------------------
# Loop control — revise_plan path
# ---------------------------------------------------------------------------

class TestRevisePlan:
    def test_revise_plan_replaces_tail(self, small_df: pd.DataFrame) -> None:
        """When critique returns revise_plan, remaining questions are replaced."""
        revise_plan_critique = json.dumps({
            "trust": "low",
            "reason": "Wrong question framing.",
            "action": "revise_plan",
        })
        with patch(CALL_LLM, side_effect=[
            '["Q1?", "Q2_bad?"]',     # initial plan
            "result = 1",             # codegen Q1
            revise_plan_critique,     # critique Q1 → revise_plan
            '["Q2_new?"]',            # replan
            "result = 2",             # codegen Q2_new
            _accept_critique(),       # critique Q2_new
            "Summary.",
        ]), patch(RUN_SANDBOXED, side_effect=[_ok_sandbox(), _ok_sandbox()]), \
           patch(WRITE_LOG):
            state = run("goal", small_df, log_dir="/tmp/test_runs")

        assert "Q2_bad?" not in [h["sub_question"] for h in state.history]
        assert state.plan_revisions == 1

    def test_plan_revision_cap(self, small_df: pd.DataFrame) -> None:
        """After MAX_PLAN_REVISIONS, revise_plan requests are ignored."""
        from agent.loop import MAX_PLAN_REVISIONS
        revise_critique = json.dumps({"trust": "low", "reason": ".", "action": "revise_plan"})

        # Build enough responses to hit the cap
        llm_calls = ['["Q1?"]']
        sandbox_calls = []
        for _ in range(MAX_PLAN_REVISIONS + 2):
            llm_calls += ["result = 1", revise_critique, '["Qnew?"]']
            sandbox_calls.append(_ok_sandbox())
        llm_calls.append("Summary.")

        with patch(CALL_LLM, side_effect=llm_calls), \
             patch(RUN_SANDBOXED, side_effect=sandbox_calls), \
             patch(WRITE_LOG):
            state = run("goal", small_df, log_dir="/tmp/test_runs")

        assert state.plan_revisions <= MAX_PLAN_REVISIONS


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class TestLogging:
    def test_log_written_per_iteration(self, small_df: pd.DataFrame) -> None:
        """write_log_entry must be called for every iteration + plan + summary."""
        with patch(CALL_LLM, side_effect=[
            '["Q?"]',
            "result = 1",
            _accept_critique(),
            "Summary.",
        ]), patch(RUN_SANDBOXED, return_value=_ok_sandbox()), \
           patch(WRITE_LOG) as mock_log:
            run("goal", small_df, log_dir="/tmp/test_runs")

        # At minimum: plan entry + 1 codegen_execute_critique entry + summary entry
        assert mock_log.call_count >= 3

    def test_run_id_stable_across_entries(self, small_df: pd.DataFrame) -> None:
        """All log entries must share the same run_id."""
        logged_entries = []
        with patch(CALL_LLM, side_effect=[
            '["Q?"]', "result = 1", _accept_critique(), "Summary.",
        ]), patch(RUN_SANDBOXED, return_value=_ok_sandbox()), \
           patch(WRITE_LOG, side_effect=lambda e, *a, **k: logged_entries.append(e)):
            state = run("goal", small_df, log_dir="/tmp/test_runs")

        run_ids = {e.run_id for e in logged_entries}
        assert len(run_ids) == 1
        assert state.run_id in run_ids



# ---------------------------------------------------------------------------
# Summary working memory — only accepted findings reach the summary LLM
# ---------------------------------------------------------------------------


class TestSummaryWorkingMemory:
    def _make_state(self, plan: list[str], history: list[dict]) -> AgentRun:
        state = AgentRun(goal="test goal", schema_summary="cols")
        state.plan = plan
        state.history = history
        return state

    def test_accepted_findings_strips_intermediate_failures(self) -> None:
        """Intermediate revise_code entries must not appear in accepted_findings."""
        q = "What is total revenue?"
        history = [
            {"sub_question": q, "iteration": 0, "action": "revise_code", "trust": "low",
             "sandbox_result": {"result": None}},
            {"sub_question": q, "iteration": 1, "action": "revise_code", "trust": "low",
             "sandbox_result": {"result": None}},
            {"sub_question": q, "iteration": 2, "action": "accept", "trust": "high",
             "sandbox_result": {"result": "{'East': 100}"}},
        ]
        state = self._make_state([q], history)
        findings = _accepted_findings(state)

        assert len(findings) == 1
        assert findings[0]["action"] == "accept"
        assert findings[0]["iteration"] == 2

    def test_accepted_findings_preserves_plan_order(self) -> None:
        """Findings are returned in plan order, not history insertion order."""
        q1, q2 = "Q1?", "Q2?"
        history = [
            {"sub_question": q1, "iteration": 0, "action": "accept", "trust": "high",
             "sandbox_result": {"result": "A1"}},
            {"sub_question": q2, "iteration": 0, "action": "accept", "trust": "high",
             "sandbox_result": {"result": "A2"}},
        ]
        state = self._make_state([q1, q2], history)
        findings = _accepted_findings(state)

        assert [f["sub_question"] for f in findings] == [q1, q2]

    def test_summary_llm_sees_only_accepted_entry(self, small_df: pd.DataFrame) -> None:
        """End-to-end: a step with 2 iterations should produce exactly 1 finding."""
        with patch(CALL_LLM, side_effect=[
            '[\"Q with retries?\"]',   # plan
            "result = bad()",          # codegen iter 0
            _revise_critique(),        # critique iter 0
            "result = df.shape[0]",    # codegen iter 1
            _accept_critique(),        # critique iter 1
            "Summary.",                # summary
        ]) as mock_llm, patch(RUN_SANDBOXED, side_effect=[_fail_sandbox(), _ok_sandbox()]), \
           patch(WRITE_LOG):
            run("goal", small_df, log_dir="/tmp/test_runs")

        # The summary call is the last one. Extract its prompt.
        summary_call_prompt = mock_llm.call_args_list[-1][0][0]
        # Only "1." should appear in the findings block — not "1." and "2."
        assert "revise_code" not in summary_call_prompt
        assert "bad()" not in summary_call_prompt
        assert "accepted after" not in summary_call_prompt

    def test_final_results_excludes_retry_metadata(self) -> None:
        """Summary payload must not expose iteration counts, code, or critiques."""
        q = "What is total revenue?"
        history = [
            {
                "sub_question": q,
                "iteration": 0,
                "action": "revise_code",
                "trust": "low",
                "code": "result = bad()",
                "critique": "ValueError",
                "sandbox_result": {"result": None},
            },
            {
                "sub_question": q,
                "iteration": 1,
                "action": "accept",
                "trust": "high",
                "code": "result = 100",
                "critique": "accepted",
                "sandbox_result": {"result": "100"},
            },
        ]
        state = self._make_state([q], history)

        assert _final_results(state) == [
            {"question": q, "answer": "100", "trust": "high"}
        ]

    def test_summary_payload_comes_from_working_memory_only(self) -> None:
        """Debug history can contain failures, but summary evidence cannot."""
        q = "What is total revenue?"
        history = [
            {
                "sub_question": q,
                "iteration": 0,
                "action": "revise_code",
                "trust": "low",
                "code": "result = broken",
                "critique": "NameError",
                "sandbox_result": {"result": None, "error_type": "NameError"},
            },
            {
                "sub_question": q,
                "iteration": 1,
                "action": "accept",
                "trust": "high",
                "code": "result = 450",
                "critique": json.dumps({
                    "trust": "high",
                    "reason": "accepted",
                    "action": "accept",
                }),
                "sandbox_result": {"result": "450"},
            },
        ]
        state = self._make_state([q], history)

        prompt_payload = _final_results(state)
        assert prompt_payload == [{"question": q, "answer": "450", "trust": "high"}]
        assert all("NameError" not in str(row) for row in prompt_payload)
        assert all("broken" not in str(row) for row in prompt_payload)


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_build_schema_summary_contains_columns(self, small_df: pd.DataFrame) -> None:
        summary = _build_schema_summary(small_df)
        assert "region" in summary
        assert "revenue" in summary
        assert "date" in summary
        assert "Rows:" in summary

    def test_build_schema_summary_shows_null_pct(self) -> None:
        df = pd.DataFrame({"x": [1.0, None, 3.0]})
        summary = _build_schema_summary(df)
        assert "33.3%" in summary or "nulls=" in summary

    def test_parse_json_list_clean(self) -> None:
        raw = '["What is X?", "How does Y compare?"]'
        result = _parse_json_list(raw)
        assert result == ["What is X?", "How does Y compare?"]

    def test_parse_json_list_with_prose(self) -> None:
        raw = 'Here are the questions:\n["Q1?", "Q2?"]'
        result = _parse_json_list(raw)
        assert "Q1?" in result

    def test_parse_critique_json_valid(self) -> None:
        raw = '{"trust": "high", "reason": "Looks good.", "action": "accept"}'
        parsed = _parse_critique_json(raw)
        assert parsed["trust"] == "high"
        assert parsed["action"] == "accept"

    def test_parse_critique_json_with_fences(self) -> None:
        raw = '```json\n{"trust": "low", "reason": "Bad.", "action": "revise_code"}\n```'
        parsed = _parse_critique_json(raw)
        assert parsed["action"] == "revise_code"

    def test_parse_critique_json_malformed_defaults_to_revise(self) -> None:
        """Malformed critique JSON must never silently accept."""
        parsed = _parse_critique_json("this is not json at all")
        assert parsed["action"] == "revise_code"
        assert parsed["trust"] == "low"

    def test_parse_critique_json_invalid_action_clamped(self) -> None:
        raw = '{"trust": "high", "reason": "ok", "action": "something_weird"}'
        parsed = _parse_critique_json(raw)
        assert parsed["action"] == "revise_code"

    def test_parse_critique_json_invalid_trust_clamped(self) -> None:
        raw = '{"trust": "medium", "reason": "ok", "action": "accept"}'
        parsed = _parse_critique_json(raw)
        assert parsed["trust"] == "low"


# ---------------------------------------------------------------------------
# Prompt template smoke tests
# ---------------------------------------------------------------------------

class TestPromptTemplates:
    def test_plan_prompt_contains_goal(self) -> None:
        p = plan_prompt("my special goal", "col1, col2")
        assert "my special goal" in p
        assert "JSON array" in p

    def test_codegen_prompt_contains_result_convention(self) -> None:
        from agent.environment import EnvironmentManifest
        p = codegen_prompt("goal", "How much revenue?", "col1 (float)", environment_manifest=EnvironmentManifest().as_prompt_text())
        assert "result" in p
        assert "import" in p.lower()  # warns against import

    def test_repair_prompt_includes_retry_context(self) -> None:
        from agent.prompts import repair_prompt
        p = repair_prompt(
            "goal",
            "Q?",
            "schema",
            completed_findings="",
            environment_manifest="",
            task_constraints="",
            previous_code="result = bad()",
            error_reason="nulls ignored",
        )
        assert "nulls ignored" in p
        assert "result = bad()" in p

    def test_codegen_prompt_includes_goal_scope_and_findings(self) -> None:
        p = codegen_prompt(
            "Explain revenue over the last quarter.",
            "Which region is top?",
            "schema",
            completed_findings="• Revenue by region: {'West': 10, 'South': 9}",
        )
        assert "last quarter" in p.lower()
        assert "Previously accepted findings".lower() in p.lower()
        assert "West" in p

    def test_critique_prompt_contains_checklist_items(self) -> None:
        sandbox_result = _ok_sandbox()
        p = critique_prompt("goal", "Q?", "result = df.sum()", sandbox_result, "")
        # Data quality checks still present
        assert "null" in p.lower()
        assert "outlier" in p.lower()
        assert "duplicate" in p.lower()
        # Code verification is now STEP 0
        assert "CODE VERIFICATION" in p or "STEP 0" in p
        # Contradiction check must warn about scope differences
        assert "scope" in p.lower() or "filter" in p.lower()
        # Must not instruct critic to blindly flag number differences
        assert "verify" in p.lower()

    def test_critique_prompt_requires_goal_scope_consistency(self) -> None:
        p = critique_prompt(
            "Explain revenue over the last quarter.",
            "Which region has the highest total revenue?",
            "result = df.groupby('region')['revenue'].sum().idxmax()",
            _ok_sandbox(),
            "[1] [accept/high] Revenue by region → {'West': 10, 'South': 9}",
        )
        assert "last quarter" in p.lower()
        assert "revise_code" in p
        assert "previous accepted finding" in p.lower() or "previously accepted" in p.lower()


# ---------------------------------------------------------------------------
# Markdown fence stripping (fixes recurring SyntaxError in demo run)
# ---------------------------------------------------------------------------


class TestMarkdownFenceStripping:
    def test_strips_python_fence(self) -> None:
        raw = "```python\nresult = df['revenue'].sum()\n```"
        assert _strip_markdown_fences(raw) == "result = df['revenue'].sum()"

    def test_strips_plain_fence(self) -> None:
        raw = "```\nresult = 42\n```"
        assert _strip_markdown_fences(raw) == "result = 42"

    def test_strips_py_fence(self) -> None:
        raw = "```py\nresult = df.shape\n```"
        assert _strip_markdown_fences(raw) == "result = df.shape"

    def test_no_fences_unchanged(self) -> None:
        code = "result = df.groupby('region')['revenue'].sum()"
        assert _strip_markdown_fences(code) == code

    def test_multiline_code_preserved(self) -> None:
        raw = "```python\ntotal = df['revenue'].sum()\nresult = round(total, 2)\n```"
        stripped = _strip_markdown_fences(raw)
        assert "total = df['revenue'].sum()" in stripped
        assert "result = round(total, 2)" in stripped
        assert "```" not in stripped

    def test_codegen_strips_fences_before_sandbox(self, small_df: pd.DataFrame) -> None:
        """End-to-end: even if LLM wraps code in fences, sandbox receives clean code."""
        fenced_code = "```python\nresult = df['revenue'].sum()\n```"
        with patch(CALL_LLM, side_effect=[
            '["Q?"]',
            fenced_code,        # codegen returns fenced output
            _accept_critique(),
            "Summary.",
        ]), patch(RUN_SANDBOXED, return_value=_ok_sandbox()) as mock_sandbox, \
           patch(WRITE_LOG):
            run("goal", small_df, log_dir="/tmp/test_runs")

        # Sandbox must receive the clean code, not the fenced version
        actual_code = mock_sandbox.call_args[0][0]
        assert "```" not in actual_code
        assert "result = df['revenue'].sum()" in actual_code
