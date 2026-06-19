"""
Prompt-template functions for every stage of the pipeline.

Rules for this module:
- Each function returns a complete, ready-to-send prompt string.
- No prompt strings live anywhere else in the codebase.
- Keep templates explicit and specific — vague prompts produce vague outputs.
- Templates are versioned implicitly by git history; add a comment if you
  make a substantive change so the blog post can reference it.

Phase 3 hard dependency from sandbox (Phase 2):
  CODE-GEN prompts MUST instruct the model to assign to `result`.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# PLAN
# ---------------------------------------------------------------------------


def plan_prompt(goal: str, schema_summary: str) -> str:
    """
    Build the PLAN prompt.

    Instructs the model to decompose the analytical goal into 3-6 ordered
    sub-questions that can each be answered with a short pandas snippet.

    Args:
        goal:           High-level analytical question from the user.
        schema_summary: Column names, dtypes, and sample values (NOT full data).

    Returns:
        Complete prompt string ready for :func:`agent.llm_client.call_llm`.
    """
    return f"""You are a senior data analyst planning an investigation.

DATASET SCHEMA:
{schema_summary}

GOAL: {goal}

Decompose this goal into 3-6 specific, ordered sub-questions. Each sub-question:
- Must be answerable with a single pandas code snippet operating on the dataset above.
- Should build logically on the previous one (funnel from broad to specific).
- Should be concrete enough that the answer is a number, table, or ranked list.

Return ONLY a JSON array of strings. No explanation, no markdown fences.
Example format: ["What is X?", "Which region has Y?", "How does Z compare?"]

JSON array:"""


def replan_prompt(
    goal: str,
    schema_summary: str,
    completed_findings: str,
    critique_reason: str,
) -> str:
    """
    Build the RE-PLAN prompt when critique returns ``"revise_plan"``.

    Carries forward what's already been established so the new plan
    doesn't re-investigate things already answered confidently.

    Args:
        goal:               Original analytical goal.
        schema_summary:     Dataset schema.
        completed_findings: Summary of high-trust steps completed so far.
        critique_reason:    The critique's reason for requesting a plan revision.

    Returns:
        Complete prompt string.
    """
    return f"""You are a senior data analyst revising an investigation plan mid-stream.

DATASET SCHEMA:
{schema_summary}

ORIGINAL GOAL: {goal}

FINDINGS ALREADY ESTABLISHED (high-trust, do not re-investigate):
{completed_findings}

REASON THE PREVIOUS PLAN NEEDS REVISION:
{critique_reason}

Generate a new list of 2-5 sub-questions to complete the investigation.
These must address the gap identified in the critique reason.
Do NOT repeat questions whose findings are already established above.

Return ONLY a JSON array of strings. No explanation, no markdown fences.

JSON array:"""


# ---------------------------------------------------------------------------
# CODE-GEN
# ---------------------------------------------------------------------------


def codegen_prompt(
    goal: str,
    sub_question: str,
    schema_summary: str,
    completed_findings: str = "",
    environment_manifest: str = "",
    task_constraints: str = "",
) -> str:
    """
    Build the CODE-GEN prompt.
    """
    return f"""You are a senior data analyst writing Python/pandas code.

DATASET SCHEMA:
{schema_summary}

{environment_manifest}

ORIGINAL ANALYTICAL GOAL:
{goal}

PREVIOUSLY ACCEPTED FINDINGS:
{completed_findings if completed_findings else "(none yet)"}

CUMULATIVE TASK CONSTRAINTS (Do not violate these):
{task_constraints}

SUB-QUESTION TO ANSWER: {sub_question}

Write a Python snippet that answers the sub-question using the DataFrame `df`.

STRICT RULES — violating any of these will cause execution to fail:
1. Handle nulls explicitly (use .dropna(), .fillna(), or .notna()) — do not assume clean data.
2. If computing an average or sum, check that the denominator / count is non-zero.
3. Print intermediate DataFrames with print(df_name.head()) for the observer to inspect.
4. Preserve the ORIGINAL ANALYTICAL GOAL scope. If the goal says "last quarter",
   every dependent revenue/trend/top-region calculation must use that same scope
   unless the sub-question explicitly says otherwise.
5. If this sub-question depends on a previously accepted finding (for example
   "top-performing region"), derive it consistently from that accepted scope.

Return ONLY the Python code. No explanation, no markdown fences, no comments beyond inline ones.

Python code:"""


def repair_prompt(
    goal: str,
    sub_question: str,
    schema_summary: str,
    completed_findings: str,
    environment_manifest: str,
    task_constraints: str,
    previous_code: str,
    error_reason: str,
) -> str:
    """
    Build the REPAIR prompt for localized fixes on a failed attempt.
    """
    return f"""You are a senior data analyst repairing a failing Python/pandas script.

DATASET SCHEMA:
{schema_summary}

{environment_manifest}

ORIGINAL ANALYTICAL GOAL:
{goal}

PREVIOUSLY ACCEPTED FINDINGS:
{completed_findings if completed_findings else "(none yet)"}

CUMULATIVE TASK CONSTRAINTS (Do not violate these):
{task_constraints}

SUB-QUESTION TO ANSWER: {sub_question}

PREVIOUS CODE ATTEMPT:
```python
{previous_code}
```

ERROR / CRITIQUE REASON:
{error_reason}

Write a REPAIRED version of the Python snippet.
DO NOT rewrite the entire logic from scratch unless absolutely necessary.
Focus ONLY on fixing the specific issue identified in the ERROR / CRITIQUE REASON,
while keeping the working parts of the previous code intact.

STRICT RULES:
1. Handle nulls explicitly — do not assume clean data.
2. Preserve the ORIGINAL ANALYTICAL GOAL scope.
3. Return ONLY the complete, repaired Python code. No explanation, no markdown fences.

Python code:"""


# ---------------------------------------------------------------------------
# CRITIQUE
# ---------------------------------------------------------------------------


def critique_prompt(
    goal: str,
    sub_question: str,
    code: str,
    sandbox_result: dict,
    history_summary: str,
) -> str:
    """
    Build the CRITIQUE prompt.

    Instructs the model to be a skeptical auditor looking for specific
    failure modes, not just "does this look right?"

    The critic's primary job is to verify the CODE ITSELF — not to
    compare result strings against memory. History is provided only to
    detect genuine contradictions (same scope, different answer), NOT to
    penalise correct code that uses a different filter or time window.

    Args:
        goal:           Original analytical goal, including global scope.
        sub_question:    The question the code was trying to answer.
        code:            The Python code that was executed.
        sandbox_result:  The full sandbox output dict
                         (success, result, stdout, error, error_type).
        history_summary: One-line summaries of previously accepted findings,
                         so the critique can detect genuine contradictions.

    Returns:
        Complete prompt string.
    """
    exec_status = "SUCCESS" if sandbox_result.get("success") else "FAILED"
    exec_result = sandbox_result.get("result") or "(no result variable set)"
    exec_stdout = sandbox_result.get("stdout", "") or "(no stdout)"
    exec_error = sandbox_result.get("error") or ""

    error_block = f"\nEXECUTION ERROR:\n{exec_error}" if exec_error else ""

    return f"""You are an independent data quality auditor reviewing an AI agent's work.
You did NOT write this code. Your PRIMARY job is to verify the CODE LOGIC — not to
compare result strings against memory.

ORIGINAL ANALYTICAL GOAL:
{goal}

SUB-QUESTION: {sub_question}

CODE EXECUTED:
```python
{code}
```

EXECUTION STATUS: {exec_status}
RESULT (repr of `result` variable): {exec_result}
STDOUT OUTPUT: {exec_stdout}{error_block}

PREVIOUSLY ACCEPTED FINDINGS (for contradiction detection only — see Step 4):
{history_summary if history_summary else "(none yet — this is the first step)"}

══════════════════════════════════════════════════════════════
AUDIT PROCESS — follow this order exactly:
══════════════════════════════════════════════════════════════

STEP 0 — CODE VERIFICATION (do this FIRST, before looking at the result):
  Read the code line by line. Mentally trace what it computes.
  Ask: "Given the schema and this code, what value SHOULD `result` hold?"
  If the code is logically correct and would produce a valid answer to the
  sub-question, that is strong evidence the result is trustworthy —
  even if the number differs from a prior finding.

STEP 1 — EXECUTION FAILURE:
  If status is FAILED or result is "(no result variable set)", action MUST be
  "revise_code". Do not evaluate anything else.

STEP 2 — CODE LOGIC ERRORS:
  Does the code contain a verifiable logic mistake? Examples:
  - Aggregating before filtering (wrong scope)
  - Using the wrong column (e.g. units_sold instead of revenue)
  - Off-by-one in date comparisons
  - Dividing by the wrong denominator
  If yes → revise_code.

STEP 2B — GOAL SCOPE AND DEPENDENCY CONSISTENCY:
  Preserve the ORIGINAL ANALYTICAL GOAL scope across dependent steps.
  If the goal is scoped to "last quarter" and this code recomputes revenue,
  top region, trend, category, or averages over the full dataset, action MUST
  be "revise_code" unless the sub-question explicitly asks for all-time data.

  If a previous accepted finding directly establishes a parent value, this
  result must be consistent with it. Example: if accepted total revenue by
  region is {{"West": 46384.76, "South": 45317.93}}, then a later "top region"
  answer of "South" is a genuine contradiction and MUST revise_code.

STEP 3 — DATA QUALITY CHECKS (verify using the code, not guesswork):
  a. NULL HANDLING: Does the code silently drop nulls without filling them?
     Check whether .dropna() is called on a column with significant nulls
     before summing or averaging.
  b. DUPLICATE ROWS: Does the code aggregate without first calling
     .drop_duplicates() on transaction_id? This inflates sums/counts.
  c. SAMPLE SIZE: Does a filter in the code reduce rows to a suspiciously
     small subset? Check date/category filter conditions explicitly.
  d. OUTLIER DOMINANCE: Would a single extreme value dominate a sum/mean
     given the filter the code applies?

STEP 4 — CONTRADICTION CHECK (read the prior code snippets before deciding):
  The "Previously Accepted Findings" above include the code each prior step
  actually ran. Use that to verify scope, not to guess.

  If this result appears to differ from a previously accepted finding:
  a. READ the prior step's code snippet in the history. What date filter or
     groupby scope does it use? Read THIS code too. Are the scopes the same?
     - Same scope, different answer → genuine contradiction → revise_code.
     - Different scope (e.g. prior filtered to last quarter, this uses full
       dataset) → both may be correct. Note the scope difference in your
       reason and accept if the code logic is otherwise sound.
  b. ANALYTICAL CONTINUITY: if this step re-derives a value that a prior
     step already established (e.g., "top region" was already found in step
     2 and this step independently computes it again), check that both codes
     arrive at the same answer. If they disagree, one of them has a bug.
     Read both code snippets to determine which derivation is correct.
  c. Do NOT rely on memory or guess what prior code did — the code preview
     is in the history; read it.

STEP 5 — LOGICAL CONSISTENCY:
  Is the result logically impossible regardless of scope?
  (Negative revenue, average > total, units_sold=0 with revenue>0, etc.)

══════════════════════════════════════════════════════════════

Return ONLY this JSON object, with no markdown fences:
{{"trust": "high" or "low", "reason": "one specific sentence — cite the code line if flagging an error", "action": "accept" or "revise_code" or "revise_plan"}}

Rules for action:
- "accept": code is logically correct and result is trustworthy
- "revise_code": code has a verifiable logic error or data quality flaw
- "revise_plan": the sub-question itself is wrong framing
- If execution FAILED, action MUST be "revise_code"
- Do NOT flag "revise_code" solely because a number differs from a prior
  finding — always verify the code's scope first.

JSON:"""


# ---------------------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------------------


def summary_prompt(goal: str, history_entries: list[dict]) -> str:
    """
    Build the FINAL SUMMARY prompt.

    Receives exactly one final result per sub-question. The summary stage is a
    reporter, not an auditor: it must not infer new trust levels, reinterpret
    retry counts, or add speculative caveats.

    Args:
        goal:            The original analytical goal.
        history_entries: One final entry per sub-question, in plan order.
                         Fields used: question, answer, trust.

    Returns:
        Complete prompt string.
    """
    import json

    findings_payload = []
    for entry in history_entries:
        findings_payload.append({
            "question": entry.get("question", ""),
            "answer": entry.get("answer", "(no result)"),
            "trust": entry.get("trust", "unknown"),
        })
    findings_json = json.dumps(findings_payload, indent=2)

    return f"""You are a senior data analyst writing the conclusion of an investigation.
You are a reporter, not a critic. Use only the final results provided below.

ORIGINAL GOAL: {goal}

FINAL RESULTS JSON:
{findings_json}

STRICT REPORTING RULES:
1. Use only the question, answer, and trust fields in FINAL RESULTS JSON.
2. Do not infer new trust levels.
3. Do not reinterpret retries or failed attempts; they are not provided.
4. Do not add caveats unless trust is exactly "low" or the answer explicitly
   contains the caveat.
5. Treat np.float64-style reprs as normal numeric answers; round for readability.
6. If two final results appear inconsistent, state the inconsistency plainly
   without inventing a resolution.

Write a concise analytical summary (150-300 words) that:
1. Directly answers the original goal using the findings above.
2. Quantifies claims where possible (use the actual numbers from results).
3. Explicitly flags findings whose trust field is exactly "low".
4. Notes any gaps — sub-questions with no result.
5. Ends with one actionable recommendation.

Do not hedge everything — be direct. If the data clearly shows X, say "The data shows X."

Summary:"""
