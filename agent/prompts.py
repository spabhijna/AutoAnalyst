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
# Shared prompt sections
# ---------------------------------------------------------------------------

# Generic example used when no plan examples are supplied: column names are
# placeholders, so no dataset-specific scope (e.g. a "last quarter") leaks in.
DEFAULT_PLAN_EXAMPLES = """[
  {"question": "What are the row count, mean and median of <numeric_column> for each <category_column>?", "capability": null, "produces_artifact": "metric_by_category", "consumes_artifacts": [], "output_schema": ["<category_column>", "row_count", "mean", "median"]},
  {"question": "Which <category_column> has the highest mean <numeric_column>?", "capability": null, "produces_artifact": null, "consumes_artifacts": ["metric_by_category"], "output_schema": ["<category_column>", "mean"]}
]"""


def data_dictionary_section(data_dictionary: str) -> str:
    """The DATA DICTIONARY block, or ``""`` when no dictionary was given."""
    if not data_dictionary:
        return ""
    return (
        "\nDATA DICTIONARY (authoritative meanings of columns and coded values — "
        "use these labels; do not guess others):\n"
        f"{data_dictionary}\n"
    )


def _coded_values_rule(data_dictionary: str) -> str:
    if data_dictionary:
        return (
            "Name coded values (integer codes, flags, abbreviations) only with the\n"
            "labels in the DATA DICTIONARY. Codes it does not list are unknown —\n"
            "say so instead of guessing, and honour any caveats it states (for\n"
            "example values that are not recorded for some rows)."
        )
    return (
        "No data dictionary was provided: do not assign meanings to integer codes\n"
        "or flags (e.g. do not decide what code 1 stands for). Refer to them by column\n"
        "and value (e.g. \"status = 1\")."
    )


def _scope_rules(time_scoped: bool) -> str:
    if time_scoped:
        return (
            "- The FIRST task should filter/prepare the main DataFrame for the goal scope\n"
            "  (e.g. apply the last-quarter window) and produce it as a named artifact.\n"
            "- Every subsequent task that needs that same scope MUST consume the artifact\n"
            "  rather than re-filtering df from scratch."
        )
    return (
        "- Only add a filtering/preparation task when the GOAL names a scope (a time\n"
        "  window, a subset of rows). Do NOT invent a time window or date filter the\n"
        "  goal does not ask for — analyse the full dataset by default.\n"
        "- When several tasks need the same prepared DataFrame, produce it once as a\n"
        "  named artifact and have later tasks consume it."
    )


# ---------------------------------------------------------------------------
# PLAN
# ---------------------------------------------------------------------------


def plan_prompt(
    goal: str,
    schema_summary: str,
    *,
    capabilities: str = "",
    examples: str = "",
    time_scoped: bool = False,
    data_dictionary: str = "",
) -> str:
    """
    Build the PLAN prompt.

    Instructs the model to decompose the analytical goal into 3-6 ordered
    tasks expressed as a dependency-aware workflow, not just a flat list of
    questions. Each task declares which DataFrame artifact it produces (if any)
    and which artifacts from prior tasks it consumes.

    Nothing here assumes a particular dataset: the capabilities and examples
    are built by :func:`agent.planning.plan_guidance` from the capabilities
    that apply to the actual columns, and the "filter to the goal's time
    window first" rule is only included when *time_scoped* is set.

    Args:
        goal:            High-level analytical question from the user.
        schema_summary:  Column names, dtypes, and sample values (NOT full data).
        capabilities:    Description of the registered capabilities that apply
                         to this dataset ("" when none apply).
        examples:        JSON example plan; defaults to a generic placeholder one.
        time_scoped:     The goal asks for a quarter window and a datetime
                         column exists.
        data_dictionary: Column / code descriptions supplied by the user.

    Returns:
        Complete prompt string ready for :func:`agent.llm_client.call_llm`.
    """
    capabilities_section = (
        f"\n{capabilities}\n"
        "Use one of these capability identifiers when a task matches it exactly;\n"
        "otherwise set \"capability\" to null and the task runs as generated code.\n"
        if capabilities
        else "\nNo registered capability applies to these columns: set \"capability\" "
        "to null for every task (tasks run as generated code).\n"
    )
    return f"""You are a senior data analyst designing a data investigation workflow.

DATASET SCHEMA:
{schema_summary}
{data_dictionary_section(data_dictionary)}{capabilities_section}
GOAL: {goal}

Decompose this goal into 3-6 ordered tasks. Design for MAXIMUM REUSE:
{_scope_rules(time_scoped)}
- Tasks should build logically (broad → specific), each consuming artifacts
  from prior steps instead of repeating work.
- Use only columns that exist in the DATASET SCHEMA.
- {_coded_values_rule(data_dictionary).replace(chr(10), chr(10) + "  ")}

Each task object must have these fields:
  "question":          A specific, concrete question answerable with one pandas snippet.
  "capability":        A capability identifier from the list above, or null.
  "produces_artifact": A snake_case DataFrame variable name this task will create and
                       register. Use null if the task returns only a scalar, string,
                       or ranked list — not a DataFrame.
  "consumes_artifacts": A JSON array of artifact names from PRIOR tasks this task will
                        use. Empty array [] if none.
  "output_schema":     A JSON array of expected output columns, or ["<scalar>"] for
                       scalar answers.

Return ONLY a JSON array of task objects. No explanation, no markdown fences.
Example of the format (adapt it to the goal and the actual columns):
{examples or DEFAULT_PLAN_EXAMPLES}

JSON array:"""


def replan_prompt(
    goal: str,
    schema_summary: str,
    completed_findings: str,
    critique_reason: str,
    available_artifacts: str = "",
    *,
    capabilities: str = "",
    examples: str = "",
    data_dictionary: str = "",
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
        available_artifacts: Description of the DataFrame artifacts already
                            registered (``ArtifactStore.describe()``); new tasks
                            may consume them by name.
        capabilities:       Applicable capabilities (see :func:`plan_prompt`).
        examples:           JSON example plan; defaults to a generic one.
        data_dictionary:    Column / code descriptions supplied by the user.

    Returns:
        Complete prompt string.
    """
    artifacts_section = (
        f"\nARTIFACTS ALREADY AVAILABLE (consume these by name, do not recreate them):\n"
        f"{available_artifacts}\n"
        if available_artifacts
        and available_artifacts != "(no pre-computed artifacts available yet)"
        else ""
    )
    capabilities_section = f"\n{capabilities}\n" if capabilities else ""
    return f"""You are a senior data analyst revising an investigation plan mid-stream.

DATASET SCHEMA:
{schema_summary}
{data_dictionary_section(data_dictionary)}{capabilities_section}
ORIGINAL GOAL: {goal}

FINDINGS ALREADY ESTABLISHED (high-trust, do not re-investigate):
{completed_findings}
{artifacts_section}
REASON THE PREVIOUS PLAN NEEDS REVISION:
{critique_reason}

Generate a new list of 2-5 tasks to complete the investigation.
These must address the gap identified in the critique reason.
Do NOT repeat questions whose findings are already established above.
Do NOT add a time window or date filter the original goal does not ask for.

Each task object must have these fields:
  "question":          A specific, concrete question answerable with one pandas snippet.
  "capability":        A capability identifier from the list above, or null.
  "produces_artifact": A snake_case DataFrame variable name this task will create and
                       register, or null if it returns only a scalar, string, or list.
  "consumes_artifacts": A JSON array of artifact names this task will use (from the
                        available artifacts above or earlier tasks in this list).
                        Empty array [] if none.
  "output_schema":     A JSON array of expected output columns, or ["<scalar>"] for
                       scalar answers.

Return ONLY a JSON array of task objects. No explanation, no markdown fences.
Example of the format (adapt it to the goal and the actual columns):
{examples or DEFAULT_PLAN_EXAMPLES}

JSON array:"""


# ---------------------------------------------------------------------------
# CODE-GEN
# ---------------------------------------------------------------------------


def _codegen_scope_rules(time_scoped: bool, start: int) -> str:
    """Numbered goal-scope / date-filter / artifact-reuse rules for CODE-GEN.

    The "last quarter" date-filter rule is only included when *time_scoped*
    (the goal asks about a quarter and a datetime column exists).
    """
    rules: list[str] = []
    if time_scoped:
        rules.append(
            "Preserve the ORIGINAL ANALYTICAL GOAL scope. If the goal says \"last quarter\",\n"
            "   every dependent calculation must use that same scope unless the\n"
            "   sub-question explicitly says otherwise."
        )
        rules.append(
            "DATE FILTER — CRITICAL: The DATASET SCHEMA above contains a\n"
            "   PRE-COMPUTED DATE WINDOW section with LAST_QUARTER_START and LAST_QUARTER_END.\n"
            "   You MUST use those exact pd.Timestamp values verbatim and the filter shown\n"
            "   there. Do NOT re-derive the boundaries using pd.DateOffset, datetime.date,\n"
            "   or any other arithmetic. Deviating from these values is a logic error."
        )
    else:
        rules.append(
            "Preserve the ORIGINAL ANALYTICAL GOAL scope. Do NOT filter by date or\n"
            "   time unless the goal or sub-question names a time window; if it does,\n"
            "   anchor the window on the latest date in the data, never today's date."
        )
    rules.append(
        "ARTIFACT REUSE — CRITICAL: If the AVAILABLE ARTIFACTS section above lists\n"
        "   a DataFrame that already covers the required scope, read from it instead\n"
        "   of recomputing it from df. Use the safe None-check pattern:\n"
        "     prepared = artifacts.get(\"<artifact_name>\")     # try artifact first\n"
        "     if prepared is None:                             # explicit None check (not `or`)\n"
        "         prepared = <recompute from df>               # fallback\n"
        "   Do NOT write `artifacts.get(\"key\") or df[...]` — that raises ValueError\n"
        "   when the artifact is a DataFrame (ambiguous boolean context)."
    )
    return "\n".join(f"{start + i}. {rule}" for i, rule in enumerate(rules))


def codegen_prompt(
    goal: str,
    sub_question: str,
    schema_summary: str,
    environment_manifest: str = "",
    task_constraints: str = "",
    available_artifacts: str = "",
    artifact_contract: str = "",
    *,
    time_scoped: bool = False,
    data_dictionary: str = "",
) -> str:
    """
    Build the CODE-GEN prompt.

    Args:
        goal:               The original analytical goal.
        sub_question:       The specific sub-question to answer.
        schema_summary:     Dataset schema with pre-computed date constants.
        environment_manifest: Sandbox capabilities and constraints.
        task_constraints:   Accumulated per-task constraints from failed retries.
        available_artifacts: Human-readable description of pre-validated DataFrames
                             available in the ``artifacts`` sandbox global. Generated
                             by :meth:`ArtifactStore.describe`.
        artifact_contract:  Optional OUTPUT CONTRACT block (variable name and
                            columns of the DataFrame this task must produce);
                            rendered only when non-empty.
        time_scoped:        Include the "last quarter" date-filter rules (the goal
                            asks about a quarter and a datetime column exists).
        data_dictionary:    Column / code descriptions supplied by the user.
    """
    contract_section = f"\n{artifact_contract}\n" if artifact_contract else ""
    artifacts_section = (
        f"\n{available_artifacts}\n"
        if available_artifacts
        and available_artifacts != "(no pre-computed artifacts available yet)"
        else ""
    )
    return f"""You are a senior data analyst writing Python/pandas code.

DATASET SCHEMA:
{schema_summary}
{data_dictionary_section(data_dictionary)}
{environment_manifest}
{artifacts_section}
ORIGINAL ANALYTICAL GOAL:
{goal}

CUMULATIVE TASK CONSTRAINTS (Do not violate these):
{task_constraints}

SUB-QUESTION TO ANSWER: {sub_question}
{contract_section}
Write a Python snippet that answers the sub-question using the DataFrame `df`.

STRICT RULES — violating any of these will cause execution to fail:
1. Handle nulls explicitly (use .dropna(), .fillna(), or .notna()) — do not assume clean data.
2. If computing an average or sum, check that the denominator / count is non-zero.
3. Print intermediate DataFrames with print(df_name.head()) for the observer to inspect.
4. If this sub-question depends on a previously accepted finding, derive it
   consistently from that finding's scope.
{_codegen_scope_rules(time_scoped, 5)}

Return ONLY the Python code. No explanation, no markdown fences, no comments beyond inline ones.

Python code:"""


def repair_prompt(
    goal: str,
    sub_question: str,
    schema_summary: str,
    environment_manifest: str,
    task_constraints: str,
    previous_code: str,
    error_reason: str,
    available_artifacts: str = "",
    artifact_contract: str = "",
    *,
    time_scoped: bool = False,
    data_dictionary: str = "",
) -> str:
    """
    Build the REPAIR prompt for localized fixes on a failed attempt.

    Args:
        goal:               The original analytical goal.
        sub_question:       The specific sub-question to answer.
        schema_summary:     Dataset schema with pre-computed date constants.
        environment_manifest: Sandbox capabilities and constraints.
        task_constraints:   Accumulated per-task constraints.
        previous_code:      The code from the last failed attempt.
        error_reason:       The critique reason explaining what went wrong.
        available_artifacts: Human-readable description of available pre-validated
                             DataFrames (from ArtifactStore.describe()).
        artifact_contract:  Optional OUTPUT CONTRACT block; rendered only when
                            non-empty.
        time_scoped:        Include the "last quarter" date-filter rules.
        data_dictionary:    Column / code descriptions supplied by the user.
    """
    contract_section = f"\n{artifact_contract}\n" if artifact_contract else ""
    artifacts_section = (
        f"\n{available_artifacts}\n"
        if available_artifacts
        and available_artifacts != "(no pre-computed artifacts available yet)"
        else ""
    )
    return f"""You are a senior data analyst repairing a failing Python/pandas script.

DATASET SCHEMA:
{schema_summary}
{data_dictionary_section(data_dictionary)}
{environment_manifest}
{artifacts_section}
ORIGINAL ANALYTICAL GOAL:
{goal}

CUMULATIVE TASK CONSTRAINTS (Do not violate these):
{task_constraints}

SUB-QUESTION TO ANSWER: {sub_question}
{contract_section}
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
{_codegen_scope_rules(time_scoped, 2)}
{5 if time_scoped else 4}. Return ONLY the complete, repaired Python code. No explanation, no markdown fences.

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
    observation_text: str = "",
    *,
    time_scoped: bool = False,
    data_dictionary: str = "",
) -> str:
    """
    Build the CRITIQUE prompt.

    Instructs the model to be a skeptical auditor looking for specific
    failure modes, not just "does this look right?"

    The critic's primary job is to verify the CODE ITSELF — not to
    compare result strings against memory. History is provided only to
    detect genuine contradictions (same scope, different answer), NOT to
    penalise correct code that uses a different filter or time window.

    When *observation_text* is provided (as of the OBSERVE stage), STEP 3
    uses computed data-quality facts instead of LLM speculation.

    Args:
        goal:             Original analytical goal, including global scope.
        sub_question:      The question the code was trying to answer.
        code:              The Python code that was executed.
        sandbox_result:    The full sandbox output dict
                           (success, result, stdout, error, error_type).
        history_summary:   One-line summaries of previously accepted findings,
                           so the critique can detect genuine contradictions.
        observation_text:  Structured data-quality report from the OBSERVE
                           stage (:meth:`ObservationReport.as_critique_text`).
                           Deterministic null / duplicate / outlier counts.
        time_scoped:       Include the "last quarter" scope and pre-computed
                           date-constant guidance (the goal asks about a quarter
                           and a datetime column exists).
        data_dictionary:   Column / code descriptions supplied by the user.

    Returns:
        Complete prompt string.
    """
    exec_status = "SUCCESS" if sandbox_result.get("success") else "FAILED"
    exec_result = sandbox_result.get("result") or "(no result variable set)"
    exec_stdout = sandbox_result.get("stdout", "") or "(no stdout)"
    exec_error = sandbox_result.get("error") or ""

    error_block = f"\nEXECUTION ERROR:\n{exec_error}" if exec_error else ""

    obs_block = ""
    if observation_text:
        obs_block = f"""
DATA-QUALITY OBSERVATION (computed deterministically from the input DataFrame —
these are hard facts, not estimates):

{observation_text}
"""

    if time_scoped:
        scope_block = """  Preserve the ORIGINAL ANALYTICAL GOAL scope across dependent steps.
  If the goal is scoped to "last quarter" and this code recomputes values over
  the full dataset, action MUST be "revise_code" unless the sub-question
  explicitly asks for all-time data.

  IMPORTANT — PRE-COMPUTED DATE CONSTANTS: The DATASET SCHEMA (provided to the
  codegen model) includes pre-computed LAST_QUARTER_START and LAST_QUARTER_END
  pd.Timestamp constants. If the code defines and uses these exact constants
  to filter the date column, the date window IS correct by construction — do NOT
  flag it as an off-by-one or extra-month error. Only flag the date filter
  if the code ignores the schema constants and derives its own date arithmetic
  that produces a provably different window."""
    else:
        scope_block = """  Preserve the ORIGINAL ANALYTICAL GOAL scope across dependent steps.
  The goal names no quarter: if the code filters to a time window (e.g. a "last
  quarter") that neither the goal nor the sub-question asks for, action MUST be
  "revise_code". If the goal names another scope (a subset, a time window),
  dependent steps must use that same scope."""

    return f"""You are an independent data quality auditor reviewing an AI agent's work.
You did NOT write this code. Your PRIMARY job is to verify the CODE LOGIC — not to
compare result strings against memory.

ORIGINAL ANALYTICAL GOAL:
{goal}
{data_dictionary_section(data_dictionary)}
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
{obs_block}
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
  - Using the wrong column (e.g. a count column instead of an amount column)
  - Misreading a coded column (check the DATA DICTIONARY when present)
  - Off-by-one in date comparisons
  - Dividing by the wrong denominator
  If yes → revise_code.

STEP 2B — GOAL SCOPE AND DEPENDENCY CONSISTENCY:
{scope_block}

  If a previous accepted finding directly establishes a parent value, this
  result must be consistent with it. Example: if an accepted total by group
  is {{"A": 1200.00, "B": 950.00}}, then a later "top group" answer of "B"
  is a genuine contradiction and MUST revise_code.

STEP 3 — DATA QUALITY CHECKS:
  If a "DATA-QUALITY OBSERVATION" block is present above, those null/duplicate/outlier
  counts are hard computed facts — use them as ground truth for this step.
  a. NULL HANDLING: Does the code silently drop nulls without filling them?
     Check whether .dropna() is called on a column with significant nulls
     (see OBSERVATION for actual null counts) before summing or averaging.
  b. DUPLICATES: If the OBSERVATION reports duplicate IDs or duplicate rows,
     does the code aggregate without de-duplicating (e.g. .drop_duplicates()
     on the ID column)? This inflates sums/counts.
  c. SAMPLE SIZE: Does a filter in the code reduce rows to a suspiciously
     small subset? Check date/category filter conditions explicitly.
  d. OUTLIER DOMINANCE: Would a single extreme value dominate a sum/mean
     given the filter the code applies? (See OBSERVATION for IQR outliers.)

STEP 4 — CONTRADICTION CHECK (read the prior code snippets before deciding):
  The "Previously Accepted Findings" above include the code each prior step
  actually ran. Use that to verify scope, not to guess.

  If this result appears to differ from a previously accepted finding:
  a. READ the prior step's code snippet in the history. What date filter or
     groupby scope does it use? Read THIS code too. Are the scopes the same?
     - Same scope, different answer → genuine contradiction → revise_code.
     - Different scope (e.g. prior filtered to a time window, this uses
       the full dataset) → both may be correct. Note the scope difference in your
       reason and accept if the code logic is otherwise sound.
  b. ANALYTICAL CONTINUITY: if this step re-derives a value that a prior
     step already established (e.g., a "top group" was already found in step
     2 and this step independently computes it again), check that both codes
     arrive at the same answer. If they disagree, one of them has a bug.
     Read both code snippets to determine which derivation is correct.
  c. Do NOT rely on memory or guess what prior code did — the code preview
     is in the history; read it.

STEP 5 — LOGICAL CONSISTENCY:
  Is the result logically impossible regardless of scope?
  (Negative amounts or counts that cannot be negative, average > total,
  zero quantity with a positive amount, timestamps far outside the data's
  range, etc.)

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


def summary_prompt(
    goal: str, history_entries: list[dict], *, data_dictionary: str = ""
) -> str:
    """
    Build the FINAL SUMMARY prompt.

    Receives exactly one final result per sub-question. The summary stage is a
    reporter, not an auditor: it must not infer new trust levels, reinterpret
    retry counts, or add speculative caveats.

    Every factual claim in the summary must be traceable to a specific finding
    via its ``artifact_key``. Recommendations are permitted only when a finding
    explicitly provides causal evidence — they must not be synthesised from
    generic patterns.

    Args:
        goal:            The original analytical goal.
        history_entries: One final entry per sub-question, in plan order.
                         Fields used: question, answer, trust, artifact_key.
        data_dictionary: Column / code descriptions supplied by the user; coded
                         values are labelled only from it.

    Returns:
        Complete prompt string.
    """
    import json

    findings_payload = []
    for entry in history_entries:
        findings_payload.append(
            {
                "question": entry.get("question", ""),
                "answer": entry.get("answer", "(no result)"),
                "trust": entry.get("trust", "unknown"),
                "artifact_key": entry.get("artifact_key", ""),
            }
        )
    findings_json = json.dumps(findings_payload, indent=2)

    return f"""You are a senior data analyst writing the conclusion of an investigation.
You are a reporter, not a critic. Use only the final results provided below.

ORIGINAL GOAL: {goal}
{data_dictionary_section(data_dictionary)}
FINAL RESULTS JSON:
{findings_json}

STRICT REPORTING RULES:
1. Use only the question, answer, trust, and artifact_key fields in FINAL RESULTS JSON.
2. Do not infer new trust levels.
3. Do not reinterpret retries or failed attempts; they are not provided.
4. Do not add caveats unless trust is exactly "low" or the answer explicitly
   contains the caveat.
5. Treat np.float64-style reprs as normal numeric answers; round for readability.
6. If two final results appear inconsistent, state the inconsistency plainly
   without inventing a resolution.
7. GROUNDING REQUIREMENT: Every factual claim must be derivable from exactly one
   finding in the FINAL RESULTS JSON. Do not synthesise claims that combine multiple
   findings into a single assertion that none of them individually supports.
8. CODED VALUES: {_coded_values_rule(data_dictionary).replace(chr(10), chr(10) + "   ")}

Write a concise analytical summary (150-300 words) that:
1. Directly answers the original goal using the findings above.
2. Quantifies claims where possible (use the actual numbers from results).
3. Explicitly flags findings whose trust field is exactly "low".
4. Notes any gaps — sub-questions with no result.

Do not hedge everything — be direct. If the data clearly shows X, say "The data shows X."

Summary:"""
