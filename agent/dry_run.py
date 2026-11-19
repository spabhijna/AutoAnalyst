"""
Canned LLM responses for ``--dry-run``: the pipeline runs without API keys.

``dry_run_llm`` has the same call shape as ``agent.llm_client.call_llm`` and is
passed to ``agent.loop.run`` as its ``llm`` callable; nothing is monkeypatched.
"""

from __future__ import annotations


def dry_run_llm(prompt: str, role: str, model: str | None = None) -> str:
    plan_json = (
        '[{"question": "Filter to last quarter and return quarter_df", "capability": "FilterByDateRange", "produces_artifact": "quarter_df", "consumes_artifacts": [], "output_schema": ["transaction_id", "date", "region", "product_category", "units_sold", "unit_price", "revenue"]}, '
        '{"question": "What is total revenue by region in the last quarter?", "capability": "GroupRevenueByRegion", "produces_artifact": "region_revenue", "consumes_artifacts": ["quarter_df"], "output_schema": ["region", "total_revenue"]}, '
        '{"question": "Which region had highest revenue and which category drove it?", "capability": "FindMaximumRevenueRegion", "produces_artifact": "top_region", "consumes_artifacts": ["region_revenue"], "output_schema": ["region", "total_revenue"]}, '
        '{"question": "Check if top region performance was consistent month-over-month", "capability": "MonthlyRevenueTrend", "produces_artifact": "monthly_trend", "consumes_artifacts": ["quarter_df"], "output_schema": ["month", "total_revenue"]}, '
        '{"question": "What was the overall revenue trend by region over the quarter?", "capability": "RevenueShare", "produces_artifact": "revenue_share", "consumes_artifacts": ["region_revenue"], "output_schema": ["region", "total_revenue", "revenue_share"]}]'
    )
    # Any other dataset gets a neutral plan (all tasks run as generated code).
    generic_plan_json = (
        '[{"question": "Summarise the numeric columns (count, mean, min, max)", "capability": null, "produces_artifact": null, "consumes_artifacts": [], "output_schema": ["<scalar>"]}, '
        '{"question": "Which columns have missing values, and how many?", "capability": null, "produces_artifact": null, "consumes_artifacts": [], "output_schema": ["<scalar>"]}]'
    )
    code = 'result = df.describe() if len(df) > 0 else "(empty DataFrame)"'
    critique = (
        '{"trust": "high", "action": "accept", "reason": "Dry-run: verified."}'
    )
    summary = (
        "Dry-run summary: The pipeline executed successfully without LLM API keys. "
        "The planner produced a capability-based execution graph, and all five "
        "steps were resolved deterministically through registered capabilities."
    )
    if role == "plan":
        # The sales plan only makes sense when the schema has its columns.
        return plan_json if "'revenue'" in prompt and "'region'" in prompt else generic_plan_json
    if role == "codegen":
        return code
    if role == "critique":
        return critique
    return summary
