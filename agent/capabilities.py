"""
Capability registry for reusable analytical operations.

The registry is the declarative layer between the planner and execution engine.
Each capability advertises what it needs, what it produces, and which
 deterministic executor implements it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from agent.dates import (
    DATE_COLUMN_TOKEN,
    detect_date_column,
    goal_mentions_quarter,
    last_quarter_window,
)
from agent.state import TaskNode


@dataclass
class DeterministicExecutionResult:
    success: bool
    result: str | None
    stdout: str
    error: str | None
    error_type: str | None
    error_category: str
    artifact: pd.DataFrame | None
    artifact_key: str | None

    def to_sandbox_result(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "result": self.result,
            "stdout": self.stdout,
            "error": self.error,
            "error_type": self.error_type,
            "error_category": self.error_category,
            "artifact": self.artifact,
            "artifact_key": self.artifact_key,
        }


@dataclass(frozen=True)
class CapabilityExecutionContext:
    """Inputs available to a deterministic capability executor."""

    task: TaskNode
    df: pd.DataFrame
    artifacts: dict[str, pd.DataFrame]


@dataclass(frozen=True)
class Capability:
    """
    Reusable analytical operation registered with the engine.

    ``input_schema`` lists the columns the executor reads; the token ``"date"``
    (:data:`agent.dates.DATE_COLUMN_TOKEN`) means "the dataset's detected
    datetime column", whatever it is called. ``example_question`` is used to
    build the planner's example plan from the capabilities that apply to the
    data. ``requires_time_scope`` capabilities only apply when the goal asks
    about a quarter (``TimeScope.quarter``).
    """

    capability_id: str
    description: str
    required_artifacts: tuple[str, ...]
    produced_artifacts: tuple[str, ...]
    input_schema: tuple[str, ...]
    output_schema: tuple[str, ...]
    executor: Callable[[CapabilityExecutionContext], DeterministicExecutionResult]
    match_terms: tuple[str, ...] = ()
    example_question: str = ""
    requires_time_scope: bool = False


@dataclass
class CapabilityRegistry:
    """Registry of reusable capabilities and their deterministic executors."""

    _capabilities: dict[str, Capability] = field(default_factory=dict)

    def register(self, capability: Capability) -> None:
        self._capabilities[capability.capability_id] = capability

    def get(self, capability_id: str | None) -> Capability | None:
        if not capability_id:
            return None
        return self._capabilities.get(capability_id)

    def list(self) -> list[Capability]:
        return list(self._capabilities.values())

    def resolve(self, task: TaskNode) -> Capability | None:
        """Resolve a task to the best matching capability."""
        if task.capability:
            capability = self.get(task.capability)
            if capability is not None:
                return capability

        text = " ".join(
            [
                task.question,
                task.produces_artifact or "",
                " ".join(task.consumes_artifacts),
            ]
        ).lower()
        scored: list[tuple[int, Capability]] = []
        for capability in self._capabilities.values():
            score = 0
            
            # Reject if the task does NOT expect an artifact, but the capability produces one
            if not task.produces_artifact and capability.produced_artifacts:
                continue

            for term in capability.match_terms:
                if term.lower() in text:
                    score += 3
            if (
                task.produces_artifact
                and task.produces_artifact in capability.produced_artifacts
            ):
                score += 5
            if any(
                required in task.consumes_artifacts
                for required in capability.required_artifacts
            ):
                score += 2
            if capability.capability_id.lower() in text:
                score += 10
                
            if task.output_schema and list(capability.output_schema) == task.output_schema:
                score += 10

            if score > 0:
                scored.append((score, capability))
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], item[1].capability_id))
        return scored[0][1]

    def describe(self, capabilities: list[Capability] | None = None) -> str:
        """Describe *capabilities* (default: all registered) for a prompt."""
        selected = self.list() if capabilities is None else capabilities
        if not selected:
            return "(no capabilities registered)"
        lines = ["AVAILABLE CAPABILITIES:"]
        for capability in selected:
            lines.append(
                f"- {capability.capability_id}: {capability.description} "
                f"requires={list(capability.required_artifacts)} "
                f"produces={list(capability.produced_artifacts)} "
                f"input_columns={list(capability.input_schema)}"
            )
        return "\n".join(lines)

    def applicable(self, columns: list[str], time_scoped: bool) -> list[Capability]:
        """
        Capabilities that can run on a dataset with *columns*, in registry order.

        A capability applies when every input column is available — in the
        dataset, or produced by another applicable capability — and every
        required artifact is produced by an applicable capability.
        ``requires_time_scope`` capabilities apply only when *time_scoped*; the
        ``"date"`` input token is satisfied by a detected datetime column, which
        callers signal by including :data:`DATE_COLUMN_TOKEN` in *columns*.
        """
        available_columns = set(columns)
        produced: set[str] = set()
        chosen: list[Capability] = []
        changed = True
        while changed:
            changed = False
            for capability in self.list():
                if capability in chosen:
                    continue
                if capability.requires_time_scope and not time_scoped:
                    continue
                if not set(capability.input_schema) <= available_columns:
                    continue
                if not set(capability.required_artifacts) <= produced:
                    continue
                chosen.append(capability)
                produced.update(capability.produced_artifacts)
                available_columns.update(capability.output_schema)
                changed = True
        order = {c.capability_id: i for i, c in enumerate(self.list())}
        return sorted(chosen, key=lambda c: order[c.capability_id])


def _determine_scope_df(
    context: CapabilityExecutionContext, artifact_name: str = "quarter_df"
) -> pd.DataFrame:
    scoped = context.artifacts.get(artifact_name)
    if scoped is not None:
        return scoped
    date_col = detect_date_column(context.df)
    window = last_quarter_window(context.df, date_col)
    if window is None:
        return context.df.copy(deep=False)
    start_ts, end_ts = window
    date_series = pd.to_datetime(context.df[date_col], errors="coerce")
    mask = (date_series >= start_ts) & (date_series <= end_ts)
    return context.df.loc[mask].copy(deep=False)


def _df_result(
    df: pd.DataFrame, artifact_key: str, result_text: str | None = None
) -> DeterministicExecutionResult:
    """Wrap a produced DataFrame as a successful deterministic result."""
    return DeterministicExecutionResult(
        success=True,
        result=df.head().to_string(index=False) if result_text is None else result_text,
        stdout="",
        error=None,
        error_type=None,
        error_category="none",
        artifact=df,
        artifact_key=artifact_key,
    )


def _failed_result(error: str) -> DeterministicExecutionResult:
    """A failed result: the engine returns None and the loop falls back to CODE-GEN."""
    return DeterministicExecutionResult(
        success=False,
        result=None,
        stdout="",
        error=error,
        error_type="ValueError",
        error_category="capability_error",
        artifact=None,
        artifact_key=None,
    )


def execute_filter_by_date_range(
    context: CapabilityExecutionContext,
) -> DeterministicExecutionResult:
    task = context.task
    # Only a task that asks for the last quarter may be answered by this
    # filter; any other filter (a month, a subset) must go to CODE-GEN.
    if not (
        goal_mentions_quarter(task.question)
        or "quarter" in (task.produces_artifact or "")
    ):
        return _failed_result(
            "FilterByDateRange only applies to tasks that ask for the last quarter"
        )
    quarter_df = _determine_scope_df(context, "quarter_df")
    return _df_result(
        quarter_df,
        context.task.produces_artifact or "quarter_df",
        f"quarter_df: {len(quarter_df):,} rows",
    )


def execute_group_revenue_by_region(
    context: CapabilityExecutionContext,
) -> DeterministicExecutionResult:
    quarter_df = _determine_scope_df(context, "quarter_df")
    grouped = (
        quarter_df.groupby("region", as_index=False)["revenue"]
        .sum()
        .rename(columns={"revenue": "total_revenue"})
        .sort_values("total_revenue", ascending=False)
        .reset_index(drop=True)
    )
    return _df_result(grouped, context.task.produces_artifact or "region_revenue")


def execute_find_maximum_revenue_region(
    context: CapabilityExecutionContext,
) -> DeterministicExecutionResult:
    region_revenue = context.artifacts.get("region_revenue")
    if region_revenue is None:
        region_revenue = execute_group_revenue_by_region(context).artifact
    top_region = region_revenue.head(1).copy(deep=False)
    return _df_result(top_region, context.task.produces_artifact or "top_region")


def execute_monthly_revenue_trend(
    context: CapabilityExecutionContext,
) -> DeterministicExecutionResult:
    quarter_df = _determine_scope_df(context, "quarter_df")
    monthly = quarter_df.copy(deep=False)
    date_col = detect_date_column(monthly) or DATE_COLUMN_TOKEN
    monthly["month"] = (
        pd.to_datetime(monthly[date_col], errors="coerce").dt.to_period("M").astype(str)
    )
    trend = (
        monthly.groupby("month", as_index=False)["revenue"]
        .sum()
        .rename(columns={"revenue": "total_revenue"})
        .sort_values("month")
        .reset_index(drop=True)
    )
    return _df_result(trend, context.task.produces_artifact or "monthly_trend")


def execute_revenue_share(
    context: CapabilityExecutionContext,
) -> DeterministicExecutionResult:
    region_revenue = context.artifacts.get("region_revenue")
    if region_revenue is None:
        region_revenue = execute_group_revenue_by_region(context).artifact
    if "total_revenue" not in region_revenue.columns:
        return _failed_result(
            "RevenueShare needs a 'total_revenue' column in region_revenue, "
            f"got {list(region_revenue.columns)}"
        )
    total = region_revenue["total_revenue"].sum()
    share = region_revenue.copy(deep=False)
    share["revenue_share"] = (
        0.0 if total == 0 else share["total_revenue"] / total
    )
    return _df_result(share, context.task.produces_artifact or "revenue_share")


def execute_category_aggregation(
    context: CapabilityExecutionContext,
) -> DeterministicExecutionResult:
    quarter_df = _determine_scope_df(context, "quarter_df")
    category = (
        quarter_df.groupby("product_category", as_index=False)["revenue"]
        .sum()
        .rename(columns={"revenue": "total_revenue"})
        .sort_values("total_revenue", ascending=False)
        .reset_index(drop=True)
    )
    return _df_result(category, context.task.produces_artifact or "category_revenue")


def execute_average_unit_price(
    context: CapabilityExecutionContext,
) -> DeterministicExecutionResult:
    quarter_df = _determine_scope_df(context, "quarter_df")
    avg_price = (
        quarter_df.groupby("product_category", as_index=False)["unit_price"]
        .mean()
        .round(2)
        .rename(columns={"unit_price": "average_price"})
        .sort_values("average_price", ascending=False)
        .reset_index(drop=True)
    )
    return _df_result(avg_price, context.task.produces_artifact or "average_prices")


def execute_top_selling_product_category(
    context: CapabilityExecutionContext,
) -> DeterministicExecutionResult:
    quarter_df = _determine_scope_df(context, "quarter_df")
    top_selling = (
        quarter_df.groupby("product_category", as_index=False)["units_sold"]
        .sum()
        .rename(columns={"units_sold": "total_units_sold"})
        .sort_values("total_units_sold", ascending=False)
        .reset_index(drop=True)
    )
    return _df_result(
        top_selling, context.task.produces_artifact or "top_selling_category"
    )


_DEFAULT_CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        capability_id="FilterByDateRange",
        description="Filter the working dataset to the analysis date window.",
        required_artifacts=(),
        produced_artifacts=("quarter_df",),
        input_schema=("date",),
        output_schema=("same_as_input",),
        executor=execute_filter_by_date_range,
        match_terms=("last quarter", "date range", "quarter_df"),
        example_question="Filter df to the last quarter and return the filtered DataFrame",
        requires_time_scope=True,
    ),
    Capability(
        capability_id="GroupRevenueByRegion",
        description="Aggregate revenue by region in the scoped dataset.",
        required_artifacts=("quarter_df",),
        produced_artifacts=("region_revenue",),
        input_schema=("region", "revenue"),
        output_schema=("region", "total_revenue"),
        executor=execute_group_revenue_by_region,
        match_terms=("region revenue", "revenue by region", "groupby region"),
        example_question="What is the total revenue by region in the last quarter?",
    ),
    Capability(
        capability_id="FindMaximumRevenueRegion",
        description="Find the region with the highest total revenue.",
        required_artifacts=("region_revenue",),
        produced_artifacts=("top_region",),
        input_schema=("region", "total_revenue"),
        output_schema=("region", "total_revenue"),
        executor=execute_find_maximum_revenue_region,
        match_terms=("highest revenue", "top region", "maximum revenue"),
        example_question="Which region has the highest total revenue?",
    ),
    Capability(
        capability_id="MonthlyRevenueTrend",
        description="Summarize revenue trends by month.",
        required_artifacts=("quarter_df",),
        produced_artifacts=("monthly_trend",),
        input_schema=("date", "revenue"),
        output_schema=("month", "total_revenue"),
        executor=execute_monthly_revenue_trend,
        match_terms=("month-over-month", "monthly trend", "monthly revenue trend"),
        example_question="How did total revenue change month by month?",
    ),
    Capability(
        capability_id="RevenueShare",
        description="Compute revenue share across regions or categories.",
        required_artifacts=("region_revenue",),
        produced_artifacts=("revenue_share",),
        input_schema=("region", "total_revenue"),
        output_schema=("region", "total_revenue", "revenue_share"),
        executor=execute_revenue_share,
        match_terms=("share", "percentage", "revenue share"),
        example_question="What share of total revenue does each region contribute?",
    ),
    Capability(
        capability_id="CategoryAggregation",
        description="Aggregate revenue by product category.",
        required_artifacts=("quarter_df",),
        produced_artifacts=("category_revenue",),
        input_schema=("product_category", "revenue"),
        output_schema=("product_category", "total_revenue"),
        executor=execute_category_aggregation,
        match_terms=("category", "product category", "category aggregation"),
        example_question="What is the total revenue by product category?",
    ),
    Capability(
        capability_id="AverageUnitPrice",
        description="Compute the average unit price by product category.",
        required_artifacts=("quarter_df",),
        produced_artifacts=("average_prices",),
        input_schema=("product_category", "unit_price"),
        output_schema=("product_category", "average_price"),
        executor=execute_average_unit_price,
        match_terms=("average price", "unit price", "mean price"),
        example_question="What is the average unit price by product category?",
    ),
    Capability(
        capability_id="TopSellingProductCategory",
        description="Find the product category with the highest units sold.",
        required_artifacts=("quarter_df",),
        produced_artifacts=("top_selling_category",),
        input_schema=("product_category", "units_sold"),
        output_schema=("product_category", "total_units_sold"),
        executor=execute_top_selling_product_category,
        match_terms=("top selling", "most units sold", "highest volume"),
        example_question="Which product category sold the most units?",
    ),
)


def build_default_capability_registry() -> CapabilityRegistry:
    registry = CapabilityRegistry()
    for capability in _DEFAULT_CAPABILITIES:
        registry.register(capability)
    return registry
