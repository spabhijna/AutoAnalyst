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

from agent.executors import DeterministicExecutionResult
from agent.state import TaskNode


@dataclass(frozen=True)
class CapabilityExecutionContext:
    """Inputs available to a deterministic capability executor."""

    task: TaskNode
    df: pd.DataFrame
    artifacts: dict[str, pd.DataFrame]


@dataclass(frozen=True)
class Capability:
    """Reusable analytical operation registered with the engine."""

    capability_id: str
    description: str
    required_artifacts: tuple[str, ...]
    produced_artifacts: tuple[str, ...]
    input_schema: tuple[str, ...]
    output_schema: tuple[str, ...]
    executor: Callable[[CapabilityExecutionContext], DeterministicExecutionResult]
    match_terms: tuple[str, ...] = ()


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
            capability_text = " ".join(
                (capability.capability_id, *capability.match_terms)
            ).lower()
            for term in capability_text.split():
                if term and term in text:
                    score += 1
            if (
                task.produces_artifact
                and task.produces_artifact in capability.produced_artifacts
            ):
                score += 3
            if any(
                required in task.consumes_artifacts
                for required in capability.required_artifacts
            ):
                score += 2
            if capability.capability_id.lower() in text:
                score += 4
            if score > 0:
                scored.append((score, capability))
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], item[1].capability_id))
        return scored[0][1]

    def describe(self) -> str:
        if not self._capabilities:
            return "(no capabilities registered)"
        lines = ["AVAILABLE CAPABILITIES:"]
        for capability in self.list():
            lines.append(
                f"- {capability.capability_id}: {capability.description} "
                f"requires={list(capability.required_artifacts)} produces={list(capability.produced_artifacts)}"
            )
        return "\n".join(lines)


def _last_quarter_window(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if "date" not in df.columns:
        return None
    date_series = pd.to_datetime(df["date"], errors="coerce").dropna()
    if date_series.empty:
        return None
    end_ts = date_series.max()
    start_ts = (end_ts - pd.DateOffset(months=2)).replace(day=1)
    return start_ts, end_ts


def _determine_scope_df(
    context: CapabilityExecutionContext, artifact_name: str = "quarter_df"
) -> pd.DataFrame:
    scoped = context.artifacts.get(artifact_name)
    if scoped is not None:
        return scoped
    window = _last_quarter_window(context.df)
    if window is None:
        return context.df.copy(deep=False)
    start_ts, end_ts = window
    date_series = pd.to_datetime(context.df["date"], errors="coerce")
    mask = (date_series >= start_ts) & (date_series <= end_ts)
    return context.df.loc[mask].copy(deep=False)


def _single_row_result(
    df: pd.DataFrame, artifact_key: str, result_text: str
) -> DeterministicExecutionResult:
    return DeterministicExecutionResult(
        success=True,
        result=result_text,
        stdout="",
        error=None,
        error_type=None,
        error_category="none",
        artifact=df,
        artifact_key=artifact_key,
    )


def execute_filter_by_date_range(
    context: CapabilityExecutionContext,
) -> DeterministicExecutionResult:
    quarter_df = _determine_scope_df(context, "quarter_df")
    return DeterministicExecutionResult(
        success=True,
        result=f"quarter_df: {len(quarter_df):,} rows",
        stdout="",
        error=None,
        error_type=None,
        error_category="none",
        artifact=quarter_df,
        artifact_key=context.task.produces_artifact or "quarter_df",
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
    return DeterministicExecutionResult(
        success=True,
        result=grouped.head().to_string(index=False),
        stdout="",
        error=None,
        error_type=None,
        error_category="none",
        artifact=grouped,
        artifact_key=context.task.produces_artifact or "region_revenue",
    )


def execute_find_maximum_revenue_region(
    context: CapabilityExecutionContext,
) -> DeterministicExecutionResult:
    region_revenue = context.artifacts.get("region_revenue")
    if region_revenue is None:
        region_revenue = execute_group_revenue_by_region(context).artifact
    top_region = region_revenue.head(1).copy(deep=False)
    return _single_row_result(
        top_region,
        context.task.produces_artifact or "top_region",
        top_region.to_string(index=False),
    )


def execute_monthly_revenue_trend(
    context: CapabilityExecutionContext,
) -> DeterministicExecutionResult:
    quarter_df = _determine_scope_df(context, "quarter_df")
    monthly = quarter_df.copy(deep=False)
    monthly["month"] = (
        pd.to_datetime(monthly["date"], errors="coerce").dt.to_period("M").astype(str)
    )
    trend = (
        monthly.groupby("month", as_index=False)["revenue"]
        .sum()
        .rename(columns={"revenue": "total_revenue"})
        .sort_values("month")
        .reset_index(drop=True)
    )
    return DeterministicExecutionResult(
        success=True,
        result=trend.head().to_string(index=False),
        stdout="",
        error=None,
        error_type=None,
        error_category="none",
        artifact=trend,
        artifact_key=context.task.produces_artifact or "monthly_trend",
    )


def execute_revenue_share(
    context: CapabilityExecutionContext,
) -> DeterministicExecutionResult:
    region_revenue = context.artifacts.get("region_revenue")
    if region_revenue is None:
        region_revenue = execute_group_revenue_by_region(context).artifact
    total = region_revenue.iloc[:, 1].sum()
    share = region_revenue.copy(deep=False)
    share["revenue_share"] = 0.0 if total == 0 else share.iloc[:, 1] / total
    return DeterministicExecutionResult(
        success=True,
        result=share.head().to_string(index=False),
        stdout="",
        error=None,
        error_type=None,
        error_category="none",
        artifact=share,
        artifact_key=context.task.produces_artifact or "revenue_share",
    )


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
    return DeterministicExecutionResult(
        success=True,
        result=category.head().to_string(index=False),
        stdout="",
        error=None,
        error_type=None,
        error_category="none",
        artifact=category,
        artifact_key=context.task.produces_artifact or "category_revenue",
    )


def build_default_capability_registry() -> CapabilityRegistry:
    registry = CapabilityRegistry()
    registry.register(
        Capability(
            capability_id="FilterByDateRange",
            description="Filter the working dataset to the analysis date window.",
            required_artifacts=(),
            produced_artifacts=("quarter_df",),
            input_schema=("date",),
            output_schema=("same_as_input",),
            executor=execute_filter_by_date_range,
            match_terms=("last quarter", "date range", "filter", "quarter_df"),
        )
    )
    registry.register(
        Capability(
            capability_id="GroupRevenueByRegion",
            description="Aggregate revenue by region in the scoped dataset.",
            required_artifacts=("quarter_df",),
            produced_artifacts=("region_revenue",),
            input_schema=("region", "revenue"),
            output_schema=("region", "total_revenue"),
            executor=execute_group_revenue_by_region,
            match_terms=("region revenue", "revenue by region", "groupby region"),
        )
    )
    registry.register(
        Capability(
            capability_id="FindMaximumRevenueRegion",
            description="Find the region with the highest total revenue.",
            required_artifacts=("region_revenue",),
            produced_artifacts=("top_region",),
            input_schema=("region", "total_revenue"),
            output_schema=("region", "total_revenue"),
            executor=execute_find_maximum_revenue_region,
            match_terms=("highest revenue", "top region", "maximum revenue"),
        )
    )
    registry.register(
        Capability(
            capability_id="MonthlyRevenueTrend",
            description="Summarize revenue trends by month.",
            required_artifacts=("quarter_df",),
            produced_artifacts=("monthly_trend",),
            input_schema=("date", "revenue"),
            output_schema=("month", "total_revenue"),
            executor=execute_monthly_revenue_trend,
            match_terms=("month-over-month", "monthly trend", "monthly revenue trend"),
        )
    )
    registry.register(
        Capability(
            capability_id="RevenueShare",
            description="Compute revenue share across regions or categories.",
            required_artifacts=("region_revenue",),
            produced_artifacts=("revenue_share",),
            input_schema=("region", "total_revenue"),
            output_schema=("region", "total_revenue", "revenue_share"),
            executor=execute_revenue_share,
            match_terms=("share", "percentage", "revenue share"),
        )
    )
    registry.register(
        Capability(
            capability_id="CategoryAggregation",
            description="Aggregate revenue by product category.",
            required_artifacts=("quarter_df",),
            produced_artifacts=("category_revenue",),
            input_schema=("product_category", "revenue"),
            output_schema=("product_category", "total_revenue"),
            executor=execute_category_aggregation,
            match_terms=("category", "product category", "category aggregation"),
        )
    )
    return registry
