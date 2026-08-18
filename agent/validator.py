"""
Validation helpers for capability-driven execution.

The validator sits between deterministic execution and artifact registration.
It checks dependency satisfaction, output schema contracts, and basic data
integrity so the engine can reject contract violations before freezing artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from agent.capabilities import Capability, CapabilityExecutionContext
from agent.state import TaskNode


@dataclass
class ValidationResult:
    success: bool
    reasons: list[str] = field(default_factory=list)
    expected_schema: list[str] = field(default_factory=list)
    actual_schema: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class CapabilityValidator:
    """Validate capability execution results and their artifact contracts."""

    def validate_context(
        self, capability: Capability, context: CapabilityExecutionContext
    ) -> ValidationResult:
        missing = [
            name
            for name in capability.required_artifacts
            if name not in context.artifacts
        ]
        if missing:
            return ValidationResult(
                success=False,
                reasons=[f"Missing required artifacts: {', '.join(missing)}"],
                metadata={"required_artifacts": list(capability.required_artifacts)},
            )
        return ValidationResult(
            success=True,
            metadata={"required_artifacts": list(capability.required_artifacts)},
        )

    def validate_output(
        self,
        capability: Capability,
        task: TaskNode,
        context: CapabilityExecutionContext,
        result_df: pd.DataFrame | None,
    ) -> ValidationResult:
        if result_df is None:
            return ValidationResult(
                success=False,
                reasons=[
                    f"Capability {capability.capability_id} did not produce an artifact"
                ],
                metadata={"task_id": task.task_id},
            )

        reasons: list[str] = []
        actual_schema = list(result_df.columns)

        if result_df.empty:
            reasons.append("Produced artifact is empty")

        if not task.produces_artifact:
            reasons.append(f"Task does not expect an artifact, but capability {capability.capability_id} produced one.")
        else:
            task_expected_schema = list(task.output_schema) if task.output_schema else []
            if task_expected_schema and task_expected_schema != ["same_as_input"]:
                if actual_schema != task_expected_schema:
                    reasons.append(
                        f"Output schema mismatch: task expected {task_expected_schema}, capability produced {actual_schema}"
                    )
            else:
                expected_schema = list(capability.output_schema)
                if expected_schema and expected_schema != ["same_as_input"]:
                    if actual_schema != expected_schema:
                        reasons.append(
                            f"Output schema mismatch: capability expected {expected_schema}, got {actual_schema}"
                        )
                elif expected_schema == ["same_as_input"]:
                    input_schema = [column for column in context.df.columns]
                    if actual_schema != input_schema:
                        reasons.append(
                            f"Output schema mismatch: expected same_as_input={input_schema}, got {actual_schema}"
                        )

        return ValidationResult(
            success=not reasons,
            reasons=reasons,
            expected_schema=task.output_schema if task.output_schema else list(capability.output_schema),
            actual_schema=actual_schema,
            metadata={
                "task_id": task.task_id,
                "capability_id": capability.capability_id,
            },
        )

    def validate_semantic_contract(
        self,
        task_id: str,
        artifact: Any,
        result_str: Any,
        artifacts: dict[str, pd.DataFrame],
    ) -> ValidationResult:
        reasons: list[str] = []
        quarter_df = artifacts.get("quarter_df")
        
        # Calculate expected top region revenue if quarter_df is available
        expected_top_revenue = None
        if quarter_df is not None and "region" in quarter_df.columns and "revenue" in quarter_df.columns:
            region_totals = quarter_df.groupby("region")["revenue"].sum()
            if not region_totals.empty:
                expected_top_revenue = float(region_totals.max())

        if task_id == "step_2":
            if not isinstance(artifact, pd.DataFrame):
                reasons.append("Step 2 requires a DataFrame artifact.")
            else:
                missing = [c for c in ["month", "region"] if c not in artifact.columns]
                has_rev = "revenue" in artifact.columns or "total_revenue" in artifact.columns
                if missing or not has_rev:
                    reasons.append(f"Step 2 missing dimensions (month, region, revenue). Got: {list(artifact.columns)}")

        elif task_id == "step_4":
            if not isinstance(artifact, pd.DataFrame):
                reasons.append("Step 4 requires a DataFrame artifact.")
            else:
                missing = [c for c in ["month"] if c not in artifact.columns]
                has_rev = "revenue" in artifact.columns or "total_revenue" in artifact.columns
                if missing or not has_rev:
                    reasons.append(f"Step 4 missing dimensions (month, revenue). Got: {list(artifact.columns)}")
                
                if has_rev and expected_top_revenue is not None:
                    rev_col = "total_revenue" if "total_revenue" in artifact.columns else "revenue"
                    actual_total = float(artifact[rev_col].sum())
                    if abs(actual_total - expected_top_revenue) > 0.01:
                        reasons.append(f"Step 4 artifact must represent ONLY the top region. Expected total revenue ~{expected_top_revenue:.2f}, got ~{actual_total:.2f}")

        elif task_id == "step_5":
            str_val = str(result_str).lower().strip()
            is_bool = str_val in ["true", "false", "np.true_", "np.false_"] or isinstance(result_str, (bool, np.bool_))
            if not is_bool:
                reasons.append(f"Step 5 requires a Boolean result. Got: {result_str}")

        elif task_id == "step_6":
            if not isinstance(artifact, pd.DataFrame):
                reasons.append("Step 6 requires a DataFrame artifact.")
            else:
                has_rev = "revenue" in artifact.columns or "total_revenue" in artifact.columns
                if has_rev and expected_top_revenue is not None:
                    rev_col = "total_revenue" if "total_revenue" in artifact.columns else "revenue"
                    actual_total = float(artifact[rev_col].sum())
                    if abs(actual_total - expected_top_revenue) > 0.01:
                        reasons.append(f"Step 6 aggregation must be restricted to the top region and last-quarter data. Expected total revenue ~{expected_top_revenue:.2f}, got ~{actual_total:.2f}")

        return ValidationResult(
            success=not reasons,
            reasons=reasons,
            metadata={"task_id": task_id}
        )

