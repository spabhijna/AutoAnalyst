"""
Validation helpers for capability-driven execution.

The validator sits between deterministic execution and artifact registration.
It checks dependency satisfaction, output schema contracts, and basic data
integrity so the engine can reject contract violations before freezing artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
        expected_schema = list(capability.output_schema)
        actual_schema = list(result_df.columns)

        if result_df.empty:
            reasons.append("Produced artifact is empty")

        if expected_schema and expected_schema != ["same_as_input"]:
            if actual_schema != expected_schema:
                reasons.append(
                    f"Output schema mismatch: expected {expected_schema}, got {actual_schema}"
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
            expected_schema=expected_schema,
            actual_schema=actual_schema,
            metadata={
                "task_id": task.task_id,
                "capability_id": capability.capability_id,
            },
        )
