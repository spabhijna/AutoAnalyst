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
from agent.state import SCHEMA_PLACEHOLDERS, TaskNode


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

        # The capability reads its input_schema columns from its first required
        # artifact, or from the input DataFrame when it requires none.
        source_name = (
            capability.required_artifacts[0] if capability.required_artifacts else "df"
        )
        source = (
            context.artifacts[source_name]
            if capability.required_artifacts
            else context.df
        )
        missing_columns = [
            column for column in capability.input_schema if column not in source.columns
        ]
        if missing_columns:
            return ValidationResult(
                success=False,
                reasons=[
                    f"{capability.capability_id} needs columns {missing_columns} "
                    f"that are missing from {source_name}"
                ],
                expected_schema=list(capability.input_schema),
                actual_schema=list(source.columns),
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
        task: TaskNode,
        artifact: Any,
        artifact_info: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """
        Generic, plan-independent contract checks for a task's artifact.

        Applies only when the task declares ``produces_artifact``:
          * the artifact must be a non-empty DataFrame;
          * its columns must match ``task.output_schema`` as a set (order-insensitive)
            when one is declared
            (``SCHEMA_PLACEHOLDERS`` are skipped).

        An artifact the sandbox produced but did not send back because it is too
        large (``artifact_info["stored"] is False``) still satisfies the contract:
        its columns are checked from ``artifact_info``, and
        ``metadata["artifact_not_stored"]`` carries the reason. Downstream tasks
        then recompute from ``df``.

        Nothing here depends on the task's position or ID in the plan.
        """
        reasons: list[str] = []
        metadata: dict[str, Any] = {"task_id": task.task_id}
        if task.produces_artifact:
            actual: list[str] | None = None
            if isinstance(artifact, pd.DataFrame):
                if artifact.empty:
                    reasons.append(
                        f"Artifact {task.produces_artifact!r} is an empty DataFrame."
                    )
                else:
                    actual = list(artifact.columns)
            elif _too_large_to_store(artifact_info):
                actual = list(artifact_info["columns"])
                metadata["artifact_not_stored"] = artifact_info["reason"]
            else:
                reasons.append(
                    f"Task {task.task_id} declares produces_artifact="
                    f"{task.produces_artifact!r} but no DataFrame artifact was produced."
                )
            if actual is not None:
                expected = list(task.output_schema)
                if expected and expected not in SCHEMA_PLACEHOLDERS:
                    missing = [c for c in expected if c not in actual]
                    extra = [c for c in actual if c not in expected]
                    if missing or extra:
                        reasons.append(
                            f"Artifact {task.produces_artifact!r} columns {actual} "
                            f"do not match the declared output_schema {expected} "
                            f"(missing: {missing}, extra: {extra})."
                        )

        return ValidationResult(
            success=not reasons,
            reasons=reasons,
            metadata=metadata,
        )


def _too_large_to_store(artifact_info: dict[str, Any] | None) -> bool:
    """True when the sandbox built the artifact but kept it because of its size."""
    return bool(
        artifact_info
        and not artifact_info.get("stored")
        and artifact_info.get("bytes") is not None
        and artifact_info.get("columns") is not None
    )
