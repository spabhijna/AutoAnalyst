"""
Generic execution engine for capability-driven analytical workflows.

The engine resolves a task to a registered capability, fetches required
artifacts, invokes the deterministic executor and validates the contract.

The engine never writes to the ArtifactStore: the loop's ``_accept`` is the single
owner of artifact registration (so an artifact is registered exactly once).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from agent.artifact_store import ArtifactStore
from agent.capabilities import (
    CapabilityRegistry,
    CapabilityExecutionContext,
    build_default_capability_registry,
)
from agent.state import TaskNode
from agent.validator import CapabilityValidator


@dataclass
class ExecutionEngine:
    """Capability-driven execution engine."""

    registry: CapabilityRegistry = field(
        default_factory=build_default_capability_registry
    )
    validator: CapabilityValidator = field(default_factory=CapabilityValidator)

    def execute(
        self,
        task: TaskNode,
        df: pd.DataFrame,
        artifact_store: ArtifactStore,
    ) -> dict[str, Any] | None:
        capability = self.registry.resolve(task)
        if capability is None:
            return None

        required_namespace = artifact_store.as_namespace_for(
            list(capability.required_artifacts)
        )
        if len(required_namespace) < len(capability.required_artifacts):
            return None

        context = CapabilityExecutionContext(
            task=task,
            df=df,
            artifacts=required_namespace,
        )
        context_validation = self.validator.validate_context(capability, context)
        if not context_validation.success:
            return None
            
        result = capability.executor(context)
        if not result.success:
            return None

        output_validation = self.validator.validate_output(
            capability, task, context, result.artifact
        )
        if not output_validation.success:
            return None

        semantic_validation = self.validator.validate_semantic_contract(
            task.task_id, result.artifact, result.result, artifact_store.as_namespace()
        )
        if not semantic_validation.success:
            return None

        return result.to_sandbox_result()
