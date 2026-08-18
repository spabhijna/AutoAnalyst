"""
AnalysisRuntime — Phase 1 persistent runtime for a single AutoAnalyst run.

Architecture
------------
One AnalysisRuntime is created per ``run()`` call and lives for the entire
lifetime of that analysis.  It owns three things:

1. **ArtifactRegistry** — typed registry mapping stable IDs to Artifact objects.
   Artifact IDs are controlled by the system (e.g. ``artifact_step_1``), never
   by the LLM.  Human-readable variable names may coexist in the namespace but
   are NOT part of the correctness contract.

2. **Persistent namespace** — a dict that accumulates variables across all
   executed task steps.  The sandbox receives a snapshot of this dict on every
   call so each step can reference objects produced by earlier steps.  The
   runtime also injects ``df``, ``pd``, and ``np`` at initialisation time.

3. **Input DataFrame reference** — the original ``df`` passed to ``run()`` is
   stored here.  It is never copied; all downstream steps get a reference.

ArtifactRegistry is intentionally separate from the namespace: the registry
stores typed, self-describing artifacts with provenance metadata.  The namespace
is a raw Python dict for execution convenience.

Design principles
-----------------
* **LLM does not manage the registry.**  Artifacts are captured automatically
  after every accepted execution step.
* **Stable IDs beat human-readable names.**  ``artifact_step_1`` will always
  refer to the output of step 1 regardless of what variable name the LLM chose.
* **No cross-run state.**  Each AnalysisRuntime is an independent object; there
  is no module-level or class-level shared state.
* **Lightweight.**  Phase 1 implements the smallest clean abstraction.  Later
  phases will add capability binding, invariant checking, and summary grounding
  on top of this foundation.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MAX_ARTIFACT_BYTES: int = 50 * 1024 * 1024  # 50 MB


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Artifact types
# ---------------------------------------------------------------------------

#: The superset of types an Artifact value can hold.
ArtifactValue = Any


def _classify_type(value: ArtifactValue) -> str:
    """Return a human-readable type label for *value*."""
    if isinstance(value, pd.DataFrame):
        if len(value) == 1 and len(value.columns) == 1:
            return "scalar_df"
        if len(value.columns) == 1:
            return "Series"
        return "DataFrame"
    if isinstance(value, pd.Series):
        return "Series"
    if isinstance(value, bool):
        # bool must come before int (bool is a subclass of int)
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, list):
        return "list"
    return type(value).__name__


# ---------------------------------------------------------------------------
# Artifact model
# ---------------------------------------------------------------------------


@dataclass
class Artifact:
    """
    A typed, self-describing analytical result produced by one pipeline step.

    Every result registered in :class:`ArtifactRegistry` is an Artifact.
    The ``artifact_id`` is system-assigned (e.g. ``artifact_step_2``);
    it is stable and independent of any variable name the LLM chose.

    DataFrame-specific metadata (shape, columns, dtypes) is captured at
    registration time.  The raw ``value`` object remains live in memory for
    downstream consumption; it is never serialised into prompts.
    """

    artifact_id: str          # Stable system ID, e.g. "artifact_step_2"
    task_id: str              # ID of the step that produced this artifact
    type: str                 # "DataFrame" | "Series" | "bool" | "number" | …
    value: ArtifactValue      # The actual Python object (live reference)
    creation_order: int       # Monotonically increasing registration counter
    parent_ids: list[str] = field(default_factory=list)  # Source artifact IDs

    # DataFrame-specific metadata (None for non-DataFrame types)
    shape: tuple[int, int] | None = None
    columns: list[str] | None = None
    dtypes: dict[str, str] | None = None

    # Lifecycle metadata
    created_at: str = field(default_factory=_utc_now)

    # ── Human-readable alias (optional) ─────────────────────────────────────
    # This is the variable name the LLM or deterministic executor used.  It is
    # preserved for debugging but must NOT be used as a correctness constraint.
    variable_name: str | None = None

    def describe(self) -> str:
        """Return a concise multi-line description for debugging."""
        lines = [f"{self.artifact_id}  [task={self.task_id}]"]
        lines.append(f"  type={self.type}")
        if self.shape is not None:
            lines.append(f"  shape={self.shape}")
        if self.columns:
            truncated = self.columns[:8]
            suffix = f" … (+{len(self.columns) - 8} more)" if len(self.columns) > 8 else ""
            lines.append(f"  columns=[{', '.join(truncated)}{suffix}]")
        if self.variable_name:
            lines.append(f"  variable_name={self.variable_name!r}")
        if self.parent_ids:
            lines.append(f"  parents={self.parent_ids}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ArtifactRegistry
# ---------------------------------------------------------------------------


class ArtifactRegistry:
    """
    Runtime-owned registry mapping stable artifact IDs to :class:`Artifact` objects.

    The LLM never interacts with this registry directly.  The execution layer
    registers results after every accepted step.

    Stable IDs are generated as ``artifact_{task_id}`` so they are both
    deterministic and human-readable without relying on LLM invention.
    """

    def __init__(self) -> None:
        self._artifacts: dict[str, Artifact] = {}
        self._counter: int = 0

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def register(
        self,
        artifact_id: str,
        task_id: str,
        value: ArtifactValue,
        parent_ids: list[str] | None = None,
        variable_name: str | None = None,
    ) -> Artifact | None:
        """
        Register *value* under *artifact_id*.

        For DataFrames, checks that the value is non-empty and within the
        size limit before accepting.  Non-DataFrame values are always accepted.

        Args:
            artifact_id:   Stable system ID (e.g. ``"artifact_step_2"``).
            task_id:       The task that produced this value.
            value:         The Python object to register.
            parent_ids:    Optional list of source artifact IDs.
            variable_name: Optional human-readable alias for debugging.

        Returns:
            The registered :class:`Artifact`, or ``None`` if rejected.
        """
        # DataFrame-specific validation
        if isinstance(value, pd.DataFrame):
            if value.empty:
                logger.warning(
                    "ArtifactRegistry.register(%r): DataFrame is empty — skipping",
                    artifact_id,
                )
                return None
            try:
                size_bytes = len(pickle.dumps(value, protocol=4))
            except Exception:  # noqa: BLE001
                size_bytes = 0
            if size_bytes > MAX_ARTIFACT_BYTES:
                logger.warning(
                    "ArtifactRegistry.register(%r): too large (%d MB) — skipping",
                    artifact_id,
                    size_bytes // (1024 * 1024),
                )
                return None

        self._counter += 1
        artifact_type = _classify_type(value)

        # Capture DataFrame-specific metadata
        shape: tuple[int, int] | None = None
        columns: list[str] | None = None
        dtypes: dict[str, str] | None = None
        if isinstance(value, pd.DataFrame):
            shape = (len(value), len(value.columns))
            columns = list(value.columns)
            dtypes = {col: str(value[col].dtype) for col in value.columns}

        artifact = Artifact(
            artifact_id=artifact_id,
            task_id=task_id,
            type=artifact_type,
            value=value,
            creation_order=self._counter,
            parent_ids=list(parent_ids or []),
            shape=shape,
            columns=columns,
            dtypes=dtypes,
            variable_name=variable_name,
        )
        self._artifacts[artifact_id] = artifact
        logger.debug(
            "ArtifactRegistry: registered %r (type=%s, task=%s)",
            artifact_id,
            artifact_type,
            task_id,
        )
        return artifact

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get(self, artifact_id: str) -> Artifact | None:
        """Return the :class:`Artifact` for *artifact_id*, or ``None``."""
        return self._artifacts.get(artifact_id)

    def get_value(self, artifact_id: str) -> ArtifactValue | None:
        """Return the raw value for *artifact_id*, or ``None``."""
        a = self._artifacts.get(artifact_id)
        return a.value if a is not None else None

    def exists(self, artifact_id: str) -> bool:
        """Return True if *artifact_id* is registered."""
        return artifact_id in self._artifacts

    def list(self) -> list[str]:
        """Return all registered artifact IDs in creation order."""
        return sorted(
            self._artifacts,
            key=lambda k: self._artifacts[k].creation_order,
        )

    def remove(self, artifact_id: str) -> bool:
        """
        Remove the artifact with *artifact_id*.

        This is provided for completeness; the current pipeline never removes
        artifacts mid-run.  Returns True if removed, False if not found.
        """
        if artifact_id in self._artifacts:
            del self._artifacts[artifact_id]
            return True
        return False

    def __len__(self) -> int:
        return len(self._artifacts)

    def __contains__(self, artifact_id: str) -> bool:
        return artifact_id in self._artifacts


# ---------------------------------------------------------------------------
# AnalysisRuntime
# ---------------------------------------------------------------------------

#: System-assigned artifact ID prefix.
_ARTIFACT_ID_PREFIX = "artifact_"


class AnalysisRuntime:
    """
    Persistent execution environment for one AutoAnalyst analysis run.

    The runtime owns:
    * The original input DataFrame (``df``), accessible read-only.
    * A persistent Python namespace that accumulates across task steps.
    * The :class:`ArtifactRegistry` for typed, provenance-tracked results.

    Lifetime: one runtime per ``run()`` call.  Not shared across runs.

    Usage::

        runtime = AnalysisRuntime(df)
        # The sandbox receives a snapshot of runtime.namespace each step.
        # After a step is accepted, the executor calls runtime.capture_result().
    """

    def __init__(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"AnalysisRuntime expects a DataFrame, got {type(df)}")
        self._df: pd.DataFrame = df
        self.registry: ArtifactRegistry = ArtifactRegistry()
        # Persistent namespace — starts with the fundamental globals.
        # Downstream steps inherit everything previous steps placed here.
        self.namespace: dict[str, Any] = {
            "df": df,   # original DataFrame — never overwritten
            "pd": pd,
            "np": np,
        }

    # ── Input data ──────────────────────────────────────────────────────────

    @property
    def df(self) -> pd.DataFrame:
        """The original input DataFrame.  Read-only reference."""
        return self._df

    # ── Stable ID helpers ────────────────────────────────────────────────────

    @staticmethod
    def artifact_id_for(task_id: str) -> str:
        """Return the canonical stable artifact ID for a task."""
        return f"{_ARTIFACT_ID_PREFIX}{task_id}"

    # ── Capture ──────────────────────────────────────────────────────────────

    def capture_result(
        self,
        task_id: str,
        value: ArtifactValue,
        *,
        variable_name: str | None = None,
        parent_ids: list[str] | None = None,
    ) -> Artifact | None:
        """
        Register *value* as the output artifact of *task_id*.

        The stable artifact ID is derived from *task_id* via
        :meth:`artifact_id_for`.  If *variable_name* is provided it is stored
        as a human-readable alias but does NOT affect the correctness contract.

        The value is also injected back into ``self.namespace`` under the
        stable ID so subsequent sandbox steps can reference it.

        Returns the registered :class:`Artifact`, or ``None`` if rejected.
        """
        artifact_id = self.artifact_id_for(task_id)
        artifact = self.registry.register(
            artifact_id=artifact_id,
            task_id=task_id,
            value=value,
            parent_ids=parent_ids,
            variable_name=variable_name,
        )
        if artifact is not None:
            # Expose via stable ID in the namespace
            self.namespace[artifact_id] = value
            # Also expose via human-readable alias if provided (for backward compat)
            if variable_name and variable_name not in self.namespace:
                self.namespace[variable_name] = value
            logger.info(
                "AnalysisRuntime: captured %r from %s (type=%s)",
                artifact_id,
                task_id,
                artifact.type,
            )
        return artifact

    def update_namespace(self, variables: dict[str, Any]) -> None:
        """
        Merge *variables* from a completed sandbox execution into the
        persistent namespace.

        Protects the original ``df`` reference — a sandbox step can overwrite
        ``df`` inside its own local namespace without affecting the runtime.
        """
        protected = {"df", "pd", "np"}
        for k, v in variables.items():
            if k in protected:
                continue
            if not k.startswith("_"):
                self.namespace[k] = v

    # ── Namespace snapshot ───────────────────────────────────────────────────

    def namespace_snapshot(self) -> dict[str, Any]:
        """
        Return a shallow copy of the current namespace for sandbox injection.

        The sandbox receives a *copy* so it cannot mutate the runtime's own
        namespace dict.  Large DataFrame values are still shared references
        (no unnecessary copies of the data).
        """
        return dict(self.namespace)

    # ── Introspection ────────────────────────────────────────────────────────

    def describe(self) -> str:
        """
        Return concise metadata about the runtime state for debugging.

        Does NOT dump full DataFrame contents — only shapes/types.

        Example output::

            df
              type=DataFrame
              shape=(700, 7)

            artifact_step_1
              type=DataFrame
              shape=(524, 7)
              columns=[transaction_id, date, region, …]
        """
        lines: list[str] = []

        # Always show the input df
        lines.append("df")
        lines.append(f"  type=DataFrame")
        lines.append(f"  shape={self._df.shape}")
        cols_preview = ", ".join(list(self._df.columns)[:6])
        if len(self._df.columns) > 6:
            cols_preview += f" … (+{len(self._df.columns) - 6} more)"
        lines.append(f"  columns=[{cols_preview}]")

        for artifact_id in self.registry.list():
            artifact = self.registry.get(artifact_id)
            if artifact is None:
                continue
            lines.append("")
            lines.append(artifact.describe())

        return "\n".join(lines) if lines else "(runtime empty)"
