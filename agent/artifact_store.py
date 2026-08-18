"""
ArtifactStore — in-process registry for validated intermediate DataFrames.

Design decisions
----------------
1. **Keyed by planner-declared names (e.g. "quarter_df").**
   Names are declared in the structured plan so both the code-gen prompt and
   the sandbox namespace use the same identifiers — no implicit discovery needed.

2. **Shallow-copy on registration.**
   The caller's DataFrame cannot be mutated from outside after registration,
   but column data is shared (no unnecessary memory duplication for read-only use).

3. **Size guard at registration time.**
   DataFrames larger than MAX_ARTIFACT_BYTES are refused to prevent the
   multiprocessing Queue from OOM-killing the parent process during
   sandbox artifact extraction.

4. **Fully accessible via `artifacts` dict in sandbox namespace.**
   ``as_namespace()`` returns a plain dict that the sandbox injects as the
   ``artifacts`` global, so generated code can write
   ``quarter_df = artifacts["quarter_df"]``.

5. **Self-describing for CODE-GEN prompts.**
   ``describe()`` includes artifact type ("DataFrame", "Series") and schema
   so downstream prompts are self-contained — a step referencing ``quarter_df``
   sees its columns and row count without needing to know which prior step
   produced it or what filter was applied.

6. **Artifacts are first-class objects.**
   Each registered result is a :class:`Artifact` bundling name, type, schema,
   producer, and value in one place.  No separate metadata dict — the artifact
   IS the metadata.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import pandas as pd

logger = logging.getLogger(__name__)

# Refuse artifacts larger than this to keep sandbox IPC safe.
# 50 MB is generous for typical analytical DataFrames (100 k–1 M rows).
MAX_ARTIFACT_BYTES: int = 50 * 1024 * 1024  # 50 MB


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Helper — artifact classification
# ---------------------------------------------------------------------------


def classify_artifact(df: pd.DataFrame) -> tuple[str, list[str]]:
    """
    Classify a DataFrame artifact by its structure.

    Returns a (type_label, schema_list) tuple:
    - ``"DataFrame"`` if the DataFrame has multiple columns (a full table).
    - ``"Series"``   if it has exactly one column (e.g. aggregated result).
    - ``"scalar"``   if it has 1 row × 1 column (single value result).

    The *schema* list contains column names for DataFrames/Series,
    or ``["<scalar>"]`` for scalar values.

    Args:
        df: A validated, non-empty DataFrame.

    Returns:
        (type, schema) tuple.
    """
    cols = list(df.columns)
    if len(df) == 1 and len(df.columns) == 1:
        return ("scalar", ["<scalar>"])
    if len(cols) == 1:
        return ("Series", cols)
    return ("DataFrame", cols)


# Backward-compatible alias (tests use the old name).
_classify_artifact = classify_artifact


# ---------------------------------------------------------------------------
# Artifact — first-class typed object
# ---------------------------------------------------------------------------


@dataclass
class Artifact:
    """
    A first-class analytical output — value + self-describing metadata.

    Every result produced by the pipeline (DataFrames, aggregated Series,
    scalar answers) becomes an :class:`Artifact`.  The ``type`` and ``schema``
    fields make downstream CODE-GEN prompts self-contained without needing to
    cross-reference the producing task's scope or filter.
    """

    name: str  # planner-declared snake_case key (e.g. "quarter_df")
    type: str  # "DataFrame" | "Series" | "scalar"
    schema: list[str]  # column names (DataFrame/Series), ["<scalar>"] for scalar
    producer: str  # producing_task_id
    value: pd.DataFrame  # the actual data (shallow-copied on registration)
    consumers: list[str] = field(default_factory=list)
    confidence: str = "unknown"
    provenance: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now_iso)
    validated_at: str | None = None
    is_frozen: bool = False

    # ── Derived accessors ─────────────────────────────────────────────────────

    @property
    def rows(self) -> int:
        """Return the number of rows in the underlying DataFrame."""
        return len(self.value)

    def column_summary(self, max_cols: int = 8) -> str:
        """Return a compact column list, truncated for prompt embedding."""
        cols = list(self.value.columns)[:max_cols]
        suffix = (
            f", … (+{len(self.value.columns) - max_cols} more)"
            if len(self.value.columns) > max_cols
            else ""
        )
        return ", ".join(cols) + suffix

    def add_consumer(self, task_id: str) -> None:
        """Record a downstream task that consumes this artifact."""
        if task_id and task_id not in self.consumers:
            self.consumers.append(task_id)

    def freeze(self) -> None:
        """Mark the artifact immutable after validation."""
        object.__setattr__(self, "validated_at", _utc_now_iso())
        object.__setattr__(self, "is_frozen", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "is_frozen", False) and name not in {
            "validated_at",
            "is_frozen",
        }:
            raise AttributeError(
                f"Artifact {self.name!r} is frozen and cannot be modified"
            )
        object.__setattr__(self, name, value)


# ---------------------------------------------------------------------------
# ArtifactStore
# ---------------------------------------------------------------------------


@dataclass
class ArtifactStore:
    """
    Registry mapping artifact_key → :class:`Artifact`.

    Populated as tasks are accepted during the agent loop. Later steps read
    their dependency DataFrames from here instead of re-applying filters and
    aggregations from the raw input DataFrame.

    Thread safety: the agent loop is single-threaded; no locking is needed.
    """

    _artifacts: dict[str, Artifact] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def register(
        self,
        key: str,
        df: pd.DataFrame,
        producing_task_id: str = "",
        consumers: list[str] | None = None,
        confidence: str = "unknown",
        provenance: list[str] | None = None,
    ) -> bool:
        """
        Register *df* under *key* as an :class:`Artifact`.

        Performs three validations before accepting:
          1. Value is a non-None pd.DataFrame.
          2. DataFrame is non-empty.
          3. Pickled size is within MAX_ARTIFACT_BYTES.

        The artifact's ``type`` and ``schema`` are auto-detected from the
        DataFrame structure via :func:`classify_artifact`.

        Args:
            key:               Snake_case name declared by the planner.
            df:                Validated DataFrame from an accepted sandbox run.
            producing_task_id: task_id of the step that produced this artifact.

        Returns:
            True if the artifact was registered, False if rejected.
        """
        if not isinstance(df, pd.DataFrame):
            logger.warning(
                "ArtifactStore.register(%r): value is not a DataFrame — skipping", key
            )
            return False

        if df.empty:
            logger.warning(
                "ArtifactStore.register(%r): DataFrame is empty — skipping", key
            )
            return False

        try:
            size_bytes = len(pickle.dumps(df, protocol=4))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ArtifactStore.register(%r): could not estimate size (%s) — accepting anyway",
                key,
                exc,
            )
            size_bytes = 0

        if size_bytes > MAX_ARTIFACT_BYTES:
            logger.warning(
                "ArtifactStore.register(%r): too large (%d MB > %d MB limit) — skipping",
                key,
                size_bytes // (1024 * 1024),
                MAX_ARTIFACT_BYTES // (1024 * 1024),
            )
            return False

        # Classify and build the artifact atomically — no separate metadata dict.
        artifact_type, schema_info = classify_artifact(df)

        self._artifacts[key] = Artifact(
            name=key,
            type=artifact_type,
            schema=schema_info,
            producer=producing_task_id,
            consumers=list(consumers or []),
            confidence=confidence,
            provenance=list(
                provenance or [producing_task_id] if producing_task_id else []
            ),
            value=df.copy(deep=False),
        )
        self._artifacts[key].freeze()
        logger.debug(
            "ArtifactStore: registered %r (type=%s, %d rows, %d cols, producer=%s)",
            key,
            artifact_type,
            len(df),
            len(df.columns),
            producing_task_id,
        )
        return True

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get(self, key: str) -> pd.DataFrame | None:
        """Return the DataFrame for *key*, or ``None`` if not found."""
        artifact = self._artifacts.get(key)
        return artifact.value if artifact is not None else None

    def get_artifact(self, key: str) -> Artifact | None:
        """Return the full :class:`Artifact` for *key*, or ``None``."""
        return self._artifacts.get(key)

    def available_keys(self) -> list[str]:
        """Return the list of currently registered artifact keys."""
        return list(self._artifacts.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._artifacts

    # ------------------------------------------------------------------
    # Sandbox integration
    # ------------------------------------------------------------------

    def as_namespace(self) -> dict[str, pd.DataFrame]:
        """
        Return a plain dict for injection into the sandbox ``artifacts`` global.

        The sandbox injects this as::

            global_ns["artifacts"] = artifact_store.as_namespace()

        so agent-generated code can write::

            quarter_df = artifacts["quarter_df"]
        """
        return {name: a.value for name, a in self._artifacts.items()}

    def as_namespace_for(self, keys: list[str]) -> dict[str, pd.DataFrame]:
        """
        Return a subset namespace containing only the requested *keys*.

        Used when a TaskNode declares explicit ``consumes_artifacts`` so only
        the relevant DataFrames are shipped to the sandbox (smaller IPC payload).
        Falls back gracefully — missing keys are silently omitted.
        """
        return {k: self._artifacts[k].value for k in keys if k in self._artifacts}

    # ------------------------------------------------------------------
    # Prompt description
    # ------------------------------------------------------------------

    def describe(self) -> str:
        """
        Return a prompt-ready description of all registered artifacts.

        Each artifact is listed with its ``type``, schema, and row count —
        making downstream CODE-GEN prompts self-contained.  A step that reads
        ``quarter_df`` sees its columns and type without needing to know which
        prior step produced it.

        Returns ``"(no pre-computed artifacts available yet)"`` when empty.
        """
        if not self._artifacts:
            return "(no pre-computed artifacts available yet)"

        lines = [
            "AVAILABLE ARTIFACTS (pre-validated — use these instead of re-deriving):",
        ]
        for key, a in self._artifacts.items():
            type_line = (
                f'  artifacts["{key}"]  type={a.type}  confidence={a.confidence}'
            )
            if a.schema and a.schema != ["<scalar>"]:
                schema_str = ", ".join(a.schema[:8])
                if len(a.schema) > 8:
                    schema_str += " …"
                type_line += f"  schema=[{schema_str}]"
            if a.consumers:
                consumers_str = ", ".join(a.consumers[:4])
                if len(a.consumers) > 4:
                    consumers_str += " …"
                type_line += f"  consumers=[{consumers_str}]"
            type_line += f"  ({a.rows:,} rows)"
            lines.append(type_line)

        lines.append(
            '\nUSAGE: val = artifacts.get("key"); if val is None: val = df[mask]'
        )
        return "\n".join(lines)
