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
   ``describe()`` returns a human-readable block listing every registered
   artifact with its row count, column names, and producing task ID. This
   is injected into every CODE-GEN and REPAIR prompt so the LLM knows
   what pre-validated DataFrames are available without re-deriving them.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

# Refuse artifacts larger than this to keep sandbox IPC safe.
# 50 MB is generous for typical analytical DataFrames (100 k–1 M rows).
MAX_ARTIFACT_BYTES: int = 50 * 1024 * 1024  # 50 MB


@dataclass
class ArtifactMeta:
    """Lightweight metadata stored alongside each registered DataFrame."""

    key: str
    rows: int
    columns: list[str]
    dtypes: dict[str, str]
    producing_task_id: str = ""

    def column_summary(self, max_cols: int = 8) -> str:
        """Return a compact column list, truncated for prompt embedding."""
        cols = self.columns[:max_cols]
        suffix = f", … (+{len(self.columns) - max_cols} more)" if len(self.columns) > max_cols else ""
        return ", ".join(cols) + suffix


@dataclass
class ArtifactStore:
    """
    Registry mapping artifact_key → validated pd.DataFrame.

    Populated as tasks are accepted during the agent loop. Later steps read
    their dependency DataFrames from here instead of re-applying filters and
    aggregations from the raw input DataFrame.

    Thread safety: the agent loop is single-threaded; no locking is needed.
    """

    _store: dict[str, pd.DataFrame] = field(default_factory=dict)
    _meta: dict[str, ArtifactMeta] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def register(
        self,
        key: str,
        df: pd.DataFrame,
        producing_task_id: str = "",
    ) -> bool:
        """
        Register *df* under *key*.

        Performs three validations before accepting:
          1. Value is a non-None pd.DataFrame.
          2. DataFrame is non-empty.
          3. Pickled size is within MAX_ARTIFACT_BYTES.

        Args:
            key:               Snake_case name declared by the planner.
            df:                Validated DataFrame from an accepted sandbox run.
            producing_task_id: task_id of the step that produced this artifact.

        Returns:
            True if the artifact was registered, False if rejected.
        """
        if not isinstance(df, pd.DataFrame):
            logger.warning("ArtifactStore.register(%r): value is not a DataFrame — skipping", key)
            return False

        if df.empty:
            logger.warning("ArtifactStore.register(%r): DataFrame is empty — skipping", key)
            return False

        try:
            size_bytes = len(pickle.dumps(df, protocol=4))
        except Exception as exc:  # noqa: BLE001
            logger.warning("ArtifactStore.register(%r): could not estimate size (%s) — accepting anyway", key, exc)
            size_bytes = 0

        if size_bytes > MAX_ARTIFACT_BYTES:
            logger.warning(
                "ArtifactStore.register(%r): too large (%d MB > %d MB limit) — skipping",
                key,
                size_bytes // (1024 * 1024),
                MAX_ARTIFACT_BYTES // (1024 * 1024),
            )
            return False

        # Shallow copy: column data is shared but the dict and index are independent.
        self._store[key] = df.copy(deep=False)
        self._meta[key] = ArtifactMeta(
            key=key,
            rows=len(df),
            columns=list(df.columns),
            dtypes={c: str(df[c].dtype) for c in df.columns},
            producing_task_id=producing_task_id,
        )
        logger.debug(
            "ArtifactStore: registered %r (%d rows, %d cols, task=%s)",
            key, len(df), len(df.columns), producing_task_id,
        )
        return True

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get(self, key: str) -> pd.DataFrame | None:
        """Return the DataFrame registered under *key*, or None."""
        return self._store.get(key)

    def available_keys(self) -> list[str]:
        """Return the list of currently registered artifact keys."""
        return list(self._store.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._store

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
        return dict(self._store)  # shallow copy of the registry

    def as_namespace_for(self, keys: list[str]) -> dict[str, pd.DataFrame]:
        """
        Return a subset namespace containing only the requested *keys*.

        Used when a TaskNode declares explicit ``consumes_artifacts`` so only
        the relevant DataFrames are shipped to the sandbox (smaller IPC payload).
        Falls back gracefully — missing keys are silently omitted.
        """
        return {k: self._store[k] for k in keys if k in self._store}

    # ------------------------------------------------------------------
    # Prompt description
    # ------------------------------------------------------------------

    def describe(self) -> str:
        """
        Return a prompt-ready description of all registered artifacts.

        Injected into every CODE-GEN and REPAIR prompt so the LLM knows which
        pre-validated DataFrames are available and how to access them.

        Returns ``"(no pre-computed artifacts available yet)"`` when empty.
        """
        if not self._meta:
            return "(no pre-computed artifacts available yet)"

        lines = [
            "AVAILABLE ARTIFACTS (pre-validated DataFrames — use these instead of re-filtering df):",
        ]
        for key, meta in self._meta.items():
            producer = f" | produced by {meta.producing_task_id}" if meta.producing_task_id else ""
            lines.append(
                f'  artifacts["{key}"]  — {meta.rows:,} rows'
                f" | columns: {meta.column_summary()}"
                f"{producer}"
            )
        lines.append(
            "\nUSAGE: read from artifacts first; only fall back to re-filtering df if the"
            " artifact you need is not listed above.\n"
            "Example:  quarter_df = artifacts.get(\"quarter_df\") or df[mask]"
        )
        return "\n".join(lines)
