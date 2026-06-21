"""
Typed state objects for the agent loop.

This module separates three concerns that must not be mixed:
- debug history: every attempt, including failed code and rejected critiques
- working memory: validated findings available to downstream stages
- summary evidence: the sanitized final answers visible to the reporter stage
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


Trust = Literal["high", "low", "unknown"]
TaskStatus = Literal["pending", "running", "accepted", "failed", "skipped"]
ErrorCategory = Literal["execution_error", "capability_error", "logic_error", "none"]



def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExecutionDifficulty:
    """
    Code-generation/execution difficulty metadata.

    This is intentionally separate from answer confidence. A result can require
    several code-generation attempts and still be a high-confidence answer once
    it is accepted by critique.
    """

    attempts: int
    had_execution_errors: bool
    final_error_type: str | None = None


@dataclass
class AttemptRecord:
    """One debug/audit record for a single codegen-execute-critique attempt."""

    task_id: str
    sub_question: str
    iteration: int
    code: str
    sandbox_result: dict[str, Any]
    critique: str
    action: str
    trust: Trust
    error_category: ErrorCategory = "none"
    timestamp: str = field(default_factory=utc_now_iso)

    def to_legacy_dict(self) -> dict[str, Any]:
        """Return the historical dict shape used by existing log tests."""
        return {
            "task_id": self.task_id,
            "sub_question": self.sub_question,
            "iteration": self.iteration,
            "code": self.code,
            "sandbox_result": self.sandbox_result,
            "critique": self.critique,
            "action": self.action,
            "trust": self.trust,
            "timestamp": self.timestamp,
        }


@dataclass
class Finding:
    """
    A validated analytical answer available to downstream stages.

    ``answer_confidence`` reflects the critic's final trust verdict only.
    ``execution_difficulty`` records how hard it was to produce the accepted
    answer and must not be used as answer confidence.

    ``produced_artifact_name`` is the key under which this step's output
    DataFrame was registered in the ArtifactStore (e.g. ``"quarter_df"``),
    or None if the step returned a scalar / string answer rather than a
    reusable DataFrame.
    """

    task_id: str
    question: str
    answer: str
    answer_confidence: Trust
    evidence: str
    artifact_key: str
    dependencies: list[str] = field(default_factory=list)
    execution_difficulty: ExecutionDifficulty | None = None
    source_iteration: int = 0
    timestamp: str = field(default_factory=utc_now_iso)
    # ArtifactStore key for the DataFrame this step emitted (None for scalars).
    produced_artifact_name: str | None = None

    def to_summary_dict(self) -> dict[str, str]:
        """
        Return the fields the summary stage is allowed to see.

        Includes ``artifact_key`` so the reporter LLM can ground each claim
        in a specific validated finding rather than synthesising from memory.
        """
        return {
            "question": self.question,
            "answer": self.answer,
            "trust": self.answer_confidence,
            "artifact_key": self.artifact_key,
        }

    def to_working_memory_line(self) -> str:
        """Return a compact line for CODE-GEN/CRITIQUE context."""
        artifact_hint = ""
        if self.produced_artifact_name:
            artifact_hint = f" [artifact: artifacts[\"{self.produced_artifact_name}\"] available]"
        return (
            f"• {self.artifact_key}: {self.question} → {self.answer} "
            f"(confidence={self.answer_confidence}){artifact_hint}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for logs."""
        return asdict(self)


@dataclass
class TaskNode:
    """
    A planned task and its explicit dependencies.

    ``produces_artifact`` and ``consumes_artifacts`` are declared by the
    planner and drive two behaviours in the execution loop:

    * ``produces_artifact``: after the step is accepted, its DataFrame variable
      (named after this value) is extracted from the sandbox and registered in
      the ArtifactStore.
    * ``consumes_artifacts``: before CODE-GEN, the named DataFrames are fetched
      from the ArtifactStore and injected into the sandbox ``artifacts`` dict,
      making them available without re-computation.
    """

    task_id: str
    question: str
    depends_on: list[str] = field(default_factory=list)
    status: TaskStatus = "pending"
    failure_reason: str = ""
    # Artifact lifecycle declarations (from structured plan)
    produces_artifact: str | None = None
    consumes_artifacts: list[str] = field(default_factory=list)


@dataclass
class WorkingMemory:
    """
    Validated operational memory used by later stages.

    Debug attempts never enter this structure. Only accepted findings are
    stored, which prevents rejected attempts from contaminating future
    reasoning or final summaries.
    """

    findings: dict[str, Finding] = field(default_factory=dict)
    artifact_index: dict[str, str] = field(default_factory=dict)

    def add_finding(self, finding: Finding) -> None:
        """Store a validated finding and index its reusable artifact key."""
        self.findings[finding.task_id] = finding
        self.artifact_index[finding.artifact_key] = finding.task_id

    def get(self, task_id: str) -> Finding | None:
        """Return the finding for *task_id*, if it has been accepted."""
        return self.findings.get(task_id)

    def dependency_failures(self, task: TaskNode) -> list[str]:
        """Return dependency IDs that are not available in working memory."""
        return [dep for dep in task.depends_on if dep not in self.findings]

    def accepted_findings(self, plan_order: list[str]) -> list[Finding]:
        """Return accepted findings in plan order."""
        return [
            self.findings[task_id]
            for task_id in plan_order
            if task_id in self.findings
        ]

    def as_prompt_text(self, plan_order: list[str]) -> str:
        """Return high-confidence working memory for CODE-GEN/CRITIQUE prompts."""
        findings = self.accepted_findings(plan_order)
        high_confidence = [
            finding.to_working_memory_line()
            for finding in findings
            if finding.answer_confidence == "high"
        ]
        if not high_confidence:
            return "(no validated findings yet)"
        return "\n".join(high_confidence)

    def summary_payload(self, plan_order: list[str]) -> list[dict[str, str]]:
        """Return sanitized final evidence for the summary stage."""
        return [
            finding.to_summary_dict()
            for finding in self.accepted_findings(plan_order)
        ]


@dataclass
class TaskMemory:
    """
    Cumulative constraints and learnings for a specific task.
    This ensures that retries do not forget past mistakes.
    """
    task_id: str
    constraints: list[str] = field(default_factory=list)

    def add_constraint(self, constraint: str) -> None:
        if constraint not in self.constraints:
            self.constraints.append(constraint)
            
    def as_prompt_text(self) -> str:
        if not self.constraints:
            return "(No constraints discovered yet)"
        return "\n".join(f"- {c}" for c in self.constraints)

