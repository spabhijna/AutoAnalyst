# AutoAnalyst Architecture Guide for Coding Agents

> **Notice to AI Agents**: This document is the definitive architectural reference for AutoAnalyst. Read this to understand how the components interact before modifying the codebase.

## 1. System Overview
AutoAnalyst is a **Self-Correcting Data Pipeline Agent** designed to run analytical queries on datasets. It has evolved from an LLM-driven execution loop into a **hybrid Execution Framework** relying heavily on deterministic capabilities, typed artifacts, and dependency-graph execution.

The high-level data flow is:
```text
User Goal
→ Planner
→ Execution Graph (TaskNodes)
→ Capability Registry & Execution Engine
→ Deterministic Executors (or Sandbox CodeGen Fallback)
→ Artifact Store & Validator
→ Working Memory
→ Summary
```

## 2. Core Execution Pipeline
The agent analyzes data using a multi-stage progressive loop (`agent/loop.py`):

1. **PLAN**: The planner parses the user goal into an `ExecutionGraph` containing `TaskNode` objects. Each node explicitly defines dependencies (`consumes_artifacts`) and expected outputs (`produces_artifact`).
2. **EXECUTE (Graph-Driven)**: 
   - The `ExecutionEngine` traverses the graph. For each node, it first checks the `CapabilityRegistry` for predefined deterministic capabilities.
   - If a capability matches, it executes deterministically, **completely bypassing LLM code generation**.
3. **CODE-GEN (Fallback)**: If no deterministic capability matches, the system falls back to generating Python code. The LLM only receives `AVAILABLE ARTIFACTS` and `WorkingMemory` context.
4. **SANDBOX**: Agent-generated code runs via `agent.sandbox.run_sandboxed()`. It runs in a process-isolated, restricted environment and explicitly extracts a `result` variable.
5. **OBSERVE (Deterministic)**: The `observer.py` performs deterministic data-quality checks (nulls, duplicates, outliers) directly on the input dataset, injecting hard facts into the critique stage.
6. **CRITIQUE**: An independent LLM call verifies the code logic (STEP 0) and reviews the sandbox results against the observation facts. It parses actions (`accept`, `revise_code`, or `revise_plan`) and fails defensively on parse errors.

## 3. State Management (Crucial Constraints)
State is strictly separated to prevent LLM hallucination and error cascading. When modifying the loop or prompts, **do not mix these state views**:
- **`debug_history` (`list[AttemptRecord]`)**: Contains *every* attempt, exception, and retry. ONLY used for the final audit log JSONL. **DO NOT pass this to the summary or downstream planning prompts**.
- **`working_memory` (`WorkingMemory`)**: Contains ONLY accepted `Finding` objects. This is the operational memory.
- **`tasks` (`list[TaskNode]`)**: The explicit dependency graph. Tasks missing parent dependencies are marked skipped/failed and execution aborts predictably.

## 4. The Artifact System
AutoAnalyst uses an in-process `ArtifactStore` to cache intermediate analytical results (e.g., `quarter_df`, `region_revenue`). 
- **`Artifact` Object**: Bundles the payload (`pd.DataFrame`), `type`, `schema`, `producer`, `consumers`, and `confidence`.
- **Integrity**: Artifacts are validated by the `CapabilityValidator` before registration (ensuring schema/contract compliance). Once registered, they are immutable.
- **Why?**: It prevents repeated recomputations (e.g., repeatedly applying date filters) and anchors summary claims to strict `artifact_key` references.

## 5. Implementing New Features (Workflow Rules)
If you are tasked with adding a new analytical feature:
1. **Prefer Capabilities over Prompts**: Do not try to solve it by tweaking the `CODE-GEN` prompt to write better code. Instead, define a new deterministic executor in `executors.py`, wire it in `capabilities.py`, and register it in the `CapabilityRegistry`.
2. **Use the Execution Graph**: Ensure your tasks declare explicit inputs and outputs so the `ExecutionEngine` can resolve them topologically.
3. **Enforce Contracts**: Update `validator.py` if your capability introduces new output schema contracts.

## 6. Hard Engineering Constraints
- **Never call LLM SDKs directly**: All LLM interactions must go through `llm_client.call_llm(prompt, role)`. 
- **No Unbounded Loops**: `max_retries_per_step` is strictly enforced. If exhausted, raise `PipelineAbortedError`. Do not force-accept broken results.
- **Test-Driven**: Update tests in `tests/` whenever you modify the loop or add capabilities. Ensure `pytest` passes with 100% success before completing your task.
- **Log Everything**: Ensure any new loop stages write their state to the structured run log (`logger.py`), as this trace is the primary project deliverable.
