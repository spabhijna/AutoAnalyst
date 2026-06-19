"""
Configuration constants for the AutoAnalyst agent.

All tunables live here — no magic strings scattered across the codebase.
Import from this module rather than hardcoding values elsewhere.
"""

# ── Groq models ───────────────────────────────────────────────────────────────

# Default model for PLAN and CODEGEN stages
GROQ_DEFAULT_MODEL: str = "llama-3.3-70b-versatile"

# Model for the CRITIQUE stage — a stronger model for independent auditing
GROQ_CRITIQUE_MODEL: str = "openai/gpt-oss-120b"

# ── Gemini fallback ───────────────────────────────────────────────────────────

# Gemini model used only when Groq is unavailable
GEMINI_FALLBACK_MODEL: str = "gemini-2.0-flash"

# ── Retry / resilience ────────────────────────────────────────────────────────

# Seconds to wait between the first Groq failure and the Groq retry attempt
GROQ_RETRY_BACKOFF_SECONDS: float = 1.5

# Hard cap on PLAN → CODEGEN → EXECUTE → OBSERVE → CRITIQUE iterations
# per sub-question (default; can be overridden via env MAX_RETRIES)
MAX_PIPELINE_RETRIES: int = 5

# Wall-clock timeout (seconds) for sandbox code execution
SANDBOX_TIMEOUT_SECONDS: float = 30.0

# ── Logging ───────────────────────────────────────────────────────────────────

# Directory where JSONL run logs are written
LOG_DIR: str = "runs"

# ── Roles (informational — used for logging, not provider routing) ────────────

ROLE_PLAN: str = "plan"
ROLE_CODEGEN: str = "codegen"
ROLE_CRITIQUE: str = "critique"
ROLE_OBSERVE: str = "observe"

VALID_ROLES: frozenset[str] = frozenset({ROLE_PLAN, ROLE_CODEGEN, ROLE_CRITIQUE, ROLE_OBSERVE})
