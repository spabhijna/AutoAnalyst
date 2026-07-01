"""
Configuration constants for the AutoAnalyst agent.

All tunables live here — no magic strings scattered across the codebase.
Import from this module rather than hardcoding values elsewhere.
"""

# ── Groq models ───────────────────────────────────────────────────────────────

# Default model for PLAN and OBSERVE stages
GROQ_DEFAULT_MODEL: str = "llama-3.3-70b-versatile"

# Model for CODE-GEN stage — llama-3.3-70b-versatile (free tier)
# Codestral is not available on Groq's free tier; use OpenRouter for Codestral.
GROQ_CODEGEN_MODEL: str = "llama-3.3-70b-versatile"

# Model for the CRITIQUE stage — a stronger model for independent auditing
GROQ_CRITIQUE_MODEL: str = "openai/gpt-oss-120b"

# ── Cerebras fallback ─────────────────────────────────────────────────────────

# Cerebras base URL (OpenAI-compatible)
CEREBRAS_BASE_URL: str = "https://api.cerebras.ai/v1"

# Model for CODE-GEN fallback on Cerebras
CEREBRAS_CODEGEN_FALLBACK_MODEL: str = "openai/gpt-oss-120b"

# Model for CRITIQUE fallback on Cerebras (same model, different host)
CEREBRAS_CRITIQUE_FALLBACK_MODEL: str = "openai/gpt-oss-120b"

# ── OpenRouter fallback ───────────────────────────────────────────────────────

# OpenRouter base URL (OpenAI-compatible)
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

# HTTP-Referer header required by OpenRouter
OPENROUTER_REFERER: str = "https://github.com/abhijnasp/AutoAnalyst"

# Model for PLAN fallback on OpenRouter
OPENROUTER_PLAN_FALLBACK_MODEL: str = "meta-llama/llama-3.3-70b-instruct:free"

# Model for CODE-GEN fallback on OpenRouter (code-focused model)
OPENROUTER_CODEGEN_FALLBACK_MODEL: str = "mistralai/codestral-2501:free"

# ── Gemini (Google AI Studio) fallback ────────────────────────────────────────

# Gemini model used only when Groq + Cerebras are both unavailable
# gemini-2.5-flash has a 1M context window — useful for late-stage critique calls
GEMINI_FALLBACK_MODEL: str = "gemini-2.5-flash"

# ── Retry / resilience ────────────────────────────────────────────────────────

# Seconds to wait between a failed provider attempt and its retry
RETRY_BACKOFF_SECONDS: float = 1.5

# Backward-compatible alias
GROQ_RETRY_BACKOFF_SECONDS: float = RETRY_BACKOFF_SECONDS

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