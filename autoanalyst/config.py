"""
Configuration module for AutoAnalyst.

Loads all settings from environment variables (via a .env file).
No other module should read os.environ or dotenv directly — import
from here instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file)
_ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(_ENV_PATH)


def _require(key: str) -> str:
    """Return the value of *key* from the environment, raising if absent."""
    value = os.environ.get(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Copy .env.example to .env and fill in your credentials."
        )
    return value


def _optional(key: str, default: str) -> str:
    """Return the value of *key* from the environment, or *default*."""
    return os.environ.get(key, default)


@dataclass(frozen=True)
class PipelineConfig:
    """
    Typed, immutable configuration for a pipeline run.

    All values are sourced from environment variables so that no secrets
    or tunables are hard-coded in source files.

    Attributes:
        groq_api_key:  API key for the Groq LLM (generation stages).
        groq_model:    Groq model identifier string.
        gemini_api_key: API key for Google Gemini (critique stage).
        gemini_model:  Gemini model identifier string.
        max_retries:   Hard cap on PLAN→EXECUTE→CRITIQUE retries per sub-question.
        log_dir:       Directory path where JSONL run logs are written.
        dry_run:       When True, LLM calls are skipped and stubs are returned.
    """

    groq_api_key: str
    groq_model: str
    gemini_api_key: str
    gemini_model: str
    max_retries: int
    log_dir: Path
    dry_run: bool = False

    @classmethod
    def from_env(cls, *, dry_run: bool = False) -> "PipelineConfig":
        """
        Build a :class:`PipelineConfig` from the current environment.

        In *dry_run* mode the API keys are not required so the pipeline
        can be exercised without live credentials.

        Args:
            dry_run: Skip LLM calls and return stub responses.

        Returns:
            A fully-populated :class:`PipelineConfig` instance.

        Raises:
            EnvironmentError: If a required env variable is missing and
                              *dry_run* is False.
        """
        if dry_run:
            groq_key = os.environ.get("GROQ_API_KEY", "dry-run-placeholder")
            gemini_key = os.environ.get("GEMINI_API_KEY", "dry-run-placeholder")
        else:
            groq_key = _require("GROQ_API_KEY")
            gemini_key = _require("GEMINI_API_KEY")

        log_dir = Path(_optional("LOG_DIR", "runs"))
        log_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            groq_api_key=groq_key,
            groq_model=_optional("GROQ_MODEL", "llama-3.3-70b-versatile"),
            gemini_api_key=gemini_key,
            gemini_model=_optional("GEMINI_MODEL", "gemini-2.0-flash"),
            max_retries=int(_optional("MAX_RETRIES", "3")),
            log_dir=log_dir,
            dry_run=dry_run,
        )
