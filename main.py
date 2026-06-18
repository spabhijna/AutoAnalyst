"""
AutoAnalyst CLI entry point.

Phase 2 will wire this to agent/pipeline.py once the sandbox and
pipeline loop are implemented. For now it prints project status.

Usage:
  python main.py
"""

from __future__ import annotations


def main() -> None:
    """Entry point — will invoke agent/pipeline.py in Phase 2."""
    print("AutoAnalyst — Self-Correcting Data Pipeline Agent")
    print("Phase 1 complete: LLM client and synthetic data ready.")
    print("Run `uv run pytest tests/ -v` to verify.")
    print("Phase 2 will add: sandbox, pipeline loop, and CLI flags.")


if __name__ == "__main__":
    main()
