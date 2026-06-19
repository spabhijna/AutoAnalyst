"""
Explicit modeling of the execution environment capabilities and constraints.
"""
from dataclasses import dataclass, field

@dataclass
class EnvironmentManifest:
    """
    A definitive source of truth about what the sandbox environment supports.
    Passed to the generator LLM so it doesn't have to guess or assume.
    """
    allowed_modules: list[str] = field(default_factory=lambda: ["pd (pandas)", "np (numpy)"])
    disallowed_operations: list[str] = field(default_factory=lambda: [
        "Network requests (e.g. requests, urllib)",
        "File system I/O (e.g. open(), os, shutil)",
        "Process execution (e.g. subprocess, sys)",
        "Dynamic code execution (e.g. eval(), exec())",
        "Importing any module not in allowed_modules"
    ])
    available_globals: list[str] = field(default_factory=lambda: ["df (the pandas DataFrame)", "pd", "np"])
    required_conventions: list[str] = field(default_factory=lambda: [
        "You MUST assign your final answer to a variable named `result`"
    ])
    
    # Known capability limits that have been discovered during execution
    discovered_limitations: list[str] = field(default_factory=list)

    def add_limitation(self, limitation: str) -> None:
        """Add a discovered limitation so future generations avoid it."""
        if limitation not in self.discovered_limitations:
            self.discovered_limitations.append(limitation)

    def as_prompt_text(self) -> str:
        """Format the manifest into a strict instruction block for the LLM."""
        lines = [
            "ENVIRONMENT CAPABILITIES & CONSTRAINTS",
            "--------------------------------------",
            "AVAILABLE MODULES: " + ", ".join(self.allowed_modules),
            "AVAILABLE GLOBALS: " + ", ".join(self.available_globals),
            "REQUIRED CONVENTIONS:",
        ]
        for conv in self.required_conventions:
            lines.append(f"  - {conv}")
        
        lines.append("DISALLOWED OPERATIONS (DO NOT USE):")
        for dis in self.disallowed_operations:
            lines.append(f"  - {dis}")

        if self.discovered_limitations:
            lines.append("KNOWN LIMITATIONS (AVOID THESE):")
            for lim in self.discovered_limitations:
                lines.append(f"  - {lim}")
        
        return "\n".join(lines)


class CapabilityError(Exception):
    """
    Raised when the agent attempts an operation that the environment explicitly
    does not support (e.g., importing a blocked module).
    """
    pass
