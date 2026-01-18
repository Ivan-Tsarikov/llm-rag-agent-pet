"""Lightweight tool registry and specs for the agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional


ToolHandler = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


@dataclass(frozen=True)
class ToolSpec:
    """Metadata and handler for a tool callable."""
    name: str
    description: str
    args_schema: Dict[str, Any]  # простой JSON schema-подобный словарь
    handler: ToolHandler
    timeout_s: float = 8.0


class ToolError(RuntimeError):
    """Tool-related error raised by the agent."""
    pass


class ToolRegistry:
    """Registry for tool specs with allowlist helpers."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        """Register a tool spec by name."""
        self._tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        """Fetch a tool spec by name if present."""
        return self._tools.get(name)

    def list_specs(self) -> list[ToolSpec]:
        """Return all registered tool specs."""
        return list(self._tools.values())

    def allowlist(self) -> set[str]:
        """Return a set of allowed tool names."""
        return set(self._tools.keys())
