"""Provider-neutral data types.

The agent loop and the tools only ever see these. Each model provider
translates between them and its own wire format.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    """What the model is told about a tool: its name, purpose and arguments."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for an object


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation requested by the model.

    `arguments` is None when the provider could not decode what the model
    sent. `error` then says why, and the registry reports that back to the
    model instead of running anything.
    """

    id: str
    name: str
    arguments: dict[str, Any] | None
    error: str | None = None


@dataclass(frozen=True)
class ToolResult:
    """The outcome of one tool call, sent back to the model."""

    call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class ModelTurn:
    """One response from the model."""

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = Usage()
    stop_reason: str | None = None  # the provider's own value, kept for logging
    truncated: bool = False  # the response was cut off by a token limit
