"""The interface every model provider implements."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from agent.types import ModelTurn, ToolResult


class ProviderError(Exception):
    """The model API failed, or returned something the agent cannot continue from."""


class ConfigurationError(Exception):
    """The provider cannot be set up: unknown name, missing API key, and so on."""


class Provider(Protocol):
    """One conversation with one model.

    A provider keeps the message history in its own wire format and returns
    neutral `ModelTurn` objects, so the agent loop is the same for every model.
    """

    name: str
    model: str

    def start(self, user_prompt: str) -> ModelTurn:
        """Send the user's request and return the model's first response."""
        ...

    def send_tool_results(self, results: Sequence[ToolResult]) -> ModelTurn:
        """Return the results of the tool calls from the previous response."""
        ...
