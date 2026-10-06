"""Test doubles shared by the test modules."""
from __future__ import annotations

from collections.abc import Iterable, Sequence

from agent.types import ModelTurn, ToolCall, ToolResult


class ScriptedProvider:
    """A provider that replays prepared model turns and records what it is sent."""

    name = "scripted"
    model = "test-model"

    def __init__(self, turns: Iterable[ModelTurn]) -> None:
        self._turns = iter(turns)
        self.prompts: list[str] = []
        self.results: list[list[ToolResult]] = []

    def start(self, user_prompt: str) -> ModelTurn:
        self.prompts.append(user_prompt)
        return next(self._turns)

    def send_tool_results(self, results: Sequence[ToolResult]) -> ModelTurn:
        self.results.append(list(results))
        return next(self._turns)


def call(name: str, call_id: str = "call_1", **arguments) -> ToolCall:
    return ToolCall(call_id, name, arguments)
