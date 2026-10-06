"""The agent loop: ask the model, run the tools it asks for, send the results back."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from agent.providers.base import Provider
from agent.tools import ToolRegistry
from agent.types import ModelTurn, ToolCall, ToolResult, Usage

DEFAULT_MAX_ITERATIONS = 20

COMPLETED = "completed"
MAX_ITERATIONS = "max_iterations"
TRUNCATED = "truncated"


@dataclass(frozen=True)
class AgentResult:
    status: str  # COMPLETED, MAX_ITERATIONS or TRUNCATED
    text: str  # the model's final answer (empty unless completed)
    model_calls: int
    tool_calls: int
    usage: Usage
    stop_reason: str | None = None


class Observer:
    """Receives what happens during a run. Subclasses override what they need."""

    def on_model_turn(self, number: int, turn: ModelTurn) -> None:
        pass

    def on_tool_result(self, call: ToolCall, result: ToolResult) -> None:
        pass


def run_agent(
    provider: Provider,
    registry: ToolRegistry,
    user_prompt: str,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    observers: Iterable[Observer] = (),
) -> AgentResult:
    """Run one request to completion, or until a limit stops it.

    `max_iterations` caps the number of model calls. When the cap is reached
    and the model is still asking for tools, those calls are not executed:
    their results could never be sent back.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    observers = list(observers)
    input_tokens = output_tokens = tool_calls = 0

    def result(status: str, turn: ModelTurn, model_calls: int) -> AgentResult:
        return AgentResult(
            status=status,
            text=turn.text if status == COMPLETED else "",
            model_calls=model_calls,
            tool_calls=tool_calls,
            usage=Usage(input_tokens, output_tokens),
            stop_reason=turn.stop_reason,
        )

    turn = provider.start(user_prompt)
    for model_calls in range(1, max_iterations + 1):
        input_tokens += turn.usage.input_tokens
        output_tokens += turn.usage.output_tokens
        for observer in observers:
            observer.on_model_turn(model_calls, turn)

        if turn.truncated:
            # A cut-off response can contain half a tool call. Never act on it.
            return result(TRUNCATED, turn, model_calls)
        if not turn.tool_calls:
            return result(COMPLETED, turn, model_calls)
        if model_calls == max_iterations:
            break

        results = []
        for call in turn.tool_calls:
            tool_result = registry.execute(call)
            tool_calls += 1
            for observer in observers:
                observer.on_tool_result(call, tool_result)
            results.append(tool_result)
        turn = provider.send_tool_results(results)

    return result(MAX_ITERATIONS, turn, max_iterations)
