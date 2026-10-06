"""Claude provider: the Anthropic Messages API through the official `anthropic` SDK."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import anthropic

from agent.providers.base import ProviderError
from agent.types import ModelTurn, ToolCall, ToolResult, ToolSpec, Usage

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"

# Stop reasons that mean the response is incomplete and must not be acted on.
_TRUNCATED_STOP_REASONS = ("max_tokens", "model_context_window_exceeded")


def to_anthropic_tool(spec: ToolSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.parameters,
    }


def to_tool_result_block(result: ToolResult) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": result.call_id,
        "content": result.content,
        "is_error": result.is_error,
    }


def parse_tool_use(block: Any) -> ToolCall:
    """Turn a `tool_use` content block into a ToolCall.

    The API delivers `input` already parsed, so there is no JSON to decode,
    but its shape still depends on the model.
    """
    if not isinstance(block.input, dict):
        return ToolCall(block.id, block.name, None, error="arguments must be a JSON object")
    return ToolCall(block.id, block.name, dict(block.input))


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        system_prompt: str,
        tools: Sequence[ToolSpec],
        max_tokens: int,
    ) -> None:
        self._client = client
        self.model = model
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._tools = [to_anthropic_tool(spec) for spec in tools]
        self._messages: list[dict[str, Any]] = []

    def start(self, user_prompt: str) -> ModelTurn:
        self._messages.append({"role": "user", "content": user_prompt})
        return self._complete()

    def send_tool_results(self, results: Sequence[ToolResult]) -> ModelTurn:
        # Every result for the previous assistant turn goes back in a single
        # user message that contains tool_result blocks and nothing else.
        self._messages.append(
            {"role": "user", "content": [to_tool_result_block(result) for result in results]}
        )
        return self._complete()

    def _complete(self) -> ModelTurn:
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self._max_tokens,
                system=self._system_prompt,
                tools=self._tools,
                messages=self._messages,
            )
        except anthropic.AnthropicError as exc:
            raise ProviderError(f"Anthropic request failed: {exc}") from exc

        # Send the assistant turn back unchanged on the next request. That keeps
        # the tool_use ids the results refer to, and any thinking blocks the
        # model produced, exactly as the API returned them.
        self._messages.append({"role": "assistant", "content": response.content})

        return ModelTurn(
            text="".join(block.text for block in response.content if block.type == "text"),
            tool_calls=tuple(
                parse_tool_use(block) for block in response.content if block.type == "tool_use"
            ),
            usage=Usage(response.usage.input_tokens, response.usage.output_tokens),
            stop_reason=response.stop_reason,
            truncated=response.stop_reason in _TRUNCATED_STOP_REASONS,
        )
