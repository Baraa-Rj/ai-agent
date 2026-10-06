"""OpenAI-compatible Chat Completions provider, used here through OpenRouter."""
from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import openai

from agent.providers.base import ProviderError
from agent.types import ModelTurn, ToolCall, ToolResult, ToolSpec, Usage

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-flash"


def to_openai_tool(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


def parse_tool_call(tool_call: Any) -> ToolCall:
    """Turn an SDK tool call into a ToolCall, without raising on bad input.

    Chat Completions delivers arguments as a JSON string written by the model,
    so it can be malformed.
    """
    if tool_call.type != "function":
        return ToolCall(tool_call.id, "", None, error=f"unsupported tool call type: {tool_call.type}")
    name = tool_call.function.name
    try:
        arguments = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError as exc:
        return ToolCall(tool_call.id, name, None, error=f"arguments were not valid JSON ({exc.msg})")
    if not isinstance(arguments, dict):
        return ToolCall(tool_call.id, name, None, error="arguments must be a JSON object")
    return ToolCall(tool_call.id, name, arguments)


class OpenAICompatProvider:
    name = "openrouter"

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        system_prompt: str,
        tools: Sequence[ToolSpec],
        max_tokens: int,
    ) -> None:
        self._client = client
        self.model = model
        self._max_tokens = max_tokens
        self._tools = [to_openai_tool(spec) for spec in tools]
        self._messages: list[Any] = [{"role": "system", "content": system_prompt}]

    def start(self, user_prompt: str) -> ModelTurn:
        self._messages.append({"role": "user", "content": user_prompt})
        return self._complete()

    def send_tool_results(self, results: Sequence[ToolResult]) -> ModelTurn:
        # One "tool" message per call. This format has no error flag, so a
        # failure is carried by the "Error: ..." text of the result.
        for result in results:
            self._messages.append(
                {"role": "tool", "tool_call_id": result.call_id, "content": result.content}
            )
        return self._complete()

    def _complete(self) -> ModelTurn:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=self._messages,
                tools=self._tools,
                max_tokens=self._max_tokens,
            )
        except openai.OpenAIError as exc:
            raise ProviderError(f"OpenRouter request failed: {exc}") from exc
        if not response.choices:
            raise ProviderError("The API returned no choices.")

        choice = response.choices[0]
        message = choice.message
        # Keep the SDK's own message object in the history, so every field the
        # provider attached to the assistant turn is sent back unchanged.
        self._messages.append(message)

        usage = response.usage
        return ModelTurn(
            text=message.content or "",
            tool_calls=tuple(parse_tool_call(call) for call in message.tool_calls or ()),
            usage=Usage(
                input_tokens=(usage.prompt_tokens or 0) if usage else 0,
                output_tokens=(usage.completion_tokens or 0) if usage else 0,
            ),
            stop_reason=choice.finish_reason,
            truncated=choice.finish_reason == "length",
        )
