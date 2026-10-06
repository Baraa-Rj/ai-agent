"""The Claude provider, exercised through the real `anthropic` SDK.

No network is used: the SDK client is given a mock HTTP transport that records
each request and replays a prepared Messages API response. That checks both
directions of the translation - what the provider sends, and how it reads
what comes back - against the SDK's own request building and response parsing.
"""
import json
from types import SimpleNamespace

import anthropic
import httpx2  # the HTTP library the anthropic SDK is built on
import pytest

from agent.providers.anthropic_messages import AnthropicProvider, parse_tool_use
from agent.providers.base import ProviderError
from agent.tools import build_tools
from agent.types import ToolCall, ToolResult, Usage

SYSTEM_PROMPT = "You are a test agent."


def message(content, stop_reason="end_turn", usage=(12, 7)):
    """A Messages API response body."""
    return 200, {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": usage[0], "output_tokens": usage[1]},
    }


class FakeAnthropicAPI:
    """Stands in for the API: records requests and replays prepared responses."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.requests = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        status, body = self._responses.pop(0)
        return httpx2.Response(status, json=body)

    def body(self, index):
        return json.loads(self.requests[index].content)


def make_provider(api, max_tokens=1000):
    client = anthropic.Anthropic(
        api_key="test-key",
        base_url="https://api.anthropic.com",
        http_client=httpx2.Client(transport=httpx2.MockTransport(api)),
        max_retries=0,
    )
    specs = [tool.spec for tool in build_tools()]
    return AnthropicProvider(client, "claude-test", SYSTEM_PROMPT, specs, max_tokens)


TEXT = {"type": "text", "text": "I will look at the project first."}
THINKING = {"type": "thinking", "thinking": "List the files, then read main.py.", "signature": "sig-abc123"}
LIST_FILES = {"type": "tool_use", "id": "toolu_01", "name": "get_files_info", "input": {"directory": "."}}
READ_MAIN = {"type": "tool_use", "id": "toolu_02", "name": "get_file_content", "input": {"file_path": "main.py"}}


def test_the_first_request_follows_the_messages_api():
    api = FakeAnthropicAPI(message([{"type": "text", "text": "Nothing to do."}]))
    make_provider(api).start("fix the bug")

    request = api.requests[0]
    assert request.method == "POST" and request.url.path == "/v1/messages"
    assert request.headers["x-api-key"] == "test-key"
    assert "anthropic-version" in request.headers

    body = api.body(0)
    assert body["model"] == "claude-test"
    assert body["max_tokens"] == 1000
    assert body["system"] == SYSTEM_PROMPT  # a top-level field, not a message
    assert body["messages"] == [{"role": "user", "content": "fix the bug"}]
    # Nothing the request does not need: no forced tool choice, no sampling settings.
    assert set(body) == {"model", "max_tokens", "system", "messages", "tools"}


def test_tools_are_sent_in_anthropic_format():
    api = FakeAnthropicAPI(message([{"type": "text", "text": "ok"}]))
    make_provider(api).start("go")

    tools = api.body(0)["tools"]
    assert [tool["name"] for tool in tools] == [
        "get_files_info", "get_file_content", "write_file", "run_python_file"]
    for tool, expected in zip(tools, build_tools()):
        assert set(tool) == {"name", "description", "input_schema"}
        assert tool["description"] == expected.spec.description
        assert tool["input_schema"] == expected.spec.parameters


def test_text_and_tool_use_blocks_become_a_model_turn():
    api = FakeAnthropicAPI(message([TEXT, LIST_FILES], stop_reason="tool_use", usage=(120, 35)))
    turn = make_provider(api).start("fix the bug")

    assert turn.text == "I will look at the project first."
    assert turn.tool_calls == (ToolCall("toolu_01", "get_files_info", {"directory": "."}),)
    assert turn.usage == Usage(120, 35)
    assert (turn.stop_reason, turn.truncated) == ("tool_use", False)


def test_tool_results_go_back_with_the_assistant_turn_echoed_unchanged():
    assistant_content = [THINKING, TEXT, LIST_FILES, READ_MAIN]
    api = FakeAnthropicAPI(
        message(assistant_content, stop_reason="tool_use"),
        message([{"type": "text", "text": "Done."}]),
    )
    provider = make_provider(api)
    first = provider.start("fix the bug")
    assert [call.id for call in first.tool_calls] == ["toolu_01", "toolu_02"]  # parallel tool use

    final = provider.send_tool_results([
        ToolResult("toolu_01", "- main.py: file_size=25 bytes, is_dir=False"),
        ToolResult("toolu_02", 'Error: "main.py" is not a file', is_error=True),
    ])
    assert final.text == "Done." and final.tool_calls == ()

    body = api.body(1)
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    # The assistant turn is returned exactly as received: same tool_use ids,
    # and the thinking block with its signature intact.
    assert body["messages"][1]["content"] == assistant_content
    # Both results travel in one user message made of tool_result blocks only.
    assert body["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_01",
         "content": "- main.py: file_size=25 bytes, is_dir=False", "is_error": False},
        {"type": "tool_result", "tool_use_id": "toolu_02",
         "content": 'Error: "main.py" is not a file', "is_error": True},
    ]
    # The system prompt and the tools are sent again with every request.
    assert body["system"] == SYSTEM_PROMPT and len(body["tools"]) == 4


def test_a_conversation_keeps_growing_across_several_tool_rounds():
    api = FakeAnthropicAPI(
        message([LIST_FILES], stop_reason="tool_use"),
        message([READ_MAIN], stop_reason="tool_use"),
        message([{"type": "text", "text": "Finished."}]),
    )
    provider = make_provider(api)
    provider.start("go")
    provider.send_tool_results([ToolResult("toolu_01", "listing")])
    provider.send_tool_results([ToolResult("toolu_02", "content")])

    roles = [m["role"] for m in api.body(2)["messages"]]
    assert roles == ["user", "assistant", "user", "assistant", "user"]
    assert api.body(2)["messages"][3]["content"] == [READ_MAIN]


def test_several_text_blocks_are_joined():
    api = FakeAnthropicAPI(message([{"type": "text", "text": "Part one. "}, THINKING,
                                    {"type": "text", "text": "Part two."}]))
    assert make_provider(api).start("go").text == "Part one. Part two."


@pytest.mark.parametrize(
    "stop_reason, truncated",
    [("end_turn", False), ("tool_use", False), ("refusal", False),
     ("max_tokens", True), ("model_context_window_exceeded", True)],
)
def test_token_limit_stop_reasons_mark_the_turn_as_truncated(stop_reason, truncated):
    api = FakeAnthropicAPI(message([LIST_FILES], stop_reason=stop_reason))
    turn = make_provider(api).start("go")
    assert (turn.stop_reason, turn.truncated) == (stop_reason, truncated)


def test_a_refusal_is_a_turn_without_text_or_tool_calls():
    api = FakeAnthropicAPI(message([], stop_reason="refusal"))
    turn = make_provider(api).start("go")
    assert (turn.text, turn.tool_calls, turn.stop_reason) == ("", (), "refusal")


def test_api_errors_are_raised_as_provider_errors():
    api = FakeAnthropicAPI((401, {"type": "error", "error": {
        "type": "authentication_error", "message": "invalid x-api-key"}}))
    with pytest.raises(ProviderError, match="Anthropic request failed.*invalid x-api-key"):
        make_provider(api).start("go")


def test_connection_failures_are_raised_as_provider_errors():
    def unreachable(request):
        raise httpx2.ConnectError("no route to host", request=request)

    with pytest.raises(ProviderError, match="Anthropic request failed"):
        make_provider(unreachable).start("go")


def test_tool_input_that_is_not_an_object_is_reported_not_executed():
    block = SimpleNamespace(type="tool_use", id="toolu_09", name="write_file", input=["oops"])
    assert parse_tool_use(block) == ToolCall(
        "toolu_09", "write_file", None, error="arguments must be a JSON object")
