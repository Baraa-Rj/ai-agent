"""The OpenRouter provider, exercised through the real `openai` SDK.

As in the Claude provider tests, the SDK client gets a mock HTTP transport, so
requests are built and responses parsed by the SDK itself without any network.
"""
import json

import httpx  # the HTTP library the openai SDK is built on
import openai
import pytest

from agent.providers.base import ProviderError
from agent.providers.openai_compat import OPENROUTER_BASE_URL, OpenAICompatProvider
from agent.tools import build_tools
from agent.types import ToolCall, ToolResult, Usage

SYSTEM_PROMPT = "You are a test agent."


def tool_call(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def completion(content=None, tool_calls=None, finish_reason="stop", usage=(11, 3)):
    """A Chat Completions response body."""
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return 200, {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test/model",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1],
                  "total_tokens": usage[0] + usage[1]},
    }


class FakeChatAPI:
    """Stands in for the API: records requests and replays prepared responses."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status, body = self._responses.pop(0)
        return httpx.Response(status, json=body)

    def body(self, index):
        return json.loads(self.requests[index].content)


def make_provider(api, max_tokens=1000):
    client = openai.OpenAI(
        api_key="test-key",
        base_url=OPENROUTER_BASE_URL,
        http_client=httpx.Client(transport=httpx.MockTransport(api)),
        max_retries=0,
    )
    specs = [tool.spec for tool in build_tools()]
    return OpenAICompatProvider(client, "test/model", SYSTEM_PROMPT, specs, max_tokens)


LIST_FILES = tool_call("call_1", "get_files_info", '{"directory": "."}')
READ_MAIN = tool_call("call_2", "get_file_content", '{"file_path": "main.py"}')


def test_the_first_request_follows_the_chat_completions_api():
    api = FakeChatAPI(completion("Nothing to do."))
    make_provider(api).start("fix the bug")

    request = api.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"

    body = api.body(0)
    assert body["model"] == "test/model" and body["max_tokens"] == 1000
    assert body["messages"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "fix the bug"},
    ]
    assert set(body) == {"model", "max_tokens", "messages", "tools"}


def test_tools_are_sent_in_function_calling_format():
    api = FakeChatAPI(completion("ok"))
    make_provider(api).start("go")

    tools = api.body(0)["tools"]
    assert len(tools) == 4
    for tool, expected in zip(tools, build_tools()):
        assert tool == {"type": "function", "function": {
            "name": expected.spec.name,
            "description": expected.spec.description,
            "parameters": expected.spec.parameters,
        }}


def test_tool_calls_become_a_model_turn_with_decoded_arguments():
    api = FakeChatAPI(completion("Looking.", [LIST_FILES, READ_MAIN], "tool_calls", usage=(90, 20)))
    turn = make_provider(api).start("fix the bug")

    assert turn.text == "Looking."
    assert turn.tool_calls == (
        ToolCall("call_1", "get_files_info", {"directory": "."}),
        ToolCall("call_2", "get_file_content", {"file_path": "main.py"}),
    )
    assert turn.usage == Usage(90, 20)
    assert (turn.stop_reason, turn.truncated) == ("tool_calls", False)


def test_tool_results_go_back_as_tool_messages_after_the_assistant_turn():
    api = FakeChatAPI(completion(None, [LIST_FILES, READ_MAIN], "tool_calls"), completion("Done."))
    provider = make_provider(api)
    provider.start("fix the bug")
    final = provider.send_tool_results([
        ToolResult("call_1", "- main.py: file_size=25 bytes, is_dir=False"),
        ToolResult("call_2", 'Error: "main.py" is not a file', is_error=True),
    ])
    assert final.text == "Done." and final.tool_calls == ()

    messages = api.body(1)["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "tool"]
    assert messages[2]["tool_calls"] == [LIST_FILES, READ_MAIN]  # same ids, names and arguments
    assert messages[3] == {"role": "tool", "tool_call_id": "call_1",
                           "content": "- main.py: file_size=25 bytes, is_dir=False"}
    # This API has no error flag: the failure is carried by the text.
    assert messages[4] == {"role": "tool", "tool_call_id": "call_2",
                           "content": 'Error: "main.py" is not a file'}


@pytest.mark.parametrize(
    "raw_arguments, error",
    [
        ('{"file_path": "a.txt", "content": ', "arguments were not valid JSON"),
        ("not json at all", "arguments were not valid JSON"),
        ('["a.txt", "x"]', "arguments must be a JSON object"),
        ('"a.txt"', "arguments must be a JSON object"),
    ],
)
def test_malformed_tool_arguments_are_captured_not_raised(raw_arguments, error):
    api = FakeChatAPI(completion(None, [tool_call("call_9", "write_file", raw_arguments)], "tool_calls"))
    (call,) = make_provider(api).start("go").tool_calls
    assert (call.id, call.name, call.arguments) == ("call_9", "write_file", None)
    assert call.error.startswith(error)


def test_empty_arguments_mean_no_arguments():
    api = FakeChatAPI(completion(None, [tool_call("call_1", "get_files_info", "")], "tool_calls"))
    assert make_provider(api).start("go").tool_calls == (ToolCall("call_1", "get_files_info", {}),)


def test_a_response_cut_off_by_the_token_limit_is_marked_truncated():
    api = FakeChatAPI(completion("partial", finish_reason="length"))
    turn = make_provider(api).start("go")
    assert (turn.stop_reason, turn.truncated) == ("length", True)


def test_a_response_without_choices_is_a_provider_error():
    status, body = completion("x")
    body["choices"] = []
    with pytest.raises(ProviderError, match="no choices"):
        make_provider(FakeChatAPI((status, body))).start("go")


def test_api_errors_are_raised_as_provider_errors():
    api = FakeChatAPI((401, {"error": {"message": "No auth credentials found", "code": 401}}))
    with pytest.raises(ProviderError, match="OpenRouter request failed"):
        make_provider(api).start("go")
