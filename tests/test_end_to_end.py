"""A whole run: command line -> agent loop -> Claude provider -> anthropic SDK -> tools.

Only the far end is replaced. Instead of api.anthropic.com there is a local
stand-in that plays a prepared sequence of model responses and, like the real
API, refuses a request whose tool results do not match the tool calls of the
previous assistant turn. Everything else is the real code path, working on a
copy of the sample calculator project with a bug planted in it.
"""
import json
import shutil
from pathlib import Path

import anthropic
import httpx2
import pytest

from agent import cli
from agent.providers.anthropic_messages import AnthropicProvider

CALCULATOR = Path(__file__).resolve().parent.parent / "calculator"
CORRECT_PRECEDENCE = '"+": 1,'
PLANTED_BUG = '"+": 3,'


class ScriptedClaude:
    """Replays model turns and checks each request the way the Messages API would."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.requests = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        problem = self._protocol_problem(body["messages"])
        if problem:
            return httpx2.Response(400, json={"type": "error", "error": {
                "type": "invalid_request_error", "message": problem}})
        content, stop_reason = self._turns.pop(0)
        return httpx2.Response(200, json={
            "id": f"msg_{len(self.requests)}", "type": "message", "role": "assistant",
            "model": body["model"], "content": content, "stop_reason": stop_reason,
            "stop_sequence": None, "usage": {"input_tokens": 100, "output_tokens": 20},
        })

    @staticmethod
    def _protocol_problem(messages):
        if [m["role"] for m in messages] != ["user", "assistant"] * (len(messages) // 2) + ["user"]:
            return "roles must alternate, starting and ending with user"
        for assistant, user in zip(messages[1::2], messages[2::2]):
            expected = [b["id"] for b in assistant["content"] if b["type"] == "tool_use"]
            blocks = user["content"] if isinstance(user["content"], list) else []
            answered = [b["tool_use_id"] for b in blocks if b["type"] == "tool_result"]
            if answered != expected or any(b["type"] != "tool_result" for b in blocks):
                return "tool_use ids were found without tool_result blocks immediately after"
        return None


def tool_use(call_id, name, **arguments):
    return {"type": "tool_use", "id": call_id, "name": name, "input": arguments}


@pytest.fixture
def buggy_calculator(tmp_path):
    project = tmp_path / "calculator"
    shutil.copytree(CALCULATOR, project, ignore=shutil.ignore_patterns("__pycache__"))
    source = project / "pkg" / "calculator.py"
    original = source.read_text(encoding="utf-8")
    assert original.count(CORRECT_PRECEDENCE) == 1
    source.write_text(original.replace(CORRECT_PRECEDENCE, PLANTED_BUG), encoding="utf-8")
    return project, original


def test_claude_finds_and_fixes_a_bug_through_the_full_stack(buggy_calculator, monkeypatch, capsys, tmp_path):
    project, fixed_source = buggy_calculator
    claude = ScriptedClaude([
        ([{"type": "thinking", "thinking": "Reproduce it first.", "signature": "sig-1"},
          tool_use("toolu_1", "run_python_file", file_path="main.py", args=["3 + 7 * 2"])], "tool_use"),
        ([{"type": "text", "text": "The result is wrong. Looking at the project."},
          tool_use("toolu_2", "get_files_info", directory="pkg")], "tool_use"),
        ([tool_use("toolu_3", "get_file_content", file_path="pkg/calculator.py")], "tool_use"),
        ([{"type": "text", "text": "Addition has a higher precedence than multiplication."},
          tool_use("toolu_4", "write_file", file_path="pkg/calculator.py", content=fixed_source)], "tool_use"),
        # Two calls in one turn: re-run the expression and the project's own tests.
        ([tool_use("toolu_5", "run_python_file", file_path="main.py", args=["3 + 7 * 2"]),
          tool_use("toolu_6", "run_python_file", file_path="tests.py")], "tool_use"),
        ([{"type": "text", "text": "Fixed: '+' had precedence 3, above '*'. 3 + 7 * 2 is now 17."}], "end_turn"),
    ])

    def create_provider(name, *, model, system_prompt, tools, max_tokens, env):
        client = anthropic.Anthropic(
            api_key="test-key", base_url="https://api.anthropic.com", max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(claude)))
        return AnthropicProvider(client, model or "claude-test", system_prompt, tools, max_tokens)

    monkeypatch.setattr(cli, "create_provider", create_provider)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    transcript = tmp_path / "run.md"

    exit_code = cli.main(["fix the bug: 3 + 7 * 2 shouldn't be 20", "--provider", "anthropic",
                          "--workdir", str(project), "--transcript", str(transcript)])

    assert exit_code == 0
    assert capsys.readouterr().out.strip().endswith("3 + 7 * 2 is now 17.")
    assert (project / "pkg" / "calculator.py").read_text(encoding="utf-8") == fixed_source

    assert len(claude.requests) == 6  # none was rejected by the protocol check

    def results(request_index):
        return claude.requests[request_index]["messages"][-1]["content"]

    # What the tools really returned, as the model received it.
    assert '"result": 20' in results(1)[0]["content"]  # the bug, reproduced
    assert "- calculator.py:" in results(2)[0]["content"]
    assert PLANTED_BUG in results(3)[0]["content"]
    assert results(4)[0]["content"].startswith('Successfully wrote to "pkg/calculator.py"')
    assert '"result": 17' in results(5)[0]["content"]  # the fix, verified
    assert "OK" in results(5)[1]["content"] and "Ran 9 tests" in results(5)[1]["content"]
    assert [block["is_error"] for i in range(1, 6) for block in results(i)] == [False] * 6
    # The thinking block from the first turn is still in the history, signature intact.
    assert claude.requests[5]["messages"][1]["content"][0] == {
        "type": "thinking", "thinking": "Reproduce it first.", "signature": "sig-1"}

    text = transcript.read_text(encoding="utf-8")
    assert "- Outcome: completed after 6 model calls and 6 tool calls" in text
    assert "- Tokens: 600 input, 120 output" in text


def test_the_stand_in_api_rejects_a_missing_tool_result():
    """The protocol check above is real: an unanswered tool call is refused."""
    claude = ScriptedClaude([])
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages", json={
        "model": "m", "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [tool_use("toolu_1", "get_files_info")]},
            {"role": "user", "content": "thanks"},
        ]})
    response = claude(request)
    assert response.status_code == 400
    assert "without tool_result blocks" in response.json()["error"]["message"]
