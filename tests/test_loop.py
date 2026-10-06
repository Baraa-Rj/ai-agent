"""The agent loop, driven by a scripted model."""
import itertools

import pytest

from agent.loop import COMPLETED, MAX_ITERATIONS, TRUNCATED, Observer, run_agent
from agent.types import ModelTurn, ToolCall, Usage
from tests.helpers import ScriptedProvider, call


def test_returns_the_answer_when_the_model_calls_no_tools(registry):
    provider = ScriptedProvider([ModelTurn("All good.", usage=Usage(10, 4), stop_reason="end_turn")])
    result = run_agent(provider, registry, "is it fine?")
    assert provider.prompts == ["is it fine?"]
    assert (result.status, result.text, result.model_calls, result.tool_calls) == (COMPLETED, "All good.", 1, 0)
    assert result.usage == Usage(10, 4)


def test_runs_a_tool_and_sends_its_result_back(registry):
    provider = ScriptedProvider([
        ModelTurn("Let me look.", (call("get_file_content", "call_a", file_path="pkg/mod.py"),), Usage(20, 5)),
        ModelTurn("VALUE is 1.", usage=Usage(30, 6)),
    ])
    result = run_agent(provider, registry, "what is VALUE?")
    assert [(r.call_id, r.content, r.is_error) for r in provider.results[0]] == [("call_a", "VALUE = 1\n", False)]
    assert (result.status, result.text, result.model_calls, result.tool_calls) == (COMPLETED, "VALUE is 1.", 2, 1)
    assert result.usage == Usage(50, 11)


def test_several_tool_calls_in_one_turn_are_answered_together_and_in_order(registry, workspace):
    provider = ScriptedProvider([
        ModelTurn("", (
            call("write_file", "call_1", file_path="a.txt", content="A"),
            call("get_files_info", "call_2"),
            call("get_file_content", "call_3", file_path="a.txt"),
        )),
        ModelTurn("done"),
    ])
    result = run_agent(provider, registry, "go")
    assert len(provider.results) == 1
    assert [r.call_id for r in provider.results[0]] == ["call_1", "call_2", "call_3"]
    assert "- a.txt:" in provider.results[0][1].content  # the write happened before the listing
    assert provider.results[0][2].content == "A"
    assert result.tool_calls == 3


def test_a_multi_step_task_edits_and_verifies_a_file(registry, workspace):
    provider = ScriptedProvider([
        ModelTurn("", (call("get_files_info", "c1"),)),
        ModelTurn("", (call("get_file_content", "c2", file_path="main.py"),)),
        ModelTurn("", (call("write_file", "c3", file_path="main.py", content='print("fixed")\n'),)),
        ModelTurn("", (call("run_python_file", "c4", file_path="main.py"),)),
        ModelTurn("Changed the greeting and ran it."),
    ])
    result = run_agent(provider, registry, "change the greeting")
    assert provider.results[3][0].content == "STDOUT:\nfixed\n"
    assert (result.status, result.model_calls, result.tool_calls) == (COMPLETED, 5, 4)


def test_stops_at_the_iteration_limit(registry, workspace):
    # A model that never stops asking for tools.
    turns = (ModelTurn("", (call("write_file", f"c{i}", file_path=f"f{i}.txt", content="x"),))
             for i in itertools.count())
    provider = ScriptedProvider(turns)
    result = run_agent(provider, registry, "loop forever", max_iterations=4)
    assert (result.status, result.text, result.model_calls) == (MAX_ITERATIONS, "", 4)
    # Four model calls, but only the first three tool calls were executed: the
    # result of a fourth could never have been sent back.
    assert result.tool_calls == 3 and len(provider.results) == 3
    assert sorted(p.name for p in workspace.root.glob("f*.txt")) == ["f0.txt", "f1.txt", "f2.txt"]


def test_a_limit_of_one_allows_a_single_model_call_and_no_tools(registry, workspace):
    provider = ScriptedProvider([ModelTurn("", (call("write_file", file_path="f.txt", content="x"),))])
    result = run_agent(provider, registry, "go", max_iterations=1)
    assert (result.status, result.model_calls, result.tool_calls) == (MAX_ITERATIONS, 1, 0)
    assert not (workspace.root / "f.txt").exists()


def test_the_iteration_limit_must_be_positive(registry):
    with pytest.raises(ValueError):
        run_agent(ScriptedProvider([]), registry, "go", max_iterations=0)


def test_a_truncated_response_is_never_acted_on(registry, workspace):
    provider = ScriptedProvider([
        ModelTurn("", (call("write_file", file_path="half.txt", content="incompl"),),
                  stop_reason="max_tokens", truncated=True),
    ])
    result = run_agent(provider, registry, "write a long file")
    assert (result.status, result.stop_reason, result.tool_calls) == (TRUNCATED, "max_tokens", 0)
    assert not (workspace.root / "half.txt").exists()


def test_malformed_tool_arguments_are_reported_to_the_model_and_the_run_continues(registry):
    broken = ToolCall("call_bad", "write_file", None, error="arguments were not valid JSON (Expecting value)")
    provider = ScriptedProvider([
        ModelTurn("", (broken,)),
        ModelTurn("", (call("get_file_content", "call_ok", file_path="main.py"),)),
        ModelTurn("recovered"),
    ])
    result = run_agent(provider, registry, "go")
    first = provider.results[0][0]
    assert (first.call_id, first.is_error) == ("call_bad", True) and "not valid JSON" in first.content
    assert provider.results[1][0].is_error is False
    assert (result.status, result.text) == (COMPLETED, "recovered")


def test_a_failing_tool_does_not_stop_the_run(registry):
    provider = ScriptedProvider([
        ModelTurn("", (call("get_file_content", file_path="../secret.txt"),)),
        ModelTurn("I cannot read that file."),
    ])
    result = run_agent(provider, registry, "read the secret")
    assert provider.results[0][0].is_error
    assert "outside the permitted working directory" in provider.results[0][0].content
    assert result.status == COMPLETED


def test_observers_see_every_model_turn_and_tool_result(registry):
    events = []

    class Recorder(Observer):
        def on_model_turn(self, number, turn):
            events.append(("turn", number, len(turn.tool_calls)))

        def on_tool_result(self, call, result):
            events.append(("tool", call.name, result.is_error))

    provider = ScriptedProvider([
        ModelTurn("", (call("get_files_info"), call("get_file_content", "c2", file_path="nope"))),
        ModelTurn("done"),
    ])
    run_agent(provider, registry, "go", observers=[Recorder()])
    assert events == [("turn", 1, 2), ("tool", "get_files_info", False),
                      ("tool", "get_file_content", True), ("turn", 2, 0)]
