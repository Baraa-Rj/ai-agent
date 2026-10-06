"""The command-line entry point, with the model replaced by a scripted provider."""
import pytest

from agent import cli
from agent.types import ModelTurn, Usage
from tests.helpers import ScriptedProvider, call


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """No real keys, no .env file and no ambient defaults leak into a test."""
    for name in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "AGENT_PROVIDER", "AGENT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)


@pytest.fixture
def scripted(monkeypatch):
    """Replace provider creation; returns a dict that records how it was called."""
    captured = {}

    def install(turns):
        def fake_create_provider(name, **kwargs):
            captured.update(kwargs, name=name)
            captured["provider"] = ScriptedProvider(turns)
            return captured["provider"]

        monkeypatch.setattr(cli, "create_provider", fake_create_provider)
        return captured

    return install


def test_a_missing_api_key_is_a_clear_configuration_error(workspace, capsys):
    for provider, variable in (("anthropic", "ANTHROPIC_API_KEY"), ("openrouter", "OPENROUTER_API_KEY")):
        assert cli.main(["hi", "--provider", provider, "--workdir", str(workspace.root)]) == 2
        assert f"Error: {variable} is not set" in capsys.readouterr().err


def test_a_missing_working_directory_is_a_configuration_error(tmp_path, capsys):
    assert cli.main(["hi", "--workdir", str(tmp_path / "nope")]) == 2
    assert "Working directory does not exist" in capsys.readouterr().err


def test_an_unknown_provider_from_the_environment_is_rejected(workspace, monkeypatch, capsys):
    monkeypatch.setenv("AGENT_PROVIDER", "skynet")
    assert cli.main(["hi", "--workdir", str(workspace.root)]) == 2
    assert 'Unknown provider "skynet"' in capsys.readouterr().err


def test_runs_a_task_and_prints_the_final_answer(workspace, scripted, capsys):
    captured = scripted([
        ModelTurn("", (call("get_file_content", file_path="pkg/mod.py"),), Usage(10, 2)),
        ModelTurn("VALUE is 1.", usage=Usage(20, 3)),
    ])
    assert cli.main(["what is VALUE?", "--workdir", str(workspace.root)]) == 0
    out = capsys.readouterr().out
    assert " - Calling function: get_file_content" in out
    assert out.strip().endswith("VALUE is 1.")
    assert captured["provider"].prompts == ["what is VALUE?"]
    # Defaults: OpenRouter, the provider's default model, all four tools.
    assert (captured["name"], captured["model"], captured["max_tokens"]) == ("openrouter", None, 4096)
    assert [spec.name for spec in captured["tools"]] == [
        "get_files_info", "get_file_content", "write_file", "run_python_file"]
    assert "Execute Python files" in captured["system_prompt"]


def test_provider_and_model_come_from_flags_then_environment(workspace, scripted, monkeypatch):
    monkeypatch.setenv("AGENT_PROVIDER", "anthropic")
    monkeypatch.setenv("AGENT_MODEL", "claude-haiku-4-5-20251001")
    captured = scripted([ModelTurn("ok")])
    cli.main(["hi", "--workdir", str(workspace.root)])
    assert (captured["name"], captured["model"]) == ("anthropic", "claude-haiku-4-5-20251001")

    captured = scripted([ModelTurn("ok")])
    cli.main(["hi", "--workdir", str(workspace.root), "--provider", "openrouter", "--model", "some/model"])
    assert (captured["name"], captured["model"]) == ("openrouter", "some/model")


def test_no_exec_removes_the_python_tool_and_its_mention_in_the_prompt(workspace, scripted):
    captured = scripted([
        ModelTurn("", (call("run_python_file", file_path="main.py"),)),
        ModelTurn("could not run it"),
    ])
    assert cli.main(["run main.py", "--workdir", str(workspace.root), "--no-exec"]) == 0
    assert [spec.name for spec in captured["tools"]] == ["get_files_info", "get_file_content", "write_file"]
    assert "Execute Python files" not in captured["system_prompt"]
    assert "cannot execute code" in captured["system_prompt"]
    # A model that asks for it anyway gets an error result instead of an execution.
    assert captured["provider"].results[0][0].content == "Error: Unknown tool: run_python_file"


def test_reaching_the_iteration_limit_exits_with_an_error(workspace, scripted, capsys):
    scripted([ModelTurn("", (call("get_files_info"),))] * 5)
    assert cli.main(["loop", "--workdir", str(workspace.root), "--max-iterations", "3"]) == 1
    assert "Reached maximum iterations (3)" in capsys.readouterr().err


def test_a_truncated_response_exits_with_an_error(workspace, scripted, capsys):
    scripted([ModelTurn("half", stop_reason="max_tokens", truncated=True)])
    assert cli.main(["go", "--workdir", str(workspace.root)]) == 1
    assert "cut off (stop reason: max_tokens)" in capsys.readouterr().err


def test_a_provider_failure_exits_with_an_error(workspace, monkeypatch, capsys):
    class Failing(ScriptedProvider):
        def start(self, user_prompt):
            raise cli.ProviderError("Anthropic request failed: overloaded")

    monkeypatch.setattr(cli, "create_provider", lambda name, **kwargs: Failing([]))
    assert cli.main(["go", "--workdir", str(workspace.root)]) == 1
    assert capsys.readouterr().err.strip() == "Error: Anthropic request failed: overloaded"


def test_an_invalid_iteration_limit_is_rejected(workspace, scripted, capsys):
    scripted([ModelTurn("ok")])
    assert cli.main(["go", "--workdir", str(workspace.root), "--max-iterations", "0"]) == 2
    assert "--max-iterations must be at least 1" in capsys.readouterr().err


def test_verbose_shows_arguments_results_and_token_usage(workspace, scripted, capsys):
    scripted([
        ModelTurn("", (call("get_file_content", file_path="pkg/mod.py"),), Usage(10, 2), "tool_use"),
        ModelTurn("done", usage=Usage(20, 3), stop_reason="end_turn"),
    ])
    cli.main(["go", "--workdir", str(workspace.root), "--verbose"])
    out = capsys.readouterr().out
    assert "User prompt: go" in out
    assert "Provider: scripted, model: test-model" in out
    assert "input tokens: 10, output tokens: 2, stop reason: tool_use" in out
    assert " - Calling function: get_file_content({'file_path': 'pkg/mod.py'})" in out
    assert "-> VALUE = 1" in out


def test_writes_a_markdown_transcript_of_the_run(workspace, scripted, tmp_path):
    scripted([
        ModelTurn("I will read the module.", (call("get_file_content", file_path="pkg/mod.py"),), Usage(10, 2)),
        ModelTurn("", (call("get_file_content", "call_2", file_path="missing.py"),), Usage(15, 2)),
        ModelTurn("VALUE is 1.", usage=Usage(20, 3)),
    ])
    transcript = tmp_path / "runs" / "example.md"
    cli.main(["what is VALUE?", "--workdir", str(workspace.root), "--transcript", str(transcript)])

    text = transcript.read_text(encoding="utf-8")
    assert "- Provider: scripted" in text and "- Model: test-model" in text
    assert "- Outcome: completed after 3 model calls and 2 tool calls" in text
    assert "- Tokens: 45 input, 7 output" in text
    assert "> what is VALUE?" in text
    assert "I will read the module." in text
    assert '**Tool call:** `get_file_content`' in text and '"file_path": "pkg/mod.py"' in text
    assert "**Result:**" in text and "VALUE = 1" in text
    assert "**Error result:**" in text and 'Error: "missing.py" is not a file' in text
    assert text.rstrip().endswith("## Final answer\n\nVALUE is 1.")
