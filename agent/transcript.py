"""Console output and Markdown transcripts of a run."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agent.loop import AgentResult, Observer
from agent.types import ModelTurn, ToolCall, ToolResult


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... {len(text) - limit} more characters]"


class ConsoleObserver(Observer):
    """Prints tool calls as they happen; with `verbose`, also arguments, results and tokens."""

    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose

    def on_model_turn(self, number: int, turn: ModelTurn) -> None:
        if self._verbose:
            print(f"[model call {number}] input tokens: {turn.usage.input_tokens}, "
                  f"output tokens: {turn.usage.output_tokens}, stop reason: {turn.stop_reason}")
        for call in turn.tool_calls:
            if self._verbose:
                print(f" - Calling function: {call.name}({call.arguments})")
            else:
                print(f" - Calling function: {call.name}")

    def on_tool_result(self, call: ToolCall, result: ToolResult) -> None:
        if self._verbose:
            print(f"-> {_shorten(result.content, 2000)}")


class TranscriptRecorder(Observer):
    """Collects one run and writes it to a Markdown file."""

    def __init__(self, *, provider: str, model: str, workdir: str, prompt: str,
                 max_result_chars: int = 2000) -> None:
        self._provider = provider
        self._model = model
        self._workdir = workdir
        self._prompt = prompt
        self._max_result_chars = max_result_chars
        self._started = datetime.now(timezone.utc)
        self._lines: list[str] = []

    def on_model_turn(self, number: int, turn: ModelTurn) -> None:
        self._lines.append(f"### Model call {number}\n")
        if turn.text.strip() and turn.tool_calls:
            self._lines.append(turn.text.strip() + "\n")

    def on_tool_result(self, call: ToolCall, result: ToolResult) -> None:
        arguments = json.dumps(call.arguments, indent=2, ensure_ascii=False)
        outcome = "Error result" if result.is_error else "Result"
        self._lines += [
            f"**Tool call:** `{call.name}`\n",
            f"```json\n{_shorten(arguments, self._max_result_chars)}\n```\n",
            f"**{outcome}:**\n",
            f"```text\n{_shorten(result.content, self._max_result_chars)}\n```\n",
        ]

    def render(self, result: AgentResult) -> str:
        header = [
            "# Agent run\n",
            f"- Date: {self._started:%Y-%m-%d %H:%M} UTC",
            f"- Provider: {self._provider}",
            f"- Model: {self._model}",
            f"- Working directory: `{self._workdir}`",
            (f"- Outcome: {result.status} after {result.model_calls} model calls and "
             f"{result.tool_calls} tool calls"),
            f"- Tokens: {result.usage.input_tokens} input, {result.usage.output_tokens} output\n",
            "## Prompt\n",
            "\n".join(f"> {line}" for line in self._prompt.splitlines()) + "\n",
            "## Steps\n",
        ]
        footer = ["## Final answer\n", result.text.strip() or f"(none: {result.status})", ""]
        return "\n".join(header + self._lines + footer)

    def write(self, path: str | Path, result: AgentResult) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.render(result), encoding="utf-8")
