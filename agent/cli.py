"""Command-line interface."""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from dotenv import load_dotenv

from agent.loop import (
    COMPLETED,
    DEFAULT_MAX_ITERATIONS,
    MAX_ITERATIONS,
    TRUNCATED,
    run_agent,
)
from agent.prompts import get_system_prompt
from agent.providers import (
    PROVIDER_NAMES,
    ConfigurationError,
    ProviderError,
    create_provider,
)
from agent.tools import ToolRegistry, build_tools
from agent.transcript import ConsoleObserver, TranscriptRecorder
from agent.workspace import Workspace

DEFAULT_PROVIDER = "openrouter"
DEFAULT_WORKDIR = "./calculator"
DEFAULT_MAX_TOKENS = 4096


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A coding agent that works on the files in one directory."
    )
    parser.add_argument("user_input", help="The task for the agent.")
    parser.add_argument(
        "--provider",
        choices=PROVIDER_NAMES,
        default=None,
        help=f"Model provider (default: $AGENT_PROVIDER, or {DEFAULT_PROVIDER}).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model id (default: $AGENT_MODEL, or the provider's default).",
    )
    parser.add_argument(
        "--workdir",
        default=DEFAULT_WORKDIR,
        help=f"Directory the agent may read, write and run files in (default: {DEFAULT_WORKDIR}).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Maximum number of model calls (default: {DEFAULT_MAX_ITERATIONS}).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum output tokens per model call (default: {DEFAULT_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--no-exec",
        action="store_true",
        help="Do not give the model the tool that runs Python files.",
    )
    parser.add_argument(
        "--transcript",
        metavar="FILE",
        default=None,
        help="Write a Markdown transcript of the run to FILE.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show tool arguments, tool results and token usage.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv()

    provider_name = args.provider or os.environ.get("AGENT_PROVIDER") or DEFAULT_PROVIDER
    allow_exec = not args.no_exec
    try:
        if args.max_iterations < 1:
            raise ConfigurationError("--max-iterations must be at least 1.")
        workspace = Workspace(args.workdir)
        registry = ToolRegistry(workspace, build_tools(allow_exec=allow_exec))
        provider = create_provider(
            provider_name,
            model=args.model or os.environ.get("AGENT_MODEL"),
            system_prompt=get_system_prompt(allow_exec=allow_exec),
            tools=registry.specs,
            max_tokens=args.max_tokens,
            env=os.environ,
        )
    except (ConfigurationError, NotADirectoryError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.verbose:
        print(f"User prompt: {args.user_input}")
        print(f"Provider: {provider.name}, model: {provider.model}, workdir: {workspace.root}")

    recorder = None
    observers = [ConsoleObserver(verbose=args.verbose)]
    if args.transcript:
        recorder = TranscriptRecorder(
            provider=provider.name,
            model=provider.model,
            workdir=args.workdir,
            prompt=args.user_input,
        )
        observers.append(recorder)

    try:
        result = run_agent(
            provider,
            registry,
            args.user_input,
            max_iterations=args.max_iterations,
            observers=observers,
        )
    except ProviderError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if recorder is not None:
        recorder.write(args.transcript, result)

    if result.status == COMPLETED:
        print(result.text or f"(The model returned no text; stop reason: {result.stop_reason}.)")
        return 0
    if result.status == MAX_ITERATIONS:
        print(
            f"Reached maximum iterations ({args.max_iterations}) without a final response.",
            file=sys.stderr,
        )
    elif result.status == TRUNCATED:
        print(
            f"The model's response was cut off (stop reason: {result.stop_reason}). "
            "Try a higher --max-tokens.",
            file=sys.stderr,
        )
    return 1
