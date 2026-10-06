"""Model providers, and the factory that builds one from the environment."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from agent.providers.base import ConfigurationError, Provider, ProviderError
from agent.types import ToolSpec

PROVIDER_NAMES = ("anthropic", "openrouter")

__all__ = ["PROVIDER_NAMES", "ConfigurationError", "Provider", "ProviderError", "create_provider"]


def create_provider(
    name: str,
    *,
    model: str | None,
    system_prompt: str,
    tools: Sequence[ToolSpec],
    max_tokens: int,
    env: Mapping[str, str],
) -> Provider:
    """Build the provider called `name`, reading its API key from `env`.

    The SDKs are imported here, so only the selected provider's package is loaded.
    """
    if name == "anthropic":
        api_key = env.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ConfigurationError("ANTHROPIC_API_KEY is not set (see .env.example).")
        import anthropic

        from agent.providers.anthropic_messages import (
            DEFAULT_ANTHROPIC_MODEL,
            AnthropicProvider,
        )

        return AnthropicProvider(
            anthropic.Anthropic(api_key=api_key),
            model or DEFAULT_ANTHROPIC_MODEL,
            system_prompt,
            tools,
            max_tokens,
        )

    if name == "openrouter":
        api_key = env.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ConfigurationError("OPENROUTER_API_KEY is not set (see .env.example).")
        import openai

        from agent.providers.openai_compat import (
            DEFAULT_OPENROUTER_MODEL,
            OPENROUTER_BASE_URL,
            OpenAICompatProvider,
        )

        return OpenAICompatProvider(
            openai.OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key),
            model or DEFAULT_OPENROUTER_MODEL,
            system_prompt,
            tools,
            max_tokens,
        )

    raise ConfigurationError(
        f'Unknown provider "{name}". Choose one of: {", ".join(PROVIDER_NAMES)}.'
    )
