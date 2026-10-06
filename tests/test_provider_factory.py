"""Building a provider from its name and the environment."""
import pytest

from agent.providers import ConfigurationError, create_provider
from agent.providers.anthropic_messages import AnthropicProvider
from agent.providers.openai_compat import OpenAICompatProvider
from agent.tools import build_tools

SPECS = [tool.spec for tool in build_tools()]
KEYS = {"ANTHROPIC_API_KEY": "test-key", "OPENROUTER_API_KEY": "test-key"}


def create(name, model=None, env=KEYS):
    return create_provider(name, model=model, system_prompt="s", tools=SPECS, max_tokens=100, env=env)


def test_anthropic_gets_the_anthropic_sdk_client_and_a_claude_default_model():
    provider = create("anthropic")
    assert isinstance(provider, AnthropicProvider)
    assert (provider.name, provider.model) == ("anthropic", "claude-sonnet-5-5")


def test_openrouter_gets_an_openai_sdk_client_pointed_at_openrouter():
    provider = create("openrouter")
    assert isinstance(provider, OpenAICompatProvider)
    assert (provider.name, provider.model) == ("openrouter", "google/gemini-2.5-flash")
    assert str(provider._client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"


def test_an_explicit_model_overrides_the_default():
    assert create("anthropic", model="claude-opus-5-5").model == "claude-opus-5-5"
    assert create("openrouter", model="some/model").model == "some/model"


@pytest.mark.parametrize("name, variable", [("anthropic", "ANTHROPIC_API_KEY"), ("openrouter", "OPENROUTER_API_KEY")])
def test_the_api_key_of_the_selected_provider_is_required(name, variable):
    other_key_only = {key: value for key, value in KEYS.items() if key != variable}
    with pytest.raises(ConfigurationError, match=f"{variable} is not set"):
        create(name, env=other_key_only)


def test_an_unknown_provider_is_rejected():
    with pytest.raises(ConfigurationError, match='Unknown provider "gemini"'):
        create("gemini")
