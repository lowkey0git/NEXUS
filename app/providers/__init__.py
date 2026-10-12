from .anthropic import AnthropicProvider
from .base import Capabilities, Provider, ProviderError, ProviderResponse
from .openai_compat import OpenAICompatProvider


def build_provider(settings) -> Provider:
    if settings.provider == "anthropic":
        return AnthropicProvider(settings.anthropic_api_key, settings.model, settings.anthropic_base_url)
    if settings.provider == "groq":
        return OpenAICompatProvider(settings.groq_api_key, settings.model, settings.groq_base_url,
                                    name="groq", key_env="GROQ_API_KEY")
    if settings.provider == "openai_compat":
        return OpenAICompatProvider(settings.openai_api_key, settings.model, settings.openai_base_url)
    raise ValueError(f"Unknown ASSISTANT_PROVIDER '{settings.provider}' (use 'anthropic', 'groq' or 'openai_compat')")
