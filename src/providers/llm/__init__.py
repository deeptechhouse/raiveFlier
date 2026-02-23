"""LLM provider adapters.

Three concrete implementations of ILLMProvider (src/interfaces/llm_provider.py):
    - OpenAILLMProvider   — gpt-4o / gpt-4o-mini (also supports OpenAI-compatible APIs)
    - AnthropicLLMProvider — Claude Sonnet (vision + text)
    - OllamaLLMProvider    — local models via Ollama server (llama3.1 / llava)

At startup, main.py creates the provider matching the available API key
(OPENAI_API_KEY or ANTHROPIC_API_KEY) or Ollama URL, and injects it into
FastAPI's app.state for dependency injection throughout the request lifecycle.
"""

from src.providers.llm.anthropic_provider import AnthropicLLMProvider
from src.providers.llm.ollama_provider import OllamaLLMProvider
from src.providers.llm.openai_provider import OpenAILLMProvider

__all__ = ["OpenAILLMProvider", "AnthropicLLMProvider", "OllamaLLMProvider"]
