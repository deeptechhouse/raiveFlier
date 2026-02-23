"""Abstract base class for LLM service providers.

Defines the contract for any large-language-model backend used for text
completion, entity extraction, and vision-based image analysis.
Implementations may wrap the Anthropic API (Claude), OpenAI, or a local
model server.  The adapter pattern (CLAUDE.md Section 6) keeps every
call-site provider-agnostic.
"""

from __future__ import annotations

# ABC = Abstract Base Class — Python's way of defining interfaces.
# abstractmethod marks methods that MUST be overridden by concrete classes.
# If a concrete class forgets to implement an abstractmethod, Python raises
# TypeError when you try to instantiate it. This catches bugs at startup.
from abc import ABC, abstractmethod


# Concrete implementations: AnthropicLLMProvider, OpenAILLMProvider, OllamaLLMProvider
# Located in: src/providers/llm/
class ILLMProvider(ABC):
    """Contract for LLM services used throughout the raiveFlier pipeline.

    Providers must support plain text completion; vision (image analysis) is
    optional and declared via :meth:`supports_vision`.
    """

    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 4000,
    ) -> str:
        """Generate a text completion from the model.

        Parameters
        ----------
        system_prompt:
            The system/instruction message that sets the model's behaviour.
        user_prompt:
            The user-facing prompt containing the actual request or data.
        temperature:
            Sampling temperature (0.0 = deterministic, 1.0 = creative).
        max_tokens:
            Upper bound on the number of tokens in the response.

        Returns
        -------
        str
            The model's text response.

        Raises
        ------
        src.core.errors.LLMError
            If the API call fails or returns an invalid response.
        """

    @abstractmethod
    async def vision_extract(self, image_bytes: bytes, prompt: str) -> str:
        """Analyse an image using the model's vision capability.

        Parameters
        ----------
        image_bytes:
            Raw bytes of the image to analyse.
        prompt:
            A natural-language instruction describing what to extract or
            identify in the image.

        Returns
        -------
        str
            The model's text response describing or extracting from the image.

        Raises
        ------
        NotImplementedError
            If the provider does not support vision (check
            :meth:`supports_vision` first).
        src.core.errors.LLMError
            If the API call fails.
        """

    @abstractmethod
    def supports_vision(self) -> bool:
        """Return ``True`` if this provider can process image inputs.

        Callers should check this before invoking :meth:`vision_extract`.
        """

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return a human-readable identifier for this LLM provider.

        Example return values: ``"claude-sonnet"``, ``"openai-gpt4o"``.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if the provider is configured and reachable.

        Implementations should verify that credentials are present without
        making a full inference call.
        """

    @abstractmethod
    async def validate_credentials(self) -> bool:
        """Perform a lightweight API call to confirm credentials are valid.

        Returns
        -------
        bool
            ``True`` if the provider accepted the credentials; ``False``
            otherwise.  Unlike :meth:`is_available`, this method actively
            contacts the remote service.
        """
