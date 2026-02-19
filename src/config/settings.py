"""Application settings loaded from environment variables via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """RaiveFlier application settings.

    Environment variables override defaults. Loaded from .env file when present.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # === LLM Providers ===
    openai_api_key: str = ""
    openai_base_url: str = ""  # Custom base URL for OpenAI-compatible APIs (TogetherAI, etc.)
    openai_text_model: str = ""  # Override text model (e.g. meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo)
    openai_vision_model: str = ""  # Override vision model (leave empty to disable vision)
    openai_embedding_model: str = ""  # Override embedding model (e.g. BAAI/bge-base-en-v1.5 for TogetherAI)
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # === Music Databases ===
    discogs_consumer_key: str = ""
    discogs_consumer_secret: str = ""
    musicbrainz_app_name: str = "raiveFlier"
    musicbrainz_app_version: str = "0.1.0"
    musicbrainz_contact: str = ""

    # === Web Search ===
    serper_api_key: str = ""

    # === RAG Configuration ===
    chromadb_persist_dir: str = "./data/chromadb"
    chromadb_collection: str = "raiveflier_corpus"
    rag_enabled: bool = False
    rag_top_k: int = 20
    rag_max_tokens: int = 30000

    # === App Config ===
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_env: str = "development"
    log_level: str = "INFO"

    def get_available_llm_providers(self) -> list[str]:
        """Return a list of LLM provider names that have non-empty API keys configured."""
        providers: list[str] = []
        if self.openai_api_key:
            providers.append("openai")
        if self.anthropic_api_key:
            providers.append("anthropic")
        if self.ollama_base_url:
            providers.append("ollama")
        return providers
