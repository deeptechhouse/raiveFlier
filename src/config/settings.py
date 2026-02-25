"""Application settings loaded from environment variables via pydantic-settings.

# ─── HOW SETTINGS WORK (Junior Developer Guide) ───────────────────────
#
# This class uses pydantic-settings to automatically read configuration
# from TWO sources (in priority order):
#
#   1. **Environment variables** — e.g., OPENAI_API_KEY=sk-abc123
#      (highest priority — always wins)
#   2. **.env file** — key=value lines in the project root .env file
#      (lower priority — used for local development)
#
# The mapping is automatic: field name `openai_api_key` maps to env var
# `OPENAI_API_KEY` (pydantic-settings uppercases and matches).
#
# Default values (the `= ""` or `= "development"`) are used when neither
# an env var nor .env entry exists for that field.
#
# SECURITY: The .env file is in .gitignore — never committed to the repo.
# Use .env.example as a template showing what variables are available.
# ──────────────────────────────────────────────────────────────────────
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """RaiveFlier application settings.

    Environment variables override defaults. Loaded from .env file when present.
    """

    # SettingsConfigDict tells pydantic-settings where to find the .env file.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # === LLM Providers ===
    # Empty string = "not configured" → the provider selection logic in
    # main.py skips providers with empty keys and falls through to the next.
    openai_api_key: str = ""
    openai_base_url: str = ""  # Custom base URL for OpenAI-compatible APIs (TogetherAI, etc.)
    openai_text_model: str = ""  # Override text model (e.g. meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo)
    openai_vision_model: str = ""  # Override vision model (leave empty to disable vision)
    openai_embedding_model: str = ""  # Override embedding model (e.g. BAAI/bge-base-en-v1.5 for TogetherAI)
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"  # Ollama always has a default URL

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
    # Corpus search defaults — overridable per-request via API params.
    # page_size controls results per "page" in Load More pagination;
    # max_results caps the total candidate pool; max_per_source limits
    # how many chunks from one source survive deduplication.
    corpus_search_default_page_size: int = 20
    corpus_search_max_results: int = 50
    corpus_search_max_per_source: int = 5
    corpus_search_default_min_similarity: float = 0.0

    # === Feedback / Ratings ===
    feedback_db_path: str = "data/feedback.db"

    # === Flier History (cross-flier recommendation data) ===
    flier_history_db_path: str = "data/flier_history.db"

    # === Session Persistence ===
    session_db_path: str = "data/session_state.db"

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
