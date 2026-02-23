"""YAML configuration loader with environment variable overrides.

# ─── CONFIGURATION HIERARCHY (Junior Developer Guide) ──────────────────
#
# Configuration is loaded in layers (later layers override earlier):
#
#   1. config/config.yaml  — Static defaults checked into the repo
#   2. .env file           — Local developer overrides (not committed)
#   3. Environment vars    — Set in Docker/Render at deploy time
#
# The load_config() function reads the YAML file first, then deep-merges
# environment-based values on top.  This means you can set a default in
# config.yaml and override it per-environment via env vars.
#
# The _deep_merge helper does recursive dict merging:
#   base = {"ocr": {"min_confidence": 0.5}}
#   overrides = {"ocr": {"provider": "tesseract"}}
#   result = {"ocr": {"min_confidence": 0.5, "provider": "tesseract"}}
# ──────────────────────────────────────────────────────────────────────
"""

from pathlib import Path

import yaml

from src.config.settings import Settings


def load_config(path: str = "config/config.yaml") -> dict:
    """Load YAML config and merge with environment-based Settings.

    Environment variables (via Settings) override YAML values where keys overlap.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Fully resolved configuration dictionary.
    """
    config_path = Path(path)
    if config_path.exists():
        with open(config_path) as f:
            # yaml.safe_load prevents arbitrary code execution from YAML.
            # Always use safe_load, never load() — security best practice.
            yaml_config = yaml.safe_load(f) or {}
    else:
        yaml_config = {}

    # Build env-based overrides from the pydantic-settings Settings object.
    # Settings reads from .env and environment variables automatically.
    settings = Settings()
    env_overrides = {
        "app": {
            "host": settings.app_host,
            "port": settings.app_port,
            "env": settings.app_env,
        },
        "llm": {
            "openai_api_key": settings.openai_api_key,
            "anthropic_api_key": settings.anthropic_api_key,
            "ollama_base_url": settings.ollama_base_url,
            "available_providers": settings.get_available_llm_providers(),
        },
        "music_db": {
            "discogs_consumer_key": settings.discogs_consumer_key,
            "discogs_consumer_secret": settings.discogs_consumer_secret,
            "musicbrainz_app_name": settings.musicbrainz_app_name,
            "musicbrainz_app_version": settings.musicbrainz_app_version,
            "musicbrainz_contact": settings.musicbrainz_contact,
        },
        "search": {
            "serper_api_key": settings.serper_api_key,
        },
        "logging": {
            "level": settings.log_level,
        },
    }

    _deep_merge(yaml_config, env_overrides)
    return yaml_config


def _deep_merge(base: dict, overrides: dict) -> None:
    """Recursively merge overrides into base dict, mutating base in place."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
