"""YAML configuration loader with environment variable overrides."""

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
            yaml_config = yaml.safe_load(f) or {}
    else:
        yaml_config = {}

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
