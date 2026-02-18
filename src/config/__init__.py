"""Configuration module — exports Settings, load_config, and a module-level singleton."""

from src.config.loader import load_config
from src.config.settings import Settings

settings = Settings()

__all__ = ["Settings", "load_config", "settings"]
