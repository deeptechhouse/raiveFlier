"""Shared pytest fixtures for the raiveFlier test suite."""

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_fliers_dir(project_root: Path) -> Path:
    """Return the path to sample flier fixtures."""
    return project_root / "tests" / "fixtures" / "sample_fliers"


@pytest.fixture
def mock_responses_dir(project_root: Path) -> Path:
    """Return the path to mock API response fixtures."""
    return project_root / "tests" / "fixtures" / "mock_responses"


@pytest.fixture
def mock_config() -> dict[str, Any]:
    """Return a minimal mock configuration for testing."""
    return {
        "app": {
            "name": "raiveFlier",
            "version": "0.1.0",
            "host": "127.0.0.1",
            "port": 8000,
        },
        "ocr": {
            "provider_priority": ["tesseract"],
            "min_confidence": 0.5,
        },
        "llm": {
            "default_provider": "openai",
            "temperature": 0.3,
            "max_tokens": 4000,
        },
        "music_db": {
            "primary": "discogs_api",
            "fallback": "discogs_scrape",
            "complementary": "musicbrainz",
        },
        "search": {
            "primary": "duckduckgo",
            "secondary": "serper",
        },
        "rate_limits": {
            "discogs": 60,
            "musicbrainz": 1,
            "duckduckgo": 20,
        },
        "cache": {
            "enabled": False,
            "ttl": 3600,
        },
    }
