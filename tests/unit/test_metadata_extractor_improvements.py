"""Unit tests for metadata extractor improvements (entity types + time_period).

Tests the updated _parse_response() in MetadataExtractor that now handles:
  - New format: entities as [{\"name\": \"...\", \"type\": \"...\"}]
  - Old format: entities as [\"name1\", \"name2\"] (backward compatibility)
  - time_period extraction from LLM response
"""

from __future__ import annotations

import json

import pytest

from src.services.ingestion.metadata_extractor import MetadataExtractor


class TestParseResponseEntityTypes:
    """Test _parse_response with new structured entity format."""

    def test_new_format_entity_objects(self) -> None:
        """Entities as list of {name, type} objects should extract both."""
        response = json.dumps({
            "entities": [
                {"name": "Juan Atkins", "type": "ARTIST"},
                {"name": "Tresor", "type": "VENUE"},
                {"name": "Warp Records", "type": "LABEL"},
            ],
            "places": ["Detroit", "Berlin"],
            "genres": ["techno", "electro"],
            "time_period": "1990s",
        })

        result = MetadataExtractor._parse_response(response)

        assert result["entity_tags"] == ["Juan Atkins", "Tresor", "Warp Records"]
        assert result["entity_types"] == ["ARTIST", "VENUE", "LABEL"]
        assert result["geographic_tags"] == ["Detroit", "Berlin"]
        assert result["genre_tags"] == ["techno", "electro"]
        assert result["time_period"] == "1990s"

    def test_old_format_plain_strings(self) -> None:
        """Entities as plain strings (legacy format) should still work."""
        response = json.dumps({
            "entities": ["Juan Atkins", "Tresor"],
            "places": ["Detroit"],
            "genres": ["techno"],
        })

        result = MetadataExtractor._parse_response(response)

        assert result["entity_tags"] == ["Juan Atkins", "Tresor"]
        # No type info from plain strings
        assert result["entity_types"] == []
        assert result["geographic_tags"] == ["Detroit"]

    def test_mixed_format(self) -> None:
        """Mix of objects and strings should handle both."""
        response = json.dumps({
            "entities": [
                {"name": "Carl Cox", "type": "ARTIST"},
                "Fabric",  # plain string
            ],
            "places": ["London"],
            "genres": ["techno"],
        })

        result = MetadataExtractor._parse_response(response)

        assert "Carl Cox" in result["entity_tags"]
        assert "Fabric" in result["entity_tags"]

    def test_time_period_null_string(self) -> None:
        """time_period of 'null' should be treated as None."""
        response = json.dumps({
            "entities": [],
            "places": [],
            "genres": [],
            "time_period": "null",
        })

        result = MetadataExtractor._parse_response(response)
        assert result["time_period"] is None

    def test_time_period_none_value(self) -> None:
        """time_period of JSON null should be None."""
        response = json.dumps({
            "entities": [],
            "places": [],
            "genres": [],
            "time_period": None,
        })

        result = MetadataExtractor._parse_response(response)
        assert result["time_period"] is None

    def test_time_period_valid_decade(self) -> None:
        response = json.dumps({
            "entities": [],
            "places": [],
            "genres": [],
            "time_period": "1990s",
        })

        result = MetadataExtractor._parse_response(response)
        assert result["time_period"] == "1990s"

    def test_time_period_valid_range(self) -> None:
        response = json.dumps({
            "entities": [],
            "places": [],
            "genres": [],
            "time_period": "1988-1992",
        })

        result = MetadataExtractor._parse_response(response)
        assert result["time_period"] == "1988-1992"

    def test_markdown_fenced_json(self) -> None:
        """JSON wrapped in markdown code fences should be parsed."""
        inner = json.dumps({
            "entities": [{"name": "Carl Cox", "type": "ARTIST"}],
            "places": ["Ibiza"],
            "genres": ["techno"],
            "time_period": "2000s",
        })
        response = f"```json\n{inner}\n```"

        result = MetadataExtractor._parse_response(response)
        assert result["entity_tags"] == ["Carl Cox"]
        assert result["entity_types"] == ["ARTIST"]
        assert result["time_period"] == "2000s"

    def test_json_in_prose(self) -> None:
        """JSON embedded in prose should be extracted."""
        inner = json.dumps({
            "entities": [{"name": "Derrick May", "type": "ARTIST"}],
            "places": ["Detroit"],
            "genres": ["detroit techno"],
        })
        response = f"Here are the extracted tags:\n{inner}\nHope that helps!"

        result = MetadataExtractor._parse_response(response)
        assert result["entity_tags"] == ["Derrick May"]
        assert result["entity_types"] == ["ARTIST"]

    def test_invalid_json_returns_empty(self) -> None:
        """Completely invalid JSON should return empty tag lists."""
        result = MetadataExtractor._parse_response("This is not JSON at all.")
        assert result["entity_tags"] == []
        assert result["entity_types"] == []
        assert result["geographic_tags"] == []
        assert result["genre_tags"] == []
        assert result["time_period"] is None

    def test_empty_entities_list(self) -> None:
        response = json.dumps({
            "entities": [],
            "places": [],
            "genres": [],
        })

        result = MetadataExtractor._parse_response(response)
        assert result["entity_tags"] == []
        assert result["entity_types"] == []

    def test_entity_type_uppercased(self) -> None:
        """Entity types should be uppercased."""
        response = json.dumps({
            "entities": [{"name": "Test", "type": "artist"}],
            "places": [],
            "genres": [],
        })

        result = MetadataExtractor._parse_response(response)
        assert result["entity_types"] == ["ARTIST"]

    def test_entity_type_missing_defaults_to_empty(self) -> None:
        """Entity objects without type should get empty string type."""
        response = json.dumps({
            "entities": [{"name": "Unknown Entity"}],
            "places": [],
            "genres": [],
        })

        result = MetadataExtractor._parse_response(response)
        assert result["entity_tags"] == ["Unknown Entity"]
        assert result["entity_types"] == [""]

    def test_all_entity_type_values(self) -> None:
        """All valid entity types should pass through."""
        response = json.dumps({
            "entities": [
                {"name": "A", "type": "ARTIST"},
                {"name": "B", "type": "VENUE"},
                {"name": "C", "type": "LABEL"},
                {"name": "D", "type": "EVENT"},
                {"name": "E", "type": "COLLECTIVE"},
            ],
            "places": [],
            "genres": [],
        })

        result = MetadataExtractor._parse_response(response)
        assert result["entity_types"] == ["ARTIST", "VENUE", "LABEL", "EVENT", "COLLECTIVE"]
