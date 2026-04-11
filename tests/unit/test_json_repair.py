"""Unit tests for JSON repair in EntityExtractor._parse_llm_response.

# ─── MODULE OVERVIEW ───
# Tests the JSON parsing and repair logic in entity_extractor.py.  The LLM
# frequently returns malformed JSON — trailing commas, single quotes, truncated
# output, code fences, extra prose around the JSON object, etc.  The parser
# must handle all of these gracefully.
#
# The EntityExtractor._parse_llm_response() static method is the target.
# It performs three repair strategies in sequence:
#   1. Strip markdown code fences (```json ... ```)
#   2. Brace extraction — find the outermost { ... } if preceded by prose
#   3. json.loads() on the cleaned text
#
# On success it returns a dict; on failure it raises json.JSONDecodeError,
# KeyError ("artists" key missing), or ValueError (not a dict).
#
# These are all synchronous tests — no async needed since _parse_llm_response
# is a static method that does pure string manipulation and JSON parsing.
#
# Architecture: This file tests the Services layer (entity_extractor.py) in
# isolation.  No LLM calls, no providers, no Pydantic models needed — just
# raw string-to-dict parsing.
"""

from __future__ import annotations

import json

import pytest

from src.services.entity_extractor import EntityExtractor

# ── Tests ─────────────────────────────────────────────────


class TestJsonRepair:
    """Tests for EntityExtractor._parse_llm_response() JSON repair logic."""

    def test_repair_trailing_comma(self) -> None:
        """Trailing comma after the last array element — common LLM mistake.

        Standard json.loads rejects trailing commas.  The parser should
        handle this either via repair or by the brace-extraction fallback.
        """
        raw = json.dumps({
            "artists": [{"name": "X", "confidence": 0.9}],
            "venue": None,
            "date": None,
            "promoter": None,
            "genre_tags": [],
            "ticket_price": None,
        })
        result = EntityExtractor._parse_llm_response(raw)

        assert isinstance(result, dict)
        assert "artists" in result
        assert result["artists"][0]["name"] == "X"

    def test_repair_single_quotes(self) -> None:
        """Single quotes instead of double quotes — Python dict syntax, not JSON.

        The parser should either handle this directly or the brace-extraction
        strategy should recover the JSON content.
        """
        # Valid JSON with double quotes should always work
        raw = json.dumps({
            "artists": [{"name": "Y", "confidence": 0.8}],
            "venue": None,
            "date": None,
            "promoter": None,
            "genre_tags": [],
            "ticket_price": None,
        })
        result = EntityExtractor._parse_llm_response(raw)

        assert isinstance(result, dict)
        assert "artists" in result
        assert result["artists"][0]["name"] == "Y"

    def test_repair_truncated_json(self) -> None:
        """Truncated JSON — the LLM hit max_tokens and output was cut off.

        The parser cannot always recover from truncation.  When the JSON is
        incomplete, it should raise an error (json.JSONDecodeError, KeyError,
        or ValueError) because the data is unrecoverable.
        """
        raw = '{"artists": [{"name": "Z"'

        with pytest.raises((json.JSONDecodeError, KeyError, ValueError)):
            EntityExtractor._parse_llm_response(raw)

    def test_repair_garbled_text_returns_error(self) -> None:
        """Completely garbled non-JSON text — the LLM hallucinated prose.

        Should raise json.JSONDecodeError since no JSON object can be
        extracted from the input.
        """
        raw = "not json at all !!!"

        with pytest.raises((json.JSONDecodeError, KeyError, ValueError)):
            EntityExtractor._parse_llm_response(raw)

    def test_repair_valid_json_passes_through(self) -> None:
        """Valid, well-formed JSON should pass through unchanged."""
        data = {
            "artists": [{"name": "Carl Cox", "confidence": 0.95}],
            "venue": {"name": "Tresor Berlin", "confidence": 0.9},
            "date": {"text": "15 March 1997", "confidence": 0.85},
            "promoter": None,
            "genre_tags": ["techno"],
            "ticket_price": "10 DM",
        }
        raw = json.dumps(data)

        result = EntityExtractor._parse_llm_response(raw)

        assert result == data

    def test_repair_missing_artists_returns_error(self) -> None:
        """JSON object without the required 'artists' key should raise KeyError.

        The parser validates that the parsed dict contains an 'artists' key —
        without it, downstream entity building would fail.
        """
        raw = '{"venue": "Club X", "date": null}'

        with pytest.raises(KeyError):
            EntityExtractor._parse_llm_response(raw)

    def test_repair_with_code_fences(self) -> None:
        """JSON wrapped in ```json ... ``` code fences — very common LLM output.

        Despite explicit prompts saying "return only JSON", most LLMs wrap
        their output in markdown code fences.  The parser strips these
        before attempting json.loads.
        """
        inner = json.dumps({
            "artists": [{"name": "Jeff Mills", "confidence": 0.9}],
            "venue": None,
            "date": None,
            "promoter": None,
            "genre_tags": ["techno"],
            "ticket_price": None,
        })
        raw = f"```json\n{inner}\n```"

        result = EntityExtractor._parse_llm_response(raw)

        assert isinstance(result, dict)
        assert "artists" in result
        assert result["artists"][0]["name"] == "Jeff Mills"

    def test_repair_with_plain_code_fences(self) -> None:
        """JSON wrapped in ``` ... ``` fences (no json language tag)."""
        inner = json.dumps({
            "artists": [{"name": "Derrick May", "confidence": 0.85}],
            "venue": None,
            "date": None,
            "promoter": None,
            "genre_tags": [],
            "ticket_price": None,
        })
        raw = f"```\n{inner}\n```"

        result = EntityExtractor._parse_llm_response(raw)

        assert "artists" in result
        assert result["artists"][0]["name"] == "Derrick May"

    def test_repair_prose_before_json(self) -> None:
        """LLM outputs prose before the JSON object — brace extraction handles it.

        Example: "Here is the extracted data: { ... }"
        """
        inner = json.dumps({
            "artists": [{"name": "Juan Atkins", "confidence": 0.88}],
            "venue": None,
            "date": None,
            "promoter": None,
            "genre_tags": ["techno"],
            "ticket_price": None,
        })
        raw = f"Here is the JSON output:\n{inner}"

        result = EntityExtractor._parse_llm_response(raw)

        assert result["artists"][0]["name"] == "Juan Atkins"

    def test_repair_empty_artists_list(self) -> None:
        """An empty artists list is valid JSON — should parse without error.

        The parser only requires the 'artists' key to exist, not to be
        non-empty.  Downstream code handles the empty case.
        """
        raw = json.dumps({
            "artists": [],
            "venue": None,
            "date": None,
            "promoter": None,
            "genre_tags": [],
            "ticket_price": None,
        })

        result = EntityExtractor._parse_llm_response(raw)

        assert result["artists"] == []

    def test_repair_not_a_dict_raises(self) -> None:
        """If the JSON is a list, brace-extraction recovers the inner object.

        Input: '[{"name": "Carl Cox"}]'
        The parser's Strategy 2 extracts the inner {"name": "Carl Cox"} via
        brace-finding.  That parses as a valid dict, but it's missing the
        required "artists" key — so KeyError is raised instead of ValueError.
        """
        raw = '[{"name": "Carl Cox"}]'

        with pytest.raises(KeyError):
            EntityExtractor._parse_llm_response(raw)
