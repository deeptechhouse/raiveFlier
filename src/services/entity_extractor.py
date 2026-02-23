"""LLM-based entity extraction service for rave flier text.

Sends raw OCR text to an LLM provider with a structured extraction prompt
that encodes rave flier conventions (headliner positioning, date formats,
artist separator tokens like b2b / vs / &).  The JSON response is parsed,
post-processed (artist name splitting), and mapped into immutable Pydantic
models.

Architecture: LLM-as-Parser with Two-Pass Retry
-------------------------------------------------
OCR text from rave fliers is messy, stylized, and full of domain jargon
(e.g. "b2b", stacked DJ names, promoter logos rendered as text).  Rule-based
extraction would require constant maintenance.  Instead, we delegate
parsing to an LLM via a domain-aware prompt, then validate + post-process
the JSON output.

The two-pass retry strategy balances accuracy with resilience:
  - **Pass 1** sends a detailed prompt with rave-flier layout heuristics and
    asks for a rich JSON schema.  This produces the best results when the LLM
    cooperates.
  - **Pass 2** (activated only if Pass 1 returns unparseable JSON) uses a
    stripped-down prompt with lower temperature, reducing the chance of the
    LLM producing creative but malformed output.

The final ``ExtractedEntities`` model feeds directly into the **confirmation
gate** -- a downstream step where the user reviews and corrects entities
before they are committed to the event database.  This means imperfect
extractions are acceptable as long as they are structured; the user is
always the last line of defence.
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.interfaces.llm_provider import ILLMProvider
from src.models.entities import EntityType
from src.models.flier import (
    ExtractedEntities,
    ExtractedEntity,
    FlierImage,
    OCRResult,
)
from src.utils.errors import EntityExtractionError
from src.utils.logging import get_logger
from src.utils.text_normalizer import normalize_artist_name, split_artist_names

# Matches markdown code fences (```json ... ``` or ``` ... ```) that LLMs
# frequently wrap around JSON output despite being asked not to.  The
# DOTALL flag lets the inner capture group span multiple lines.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


class EntityExtractor:
    """Extracts structured entities from raw OCR text using an LLM.

    The extraction follows a two-pass strategy:

    1. **Primary pass** — a detailed prompt that explains rave flier layout
       conventions and requests a structured JSON response.
    2. **Retry pass** — if the first attempt returns unparseable JSON, a
       simpler prompt is sent once.  If that also fails,
       :class:`EntityExtractionError` is raised.

    Post-processing splits multi-artist entries (e.g. "Artist1 b2b Artist2")
    via :func:`split_artist_names` and normalises each name.
    """

    def __init__(
        self,
        llm_provider: ILLMProvider,
        text_normalizer_split: Any = None,
    ) -> None:
        """Initialise the extractor with injected dependencies.

        Parameters
        ----------
        llm_provider:
            The LLM backend used for text completion.
        text_normalizer_split:
            Optional override for the artist-name splitting callable.
            Defaults to :func:`split_artist_names`.
        """
        # LLM provider is injected via the IOCRProvider interface, so the
        # extractor is agnostic about whether the backend is OpenAI, Anthropic,
        # a local model, etc.
        self._llm = llm_provider
        # The split function is injectable for testing (pass a mock that
        # returns the name unchanged) and for future locale-specific splitting
        # rules.  Default handles "b2b", "vs", "&", and "," separators.
        self._split_fn = text_normalizer_split or split_artist_names
        self._logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def extract(
        self,
        ocr_result: OCRResult,
        image: FlierImage | None = None,
    ) -> ExtractedEntities:
        """Extract structured entities from an OCR result.

        Parameters
        ----------
        ocr_result:
            The raw OCR output to parse.
        image:
            Optional original flier image (reserved for future
            vision-augmented extraction).

        Returns
        -------
        ExtractedEntities
            Immutable model containing artists, venue, date, promoter,
            genre tags, and ticket price.

        Raises
        ------
        EntityExtractionError
            If the LLM returns unparseable JSON on both attempts.
        """
        raw_text = ocr_result.raw_text.strip()
        if not raw_text:
            self._logger.warning("empty_ocr_text", provider=ocr_result.provider_used)
            return ExtractedEntities(raw_ocr=ocr_result)

        provider_name = self._llm.get_provider_name()
        self._logger.info(
            "entity_extraction_start",
            ocr_chars=len(raw_text),
            llm_provider=provider_name,
        )

        # ----- Pass 1: Detailed domain-aware prompt -----
        # Uses a rich system prompt that primes the LLM with rave-flier layout
        # conventions (headliner positioning, date placement, separator tokens).
        # Temperature 0.2 keeps output focused while allowing slight creativity
        # for ambiguous text.  4000 max_tokens accommodates fliers with long
        # lineups.
        prompt = self._build_extraction_prompt(raw_text)
        try:
            response = await self._llm.complete(
                system_prompt=self._system_prompt(),
                user_prompt=prompt,
                temperature=0.2,
                max_tokens=4000,
            )
            parsed = self._parse_llm_response(response)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            # Parse failure on the primary attempt is not fatal -- we fall
            # through to the simpler retry prompt below.
            self._logger.warning(
                "primary_extraction_failed",
                error=str(exc),
                provider=provider_name,
            )
            parsed = None

        # ----- Pass 2: Simplified fallback prompt -----
        # If Pass 1 returned malformed JSON (common when the LLM "hallucinates"
        # extra commentary or truncates the object), this retry uses:
        #   - A minimal system prompt (less room for the LLM to go off-script)
        #   - Lower temperature (0.1) for more deterministic output
        #   - Smaller max_tokens (2000) to reduce the chance of runaway generation
        # If this also fails, we raise -- there is no Pass 3.
        if parsed is None:
            self._logger.info("retrying_with_simple_prompt", provider=provider_name)
            simple_prompt = self._build_simple_prompt(raw_text)
            try:
                response = await self._llm.complete(
                    system_prompt="You extract data from text and return valid JSON.",
                    user_prompt=simple_prompt,
                    temperature=0.1,
                    max_tokens=2000,
                )
                parsed = self._parse_llm_response(response)
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                self._logger.error(
                    "retry_extraction_failed",
                    error=str(exc),
                    provider=provider_name,
                )
                raise EntityExtractionError(
                    message=f"LLM returned unparseable JSON after retry: {exc}",
                    provider_name=provider_name,
                ) from exc

        # Convert the validated dict into an immutable Pydantic model.
        # This is the boundary between "unstructured LLM output" and
        # "typed domain objects" -- after this point, the rest of the pipeline
        # works with strongly-typed ExtractedEntities.
        entities = self._build_entities(parsed, ocr_result)
        self._logger.info(
            "entity_extraction_complete",
            artists=len(entities.artists),
            has_venue=entities.venue is not None,
            has_date=entities.date is not None,
            genre_tags=len(entities.genre_tags),
        )
        # The returned ExtractedEntities feeds into the confirmation gate,
        # where a user reviews each entity before it enters the event DB.
        return entities

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _system_prompt() -> str:
        """Return the system prompt that frames the LLM as a flier analyst."""
        return (
            "You are an expert at reading rave, club, and electronic music "
            "event fliers.  You understand the visual and textual conventions "
            "of underground and commercial dance-music promotion.  Your task is "
            "to extract structured event information from raw OCR text and "
            "return it as a JSON object."
        )

    @staticmethod
    def _build_extraction_prompt(ocr_text: str) -> str:
        """Build the primary extraction prompt with rave-flier heuristics.

        Parameters
        ----------
        ocr_text:
            Raw OCR text from the flier image.

        Returns
        -------
        str
            The fully constructed prompt string.
        """
        return (
            "Below is raw OCR text extracted from a rave / electronic music "
            "event flier.  Parse it into the structured JSON format described.\n"
            "\n"
            "## Rave Flier Conventions\n"
            "- **Headliner names** are typically the largest text on the flier.\n"
            "- **Supporting DJs** are listed below headliners, often in smaller type.\n"
            "- **Date** is often near the top or bottom of the flier.\n"
            "- **Venue name and address** are usually near the bottom.\n"
            "- **Promoter** is the person or company organizing the event, often "
            "in small text at the top or bottom.\n"
            "- **Event name / series name** (e.g. 'Bugged Out!', 'BLOC', 'Cream', "
            "'Gatecrasher') is often the most prominent text on the flier, sometimes "
            "even larger than artist names.  It is NOT the venue name.\n"
            '- Tokens like "b2b", "vs", "&" indicate multiple artists on one line.\n'
            '- Date formats vary widely: "Saturday March 15th", "03.15.97", '
            '"15/03/1997", "SAT 15 MAR", etc.\n'
            "- Ticket price may appear as currency amounts (e.g. $10, £5, €8).\n"
            "\n"
            "## Required JSON Output\n"
            "Return **only** a JSON object (no markdown fences, no commentary) "
            "with these keys:\n"
            "```\n"
            "{\n"
            '  "artists": [\n'
            '    {"name": "<artist name>", "confidence": <0.0-1.0>}\n'
            "  ],\n"
            '  "venue": {"name": "<venue>", "confidence": <0.0-1.0>} | null,\n'
            '  "date": {"text": "<date string>", "confidence": <0.0-1.0>} | null,\n'
            '  "promoter": {"name": "<promoter>", "confidence": <0.0-1.0>} | null,\n'
            '  "event_name": {"name": "<event/series name>", "confidence": <0.0-1.0>} | null,\n'
            '  "genre_tags": ["<tag1>", "<tag2>"],\n'
            '  "ticket_price": "<price string>" | null\n'
            "}\n"
            "```\n"
            "\n"
            "- Confidence is your estimated probability that the extraction is "
            "correct (0.0 = guess, 1.0 = certain).\n"
            "- If an entity cannot be identified, use null.\n"
            "- Keep artist names exactly as they appear on the flier.\n"
            "- Include genre_tags only if the text strongly suggests a genre.\n"
            "\n"
            "## OCR Text\n"
            f"```\n{ocr_text}\n```"
        )

    @staticmethod
    def _build_simple_prompt(ocr_text: str) -> str:
        """Build a minimal fallback prompt for the retry attempt.

        Parameters
        ----------
        ocr_text:
            Raw OCR text from the flier image.

        Returns
        -------
        str
            A simpler prompt requesting the same JSON structure.
        """
        return (
            "Extract event information from this text and return valid JSON.\n"
            "\n"
            "Keys: artists (list of {name, confidence}), venue ({name, confidence} "
            "or null), date ({text, confidence} or null), promoter ({name, confidence} "
            "or null), event_name ({name, confidence} or null — the event/series name, "
            "not the venue), genre_tags (list of strings), ticket_price (string or null).\n"
            "\n"
            f"Text:\n{ocr_text}"
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_llm_response(response: str) -> dict[str, Any]:
        """Extract and validate JSON from an LLM response string.

        Handles markdown code fences (````` ``` ````` or ````` ```json `````)
        that many LLMs wrap around their output.

        Parameters
        ----------
        response:
            Raw LLM response text.

        Returns
        -------
        dict
            Parsed JSON object.

        Raises
        ------
        json.JSONDecodeError
            If no valid JSON can be extracted.
        KeyError
            If the parsed object is missing the required ``artists`` key.
        """
        text = response.strip()

        # --- Strategy 1: Strip markdown code fences ---
        # Despite explicit instructions to return bare JSON, most LLMs
        # (GPT-4, Claude, Mistral) wrap output in ```json ... ``` fences.
        # _JSON_FENCE_RE captures the content inside the fences so we can
        # parse it directly.
        fence_match = _JSON_FENCE_RE.search(text)
        if fence_match:
            text = fence_match.group(1).strip()

        # --- Strategy 2: Brace extraction ---
        # If the LLM returned preamble text before the JSON object (e.g.
        # "Here is the extracted data: { ... }"), locate the outermost
        # brace pair.  Uses rfind for the closing brace to handle nested
        # objects correctly.
        if not text.startswith("{"):
            brace_start = text.find("{")
            brace_end = text.rfind("}")
            if brace_start != -1 and brace_end > brace_start:
                text = text[brace_start : brace_end + 1]

        parsed = json.loads(text)

        # Minimal schema validation -- we only require a dict with an
        # "artists" key.  Finer-grained validation happens in _build_entities
        # where each field is handled with safe .get() defaults.
        if not isinstance(parsed, dict):
            raise ValueError("LLM response is not a JSON object")

        if "artists" not in parsed:
            raise KeyError("LLM response missing required 'artists' key")

        return parsed

    # ------------------------------------------------------------------
    # Entity building
    # ------------------------------------------------------------------

    def _build_entities(
        self,
        parsed: dict[str, Any],
        ocr_result: OCRResult,
    ) -> ExtractedEntities:
        """Convert parsed JSON into an immutable ExtractedEntities model.

        Post-processes artist names through the split function and normaliser.

        Parameters
        ----------
        parsed:
            Validated dict from :meth:`_parse_llm_response`.
        ocr_result:
            Original OCR result (embedded in the output model).

        Returns
        -------
        ExtractedEntities
        """
        # --- Artists ---
        # Artist processing is the most complex entity type because rave
        # fliers routinely list multiple artists on one line using separator
        # tokens:  "Carl Cox b2b Joseph Capriati", "Bicep & Hammer",
        # "DJ EZ vs Todd Terry".  The LLM often returns these as a single
        # artist entry, so we must split them into individual names.
        artists: list[ExtractedEntity] = []
        for entry in parsed.get("artists", []):
            raw_name = entry.get("name", "").strip()
            # Default confidence 0.5 (coin-flip) when the LLM omits the field.
            confidence = float(entry.get("confidence", 0.5))
            if not raw_name:
                continue

            # Split multi-artist entries on domain-specific separators
            # (b2b, vs, &, comma).  Each resulting name is then normalised
            # (case-corrected, whitespace-cleaned, common OCR artifacts removed)
            # via normalize_artist_name.  Both functions live in
            # src/utils/text_normalizer to keep this class focused on
            # orchestration rather than string manipulation.
            individual_names = self._split_fn(raw_name)
            for name in individual_names:
                normalised = normalize_artist_name(name)
                if normalised:
                    artists.append(
                        ExtractedEntity(
                            text=normalised,
                            entity_type=EntityType.ARTIST,
                            confidence=confidence,
                        )
                    )

        # --- Venue ---
        venue: ExtractedEntity | None = None
        venue_data = parsed.get("venue")
        if isinstance(venue_data, dict) and venue_data.get("name"):
            venue = ExtractedEntity(
                text=venue_data["name"].strip(),
                entity_type=EntityType.VENUE,
                confidence=float(venue_data.get("confidence", 0.5)),
            )

        # --- Date ---
        date: ExtractedEntity | None = None
        date_data = parsed.get("date")
        if isinstance(date_data, dict) and date_data.get("text"):
            date = ExtractedEntity(
                text=date_data["text"].strip(),
                entity_type=EntityType.DATE,
                confidence=float(date_data.get("confidence", 0.5)),
            )

        # --- Promoter ---
        promoter: ExtractedEntity | None = None
        promoter_data = parsed.get("promoter")
        if isinstance(promoter_data, dict) and promoter_data.get("name"):
            promoter = ExtractedEntity(
                text=promoter_data["name"].strip(),
                entity_type=EntityType.PROMOTER,
                confidence=float(promoter_data.get("confidence", 0.5)),
            )

        # --- Event Name ---
        event_name: ExtractedEntity | None = None
        event_name_data = parsed.get("event_name")
        if isinstance(event_name_data, dict) and event_name_data.get("name"):
            event_name = ExtractedEntity(
                text=event_name_data["name"].strip(),
                entity_type=EntityType.EVENT,
                confidence=float(event_name_data.get("confidence", 0.5)),
            )

        # --- Genre tags ---
        genre_tags: list[str] = []
        raw_tags = parsed.get("genre_tags", [])
        if isinstance(raw_tags, list):
            genre_tags = [str(t).strip() for t in raw_tags if str(t).strip()]

        # --- Ticket price ---
        ticket_price: str | None = None
        raw_price = parsed.get("ticket_price")
        if raw_price is not None:
            price_str = str(raw_price).strip()
            if price_str:
                ticket_price = price_str

        # Build the immutable Pydantic model that represents everything we
        # extracted from this flier.  The raw_ocr field preserves the original
        # OCR output so the confirmation gate UI can display it alongside the
        # parsed entities for user verification.  Each ExtractedEntity carries
        # its own confidence score, enabling the gate to highlight low-confidence
        # items that need human review.
        return ExtractedEntities(
            artists=artists,
            venue=venue,
            date=date,
            promoter=promoter,
            event_name=event_name,
            genre_tags=genre_tags,
            ticket_price=ticket_price,
            raw_ocr=ocr_result,
        )
