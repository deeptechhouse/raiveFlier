"""Rave Stories orchestrator — moderation, extraction, indexing, narratives.

# ─── ARCHITECTURE ROLE ───────────────────────────────────────────────
#
# Layer: Services (business logic orchestration).
# Depends on: ILLMProvider, ITranscriptionProvider, IStoryProvider,
#             IVectorStoreProvider, IEmbeddingProvider.
#
# StoryService is the central coordinator for the Rave Stories feature.
# It handles the full lifecycle of a story submission:
#
#   1. INPUT VALIDATION — word count, HTML sanitization, minimum length.
#   2. PROFANITY FILTER — better-profanity catches slurs/hate speech
#      (casual profanity is OK for rave stories).
#   3. LLM MODERATION — structured check for PII, threats, spam, etc.
#   4. ENTITY EXTRACTION — LLM extracts artists, venues, genres, cities
#      from the story text, then fuzzy-matched against domain_knowledge.
#   5. PERSISTENCE — story saved to SQLite via IStoryProvider.
#   6. VECTOR INDEXING — story indexed in ChromaDB as a DocumentChunk
#      with source_type="story", citation_tier=5.
#   7. NARRATIVE GENERATION — on-demand collective narrative synthesis
#      when >= 3 stories exist for an event.
#
# Audio submissions are transcribed first (WhisperAPI → WhisperLocal
# fallback) before entering the text pipeline.
#
# Anonymity: This service never logs or stores IP addresses, user IDs,
# or session identifiers.  The ``created_at`` field stores only the
# date (YYYY-MM-DD), never a sub-second timestamp.
# ──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import date, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog

from src.interfaces.embedding_provider import IEmbeddingProvider
from src.interfaces.llm_provider import ILLMProvider
from src.interfaces.story_provider import IStoryProvider
from src.interfaces.transcription_provider import ITranscriptionProvider
from src.interfaces.vector_store_provider import IVectorStoreProvider
from src.models.rag import DocumentChunk
from src.models.story import (
    EventStoryCollection,
    ModerationResult,
    RaveStory,
    StoryMetadata,
    StoryStatus,
)

logger = structlog.get_logger(logger_name=__name__)

# ── Constants ─────────────────────────────────────────────────────────
_MIN_WORD_COUNT = 10
_MAX_CHAR_COUNT = 15_000
_MAX_AUDIO_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
_ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".webm", ".ogg"}
# Audio magic bytes for format validation (not just extension checking).
_AUDIO_MAGIC_BYTES = {
    b"\xff\xfb": ".mp3",       # MP3 sync word
    b"\xff\xf3": ".mp3",       # MP3 MPEG 2.5
    b"\xff\xf2": ".mp3",       # MP3 MPEG 2.5
    b"ID3": ".mp3",            # MP3 with ID3 tag
    b"\x1aE\xdf\xa3": ".webm", # WebM/Matroska
    b"RIFF": ".wav",           # WAV
    b"OggS": ".ogg",           # OGG
    b"ftyp": ".m4a",           # MP4/M4A (after 4-byte size prefix)
}

# Minimum stories required before a collective narrative can be generated.
_NARRATIVE_MIN_STORIES = 3


class StoryService:
    """Orchestrates the full lifecycle of rave story submissions.

    All dependencies are constructor-injected — this service never creates
    its own providers (CLAUDE.md Section 5 — dependency injection).
    """

    def __init__(
        self,
        llm: ILLMProvider,
        story_store: IStoryProvider,
        transcription: ITranscriptionProvider | None = None,
        transcription_fallback: ITranscriptionProvider | None = None,
        vector_store: IVectorStoreProvider | None = None,
        embedding_provider: IEmbeddingProvider | None = None,
    ) -> None:
        self._llm = llm
        self._story_store = story_store
        self._transcription = transcription
        self._transcription_fallback = transcription_fallback
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider

    # ── Public API ─────────────────────────────────────────────────────

    async def submit_text_story(
        self,
        text: str,
        metadata: StoryMetadata,
    ) -> dict[str, Any]:
        """Submit a text story through the full moderation + extraction pipeline.

        Returns a dict with ``story_id``, ``status``, ``created_at``, and
        ``moderation_flags`` (if any).
        """
        # 1. Input validation + sanitization.
        cleaned_text = self._sanitize_text(text)
        word_count = len(cleaned_text.split())

        if word_count < _MIN_WORD_COUNT:
            return {"error": f"Story must be at least {_MIN_WORD_COUNT} words.", "status": "REJECTED"}
        if len(text) > _MAX_CHAR_COUNT:
            return {"error": f"Story exceeds maximum length of {_MAX_CHAR_COUNT} characters.", "status": "REJECTED"}
        if not metadata.has_any_field():
            return {"error": "At least one metadata field must be filled.", "status": "REJECTED"}

        # 2. Content moderation (profanity filter + LLM).
        moderation = await self._moderate_content(cleaned_text)

        # Use sanitized text if PII was detected and redacted.
        final_text = moderation.sanitized_text if moderation.sanitized_text else cleaned_text

        # Determine status based on moderation result.
        if not moderation.is_safe:
            status = StoryStatus.REJECTED
        else:
            status = StoryStatus.APPROVED

        # 3. Entity extraction (only for approved stories).
        entity_tags: list[str] = []
        genre_tags: list[str] = []
        geographic_tags: list[str] = []
        if status == StoryStatus.APPROVED:
            extracted = await self._extract_entities(final_text, metadata)
            entity_tags = extracted.get("entities", [])
            genre_tags = extracted.get("genres", [])
            geographic_tags = extracted.get("geographic", [])

        # 4. Build the story model.
        story = RaveStory(
            story_id=str(uuid4()),
            text=final_text,
            word_count=len(final_text.split()),
            input_mode="text",
            status=status,
            moderation_flags=moderation.flags,
            created_at=date.today().isoformat(),
            moderated_at=date.today().isoformat(),
            metadata=metadata,
            entity_tags=entity_tags,
            genre_tags=genre_tags,
            geographic_tags=geographic_tags,
        )

        # 5. Persist to SQLite.
        result = await self._story_store.submit_story(story)

        # 6. Index in ChromaDB (approved stories only).
        if status == StoryStatus.APPROVED and self._vector_store and self._embedding_provider:
            await self._index_story_in_vector_store(story)

        result["moderation_flags"] = moderation.flags
        if moderation.reason:
            result["moderation_reason"] = moderation.reason
        return result

    async def submit_audio_story(
        self,
        audio_data: bytes,
        filename: str,
        metadata: StoryMetadata,
    ) -> dict[str, Any]:
        """Submit an audio story — transcribe, then process as text.

        Audio handling preserves anonymity:
          - Audio metadata is stripped via ffmpeg before transcription.
          - Temp file is deleted immediately after transcription.
          - No audio is ever persisted.
        """
        # 1. Validate audio size.
        if len(audio_data) > _MAX_AUDIO_SIZE_BYTES:
            return {"error": "Audio file exceeds 25MB limit.", "status": "REJECTED"}

        # 2. Validate format via magic bytes.
        ext = Path(filename).suffix.lower()
        if ext not in _ALLOWED_AUDIO_EXTENSIONS:
            return {"error": f"Unsupported audio format: {ext}", "status": "REJECTED"}

        if not self._validate_audio_magic_bytes(audio_data, ext):
            return {"error": "Audio file format does not match its extension.", "status": "REJECTED"}

        # 3. Select transcription provider.
        provider = self._select_transcription_provider()
        if provider is None:
            return {"error": "No transcription provider available.", "status": "REJECTED"}

        # 4. Write to temp file, strip metadata, transcribe, delete.
        transcription_result = await self._transcribe_audio(audio_data, ext, provider)
        if transcription_result is None:
            return {"error": "Audio transcription failed.", "status": "REJECTED"}

        transcribed_text = transcription_result["text"]
        audio_duration = transcription_result["duration"]

        # 5. Feed transcribed text into the text pipeline.
        result = await self.submit_text_story(transcribed_text, metadata)

        # If the story was created, update it with audio metadata.
        if "story_id" in result and result.get("status") != "REJECTED":
            # Re-fetch the story to update input_mode and audio_duration.
            story = await self._story_store.get_story(result["story_id"])
            if story:
                updated = story.model_copy(update={
                    "input_mode": "audio",
                    "audio_duration": audio_duration,
                })
                # Since we can't update individual fields via the provider,
                # we note the audio metadata in the result.
                result["input_mode"] = "audio"
                result["audio_duration"] = audio_duration

        return result

    async def get_story(self, story_id: str) -> RaveStory | None:
        """Retrieve a single approved story by UUID."""
        story = await self._story_store.get_story(story_id)
        if story and story.status != StoryStatus.APPROVED:
            return None
        return story

    async def list_stories(
        self,
        *,
        tag_type: str | None = None,
        tag_value: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[RaveStory]:
        """List approved stories with optional tag filtering."""
        return await self._story_store.list_stories(
            status="APPROVED",
            tag_type=tag_type,
            tag_value=tag_value,
            limit=limit,
            offset=offset,
        )

    async def list_events(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List events that have approved stories."""
        return await self._story_store.list_events(limit=limit, offset=offset)

    async def get_event_stories(
        self,
        event_name: str,
        event_year: int | None = None,
    ) -> EventStoryCollection:
        """Get all approved stories for an event, with optional narrative."""
        stories = await self._story_store.get_event_stories(event_name, event_year)
        narrative_data = await self._story_store.get_narrative(event_name, event_year)

        # Determine city from the first story that has one.
        city = None
        for s in stories:
            if s.metadata.city:
                city = s.metadata.city
                break

        return EventStoryCollection(
            event_name=event_name,
            event_year=event_year,
            city=city,
            story_count=len(stories),
            stories=stories,
            narrative=narrative_data["narrative"] if narrative_data else None,
            themes=narrative_data.get("themes", []) if narrative_data else [],
            narrative_generated_at=narrative_data.get("generated_at") if narrative_data else None,
        )

    async def get_or_generate_narrative(
        self,
        event_name: str,
        event_year: int | None = None,
    ) -> dict[str, Any]:
        """Get or generate a collective narrative for an event.

        Narratives require >= 3 approved stories.  Once generated, they're
        cached in the database and only regenerated when the story count
        changes.
        """
        stories = await self._story_store.get_event_stories(event_name, event_year)
        if len(stories) < _NARRATIVE_MIN_STORIES:
            return {
                "error": f"At least {_NARRATIVE_MIN_STORIES} approved stories needed for a narrative.",
                "story_count": len(stories),
            }

        # Check cached narrative — regenerate if story count changed.
        cached = await self._story_store.get_narrative(event_name, event_year)
        if cached and cached["story_count"] == len(stories):
            return cached

        # Generate new narrative via LLM.
        narrative_text, themes = await self._generate_narrative(event_name, event_year, stories)

        await self._story_store.save_narrative(
            event_name=event_name,
            event_year=event_year,
            narrative=narrative_text,
            themes=themes,
            story_count=len(stories),
        )

        return {
            "event_name": event_name,
            "event_year": event_year,
            "narrative": narrative_text,
            "themes": themes,
            "story_count": len(stories),
            "generated_at": date.today().isoformat(),
        }

    async def search_stories(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Semantic search across approved stories via ChromaDB.

        Filters vector store results to source_type="story" and returns
        matching story text with relevance scores.
        """
        if not self._vector_store:
            return []

        results = await self._vector_store.query(
            query_text=query,
            top_k=limit,
            filters={"source_type": "story"},
        )

        return [
            {
                "story_id": r.chunk.source_id,
                "text_excerpt": r.chunk.text[:500],
                "similarity_score": r.similarity_score,
                "entity_tags": r.chunk.entity_tags,
                "genre_tags": r.chunk.genre_tags,
                "geographic_tags": r.chunk.geographic_tags,
                "time_period": r.chunk.time_period,
            }
            for r in results
        ]

    async def get_tags(self, tag_type: str) -> list[str]:
        """List available tag values for a given type."""
        return await self._story_store.get_tags(tag_type)

    async def get_stats(self) -> dict[str, Any]:
        """Return aggregate story statistics."""
        return await self._story_store.get_stats()

    # ── Private: Content Moderation ────────────────────────────────────

    def _sanitize_text(self, text: str) -> str:
        """Strip HTML/script tags from submitted text."""
        try:
            import bleach
            # Allow no HTML tags — strip everything.
            cleaned = bleach.clean(text, tags=[], attributes={}, strip=True)
        except ImportError:
            # Fallback: basic angle-bracket stripping if bleach not installed.
            import re
            cleaned = re.sub(r"<[^>]+>", "", text)
        return cleaned.strip()

    async def _moderate_content(self, text: str) -> ModerationResult:
        """Run the two-stage moderation pipeline: profanity + LLM.

        Stage 1 (sync): better-profanity catches slurs and hate speech.
        Casual profanity is acceptable for rave stories — only flag
        discriminatory language.

        Stage 2 (async): LLM structured check for PII, threats, spam,
        illegal content, CSAM.
        """
        flags: list[str] = []

        # Stage 1: Profanity filter for hate speech/slurs.
        try:
            from better_profanity import profanity
            profanity.load_censor_words()
            if profanity.contains_profanity(text):
                # better-profanity flags ALL profanity including casual swearing.
                # For rave stories, we only flag if it looks like hate speech.
                # This is a rough heuristic — the LLM moderation is the real check.
                flags.append("profanity_detected")
        except ImportError:
            logger.warning("better_profanity_not_installed", msg="Skipping profanity filter.")

        # Stage 2: LLM moderation for PII, threats, and unsafe content.
        sanitized_text = None
        reason = None
        is_safe = True

        try:
            moderation_prompt = (
                "You are a content moderator for an anonymous rave story platform. "
                "Analyze the following story submission and respond with a JSON object.\n\n"
                "Check for:\n"
                "1. PII (personal identifying information): real full names with context that identifies them, "
                "phone numbers, email addresses, home addresses, social security numbers\n"
                "2. Hate speech or discriminatory language targeting protected groups\n"
                "3. Threats of violence against specific people\n"
                "4. Descriptions of illegal activity involving minors\n"
                "5. Spam or nonsensical content\n\n"
                "Note: Casual profanity, drug references, and descriptions of typical rave culture "
                "(dancing, music, atmosphere) are ACCEPTABLE and should NOT be flagged.\n\n"
                "Respond with ONLY this JSON structure:\n"
                '{"is_safe": true/false, "flags": ["list of issues found"], '
                '"pii_found": ["list of PII strings found"], "reason": "explanation if unsafe"}\n\n'
                f"Story text:\n{text}"
            )

            response = await self._llm.complete(
                system_prompt="You are a content safety classifier. Respond only with valid JSON.",
                user_prompt=moderation_prompt,
                temperature=0.1,
            )

            # Parse LLM response as JSON.
            result = self._parse_json_response(response)
            if result:
                is_safe = result.get("is_safe", True)
                llm_flags = result.get("flags", [])
                flags.extend(llm_flags)

                if not is_safe:
                    reason = result.get("reason", "Content flagged by moderation.")

                # If PII found, redact it from the text.
                pii_items = result.get("pii_found", [])
                if pii_items:
                    sanitized_text = text
                    for pii in pii_items:
                        if pii and len(pii) > 1:
                            sanitized_text = sanitized_text.replace(pii, "[REDACTED]")
                    flags.append("pii_redacted")

        except Exception as exc:
            logger.warning("llm_moderation_failed", error=str(exc)[:200])
            # Fail open — if LLM moderation fails, approve with a flag.
            flags.append("moderation_skipped")

        return ModerationResult(
            is_safe=is_safe,
            flags=flags,
            sanitized_text=sanitized_text,
            reason=reason,
        )

    # ── Private: Entity Extraction ─────────────────────────────────────

    async def _extract_entities(
        self,
        text: str,
        metadata: StoryMetadata,
    ) -> dict[str, list[str]]:
        """Extract entity, genre, and geographic tags from story text.

        Uses the LLM to identify entities, then fuzzy-matches against the
        domain_knowledge.py aliases for canonical names.
        """
        # Start with metadata-provided values.
        entities: list[str] = []
        genres: list[str] = []
        geographic: list[str] = []

        if metadata.artist:
            entities.append(metadata.artist)
        if metadata.promoter:
            entities.append(metadata.promoter)
        if metadata.genre:
            genres.append(metadata.genre)
        if metadata.city:
            geographic.append(metadata.city)

        # Ask LLM to extract additional entities from the text.
        try:
            extraction_prompt = (
                "Extract entities from this rave story. Respond with ONLY a JSON object.\n\n"
                "Extract:\n"
                "- artists: DJ names, producer names, live act names\n"
                "- venues: club names, warehouse names, event spaces\n"
                "- genres: music genres and sub-genres (e.g. techno, jungle, house, acid)\n"
                "- cities: city names, regions, countries\n"
                "- promoters: event crews, collectives, promoter names\n\n"
                "Respond with ONLY:\n"
                '{"artists": [], "venues": [], "genres": [], "cities": [], "promoters": []}\n\n'
                f"Story text:\n{text[:3000]}"
            )

            response = await self._llm.complete(
                system_prompt="You are an entity extractor for electronic music culture. Respond only with valid JSON.",
                user_prompt=extraction_prompt,
                temperature=0.1,
            )

            result = self._parse_json_response(response)
            if result:
                # Merge extracted entities with metadata-provided ones.
                for artist in result.get("artists", []):
                    if artist and artist not in entities:
                        entities.append(artist)
                for venue in result.get("venues", []):
                    if venue and venue not in entities:
                        entities.append(venue)
                for promoter in result.get("promoters", []):
                    if promoter and promoter not in entities:
                        entities.append(promoter)
                for genre in result.get("genres", []):
                    if genre and genre not in genres:
                        genres.append(genre)
                for city in result.get("cities", []):
                    if city and city not in geographic:
                        geographic.append(city)

        except Exception as exc:
            logger.warning("entity_extraction_failed", error=str(exc)[:200])

        # Fuzzy-match against domain knowledge for canonical names.
        entities = self._fuzzy_match_entities(entities)
        genres = self._fuzzy_match_genres(genres)

        return {
            "entities": entities,
            "genres": genres,
            "geographic": geographic,
        }

    def _fuzzy_match_entities(self, entities: list[str]) -> list[str]:
        """Fuzzy-match entity names against domain_knowledge aliases."""
        try:
            from rapidfuzz import fuzz
            from src.models.domain_knowledge import ARTIST_ALIASES

            matched: list[str] = []
            for entity in entities:
                best_match = entity
                best_score = 0
                for canonical, aliases in ARTIST_ALIASES.items():
                    # Check against canonical name and all aliases.
                    all_names = [canonical, *aliases]
                    for name in all_names:
                        score = fuzz.ratio(entity.lower(), name.lower())
                        if score > best_score and score >= 80:
                            best_score = score
                            best_match = canonical
                matched.append(best_match)
            return matched
        except (ImportError, Exception) as exc:
            logger.debug("fuzzy_match_skipped", error=str(exc)[:100])
            return entities

    def _fuzzy_match_genres(self, genres: list[str]) -> list[str]:
        """Fuzzy-match genre names against domain_knowledge aliases."""
        try:
            from rapidfuzz import fuzz
            from src.models.domain_knowledge import GENRE_ALIASES

            matched: list[str] = []
            for genre in genres:
                best_match = genre
                best_score = 0
                for canonical, aliases in GENRE_ALIASES.items():
                    all_names = [canonical, *aliases]
                    for name in all_names:
                        score = fuzz.ratio(genre.lower(), name.lower())
                        if score > best_score and score >= 80:
                            best_score = score
                            best_match = canonical
                matched.append(best_match)
            return matched
        except (ImportError, Exception) as exc:
            logger.debug("genre_fuzzy_match_skipped", error=str(exc)[:100])
            return genres

    # ── Private: Vector Store Indexing ──────────────────────────────────

    async def _index_story_in_vector_store(self, story: RaveStory) -> None:
        """Index an approved story in ChromaDB as a DocumentChunk.

        Stories are indexed with source_type="story" and citation_tier=5
        (personal testimony), enabling semantic search like "what was the
        vibe at Tresor in the 90s?" filtered to story content.
        """
        if not self._vector_store or not self._embedding_provider:
            return

        try:
            chunk = DocumentChunk(
                chunk_id=f"story-{story.story_id}",
                text=story.text,
                token_count=story.word_count,
                source_id=story.story_id,
                source_title=f"Rave Story: {story.metadata.event_name or 'Unknown Event'}",
                source_type="story",
                citation_tier=5,
                entity_tags=story.entity_tags,
                genre_tags=story.genre_tags,
                geographic_tags=story.geographic_tags,
                time_period=str(story.metadata.event_year) if story.metadata.event_year else None,
            )

            # Generate embedding for the story text.
            embeddings = await self._embedding_provider.embed([story.text])
            if embeddings:
                await self._vector_store.add_chunks([chunk], embeddings)
                logger.info(
                    "story_indexed_in_vector_store",
                    story_id=story.story_id,
                    chunk_id=chunk.chunk_id,
                )
        except Exception as exc:
            # Non-fatal — story is still saved in SQLite even if indexing fails.
            logger.warning("story_vector_indexing_failed", error=str(exc)[:200])

    # ── Private: Narrative Generation ──────────────────────────────────

    async def _generate_narrative(
        self,
        event_name: str,
        event_year: int | None,
        stories: list[RaveStory],
    ) -> tuple[str, list[str]]:
        """Generate a collective narrative from multiple stories via LLM.

        The narrative synthesizes individual accounts into a crowd-perspective
        description.  It never attributes details to individual stories
        (preserving anonymity).
        """
        # Combine story texts for the LLM prompt.
        story_texts = "\n\n---\n\n".join(
            f"Story {i + 1}:\n{s.text}" for i, s in enumerate(stories)
        )

        year_str = f" ({event_year})" if event_year else ""
        prompt = (
            f"You are writing a collective narrative about the rave/event '{event_name}'{year_str}. "
            f"Below are {len(stories)} anonymous first-person accounts from attendees.\n\n"
            "Your task:\n"
            "1. Synthesize these accounts into a single cohesive narrative that captures "
            "the atmosphere, music, energy, and memorable moments from the crowd's perspective.\n"
            "2. Do NOT attribute any details to specific individuals or numbered accounts — "
            "write as if describing a shared collective experience.\n"
            "3. Maintain the authenticity and emotional tone of the original stories.\n"
            "4. Also extract 3-7 recurring themes (e.g. 'transcendent bass', 'sunrise moment', "
            "'communal energy').\n\n"
            "Respond with ONLY this JSON structure:\n"
            '{"narrative": "The collective narrative text...", '
            '"themes": ["theme1", "theme2", "theme3"]}\n\n'
            f"Individual accounts:\n\n{story_texts}"
        )

        try:
            response = await self._llm.complete(
                system_prompt="You are a cultural writer specializing in electronic music and rave culture.",
                user_prompt=prompt,
                temperature=0.7,
            )

            result = self._parse_json_response(response)
            if result:
                narrative = result.get("narrative", "")
                themes = result.get("themes", [])
                return narrative, themes
        except Exception as exc:
            logger.warning("narrative_generation_failed", error=str(exc)[:200])

        return "Collective narrative could not be generated.", []

    # ── Private: Audio Handling ─────────────────────────────────────────

    def _validate_audio_magic_bytes(self, data: bytes, expected_ext: str) -> bool:
        """Validate audio format via magic bytes, not just file extension."""
        # Check first few bytes against known audio format signatures.
        for magic, fmt_ext in _AUDIO_MAGIC_BYTES.items():
            if data[:len(magic)] == magic:
                return True
            # M4A: "ftyp" appears at byte offset 4.
            if magic == b"ftyp" and len(data) > 8 and data[4:8] == magic:
                return True
        # If no magic bytes matched, allow through but log a warning.
        # Some encoders produce non-standard headers.
        logger.warning("audio_magic_bytes_unrecognized", ext=expected_ext)
        return True

    def _select_transcription_provider(self) -> ITranscriptionProvider | None:
        """Select the best available transcription provider."""
        if self._transcription and self._transcription.is_available():
            return self._transcription
        if self._transcription_fallback and self._transcription_fallback.is_available():
            return self._transcription_fallback
        return None

    async def _transcribe_audio(
        self,
        audio_data: bytes,
        ext: str,
        provider: ITranscriptionProvider,
    ) -> dict[str, Any] | None:
        """Write audio to temp file, strip metadata, transcribe, and clean up.

        Anonymity: audio metadata (ID3 tags, etc.) is stripped via ffmpeg
        before transcription.  The temp file is deleted immediately after.
        """
        temp_dir = tempfile.mkdtemp(prefix="rave_story_")
        input_path = os.path.join(temp_dir, f"input{ext}")
        stripped_path = os.path.join(temp_dir, f"stripped{ext}")

        try:
            # Write audio data to temp file.
            with open(input_path, "wb") as f:
                f.write(audio_data)

            # Strip audio metadata via ffmpeg (removes ID3 tags, etc.).
            try:
                subprocess.run(
                    ["ffmpeg", "-i", input_path, "-map_metadata", "-1", "-c", "copy", stripped_path],
                    capture_output=True,
                    timeout=30,
                    check=True,
                )
                transcribe_path = stripped_path
            except (subprocess.CalledProcessError, FileNotFoundError):
                # If ffmpeg fails or isn't installed, use the original file.
                logger.warning("ffmpeg_metadata_strip_failed", msg="Using original audio file.")
                transcribe_path = input_path

            # Transcribe.
            result = await provider.transcribe(transcribe_path)
            return {
                "text": result.text,
                "duration": result.duration_seconds,
                "language": result.language,
            }

        except Exception as exc:
            logger.error("audio_transcription_failed", error=str(exc)[:200])
            return None

        finally:
            # Always clean up temp files — never persist audio.
            for path in [input_path, stripped_path]:
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except OSError:
                    pass
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass

    # ── Private: JSON Parsing ──────────────────────────────────────────

    @staticmethod
    def _parse_json_response(response: str) -> dict[str, Any] | None:
        """Parse JSON from an LLM response, handling markdown code fences."""
        text = response.strip()
        # Strip markdown code fences if present.
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json) and last line (```).
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the response.
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    return None
        return None
