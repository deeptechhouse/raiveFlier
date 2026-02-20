"""LLM-powered metadata tag extraction for document chunks.

Uses an :class:`~src.interfaces.llm_provider.ILLMProvider` to extract
entity, geographic, and genre tags from chunk text.  Tags enable filtered
semantic retrieval — e.g. retrieve only chunks mentioning a specific artist
published before a given date.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.interfaces.llm_provider import ILLMProvider
    from src.models.rag import DocumentChunk

logger = structlog.get_logger(logger_name=__name__)

_EXTRACTION_SYSTEM_PROMPT = (
    "You are a metadata extraction assistant specializing in " "electronic and dance music culture."
)

_EXTRACTION_USER_PROMPT = """\
Extract structured tags from this text about electronic/dance music.
Return JSON: {{"entities": [...], "places": [...], "genres": [...]}}
Only include items explicitly mentioned in the text. Be precise.

Text:
{text}"""


class MetadataExtractor:
    """Extracts entity, geographic, and genre tags from chunk text using an LLM.

    Parameters
    ----------
    llm:
        The LLM provider used for tag extraction prompts.
    max_concurrent:
        Maximum number of concurrent LLM calls (default 5) to avoid rate limits.
    """

    def __init__(self, llm: ILLMProvider, max_concurrent: int = 5) -> None:
        self._llm = llm
        self._semaphore = asyncio.Semaphore(max_concurrent)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def extract_tags(self, chunk_text: str) -> dict[str, list[str]]:
        """Extract entity, geographic, and genre tags from *chunk_text*.

        Returns
        -------
        dict
            Keys: ``entity_tags``, ``geographic_tags``, ``genre_tags``.
            Each value is a list of strings.  On LLM failure returns empty
            lists so ingestion is never blocked.
        """
        empty: dict[str, list[str]] = {
            "entity_tags": [],
            "geographic_tags": [],
            "genre_tags": [],
        }

        try:
            async with self._semaphore:
                response = await self._llm.complete(
                    system_prompt=_EXTRACTION_SYSTEM_PROMPT,
                    user_prompt=_EXTRACTION_USER_PROMPT.format(text=chunk_text[:3000]),
                    temperature=0.1,
                    max_tokens=500,
                )
            return self._parse_response(response)
        except Exception:  # noqa: BLE001
            logger.warning(
                "metadata_extraction_failed",
                text_preview=chunk_text[:80],
                msg="Returning empty tags — ingestion continues.",
            )
            return empty

    async def extract_batch(
        self,
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        """Extract and apply metadata tags to every chunk in *chunks*.

        Uses ``asyncio.gather`` with a semaphore to limit concurrency.

        Parameters
        ----------
        chunks:
            Document chunks to enrich with metadata tags.

        Returns
        -------
        list[DocumentChunk]
            New chunk instances with ``entity_tags``, ``geographic_tags``,
            and ``genre_tags`` populated.
        """
        if not chunks:
            return []

        tasks = [self._tag_single_chunk(chunk) for chunk in chunks]
        results: list[DocumentChunk] = await asyncio.gather(*tasks)

        tagged_count = sum(1 for c in results if c.entity_tags or c.geographic_tags or c.genre_tags)
        logger.info(
            "batch_extraction_complete",
            total=len(results),
            tagged=tagged_count,
        )
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _tag_single_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """Extract tags for a single chunk and return a copy with tags applied."""
        from src.models.rag import DocumentChunk as DocumentChunkModel

        tags = await self.extract_tags(chunk.text)
        return DocumentChunkModel(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            token_count=chunk.token_count,
            source_id=chunk.source_id,
            source_title=chunk.source_title,
            source_type=chunk.source_type,
            author=chunk.author,
            publication_date=chunk.publication_date,
            citation_tier=chunk.citation_tier,
            page_number=chunk.page_number,
            entity_tags=tags.get("entity_tags", []),
            geographic_tags=tags.get("geographic_tags", []),
            genre_tags=tags.get("genre_tags", []),
        )

    @staticmethod
    def _parse_response(response: str) -> dict[str, list[str]]:
        """Parse the LLM JSON response into a normalised tag dict.

        Handles responses wrapped in markdown code fences and partial JSON.
        """
        empty: dict[str, list[str]] = {
            "entity_tags": [],
            "geographic_tags": [],
            "genre_tags": [],
        }

        # Strip markdown code fences and any preamble text.
        cleaned = response.strip()

        # Try to extract JSON from between code fences first.
        import re

        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()
        else:
            # Fallback: extract the first top-level JSON object.
            brace_start = cleaned.find("{")
            brace_end = cleaned.rfind("}")
            if brace_start != -1 and brace_end > brace_start:
                cleaned = cleaned[brace_start : brace_end + 1]

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(
                "json_parse_failed",
                response_preview=response[:200],
            )
            return empty

        if not isinstance(data, dict):
            return empty

        def _as_str_list(val: object) -> list[str]:
            if isinstance(val, list):
                return [str(v) for v in val if v]
            return []

        return {
            "entity_tags": _as_str_list(data.get("entities")),
            "geographic_tags": _as_str_list(data.get("places")),
            "genre_tags": _as_str_list(data.get("genres")),
        }
