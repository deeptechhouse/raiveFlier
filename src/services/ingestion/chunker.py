"""Text chunking with overlapping windows and paragraph boundary preservation.

Splits source text into :class:`~src.models.rag.DocumentChunk` objects sized
for embedding models (~500 tokens each with 100-token overlap).  Paragraph
boundaries are preserved so no chunk starts or ends mid-thought.  When a
single paragraph exceeds the chunk budget, it is split at sentence boundaries.
"""

from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING

import structlog

from src.models.rag import DocumentChunk

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(logger_name=__name__)

# Common abbreviations that should NOT trigger a sentence split.
_ABBREVIATIONS = frozenset(
    {
        "Dr",
        "Mr",
        "Mrs",
        "Ms",
        "Prof",
        "Jr",
        "Sr",
        "St",
        "Ave",
        "Blvd",
        "Vol",
        "No",
        "vs",
        "etc",
        "approx",
        "dept",
        "est",
        "govt",
        "inc",
        "ltd",
        "co",
        "ft",
    }
)


class TextChunker:
    """Splits text into overlapping chunks preserving paragraph boundaries.

    Parameters
    ----------
    chunk_size:
        Target maximum token count per chunk (default 500).
    overlap:
        Number of tokens of overlap between consecutive chunks (default 100).
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 100) -> None:
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._tokenizer = self._load_tokenizer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, text: str, source_metadata: dict[str, object]) -> list[DocumentChunk]:
        """Split *text* into overlapping :class:`DocumentChunk` objects.

        Parameters
        ----------
        text:
            The full text to chunk.
        source_metadata:
            Metadata dict copied into every chunk.  Expected keys:
            ``source_id``, ``source_title``, ``source_type``, and
            optionally ``citation_tier``, ``author``, ``publication_date``,
            ``page_number``.

        Returns
        -------
        list[DocumentChunk]
            One chunk per window.  Empty input returns an empty list.
        """
        if not text or not text.strip():
            return []

        paragraphs = self._split_paragraphs(text)
        raw_chunks = self._accumulate_chunks(paragraphs)

        chunks: list[DocumentChunk] = []
        for chunk_text in raw_chunks:
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                text=chunk_text,
                source_id=str(source_metadata.get("source_id", "")),
                source_title=str(source_metadata.get("source_title", "")),
                source_type=str(source_metadata.get("source_type", "unknown")),
                citation_tier=int(source_metadata.get("citation_tier", 6)),
                author=source_metadata.get("author"),  # type: ignore[arg-type]
                publication_date=source_metadata.get("publication_date"),  # type: ignore[arg-type]
                page_number=source_metadata.get("page_number"),  # type: ignore[arg-type]
            )
            chunks.append(chunk)

        logger.debug(
            "chunking_complete",
            num_chunks=len(chunks),
            avg_tokens=self._avg_tokens(chunks),
            source_id=source_metadata.get("source_id"),
        )
        return chunks

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        """Return an accurate token count for *text*.

        Uses the HuggingFace ``tokenizers`` library when available, falling
        back to an approximate ``len(text) // 4`` heuristic.
        """
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text).ids)
        return len(text) // 4

    @staticmethod
    def _load_tokenizer():  # noqa: ANN205 – optional dep
        """Attempt to load a fast tokenizer for token counting."""
        try:
            from tokenizers import Tokenizer  # type: ignore[import-untyped]

            return Tokenizer.from_pretrained("bert-base-uncased")
        except Exception:  # noqa: BLE001
            logger.info(
                "tokenizers_unavailable",
                msg="Falling back to approximate token counting (len // 4).",
            )
            return None

    # ------------------------------------------------------------------
    # Paragraph / sentence splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        """Split *text* on double-newlines, discarding blanks."""
        parts = re.split(r"\n\s*\n", text)
        return [p.strip() for p in parts if p.strip()]

    def _split_sentences(self, text: str) -> list[str]:
        """Split *text* at sentence boundaries while respecting abbreviations.

        Handles ``.``, ``!``, ``?`` followed by whitespace or end-of-string.
        Common abbreviations (Dr., Mr., etc.) do not trigger a split.

        Uses a masking approach instead of a variable-width lookbehind
        (which Python 3.10's ``re`` module does not support).
        """
        # Temporarily mask periods after known abbreviations so they
        # don't trigger a sentence split.  Replace '.' with '\x00'
        # (same length, keeps indices aligned with the original text).
        masked = text
        for abbr in _ABBREVIATIONS:
            masked = masked.replace(f"{abbr}.", f"{abbr}\x00")

        sentences: list[str] = []
        last = 0
        for match in re.finditer(r"[.!?](?:\s|$)", masked):
            end = match.end()
            sentence = text[last:end].strip()
            if sentence:
                sentences.append(sentence)
            last = end

        # Trailing text that didn't end with punctuation.
        remainder = text[last:].strip()
        if remainder:
            sentences.append(remainder)

        return sentences if sentences else [text]

    # ------------------------------------------------------------------
    # Chunk accumulation
    # ------------------------------------------------------------------

    def _accumulate_chunks(self, paragraphs: list[str]) -> list[str]:
        """Accumulate paragraphs into chunks respecting *chunk_size* and *overlap*."""
        chunks: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self._count_tokens(para)

            # If a single paragraph exceeds the budget, split it by sentence.
            if para_tokens > self._chunk_size:
                # Flush anything accumulated so far.
                if current_parts:
                    chunks.append("\n\n".join(current_parts))
                    current_parts = []
                    current_tokens = 0

                sentence_chunks = self._chunk_long_paragraph(para)
                chunks.extend(sentence_chunks)
                continue

            # Would adding this paragraph exceed the limit?
            if current_tokens + para_tokens > self._chunk_size and current_parts:
                chunks.append("\n\n".join(current_parts))

                # Start next chunk with overlap from the tail of the previous.
                current_parts, current_tokens = self._build_overlap(current_parts)

            current_parts.append(para)
            current_tokens += para_tokens

        # Flush remaining content.
        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return chunks

    def _chunk_long_paragraph(self, paragraph: str) -> list[str]:
        """Split a paragraph that exceeds *chunk_size* at sentence boundaries."""
        sentences = self._split_sentences(paragraph)
        chunks: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self._count_tokens(sentence)
            if current_tokens + sent_tokens > self._chunk_size and current_parts:
                chunks.append(" ".join(current_parts))
                # Overlap: keep trailing sentences.
                current_parts, current_tokens = self._build_overlap_sentences(current_parts)
            current_parts.append(sentence)
            current_tokens += sent_tokens

        if current_parts:
            chunks.append(" ".join(current_parts))

        return chunks

    def _build_overlap(self, parts: list[str]) -> tuple[list[str], int]:
        """Return tail paragraphs from *parts* whose combined tokens <= *overlap*."""
        overlap_parts: list[str] = []
        overlap_tokens = 0
        for part in reversed(parts):
            part_tokens = self._count_tokens(part)
            if overlap_tokens + part_tokens > self._overlap:
                break
            overlap_parts.insert(0, part)
            overlap_tokens += part_tokens
        return overlap_parts, overlap_tokens

    def _build_overlap_sentences(self, parts: list[str]) -> tuple[list[str], int]:
        """Return tail sentences from *parts* whose combined tokens <= *overlap*."""
        return self._build_overlap(parts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _avg_tokens(self, chunks: list[DocumentChunk]) -> int:
        """Return the average token count across *chunks*."""
        if not chunks:
            return 0
        return sum(self._count_tokens(c.text) for c in chunks) // len(chunks)
