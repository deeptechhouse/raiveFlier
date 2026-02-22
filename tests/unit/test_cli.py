"""Unit tests for CLI modules — src.cli.analyze and src.cli.ingest."""

from __future__ import annotations

import hashlib
import io
import json
from argparse import Namespace
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from src.config.settings import Settings
from src.models.analysis import (
    Citation,
    EntityNode,
    InterconnectionMap,
    PatternInsight,
    RelationshipEdge,
)
from src.models.entities import (
    Artist,
    EntityType,
    Label,
    Promoter,
    Release,
    Venue,
)
from src.models.flier import (
    ExtractedEntities,
    ExtractedEntity,
    FlierImage,
    OCRResult,
)
from src.models.pipeline import PipelineError, PipelinePhase, PipelineState
from src.models.research import DateContext, ResearchResult


# ======================================================================
# Shared helpers
# ======================================================================


def _settings(**overrides) -> Settings:
    """Build a Settings instance with sensible test defaults."""
    defaults = {
        "openai_api_key": "sk-test",
        "openai_base_url": "",
        "openai_text_model": "",
        "openai_vision_model": "",
        "openai_embedding_model": "",
        "anthropic_api_key": "test-anthropic",
        "ollama_base_url": "http://localhost:11434",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_flier_image() -> FlierImage:
    """Create a minimal FlierImage with raw JPEG data attached."""
    img = Image.new("RGB", (400, 300), (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    image_data = buf.getvalue()

    flier = FlierImage(
        filename="test.jpg",
        content_type="image/jpeg",
        file_size=len(image_data),
        image_hash=hashlib.sha256(image_data).hexdigest(),
    )
    flier.__pydantic_private__["_image_data"] = image_data
    return flier


def _make_ocr_result() -> OCRResult:
    """Build a sample OCRResult."""
    return OCRResult(
        raw_text="CARL COX\nTRESOR BERLIN\nSATURDAY MARCH 15 1997",
        confidence=0.85,
        provider_used="tesseract",
        processing_time=1.23,
    )


def _make_extracted_entities(ocr: OCRResult) -> ExtractedEntities:
    """Build sample extracted entities."""
    return ExtractedEntities(
        artists=[
            ExtractedEntity(text="Carl Cox", entity_type=EntityType.ARTIST, confidence=0.95),
        ],
        venue=ExtractedEntity(text="Tresor, Berlin", entity_type=EntityType.VENUE, confidence=0.88),
        date=ExtractedEntity(
            text="Saturday March 15th 1997", entity_type=EntityType.DATE, confidence=0.85,
        ),
        promoter=ExtractedEntity(
            text="Tresor Records", entity_type=EntityType.PROMOTER, confidence=0.70,
        ),
        event_name=ExtractedEntity(
            text="Tresor Night", entity_type=EntityType.EVENT, confidence=0.60,
        ),
        genre_tags=["techno"],
        ticket_price="10 DM",
        raw_ocr=ocr,
    )


def _make_research_results() -> list[ResearchResult]:
    """Build sample research results covering artist, venue, promoter, date."""
    artist = Artist(
        name="Carl Cox",
        discogs_id=12345,
        musicbrainz_id="abc-123",
        confidence=0.95,
        profile_summary="British techno DJ and producer known for three-deck mixing.",
        releases=[
            Release(title="Phat Trax", label="React", year=1995),
            Release(title="Two Paintings and a Drum", label="Intec", year=1996),
        ],
        labels=[
            Label(name="Intec", discogs_id=100),
            Label(name="React", discogs_id=200),
        ],
    )

    venue = Venue(
        name="Tresor",
        city="Berlin",
        country="Germany",
        history="Opened in 1991 in the vault of a former department store.",
    )

    promoter = Promoter(
        name="Tresor Records",
        event_history=["Tresor nights", "Love Parade"],
    )

    date_context = DateContext(
        event_date=date(1997, 3, 15),
        scene_context="Berlin techno at its peak in 1997.",
        cultural_context="Post-reunification culture thriving.",
    )

    return [
        ResearchResult(
            entity_type=EntityType.ARTIST,
            entity_name="Carl Cox",
            artist=artist,
            confidence=0.95,
            sources_consulted=["discogs", "musicbrainz"],
            warnings=["Limited release data before 1993"],
        ),
        ResearchResult(
            entity_type=EntityType.VENUE,
            entity_name="Tresor",
            venue=venue,
            confidence=0.88,
            sources_consulted=["wikipedia"],
        ),
        ResearchResult(
            entity_type=EntityType.PROMOTER,
            entity_name="Tresor Records",
            promoter=promoter,
            confidence=0.70,
            sources_consulted=["discogs"],
        ),
        ResearchResult(
            entity_type=EntityType.DATE,
            entity_name="1997-03-15",
            date_context=date_context,
            confidence=0.85,
            sources_consulted=["ra.co"],
        ),
    ]


def _make_interconnection_map() -> InterconnectionMap:
    """Build a sample interconnection map."""
    citation = Citation(
        text="Carl Cox played at Tresor Berlin in 1997",
        source_type="press",
        source_name="Resident Advisor",
        source_url="https://ra.co/features/carl-cox-tresor",
        tier=2,
    )
    return InterconnectionMap(
        nodes=[
            EntityNode(entity_type=EntityType.ARTIST, name="Carl Cox"),
            EntityNode(entity_type=EntityType.VENUE, name="Tresor"),
        ],
        edges=[
            RelationshipEdge(
                source="Carl Cox",
                target="Tresor",
                relationship_type="performed_at",
                details="Played at Tresor Berlin in spring 1997",
                citations=[citation],
                confidence=0.9,
            ),
        ],
        patterns=[
            PatternInsight(
                pattern_type="venue_residency",
                description="Carl Cox had recurring appearances at Tresor",
                involved_entities=["Carl Cox", "Tresor"],
            ),
        ],
        narrative="Carl Cox was a key figure in the Berlin techno scene.",
        citations=[citation],
    )


def _make_full_pipeline_state() -> PipelineState:
    """Build a fully-populated PipelineState for formatting tests."""
    flier = _make_flier_image()
    ocr = _make_ocr_result()
    entities = _make_extracted_entities(ocr)
    research = _make_research_results()
    imap = _make_interconnection_map()

    return PipelineState(
        session_id="test-session-001",
        flier=flier,
        current_phase=PipelinePhase.OUTPUT,
        ocr_result=ocr,
        extracted_entities=entities,
        confirmed_entities=entities,
        research_results=research,
        interconnection_map=imap,
        completed_at=datetime(2026, 2, 22, 12, 0, 0, tzinfo=timezone.utc),
        errors=[
            PipelineError(
                phase=PipelinePhase.RESEARCH,
                message="Bandcamp rate limit exceeded",
            ),
        ],
        progress_percent=100.0,
    )


def _make_minimal_pipeline_state() -> PipelineState:
    """Build a PipelineState with only session_id and flier (no results)."""
    flier = _make_flier_image()
    return PipelineState(
        session_id="minimal-session",
        flier=flier,
    )


# ======================================================================
# TestAnalyzeBuildParser
# ======================================================================


class TestAnalyzeBuildParser:
    """Tests for src.cli.analyze._build_parser."""

    def test_parser_with_image_only(self) -> None:
        from src.cli.analyze import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["flier.jpg"])

        assert args.image == "flier.jpg"
        assert args.json_output is False
        assert args.output is None
        assert args.quiet is False

    def test_parser_with_json_flag(self) -> None:
        from src.cli.analyze import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["flier.jpg", "--json"])

        assert args.image == "flier.jpg"
        assert args.json_output is True

    def test_parser_with_output_file(self) -> None:
        from src.cli.analyze import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["flier.png", "--output", "results.json"])

        assert args.image == "flier.png"
        assert args.output == "results.json"

        # Short flag -o should also work
        args_short = parser.parse_args(["flier.png", "-o", "out.txt"])
        assert args_short.output == "out.txt"

    def test_parser_with_quiet_flag(self) -> None:
        from src.cli.analyze import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["flier.webp", "--quiet"])

        assert args.quiet is True

        # Short flag -q should also work
        args_short = parser.parse_args(["flier.webp", "-q"])
        assert args_short.quiet is True


# ======================================================================
# TestFormatTextOutput
# ======================================================================


class TestFormatTextOutput:
    """Tests for src.cli.analyze._format_text_output."""

    def test_format_text_with_full_state(self) -> None:
        from src.cli.analyze import _format_text_output

        state = _make_full_pipeline_state()
        text = _format_text_output(state)

        # Header
        assert "raiveFlier" in text
        assert "Analysis Report" in text

        # OCR section
        assert "tesseract" in text
        assert "85%" in text
        assert "1.23" in text

        # Entity section
        assert "Carl Cox" in text
        assert "Tresor, Berlin" in text
        assert "Saturday March 15th 1997" in text
        assert "Tresor Records" in text
        assert "Tresor Night" in text
        assert "techno" in text
        assert "10 DM" in text

        # Research section
        assert "RESEARCH RESULTS" in text
        assert "[ARTIST]" in text
        assert "[VENUE]" in text

        # Interconnection section
        assert "INTERCONNECTIONS" in text
        assert "performed_at" in text
        assert "venue_residency" in text

    def test_format_text_minimal_state(self) -> None:
        from src.cli.analyze import _format_text_output

        state = _make_minimal_pipeline_state()
        text = _format_text_output(state)

        # Header should still appear
        assert "raiveFlier" in text
        assert "Analysis Report" in text

        # No OCR, entity, research, or interconnection sections
        assert "EXTRACTED ENTITIES" not in text
        assert "RESEARCH RESULTS" not in text
        assert "INTERCONNECTIONS" not in text
        assert "Completed:" not in text

    def test_format_text_with_completed_at(self) -> None:
        from src.cli.analyze import _format_text_output

        state = _make_full_pipeline_state()
        text = _format_text_output(state)

        assert "2026-02-22 12:00:00 UTC" in text
        assert "Completed:" in text

    def test_format_text_with_research_results(self) -> None:
        from src.cli.analyze import _format_text_output

        state = _make_full_pipeline_state()
        text = _format_text_output(state)

        # Artist bio summary
        assert "British techno DJ" in text

        # Labels
        assert "Intec" in text
        assert "React" in text

        # Releases
        assert "Phat Trax" in text
        assert "Two Paintings and a Drum" in text
        assert "1995" in text
        assert "1996" in text

        # Venue description
        assert "former department store" in text

        # Date context
        assert "Berlin techno" in text
        assert "Post-reunification" in text

        # Warnings
        assert "Limited release data before 1993" in text


# ======================================================================
# TestFormatJsonOutput
# ======================================================================


class TestFormatJsonOutput:
    """Tests for src.cli.analyze._format_json_output."""

    def test_format_json_full_state(self) -> None:
        from src.cli.analyze import _format_json_output

        state = _make_full_pipeline_state()
        raw = _format_json_output(state)
        data = json.loads(raw)

        assert data["session_id"] == "test-session-001"
        assert "ocr" in data
        assert data["ocr"]["provider_used"] == "tesseract"
        assert data["ocr"]["confidence"] == 0.85
        assert data["ocr"]["processing_time"] == 1.23
        assert "entities" in data
        assert "research_results" in data
        assert len(data["research_results"]) == 4
        assert "interconnection_map" in data
        assert "completed_at" in data
        assert "errors" in data

    def test_format_json_minimal_state(self) -> None:
        from src.cli.analyze import _format_json_output

        state = _make_minimal_pipeline_state()
        raw = _format_json_output(state)
        data = json.loads(raw)

        assert data["session_id"] == "minimal-session"
        assert "ocr" not in data
        assert "entities" not in data
        assert "research_results" not in data
        assert "interconnection_map" not in data
        assert "completed_at" not in data
        assert "errors" not in data

    def test_format_json_with_errors(self) -> None:
        from src.cli.analyze import _format_json_output

        state = _make_full_pipeline_state()
        raw = _format_json_output(state)
        data = json.loads(raw)

        assert "errors" in data
        assert len(data["errors"]) == 1
        assert data["errors"][0]["phase"] == "RESEARCH"
        assert data["errors"][0]["message"] == "Bandcamp rate limit exceeded"
        assert data["errors"][0]["recoverable"] is True


# ======================================================================
# TestRunValidation
# ======================================================================


class TestRunValidation:
    """Tests for src.cli.analyze._run input validation and success path."""

    @pytest.mark.asyncio
    async def test_run_file_not_found(self) -> None:
        from src.cli.analyze import _run

        result = await _run(
            image_path=Path("/nonexistent/fake_flier.jpg"),
            json_output=False,
            output_file=None,
            quiet=False,
        )
        assert result == 1

    @pytest.mark.asyncio
    async def test_run_unsupported_extension(self, tmp_path: Path) -> None:
        from src.cli.analyze import _run

        bad_file = tmp_path / "flier.bmp"
        bad_file.write_bytes(b"\x00" * 100)

        result = await _run(
            image_path=bad_file,
            json_output=False,
            output_file=None,
            quiet=False,
        )
        assert result == 1

    @pytest.mark.asyncio
    async def test_run_file_too_large(self, tmp_path: Path) -> None:
        from src.cli.analyze import _run

        # Create a tiny valid JPEG header then pad to exceed 10 MB
        img = Image.new("RGB", (1, 1), (255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        tiny_jpeg = buf.getvalue()

        big_file = tmp_path / "huge.jpg"
        # Write just over the 10 MB limit
        big_file.write_bytes(tiny_jpeg + b"\x00" * (10 * 1024 * 1024 + 1))

        result = await _run(
            image_path=big_file,
            json_output=False,
            output_file=None,
            quiet=False,
        )
        assert result == 1

    @pytest.mark.asyncio
    async def test_run_success(self, tmp_path: Path) -> None:
        # Create a real small JPEG on disk
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        file_path = tmp_path / "test.jpg"
        img.save(str(file_path), format="JPEG")

        mock_state = _make_minimal_pipeline_state()

        with patch("src.main.run_pipeline", new_callable=AsyncMock, return_value=mock_state):
            from src.cli.analyze import _run

            result = await _run(
                image_path=file_path,
                json_output=False,
                output_file=None,
                quiet=False,
            )

        assert result == 0

    @pytest.mark.asyncio
    async def test_run_success_json_to_file(self, tmp_path: Path) -> None:
        """Test successful run with --json and --output flags."""
        img = Image.new("RGB", (100, 100), (0, 255, 0))
        file_path = tmp_path / "test.png"
        img.save(str(file_path), format="PNG")

        output_path = tmp_path / "output.json"
        mock_state = _make_full_pipeline_state()

        with patch("src.main.run_pipeline", new_callable=AsyncMock, return_value=mock_state):
            from src.cli.analyze import _run

            result = await _run(
                image_path=file_path,
                json_output=True,
                output_file=str(output_path),
                quiet=True,
            )

        assert result == 0
        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["session_id"] == "test-session-001"


# ======================================================================
# TestIngestBuildParser
# ======================================================================


class TestIngestBuildParser:
    """Tests for src.cli.ingest._build_parser."""

    def test_parser_book_subcommand(self) -> None:
        from src.cli.ingest import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "book",
            "--file", "/path/to/book.txt",
            "--title", "Energy Flash",
            "--author", "Simon Reynolds",
            "--year", "1998",
        ])

        assert args.command == "book"
        assert args.file == "/path/to/book.txt"
        assert args.title == "Energy Flash"
        assert args.author == "Simon Reynolds"
        assert args.year == 1998

    def test_parser_article_subcommand(self) -> None:
        from src.cli.ingest import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "article",
            "--url", "https://ra.co/features/example",
        ])

        assert args.command == "article"
        assert args.url == "https://ra.co/features/example"

    def test_parser_stats_subcommand(self) -> None:
        from src.cli.ingest import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["stats"])

        assert args.command == "stats"

    def test_parser_directory_subcommand(self) -> None:
        from src.cli.ingest import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "directory",
            "--path", "/path/to/articles",
            "--type", "book",
        ])

        assert args.command == "directory"
        assert args.path == "/path/to/articles"
        assert args.type == "book"

    def test_parser_directory_default_type(self) -> None:
        from src.cli.ingest import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "directory",
            "--path", "/path/to/articles",
        ])

        assert args.type == "article"

    def test_parser_pdf_subcommand(self) -> None:
        from src.cli.ingest import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "pdf",
            "--file", "/path/to/doc.pdf",
            "--title", "Last Night a DJ Saved My Life",
            "--author", "Bill Brewster",
            "--year", "1999",
        ])

        assert args.command == "pdf"
        assert args.file == "/path/to/doc.pdf"
        assert args.year == 1999

    def test_parser_epub_subcommand(self) -> None:
        from src.cli.ingest import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "epub",
            "--file", "/path/to/book.epub",
            "--title", "Altered State",
            "--author", "Matthew Collin",
            "--year", "1997",
        ])

        assert args.command == "epub"
        assert args.file == "/path/to/book.epub"

    def test_parser_no_subcommand(self) -> None:
        from src.cli.ingest import _build_parser

        parser = _build_parser()
        args = parser.parse_args([])

        assert args.command is None


# ======================================================================
# TestIngestBuildEmbeddingProvider
# ======================================================================


class TestIngestBuildEmbeddingProvider:
    """Tests for src.cli.ingest._build_embedding_provider."""

    def test_openai_available(self) -> None:
        """When OpenAI key is set and provider is available, returns OpenAI provider."""
        settings = _settings(openai_api_key="sk-real-key")

        mock_instance = MagicMock()
        mock_instance.is_available.return_value = True

        with patch(
            "src.providers.embedding.openai_embedding_provider.OpenAIEmbeddingProvider",
            return_value=mock_instance,
        ):
            from src.cli.ingest import _build_embedding_provider

            result = _build_embedding_provider(settings)

        assert result is mock_instance
        mock_instance.is_available.assert_called_once()

    def test_nomic_fallback(self) -> None:
        """When OpenAI is not available, falls back to Nomic."""
        settings = _settings(openai_api_key="")

        mock_nomic = MagicMock()
        mock_nomic.is_available.return_value = True

        with patch(
            "src.providers.embedding.nomic_embedding_provider.NomicEmbeddingProvider",
            return_value=mock_nomic,
        ):
            from src.cli.ingest import _build_embedding_provider

            result = _build_embedding_provider(settings)

        assert result is mock_nomic
        mock_nomic.is_available.assert_called_once()

    def test_openai_unavailable_falls_to_nomic(self) -> None:
        """When OpenAI key exists but provider is not available, falls to Nomic."""
        settings = _settings(openai_api_key="sk-test")

        mock_openai = MagicMock()
        mock_openai.is_available.return_value = False

        mock_nomic = MagicMock()
        mock_nomic.is_available.return_value = True

        with patch(
            "src.providers.embedding.openai_embedding_provider.OpenAIEmbeddingProvider",
            return_value=mock_openai,
        ), patch(
            "src.providers.embedding.nomic_embedding_provider.NomicEmbeddingProvider",
            return_value=mock_nomic,
        ):
            from src.cli.ingest import _build_embedding_provider

            result = _build_embedding_provider(settings)

        assert result is mock_nomic

    def test_none_available(self) -> None:
        """When no embedding providers are reachable, returns None."""
        settings = _settings(openai_api_key="")

        mock_nomic = MagicMock()
        mock_nomic.is_available.return_value = False

        with patch(
            "src.providers.embedding.nomic_embedding_provider.NomicEmbeddingProvider",
            return_value=mock_nomic,
        ):
            from src.cli.ingest import _build_embedding_provider

            result = _build_embedding_provider(settings)

        assert result is None


# ======================================================================
# TestIngestBuildLLMProvider
# ======================================================================


class TestIngestBuildLLMProvider:
    """Tests for src.cli.ingest._build_llm_provider."""

    def test_anthropic_priority(self) -> None:
        """When Anthropic key is set, it takes priority."""
        settings = _settings(anthropic_api_key="test-anthropic-key", openai_api_key="sk-test")

        mock_anthropic = MagicMock()

        with patch(
            "src.providers.llm.anthropic_provider.AnthropicLLMProvider",
            return_value=mock_anthropic,
        ):
            from src.cli.ingest import _build_llm_provider

            result = _build_llm_provider(settings)

        assert result is mock_anthropic

    def test_openai_fallback(self) -> None:
        """When Anthropic key is empty but OpenAI key is set, falls to OpenAI."""
        settings = _settings(anthropic_api_key="", openai_api_key="sk-test")

        mock_openai = MagicMock()

        with patch(
            "src.providers.llm.openai_provider.OpenAILLMProvider",
            return_value=mock_openai,
        ):
            from src.cli.ingest import _build_llm_provider

            result = _build_llm_provider(settings)

        assert result is mock_openai

    def test_ollama_default(self) -> None:
        """When no API keys are set, defaults to Ollama."""
        settings = _settings(anthropic_api_key="", openai_api_key="")

        mock_ollama = MagicMock()

        with patch(
            "src.providers.llm.ollama_provider.OllamaLLMProvider",
            return_value=mock_ollama,
        ):
            from src.cli.ingest import _build_llm_provider

            result = _build_llm_provider(settings)

        assert result is mock_ollama


# ======================================================================
# TestIngestBuildIngestionService
# ======================================================================


class TestIngestBuildIngestionService:
    """Tests for src.cli.ingest._build_ingestion_service."""

    def test_success(self) -> None:
        """When embedding provider is available, returns (service, status_msg)."""
        settings = _settings(openai_api_key="sk-test")

        mock_embedding = MagicMock()
        mock_embedding.is_available.return_value = True
        mock_embedding.get_provider_name.return_value = "openai_embedding"

        mock_llm = MagicMock()
        mock_chromadb = MagicMock()
        mock_chunker = MagicMock()
        mock_metadata_extractor = MagicMock()
        mock_service = MagicMock()

        with patch(
            "src.providers.embedding.openai_embedding_provider.OpenAIEmbeddingProvider",
            return_value=mock_embedding,
        ), patch(
            "src.providers.llm.anthropic_provider.AnthropicLLMProvider",
            return_value=mock_llm,
        ), patch(
            "src.providers.vector_store.chromadb_provider.ChromaDBProvider",
            return_value=mock_chromadb,
        ), patch(
            "src.services.ingestion.chunker.TextChunker",
            return_value=mock_chunker,
        ), patch(
            "src.services.ingestion.metadata_extractor.MetadataExtractor",
            return_value=mock_metadata_extractor,
        ), patch(
            "src.services.ingestion.ingestion_service.IngestionService",
            return_value=mock_service,
        ):
            from src.cli.ingest import _build_ingestion_service

            service, status = _build_ingestion_service(settings)

        assert service is mock_service
        assert "openai_embedding" in status
        assert "chromadb" in status

    def test_no_embedding_provider(self) -> None:
        """When no embedding provider is available, returns (None, error_msg)."""
        settings = _settings(openai_api_key="")

        mock_nomic = MagicMock()
        mock_nomic.is_available.return_value = False

        with patch(
            "src.providers.embedding.nomic_embedding_provider.NomicEmbeddingProvider",
            return_value=mock_nomic,
        ):
            from src.cli.ingest import _build_ingestion_service

            service, status = _build_ingestion_service(settings)

        assert service is None
        assert "No embedding provider available" in status


# ======================================================================
# TestIngestHandlers
# ======================================================================


class TestIngestHandlers:
    """Tests for src.cli.ingest handler functions."""

    @pytest.mark.asyncio
    async def test_handle_book(self) -> None:
        """_handle_book calls service.ingest_book and returns 0."""
        from src.cli.ingest import _handle_book

        mock_result = MagicMock()
        mock_result.chunks_created = 42
        mock_result.total_tokens = 10000
        mock_result.ingestion_time = 3.5
        mock_result.source_id = "book-001"

        mock_service = MagicMock()
        mock_service.ingest_book = AsyncMock(return_value=mock_result)

        args = Namespace(
            file="/path/to/book.txt",
            title="Energy Flash",
            author="Simon Reynolds",
            year=1998,
        )

        exit_code = await _handle_book(args, mock_service)

        assert exit_code == 0
        mock_service.ingest_book.assert_awaited_once_with(
            file_path="/path/to/book.txt",
            title="Energy Flash",
            author="Simon Reynolds",
            year=1998,
        )

    @pytest.mark.asyncio
    async def test_handle_article(self) -> None:
        """_handle_article calls service.ingest_article and returns 0."""
        from src.cli.ingest import _handle_article

        mock_result = MagicMock()
        mock_result.chunks_created = 8
        mock_result.total_tokens = 2000
        mock_result.ingestion_time = 1.2
        mock_result.source_id = "article-001"

        mock_service = MagicMock()
        mock_service.ingest_article = AsyncMock(return_value=mock_result)

        args = Namespace(url="https://ra.co/features/example")

        exit_code = await _handle_article(args, mock_service)

        assert exit_code == 0
        mock_service.ingest_article.assert_awaited_once_with(
            url="https://ra.co/features/example",
        )

    @pytest.mark.asyncio
    async def test_handle_pdf(self) -> None:
        """_handle_pdf calls service.ingest_pdf and returns 0."""
        from src.cli.ingest import _handle_pdf

        mock_result = MagicMock()
        mock_result.chunks_created = 100
        mock_result.total_tokens = 50000
        mock_result.ingestion_time = 12.5
        mock_result.source_id = "pdf-001"

        mock_service = MagicMock()
        mock_service.ingest_pdf = AsyncMock(return_value=mock_result)

        args = Namespace(
            file="/path/to/doc.pdf",
            title="Last Night a DJ Saved My Life",
            author="Bill Brewster",
            year=1999,
        )

        exit_code = await _handle_pdf(args, mock_service)

        assert exit_code == 0
        mock_service.ingest_pdf.assert_awaited_once_with(
            file_path="/path/to/doc.pdf",
            title="Last Night a DJ Saved My Life",
            author="Bill Brewster",
            year=1999,
        )

    @pytest.mark.asyncio
    async def test_handle_epub(self) -> None:
        """_handle_epub calls service.ingest_epub and returns 0."""
        from src.cli.ingest import _handle_epub

        mock_result = MagicMock()
        mock_result.chunks_created = 75
        mock_result.total_tokens = 30000
        mock_result.ingestion_time = 8.0
        mock_result.source_id = "epub-001"

        mock_service = MagicMock()
        mock_service.ingest_epub = AsyncMock(return_value=mock_result)

        args = Namespace(
            file="/path/to/book.epub",
            title="Altered State",
            author="Matthew Collin",
            year=1997,
        )

        exit_code = await _handle_epub(args, mock_service)

        assert exit_code == 0
        mock_service.ingest_epub.assert_awaited_once_with(
            file_path="/path/to/book.epub",
            title="Altered State",
            author="Matthew Collin",
            year=1997,
        )

    @pytest.mark.asyncio
    async def test_handle_directory(self) -> None:
        """_handle_directory calls service.ingest_directory and returns 0."""
        from src.cli.ingest import _handle_directory

        mock_result_1 = MagicMock()
        mock_result_1.chunks_created = 10
        mock_result_1.total_tokens = 3000
        mock_result_1.ingestion_time = 1.0

        mock_result_2 = MagicMock()
        mock_result_2.chunks_created = 15
        mock_result_2.total_tokens = 4000
        mock_result_2.ingestion_time = 1.5

        mock_service = MagicMock()
        mock_service.ingest_directory = AsyncMock(
            return_value=[mock_result_1, mock_result_2],
        )

        args = Namespace(
            path="/path/to/articles",
            type="article",
        )

        exit_code = await _handle_directory(args, mock_service)

        assert exit_code == 0
        mock_service.ingest_directory.assert_awaited_once_with(
            dir_path="/path/to/articles",
            source_type="article",
        )

    @pytest.mark.asyncio
    async def test_handle_stats_no_embedding(self) -> None:
        """_handle_stats returns 0 with message when no embedding provider."""
        from src.cli.ingest import _handle_stats

        settings = _settings(openai_api_key="")

        mock_nomic = MagicMock()
        mock_nomic.is_available.return_value = False

        with patch(
            "src.providers.embedding.nomic_embedding_provider.NomicEmbeddingProvider",
            return_value=mock_nomic,
        ):
            exit_code = await _handle_stats(settings)

        assert exit_code == 0
