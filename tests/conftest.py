"""Shared pytest fixtures for the raiveFlier test suite."""

from __future__ import annotations

import hashlib
import io
import struct
from datetime import date, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image, ImageDraw, ImageFont

from src.interfaces.article_provider import ArticleContent, IArticleProvider
from src.interfaces.embedding_provider import IEmbeddingProvider
from src.interfaces.llm_provider import ILLMProvider
from src.interfaces.music_db_provider import ArtistSearchResult, IMusicDatabaseProvider
from src.interfaces.vector_store_provider import IVectorStoreProvider
from src.interfaces.web_search_provider import IWebSearchProvider, SearchResult
from src.models.entities import EntityType, Label, Release
from src.models.flier import ExtractedEntities, ExtractedEntity, FlierImage, OCRResult
from src.models.rag import CorpusStats, DocumentChunk, RetrievedChunk

# ---------------------------------------------------------------------------
# Existing fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Mock provider fixtures for integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_provider() -> ILLMProvider:
    """Mock ILLMProvider that returns configurable responses.

    Default complete() returns a valid JSON entity extraction response.
    Override with mock_llm_provider.complete.return_value = "custom" or
    mock_llm_provider.complete.side_effect = [...] for specific tests.
    """
    mock = MagicMock(spec=ILLMProvider)
    mock.get_provider_name.return_value = "mock-llm"
    mock.is_available.return_value = True
    mock.supports_vision.return_value = False
    mock.validate_credentials = AsyncMock(return_value=True)
    mock.complete = AsyncMock(return_value='{"result": "ok"}')
    mock.vision_extract = AsyncMock(return_value='{"result": "ok"}')
    return mock


@pytest.fixture
def mock_music_db_provider() -> IMusicDatabaseProvider:
    """Mock IMusicDatabaseProvider with sample Discogs-like data."""
    mock = MagicMock(spec=IMusicDatabaseProvider)
    mock.get_provider_name.return_value = "mock-discogs"
    mock.is_available.return_value = True

    mock.search_artist = AsyncMock(
        return_value=[
            ArtistSearchResult(
                id="12345",
                name="Carl Cox",
                disambiguation="UK techno DJ",
                confidence=0.95,
            ),
        ]
    )
    mock.get_artist_releases = AsyncMock(
        return_value=[
            Release(
                title="Phat Trax",
                label="React",
                catalog_number="REACT-001",
                year=1995,
                format='12"',
                discogs_url="https://www.discogs.com/release/1",
                genres=["Electronic"],
                styles=["Techno"],
            ),
            Release(
                title="Two Paintings and a Drum",
                label="Intec",
                catalog_number="INTEC-002",
                year=1996,
                format='12"',
                discogs_url="https://www.discogs.com/release/2",
                genres=["Electronic"],
                styles=["Techno", "Tech House"],
            ),
        ]
    )
    mock.get_artist_labels = AsyncMock(
        return_value=[
            Label(
                name="Intec",
                discogs_id=100,
                discogs_url="https://www.discogs.com/label/100",
            ),
        ]
    )
    mock.get_release_details = AsyncMock(return_value=None)
    return mock


@pytest.fixture
def mock_search_provider() -> IWebSearchProvider:
    """Mock IWebSearchProvider with sample search results."""
    mock = MagicMock(spec=IWebSearchProvider)
    mock.get_provider_name.return_value = "mock-search"
    mock.is_available.return_value = True

    mock.search = AsyncMock(
        return_value=[
            SearchResult(
                title="Carl Cox at Tresor Berlin 1997",
                url="https://ra.co/features/carl-cox-tresor",
                snippet="Carl Cox played a legendary set at Tresor Berlin...",
                date=date(1997, 5, 1),
            ),
            SearchResult(
                title="History of Tresor Club",
                url="https://en.wikipedia.org/wiki/Tresor_(club)",
                snippet="Tresor is a techno club in Berlin, Germany...",
                date=None,
            ),
        ]
    )
    return mock


@pytest.fixture
def mock_article_provider() -> IArticleProvider:
    """Mock IArticleProvider returning sample article content."""
    mock = MagicMock(spec=IArticleProvider)
    mock.get_provider_name.return_value = "mock-article"
    mock.is_available.return_value = True

    mock.extract_content = AsyncMock(
        return_value=ArticleContent(
            title="Carl Cox at Tresor Berlin 1997",
            text=(
                "Carl Cox delivered one of his most legendary sets at Tresor "
                "Berlin in the spring of 1997. The night was part of a series "
                "of events that cemented the club's reputation."
            ),
            author="Resident Advisor Staff",
            date=date(1997, 5, 15),
            url="https://ra.co/features/carl-cox-tresor",
        )
    )
    mock.check_availability = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def sample_flier_image() -> tuple[FlierImage, bytes]:
    """Create a simple test image with rave flier text and return (FlierImage, bytes)."""
    img = Image.new("RGB", (400, 600), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except OSError:
        font = ImageFont.load_default()
        font_small = font

    draw.text((50, 50), "TRESOR PRESENTS", fill=(255, 255, 255), font=font_small)
    draw.text((50, 100), "CARL COX", fill=(255, 255, 0), font=font)
    draw.text((50, 150), "JEFF MILLS", fill=(255, 255, 0), font=font)
    draw.text((50, 200), "DERRICK MAY", fill=(255, 255, 0), font=font)
    draw.text((50, 280), "TRESOR BERLIN", fill=(200, 200, 200), font=font_small)
    draw.text((50, 320), "SATURDAY MARCH 15TH 1997", fill=(200, 200, 200), font=font_small)
    draw.text((50, 360), "10PM - 6AM", fill=(150, 150, 150), font=font_small)
    draw.text((50, 400), "10 DM", fill=(150, 150, 150), font=font_small)

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    image_data = buf.getvalue()
    image_hash = hashlib.sha256(image_data).hexdigest()

    flier = FlierImage(
        id="test-session-001",
        filename="tresor_1997.jpg",
        content_type="image/jpeg",
        file_size=len(image_data),
        image_hash=image_hash,
    )
    flier.__pydantic_private__["_image_data"] = image_data

    return flier, image_data


@pytest.fixture
def sample_ocr_result() -> OCRResult:
    """Pre-built OCRResult with sample rave flier text."""
    return OCRResult(
        raw_text=(
            "TRESOR PRESENTS\n"
            "CARL COX\n"
            "JEFF MILLS\n"
            "DERRICK MAY\n"
            "TRESOR BERLIN\n"
            "SATURDAY MARCH 15TH 1997\n"
            "10PM - 6AM\n"
            "10 DM"
        ),
        confidence=0.85,
        provider_used="tesseract",
        processing_time=1.2,
    )


@pytest.fixture
def sample_extracted_entities(sample_ocr_result: OCRResult) -> ExtractedEntities:
    """Pre-built ExtractedEntities with 3 artists + venue + date."""
    return ExtractedEntities(
        artists=[
            ExtractedEntity(text="Carl Cox", entity_type=EntityType.ARTIST, confidence=0.95),
            ExtractedEntity(text="Jeff Mills", entity_type=EntityType.ARTIST, confidence=0.92),
            ExtractedEntity(text="Derrick May", entity_type=EntityType.ARTIST, confidence=0.90),
        ],
        venue=ExtractedEntity(text="Tresor, Berlin", entity_type=EntityType.VENUE, confidence=0.88),
        date=ExtractedEntity(
            text="Saturday March 15th 1997", entity_type=EntityType.DATE, confidence=0.85
        ),
        promoter=ExtractedEntity(
            text="Tresor Records", entity_type=EntityType.PROMOTER, confidence=0.70
        ),
        genre_tags=["techno"],
        ticket_price="10 DM",
        raw_ocr=sample_ocr_result,
    )


# ---------------------------------------------------------------------------
# RAG fixtures
# ---------------------------------------------------------------------------

_EMBEDDING_DIM = 128


def _hash_to_vector(text: str, dim: int = _EMBEDDING_DIM) -> list[float]:
    """Generate a deterministic fixed-length vector by hashing *text*.

    Uses SHA-256 to hash the text, then unpacks bytes into floats and
    normalises to unit length.  Deterministic — same text always produces
    the same vector.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Extend the digest to cover `dim` floats (4 bytes each)
    raw = digest
    while len(raw) < dim * 4:
        raw += hashlib.sha256(raw).digest()
    raw = raw[: dim * 4]
    values = list(struct.unpack(f"<{dim}f", raw))
    # Normalise to unit length
    magnitude = max(sum(v * v for v in values) ** 0.5, 1e-10)
    return [v / magnitude for v in values]


class MockEmbeddingProvider(IEmbeddingProvider):
    """In-memory deterministic embedding provider for tests."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [_hash_to_vector(t) for t in texts]

    async def embed_single(self, text: str) -> list[float]:
        return _hash_to_vector(text)

    def get_dimension(self) -> int:
        return _EMBEDDING_DIM

    def get_provider_name(self) -> str:
        return "mock-embedding"

    def is_available(self) -> bool:
        return True


class MockVectorStore(IVectorStoreProvider):
    """In-memory vector store backed by a simple dict.

    Stores chunks and returns all on query, sorted by simulated cosine
    similarity (dot product of hash-based vectors).
    """

    def __init__(self) -> None:
        self._store: dict[str, tuple[DocumentChunk, list[float]]] = {}
        self._embedding = MockEmbeddingProvider()

    async def query(
        self,
        query_text: str,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if not self._store:
            return []

        query_vec = await self._embedding.embed_single(query_text)
        scored: list[tuple[float, DocumentChunk]] = []

        query_lower = query_text.lower()
        for chunk, vec in self._store.values():
            # Apply basic filters
            if filters and not self._matches_filters(chunk, filters):
                continue
            dot = sum(a * b for a, b in zip(query_vec, vec, strict=False))
            similarity = max(0.0, min(1.0, (dot + 1.0) / 2.0))

            # Boost score when entity tags match words in the query
            for tag in chunk.entity_tags:
                if tag.lower() in query_lower:
                    similarity = min(1.0, similarity + 0.35)
                    break

            scored.append((similarity, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[RetrievedChunk] = []
        for sim, chunk in scored[:top_k]:
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    similarity_score=sim,
                    formatted_citation=f"{chunk.source_title} [Tier {chunk.citation_tier}]",
                )
            )
        return results

    async def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        for chunk, emb in zip(chunks, embeddings, strict=True):
            self._store[chunk.chunk_id] = (chunk, emb)
        return len(chunks)

    async def delete_by_source(self, source_id: str) -> int:
        to_delete = [cid for cid, (chunk, _) in self._store.items() if chunk.source_id == source_id]
        for cid in to_delete:
            del self._store[cid]
        return len(to_delete)

    async def delete_by_source_type(self, source_type: str) -> int:
        to_delete = [cid for cid, (chunk, _) in self._store.items() if chunk.source_type == source_type]
        for cid in to_delete:
            del self._store[cid]
        return len(to_delete)

    async def get_source_ids(self, source_type: str | None = None) -> set[str]:
        ids: set[str] = set()
        for chunk, _ in self._store.values():
            if source_type is None or chunk.source_type == source_type:
                ids.add(chunk.source_id)
        return ids

    async def get_stats(self) -> CorpusStats:
        sources_by_type: dict[str, int] = {}
        source_ids: set[str] = set()
        entity_tags: set[str] = set()
        geo_tags: set[str] = set()

        for chunk, _ in self._store.values():
            source_ids.add(chunk.source_id)
            sources_by_type[chunk.source_type] = sources_by_type.get(chunk.source_type, 0) + 1
            for tag in chunk.entity_tags:
                entity_tags.add(tag)
            for tag in chunk.geographic_tags:
                geo_tags.add(tag)

        return CorpusStats(
            total_chunks=len(self._store),
            total_sources=len(source_ids),
            sources_by_type=sources_by_type,
            entity_tag_count=len(entity_tags),
            geographic_tag_count=len(geo_tags),
        )

    async def update_chunk_metadata(
        self, chunk_id: str, metadata: dict[str, Any]
    ) -> bool:
        """Update metadata fields on an existing chunk in the mock store."""
        entry = self._store.get(chunk_id)
        if entry is None:
            return False
        chunk, emb = entry
        update_fields: dict[str, Any] = {}
        for key, value in metadata.items():
            if hasattr(chunk, key):
                update_fields[key] = value
        if update_fields:
            self._store[chunk_id] = (chunk.model_copy(update=update_fields), emb)
        return True

    def get_provider_name(self) -> str:
        return "mock-vector-store"

    def is_available(self) -> bool:
        return True

    @staticmethod
    def _matches_filters(chunk: DocumentChunk, filters: dict[str, Any]) -> bool:
        """Apply basic filter matching for test queries."""
        entity_filter = filters.get("entity_tags")
        if entity_filter and isinstance(entity_filter, dict):
            contains = entity_filter.get("$contains")
            if contains and not any(contains.lower() in t.lower() for t in chunk.entity_tags):
                return False

        source_filter = filters.get("source_type")
        if source_filter and isinstance(source_filter, dict):
            in_val = source_filter.get("$in")
            if in_val and chunk.source_type not in in_val:
                return False

        return True


@pytest.fixture
def mock_embedding_provider() -> MockEmbeddingProvider:
    """Mock IEmbeddingProvider returning deterministic hash-based vectors."""
    return MockEmbeddingProvider()


@pytest.fixture
def mock_vector_store() -> MockVectorStore:
    """Mock IVectorStoreProvider backed by an in-memory dict."""
    return MockVectorStore()


@pytest.fixture
def sample_book_text() -> str:
    """Multi-paragraph text about electronic music for chunker tests."""
    return (
        "The history of electronic dance music is deeply intertwined with "
        "the development of synthesizer technology and the club culture of "
        "the late twentieth century. Detroit techno emerged in the mid-1980s, "
        "pioneered by the Belleville Three: Juan Atkins, Derrick May, and "
        "Kevin Saunderson. Their futuristic sound drew from Kraftwerk, "
        "Parliament-Funkadelic, and the post-industrial landscape of Detroit.\n\n"
        "Across the Atlantic, acid house exploded in the UK during the late "
        "1980s. DJs like Danny Rampling and Paul Oakenfold brought the "
        "Balearic sound from Ibiza to London warehouses. The Second Summer "
        "of Love in 1988 transformed British youth culture forever, spawning "
        "massive outdoor raves and a new generation of producers.\n\n"
        "Chicago house music, the precursor to much of what followed, was "
        "built in the clubs of the South Side. Frankie Knuckles, the "
        '"Godfather of House," held legendary residencies at the Warehouse '
        "and the Power Plant. His blending of disco, European electronic "
        "music, and drum machines created a template that reverberates "
        "through dance floors to this day.\n\n"
        "Carl Cox, one of the most enduring figures in electronic music, "
        "began DJing in the early 1980s and rose to prominence through the "
        "UK rave scene. His technical prowess with three decks and a drum "
        "machine set him apart, and his residencies at Space Ibiza became "
        "the stuff of legend. Cox has consistently championed underground "
        "techno and house music throughout his career.\n\n"
        "The Berlin techno scene developed its own distinct character after "
        "the fall of the Wall in 1989. Abandoned buildings in the former "
        "East became improvised clubs, with Tresor opening in a former "
        "department store vault. The raw, minimal sound of Berlin techno "
        "attracted artists and ravers from around the world, establishing "
        "the city as the global capital of electronic music."
    )


@pytest.fixture
def tmp_chromadb(tmp_path: Path):
    """Create a temporary ChromaDB instance for integration tests."""
    import chromadb

    persist_dir = str(tmp_path / "chromadb_test")
    client = chromadb.PersistentClient(path=persist_dir)
    return client, persist_dir


# ---------------------------------------------------------------------------
# Shared utility fixtures (Phase 1 — coverage push)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_settings() -> Any:
    """Return a Settings-like object with dummy API keys for provider tests."""
    from src.config.settings import Settings

    return Settings(
        openai_api_key="sk-test-key",
        anthropic_api_key="test-anthropic-key",
        discogs_consumer_key="test-discogs-key",
        discogs_consumer_secret="test-discogs-secret",
        serper_api_key="test-serper-key",
        rag_enabled=True,
        chromadb_persist_dir="/tmp/test_chromadb",
        app_env="test",
    )


@pytest.fixture
def mock_cache_provider() -> Any:
    """Return a MagicMock(spec=ICacheProvider) with AsyncMock methods."""
    from src.interfaces.cache_provider import ICacheProvider

    mock = MagicMock(spec=ICacheProvider)
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=None)
    mock.delete = AsyncMock(return_value=None)
    mock.exists = AsyncMock(return_value=False)
    return mock


@pytest.fixture
def sample_pipeline_state(
    sample_flier_image: tuple[FlierImage, bytes],
    sample_ocr_result: OCRResult,
    sample_extracted_entities: ExtractedEntities,
) -> Any:
    """Build a fully-populated PipelineState for output formatter tests."""
    from datetime import timezone

    from src.models.analysis import (
        Citation,
        EntityNode,
        InterconnectionMap,
        PatternInsight,
        RelationshipEdge,
    )
    from src.models.entities import Artist, Label, Promoter, Release, Venue
    from src.models.pipeline import PipelineError, PipelinePhase, PipelineState
    from src.models.research import DateContext, ResearchResult

    flier, _ = sample_flier_image

    citation = Citation(
        text="Carl Cox played at Tresor Berlin in 1997",
        source_type="press",
        source_name="Resident Advisor",
        source_url="https://ra.co/features/carl-cox-tresor",
        tier=2,
    )

    artist = Artist(
        name="Carl Cox",
        discogs_id=12345,
        musicbrainz_id="abc-123",
        confidence=0.95,
        releases=[
            Release(title="Phat Trax", label="React", year=1995),
        ],
        labels=[
            Label(name="Intec", discogs_id=100),
        ],
    )

    venue = Venue(
        name="Tresor",
        city="Berlin",
        country="Germany",
        history="Opened in 1991 in the vault of a former department store.",
        notable_events=["Love Parade afterparty", "Tresor Records launch"],
        articles=[],
    )

    promoter = Promoter(
        name="Tresor Records",
        event_history=["Tresor nights", "Love Parade"],
        articles=[],
    )

    date_context = DateContext(
        event_date=date(1997, 3, 15),
        scene_context="Berlin techno was at its peak in 1997.",
        city_context="Berlin was the capital of electronic music.",
        cultural_context="Post-reunification culture thriving.",
    )

    research_results = [
        ResearchResult(
            entity_type=EntityType.ARTIST,
            entity_name="Carl Cox",
            artist=artist,
            confidence=0.95,
            warnings=["Limited release data before 1993"],
        ),
        ResearchResult(
            entity_type=EntityType.VENUE,
            entity_name="Tresor",
            venue=venue,
            confidence=0.88,
        ),
        ResearchResult(
            entity_type=EntityType.PROMOTER,
            entity_name="Tresor Records",
            promoter=promoter,
            confidence=0.70,
        ),
        ResearchResult(
            entity_type=EntityType.DATE,
            entity_name="1997-03-15",
            date_context=date_context,
            confidence=0.85,
        ),
    ]

    interconnection_map = InterconnectionMap(
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

    errors = [
        PipelineError(
            phase=PipelinePhase.RESEARCH,
            message="Bandcamp rate limit exceeded",
        ),
    ]

    return PipelineState(
        session_id="test-session-001",
        flier=flier,
        current_phase=PipelinePhase.OUTPUT,
        ocr_result=sample_ocr_result,
        extracted_entities=sample_extracted_entities,
        confirmed_entities=sample_extracted_entities,
        research_results=research_results,
        interconnection_map=interconnection_map,
        completed_at=datetime(2026, 2, 22, 12, 0, 0, tzinfo=timezone.utc),
        errors=errors,
        progress_percent=100.0,
    )
