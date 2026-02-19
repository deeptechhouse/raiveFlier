"""RaiveFlier pipeline entry point.

Wires up all services and providers via dependency injection, then
exposes an ``async run_pipeline`` function that executes the full
flier-analysis pipeline: OCR -> Entity Extraction -> Research ->
Interconnection Analysis -> Output.
"""

from __future__ import annotations

from uuid import uuid4

import httpx

from src.config.settings import Settings
from src.models.flier import FlierImage
from src.models.pipeline import PipelinePhase, PipelineState
from src.models.research import ResearchResult
from src.providers.cache.memory_cache import MemoryCacheProvider
from src.providers.llm.anthropic_provider import AnthropicLLMProvider
from src.providers.llm.ollama_provider import OllamaLLMProvider
from src.providers.llm.openai_provider import OpenAILLMProvider
from src.providers.music_db.discogs_api_provider import DiscogsAPIProvider
from src.providers.music_db.discogs_scrape_provider import DiscogsScrapeProvider
from src.providers.music_db.musicbrainz_provider import MusicBrainzProvider
from src.providers.ocr.easyocr_provider import EasyOCRProvider
from src.providers.ocr.llm_vision_provider import LLMVisionOCRProvider
from src.providers.ocr.tesseract_provider import TesseractOCRProvider
from src.providers.search.duckduckgo_provider import DuckDuckGoSearchProvider
from src.services.artist_researcher import ArtistResearcher
from src.services.citation_service import CitationService
from src.services.date_context_researcher import DateContextResearcher
from src.services.entity_extractor import EntityExtractor
from src.services.interconnection_service import InterconnectionService
from src.services.ocr_service import OCRService
from src.services.promoter_researcher import PromoterResearcher
from src.services.venue_researcher import VenueResearcher
from src.utils.image_preprocessor import ImagePreprocessor
from src.utils.logging import configure_logging, get_logger

_logger = get_logger(__name__)


def _build_llm_provider(settings: Settings):
    """Select the first available LLM provider based on configured API keys.

    Priority order: Anthropic -> OpenAI -> Ollama.
    """
    if settings.anthropic_api_key:
        return AnthropicLLMProvider(settings=settings)
    if settings.openai_api_key:
        return OpenAILLMProvider(settings=settings)
    return OllamaLLMProvider(settings=settings)


def build_pipeline(settings: Settings | None = None):
    """Construct and return all pipeline services with injected dependencies.

    Parameters
    ----------
    settings:
        Application settings.  Loaded from environment if not provided.

    Returns
    -------
    dict
        A dictionary of service instances keyed by role name.
    """
    if settings is None:
        settings = Settings()

    configure_logging(log_level=settings.log_level)

    # -- Providers --
    llm = _build_llm_provider(settings)
    cache = MemoryCacheProvider()
    web_search = DuckDuckGoSearchProvider()
    http_client = httpx.AsyncClient()
    preprocessor = ImagePreprocessor()

    # Article scraper
    from src.providers.article.web_scraper_provider import WebScraperProvider

    article_scraper = WebScraperProvider()

    # Music databases
    music_dbs = []
    if settings.discogs_consumer_key:
        music_dbs.append(DiscogsAPIProvider(settings=settings))
    music_dbs.append(DiscogsScrapeProvider(http_client=http_client))
    music_dbs.append(MusicBrainzProvider(settings=settings))

    # OCR providers (priority order)
    ocr_providers = []
    if llm.supports_vision():
        ocr_providers.append(LLMVisionOCRProvider(llm_provider=llm))
    ocr_providers.append(EasyOCRProvider(preprocessor=preprocessor))
    ocr_providers.append(TesseractOCRProvider(preprocessor=preprocessor))

    # -- Services --
    ocr_service = OCRService(providers=ocr_providers)
    entity_extractor = EntityExtractor(llm_provider=llm)

    artist_researcher = ArtistResearcher(
        music_dbs=music_dbs,
        web_search=web_search,
        article_scraper=article_scraper,
        llm=llm,
        cache=cache,
    )
    venue_researcher = VenueResearcher(
        web_search=web_search,
        article_scraper=article_scraper,
        llm=llm,
        cache=cache,
    )
    promoter_researcher = PromoterResearcher(
        web_search=web_search,
        article_scraper=article_scraper,
        llm=llm,
        cache=cache,
    )
    date_context_researcher = DateContextResearcher(
        web_search=web_search,
        article_scraper=article_scraper,
        llm=llm,
        cache=cache,
    )

    citation_service = CitationService()
    interconnection_service = InterconnectionService(
        llm_provider=llm,
        citation_service=citation_service,
    )

    return {
        "ocr_service": ocr_service,
        "entity_extractor": entity_extractor,
        "artist_researcher": artist_researcher,
        "venue_researcher": venue_researcher,
        "promoter_researcher": promoter_researcher,
        "date_context_researcher": date_context_researcher,
        "citation_service": citation_service,
        "interconnection_service": interconnection_service,
        "settings": settings,
    }


async def run_pipeline(flier: FlierImage) -> PipelineState:
    """Execute the full flier-analysis pipeline.

    Phases: UPLOAD -> OCR -> ENTITY_EXTRACTION -> RESEARCH ->
    INTERCONNECTION -> OUTPUT.

    Parameters
    ----------
    flier:
        The uploaded flier image to analyse.

    Returns
    -------
    PipelineState
        Final pipeline state containing all results.
    """
    services = build_pipeline()
    session_id = str(uuid4())

    state = PipelineState(session_id=session_id)

    # -- Phase 1: OCR --
    _logger.info("pipeline_phase", phase="OCR", session=session_id)
    state = state.model_copy(update={"current_phase": PipelinePhase.OCR})

    ocr_result = await services["ocr_service"].extract_text(flier)
    state = state.model_copy(update={"ocr_result": ocr_result})

    # -- Phase 2: Entity Extraction --
    _logger.info("pipeline_phase", phase="ENTITY_EXTRACTION", session=session_id)
    state = state.model_copy(update={"current_phase": PipelinePhase.ENTITY_EXTRACTION})

    extracted = await services["entity_extractor"].extract(ocr_result, flier)
    state = state.model_copy(
        update={
            "extracted_entities": extracted,
            "confirmed_entities": extracted,
        }
    )

    # -- Phase 3: Research --
    _logger.info("pipeline_phase", phase="RESEARCH", session=session_id)
    state = state.model_copy(update={"current_phase": PipelinePhase.RESEARCH})

    research_results: list[ResearchResult] = []

    for artist_entity in extracted.artists:
        result = await services["artist_researcher"].research(artist_entity.text)
        research_results.append(result)

    if extracted.venue:
        result = await services["venue_researcher"].research(extracted.venue.text)
        research_results.append(result)

    if extracted.promoter:
        result = await services["promoter_researcher"].research(extracted.promoter.text)
        research_results.append(result)

    if extracted.date:
        result = await services["date_context_researcher"].research(extracted.date.text)
        research_results.append(result)

    state = state.model_copy(update={"research_results": research_results})

    # -- Phase 4: Interconnection Analysis --
    _logger.info("pipeline_phase", phase="INTERCONNECTION", session=session_id)
    state = state.model_copy(update={"current_phase": PipelinePhase.INTERCONNECTION})

    interconnection_map = await services["interconnection_service"].analyze(
        research_results=research_results,
        entities=extracted,
    )
    state = state.model_copy(update={"interconnection_map": interconnection_map})

    # -- Phase 5: Output --
    _logger.info("pipeline_phase", phase="OUTPUT", session=session_id)
    from datetime import datetime, timezone

    state = state.model_copy(
        update={
            "current_phase": PipelinePhase.OUTPUT,
            "completed_at": datetime.now(tz=timezone.utc),  # noqa: UP017
            "progress_percent": 100.0,
        }
    )

    _logger.info(
        "pipeline_complete",
        session=session_id,
        entities=len(extracted.artists),
        edges=len(interconnection_map.edges),
        patterns=len(interconnection_map.patterns),
    )

    return state
