"""raiveFlier domain models — re-exports all public model classes.

This __init__.py acts as the single public entry point for all domain models
in the raiveFlier project. Instead of importing models from their individual
module files (e.g. ``from src.models.flier import FlierImage``), other parts
of the codebase can import directly from ``src.models`` (e.g.
``from src.models import FlierImage``).

The models are organized across six submodules by domain concern:
    - analysis.py   — Interconnection graph (nodes, edges, patterns, citations)
    - entities.py   — Core domain entities (Artist, Venue, Promoter, etc.)
    - flier.py      — Flier image + OCR extraction data structures
    - pipeline.py   — Pipeline lifecycle state management
    - rag.py        — RAG vector-store document chunks and retrieval results
    - research.py   — Research results + date/scene context

The ``__all__`` list at the bottom controls what ``from src.models import *``
exports. If you add a new model class, remember to add it here too.
"""

# Enables PEP 604 union syntax (X | Y) in type hints for Python 3.9 compat.
from __future__ import annotations

# --- Analysis models: represent the interconnection graph that maps
# relationships between entities discovered on a rave flier. ---
from src.models.analysis import (
    Citation,
    EntityNode,
    InterconnectionMap,
    PatternInsight,
    RelationshipEdge,
)
# --- Entity models: the core domain objects that the entire pipeline revolves
# around — artists, venues, promoters, releases, labels, events. ---
from src.models.entities import (
    ArticleReference,
    Artist,
    ConfidenceLevel,
    EntityType,
    EventAppearance,
    Label,
    Promoter,
    Release,
    Venue,
)
# --- Flier models: represent the uploaded image, OCR output, and the
# entities extracted from the OCR text before research begins. ---
from src.models.flier import (
    ExtractedEntities,
    ExtractedEntity,
    FlierImage,
    OCRResult,
    TextRegion,
)
# --- Pipeline models: track the state machine that drives the 5-phase
# analysis workflow (Upload → OCR → Entity Extraction → Research → Output). ---
from src.models.pipeline import (
    PipelineError,
    PipelinePhase,
    PipelineState,
)
# --- RAG models: support the Retrieval-Augmented Generation layer, where
# curated reference documents (books, articles) are chunked, embedded into
# vectors, and stored in ChromaDB for semantic search during analysis. ---
from src.models.rag import (
    CorpusStats,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
# --- Research models: wrap the final output of the entity research phase,
# including date/scene context and per-entity research results. ---
from src.models.research import (
    DateContext,
    ResearchResult,
)
# --- Story models: anonymous first-person rave experience accounts,
# with moderation results, metadata, and event story collections. ---
from src.models.story import (
    EventStoryCollection,
    ModerationResult,
    RaveStory,
    StoryMetadata,
    StoryStatus,
)

# __all__ defines the public API of this package. Any model listed here can
# be imported with ``from src.models import ModelName``. If you create a new
# model class, add its name to the appropriate section below.
__all__ = [
    # entities
    "ArticleReference",
    "Artist",
    "ConfidenceLevel",
    "EntityType",
    "EventAppearance",
    "Label",
    "Promoter",
    "Release",
    "Venue",
    # flier
    "ExtractedEntities",
    "ExtractedEntity",
    "FlierImage",
    "OCRResult",
    "TextRegion",
    # research
    "DateContext",
    "ResearchResult",
    # analysis
    "Citation",
    "EntityNode",
    "InterconnectionMap",
    "PatternInsight",
    "RelationshipEdge",
    # pipeline
    "PipelineError",
    "PipelinePhase",
    "PipelineState",
    # rag
    "CorpusStats",
    "DocumentChunk",
    "IngestionResult",
    "RetrievedChunk",
    # story
    "EventStoryCollection",
    "ModerationResult",
    "RaveStory",
    "StoryMetadata",
    "StoryStatus",
]
