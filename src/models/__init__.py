"""raiveFlier domain models — re-exports all public model classes."""

from __future__ import annotations

from src.models.analysis import (
    Citation,
    EntityNode,
    InterconnectionMap,
    PatternInsight,
    RelationshipEdge,
)
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
from src.models.flier import (
    ExtractedEntities,
    ExtractedEntity,
    FlierImage,
    OCRResult,
    TextRegion,
)
from src.models.pipeline import (
    PipelineError,
    PipelinePhase,
    PipelineState,
)
from src.models.research import (
    DateContext,
    ResearchResult,
)

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
]
