"""Interconnection analysis models for the raiveFlier pipeline.

Defines Pydantic v2 models for citations, entity graphs, relationship edges,
pattern insights, and the interconnection map. All models use frozen config
to enforce immutability.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.models.entities import EntityType


class Citation(BaseModel):
    """A citation backing a claim in the interconnection analysis."""

    model_config = ConfigDict(frozen=True)

    text: str
    source_type: str
    source_name: str
    source_url: str | None = None
    source_date: date | None = None
    tier: int = Field(default=6, ge=1, le=6)
    page_number: str | None = None


class EntityNode(BaseModel):
    """A node in the interconnection graph representing an entity."""

    model_config = ConfigDict(frozen=True)

    entity_type: EntityType
    name: str
    properties: dict[str, Any] = Field(default_factory=dict)


class RelationshipEdge(BaseModel):
    """An edge in the interconnection graph representing a relationship between entities."""

    model_config = ConfigDict(frozen=True)

    source: str
    target: str
    relationship_type: str
    details: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    dismissed: bool = False


class PatternInsight(BaseModel):
    """A pattern or insight discovered during interconnection analysis."""

    model_config = ConfigDict(frozen=True)

    pattern_type: str
    description: str
    involved_entities: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class InterconnectionMap(BaseModel):
    """The complete interconnection analysis of entities from a rave flier."""

    model_config = ConfigDict(frozen=True)

    nodes: list[EntityNode] = Field(default_factory=list)
    edges: list[RelationshipEdge] = Field(default_factory=list)
    patterns: list[PatternInsight] = Field(default_factory=list)
    narrative: str | None = None
    citations: list[Citation] = Field(default_factory=list)
