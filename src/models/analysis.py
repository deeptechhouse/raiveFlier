"""Interconnection analysis models for the raiveFlier pipeline.

Defines Pydantic v2 models for citations, entity graphs, relationship edges,
pattern insights, and the interconnection map. All models use frozen config
to enforce immutability.

These models represent the output of Phase 3 (INTERCONNECTION) of the pipeline.
After all entities have been individually researched, the interconnection service
(src/services/interconnection_service.py) uses an LLM to discover relationships
*between* them — e.g. "Artist X released on Venue Y's in-house label" or
"Promoter Z has booked all three artists on this flier before."

The data structure is a graph:
    - EntityNode  = a node (artist, venue, promoter, etc.)
    - RelationshipEdge = an edge connecting two nodes
    - PatternInsight = a higher-level observation spanning multiple nodes
    - InterconnectionMap = the complete graph + narrative summary

All claims are backed by Citation objects that track source provenance
and reliability tier (1 = published book … 6 = unverified web content).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any  # Used for the flexible 'properties' dict on EntityNode

from pydantic import BaseModel, ConfigDict, Field

from src.models.entities import EntityType


# ---------------------------------------------------------------------------
# Citation — provenance tracking for every claim in the analysis.
# ---------------------------------------------------------------------------
class Citation(BaseModel):
    """A citation backing a claim in the interconnection analysis.

    Every relationship edge and pattern insight carries a list of citations
    so the user can verify claims. The tier system (1–6) ranks reliability:
        1 = Published book (e.g. Energy Flash)
        2 = Academic paper
        3 = Major music publication (RA, Mixmag)
        4 = Blog / independent publication
        5 = Forum post / social media
        6 = Unverified web content
    """

    model_config = ConfigDict(frozen=True)

    # The specific claim or quote being cited.
    text: str
    # Category of the source (e.g. "book", "article", "database").
    source_type: str
    # Name of the source (e.g. "Resident Advisor", "Energy Flash").
    source_name: str
    # URL to the source, if available (None for offline sources like books).
    source_url: str | None = None
    # Publication date of the source (helps assess currency of information).
    source_date: date | None = None
    # Reliability tier (1 = most reliable, 6 = least). Matches the
    # citation_tier system used in ArticleReference and DocumentChunk.
    tier: int = Field(default=6, ge=1, le=6)
    # Page number or section reference (useful for book citations).
    page_number: str | None = None


# ---------------------------------------------------------------------------
# EntityNode — a vertex in the interconnection graph.
# ---------------------------------------------------------------------------
class EntityNode(BaseModel):
    """A node in the interconnection graph representing an entity.

    Each entity from the research phase becomes a node. The 'properties'
    dict stores entity-specific metadata (e.g. city, genre) without
    requiring separate node classes per entity type.
    """

    model_config = ConfigDict(frozen=True)

    # What kind of entity this node represents (ARTIST, VENUE, etc.).
    entity_type: EntityType
    # Display name (matches the entity_name from ResearchResult).
    name: str
    # Flexible key-value store for entity-specific attributes.
    # Example: {"city": "Berlin", "genres": ["techno", "ambient"]}
    # Using dict[str, Any] keeps the graph schema flexible without
    # needing a different node class for each entity type.
    properties: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# RelationshipEdge — a connection between two entities.
# ---------------------------------------------------------------------------
class RelationshipEdge(BaseModel):
    """An edge in the interconnection graph representing a relationship between entities.

    Examples of relationships the LLM might discover:
        - "released_on": Artist released music on Venue's label
        - "booked_by": Promoter has booked this artist
        - "shared_label": Two artists share a record label
        - "resident_at": Artist is/was a resident DJ at this venue
    """

    model_config = ConfigDict(frozen=True)

    # Names of the two connected entities (must match EntityNode.name values).
    source: str                     # "from" entity
    target: str                     # "to" entity
    # Type of relationship (e.g. "shared_label", "booked_by", "released_on").
    relationship_type: str
    # Human-readable description of the relationship.
    details: str | None = None
    # Citations backing this relationship claim.
    citations: list[Citation] = Field(default_factory=list)
    # How confident we are in this relationship (0.0–1.0).
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # If True, the user or system has dismissed this edge as incorrect.
    # Edges are soft-deleted (marked dismissed) rather than removed, so
    # the analysis can be re-evaluated without losing information.
    dismissed: bool = False


# ---------------------------------------------------------------------------
# PatternInsight — a higher-level observation spanning multiple entities.
# ---------------------------------------------------------------------------
class PatternInsight(BaseModel):
    """A pattern or insight discovered during interconnection analysis.

    Patterns are broader observations than individual edges. Examples:
        - "scene_cluster": All artists on this flier are from the Berlin techno scene
        - "label_network": Three artists share the same two record labels
        - "era_marker": This flier represents the peak of UK garage (1999-2001)
    """

    model_config = ConfigDict(frozen=True)

    # Category of the pattern (e.g. "scene_cluster", "label_network").
    pattern_type: str
    # Human-readable description of what was discovered.
    description: str
    # Names of all entities involved in this pattern.
    involved_entities: list[str] = Field(default_factory=list)
    # Citations supporting this pattern observation.
    citations: list[Citation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# InterconnectionMap — the complete graph + narrative for a flier analysis.
# ---------------------------------------------------------------------------
class InterconnectionMap(BaseModel):
    """The complete interconnection analysis of entities from a rave flier.

    This is the final analytical product of the pipeline. It contains:
        - A graph of entities (nodes) and their relationships (edges)
        - Pattern insights that span multiple relationships
        - An LLM-generated narrative that tells the "story" of this flier
        - All citations backing every claim

    Stored on PipelineState.interconnection_map and served to the frontend
    via the /api/v1/fliers/{session_id}/results endpoint.
    """

    model_config = ConfigDict(frozen=True)

    # All entities from the research phase as graph nodes.
    nodes: list[EntityNode] = Field(default_factory=list)
    # Discovered relationships between entities.
    edges: list[RelationshipEdge] = Field(default_factory=list)
    # Higher-level patterns and insights.
    patterns: list[PatternInsight] = Field(default_factory=list)
    # LLM-generated narrative summary — a readable "story" that weaves
    # together the entities, relationships, and patterns into a cohesive
    # description of the flier's cultural context and significance.
    narrative: str | None = None
    # Master citation list — may include citations not attached to specific
    # edges or patterns (e.g. general scene context).
    citations: list[Citation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Persistent analysis storage models — survive beyond session TTL.
# These models support permanent storage of InterconnectionMaps in
# flier_history.db and aggregation into a combined connection map.
# ---------------------------------------------------------------------------


class EdgeDismissal(BaseModel):
    """A permanent record of a user dismissing an interconnection edge.

    Stored in the edge_dismissals table and applied as a layer on top of
    any analysis snapshot, so dismissals survive re-analysis.
    """

    model_config = ConfigDict(frozen=True)

    id: int | None = None
    flier_id: int
    source_entity: str
    target_entity: str
    relationship_type: str
    reason: str | None = None
    dismissed_at: datetime | None = None


class AnalysisAnnotation(BaseModel):
    """A user-authored note attached to a stored analysis.

    Supports both general annotations (target_type="analysis") and
    targeted annotations on specific entities or edges.
    """

    model_config = ConfigDict(frozen=True)

    id: int | None = None
    flier_id: int
    target_type: str  # "analysis", "entity", "edge"
    target_key: str | None = None  # entity name or "source->target" for edges
    note: str
    created_at: datetime | None = None
    updated_at: datetime | None = None


class CombinedNode(BaseModel):
    """A deduplicated entity node in the combined connection map.

    Merges appearances of the same entity across multiple flier analyses.
    appearance_count tracks how many distinct fliers featured this entity.
    """

    model_config = ConfigDict(frozen=True)

    entity_type: str
    name: str
    appearance_count: int = 1
    source_sessions: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)


class CombinedEdge(BaseModel):
    """A merged relationship edge in the combined connection map.

    Aggregates the same relationship across multiple analyses, tracking
    average confidence and how many times the relationship was observed.
    """

    model_config = ConfigDict(frozen=True)

    source: str
    target: str
    relationship_type: str
    avg_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    occurrence_count: int = 1
    source_sessions: list[str] = Field(default_factory=list)
    details: str | None = None


class CombinedConnectionMap(BaseModel):
    """The unified graph aggregating all stored flier analyses.

    Built on-demand by GraphAggregationService from all active analysis
    snapshots.  Nodes are deduplicated via fuzzy matching, edges are
    merged, and user dismissals are applied.
    """

    model_config = ConfigDict(frozen=True)

    nodes: list[CombinedNode] = Field(default_factory=list)
    edges: list[CombinedEdge] = Field(default_factory=list)
    total_analyses: int = 0
    generated_at: datetime | None = None
