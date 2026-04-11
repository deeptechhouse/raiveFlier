"""Pre-computed relationship graph edges between music entities.

Materialised during flier analysis so downstream consumers (especially
the recommendation service) can query relationships without live API
calls.  Edges accumulate strength over time as more fliers confirm
the same relationship.

# ─── ARCHITECTURE CONTEXT ────────────────────────────────────────────
#
# Layer: Models (bottom of the stack)
# Consumers: GraphBuilderService (writes), RecommendationService (reads)
# Storage:   Serialised to the ``relationship_edges`` table in
#            flier_history.db via SQLiteFlierHistoryProvider.
#
# Three relationship types are tracked:
#   - label_mate:    Two artists who released on the same record label.
#                    Strongest data-driven signal after co-billing.
#   - co_billing:    Two artists on the same flier.  Strongest signal
#                    of real-world scene proximity.
#   - venue_artist:  An artist who has played at a specific venue.
#                    Useful for venue-centric recommendations.
#
# Edges use UPSERT semantics — repeated evidence from new fliers
# increases the ``strength`` field, making frequently-confirmed
# relationships rank higher in recommendation results.
#
# Frozen Pydantic model (immutable) — follows the project convention
# where all data objects use ``frozen=True`` and new instances are
# created via model_copy(update={...}) instead of field mutation.
# ──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class RelationshipEdge(BaseModel, frozen=True):
    """A single directed edge in the pre-computed relationship graph.

    Attributes
    ----------
    source_entity:
        The originating entity name (e.g. an artist or venue name).
    target_entity:
        The destination entity name.
    relationship_type:
        One of ``"label_mate"``, ``"co_billing"``, or ``"venue_artist"``.
    evidence:
        Supporting metadata — varies by type.  For label_mate this
        contains ``{"shared_labels": [...]}``, for co_billing it holds
        event/venue info, etc.
    strength:
        Accumulates with each confirming flier.  Higher values mean
        the relationship has been observed more frequently.
    created_at:
        ISO-8601 timestamp of when the edge was first created.
    """

    source_entity: str
    target_entity: str
    # "label_mate", "co_billing", or "venue_artist"
    relationship_type: str
    # Supporting metadata (label names, event names, etc.)
    evidence: dict = Field(default_factory=dict)
    # Accumulates with each confirming flier — higher = more confirmed
    strength: float = 1.0
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
