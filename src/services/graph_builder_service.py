"""Builds pre-computed relationship graphs from flier analysis results.

Called after the pipeline completes to materialise edges that the
recommendation service can query without live API calls.  Edges
use UPSERT semantics -- repeated evidence increases strength.

# ─── ARCHITECTURE CONTEXT ────────────────────────────────────────────
#
# Layer: Services (business logic)
# Depends on: IFlierHistoryProvider (interface), RelationshipEdge (model),
#             text_normalizer (utility)
# Called by:  FlierAnalysisPipeline orchestrator (after Phase 5)
#
# This service bridges the gap between the pipeline's research output
# and the recommendation service's need for pre-computed relationship
# data.  Without this, the recommendation service must query external
# APIs (Discogs at 1 req/sec) to discover label-mates — a bottleneck
# that this pre-computation eliminates.
#
# Edge extraction logic:
#   1. label_mate:   Compares artist label lists from research results.
#                    Two artists sharing a label → label_mate edge.
#   2. co_billing:   Every pair of artists on the current flier → edge.
#                    This is the strongest scene-proximity signal.
#   3. venue_artist: Each artist linked to the flier's venue → edge.
#                    Enables venue-centric discovery.
#
# Failure in this service never crashes the pipeline — the orchestrator
# wraps the call in try/except and logs a warning on failure.
#
# Design pattern: Service layer — pure business logic, no I/O except
# through the injected flier_history provider (adapter pattern).
# ──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Any

from src.models.relationship_graph import RelationshipEdge
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.interfaces.flier_history_provider import IFlierHistoryProvider
    from src.models.flier import ExtractedEntities
    from src.models.research import ResearchResult


class GraphBuilderService:
    """Extracts and stores pre-computed relationship edges from pipeline output.

    Injected with a flier history provider for persistence.  All edge
    extraction is performed in-memory from the research results and
    confirmed entities — no external API calls are made.
    """

    def __init__(self, flier_history: IFlierHistoryProvider) -> None:
        """Initialise with the flier history provider for edge persistence.

        Parameters
        ----------
        flier_history:
            Provider implementing ``store_relationship_edges()`` for
            UPSERT-style edge persistence.
        """
        self._flier_history = flier_history
        self._logger = get_logger(__name__)

    async def build_edges_from_research(
        self,
        research_results: list[ResearchResult],
        entities: ExtractedEntities,
    ) -> int:
        """Extract and store relationship edges from research results.

        Edge types:
        - label_mate:   Two artists who released on the same label.
        - co_billing:   Two artists on the same flier (current flier).
        - venue_artist: An artist who has played at this venue.

        Parameters
        ----------
        research_results:
            Completed research profiles from the pipeline.
        entities:
            The user-confirmed extracted entities from the flier.

        Returns
        -------
        int
            Number of edges stored (new + updated).
        """
        edges: list[RelationshipEdge] = []

        # -- Label-mate edges --
        # Compare every pair of artist research results.  If two artists
        # share any label (case-insensitive), create a label_mate edge
        # with evidence listing the shared labels.
        label_mate_edges = self._extract_label_mate_edges(research_results)
        edges.extend(label_mate_edges)

        # -- Co-billing edges --
        # Every pair of artists on the current flier gets a co_billing edge.
        # This is the strongest real-world signal of scene proximity.
        co_billing_edges = self._extract_co_billing_edges(entities)
        edges.extend(co_billing_edges)

        # -- Venue-artist edges --
        # Link each artist on the flier to the venue (if known).
        venue_artist_edges = self._extract_venue_artist_edges(entities)
        edges.extend(venue_artist_edges)

        if not edges:
            self._logger.debug("graph_builder_no_edges_extracted")
            return 0

        # Serialise edges to dicts for the provider's store method
        edge_dicts = [edge.model_dump() for edge in edges]

        stored_count = await self._flier_history.store_relationship_edges(edge_dicts)

        self._logger.info(
            "graph_builder_edges_stored",
            label_mate=len(label_mate_edges),
            co_billing=len(co_billing_edges),
            venue_artist=len(venue_artist_edges),
            total_stored=stored_count,
        )
        return stored_count

    # ------------------------------------------------------------------
    # Private edge extractors
    # ------------------------------------------------------------------

    def _extract_label_mate_edges(
        self,
        research_results: list[ResearchResult],
    ) -> list[RelationshipEdge]:
        """Find label-mate relationships between researched artists.

        Iterates all artist-pair combinations from the research results.
        For each pair, compares their label lists (case-insensitive).
        Shared labels produce a label_mate edge whose evidence includes
        the list of shared label names.
        """
        # Collect artist results that have label data
        artist_results = [
            r for r in research_results
            if r.artist is not None and r.artist.labels
        ]

        edges: list[RelationshipEdge] = []

        # Compare every unique pair of artists with label data
        for a, b in combinations(artist_results, 2):
            # Case-insensitive label name comparison
            a_labels = {label.name.lower(): label.name for label in a.artist.labels}
            b_labels = {label.name.lower(): label.name for label in b.artist.labels}

            shared_keys = set(a_labels.keys()) & set(b_labels.keys())
            if not shared_keys:
                continue

            # Use original-cased names from the first artist's labels
            shared_label_names = sorted(a_labels[k] for k in shared_keys)

            edges.append(
                RelationshipEdge(
                    source_entity=a.artist.name,
                    target_entity=b.artist.name,
                    relationship_type="label_mate",
                    evidence={"shared_labels": shared_label_names},
                )
            )

        return edges

    def _extract_co_billing_edges(
        self,
        entities: ExtractedEntities,
    ) -> list[RelationshipEdge]:
        """Create co-billing edges for every artist pair on the flier.

        Co-billing (appearing on the same flier) is the strongest signal
        of scene proximity in the rave/electronic music context.  Each
        edge records the event name, venue, and date as evidence.
        """
        artist_names = [e.text for e in entities.artists]
        if len(artist_names) < 2:
            return []

        # Build evidence dict from flier metadata
        evidence: dict[str, Any] = {}
        if entities.event_name is not None:
            evidence["event_name"] = entities.event_name.text
        if entities.venue is not None:
            evidence["venue"] = entities.venue.text
        if entities.date is not None:
            evidence["date"] = entities.date.text

        edges: list[RelationshipEdge] = []
        for a_name, b_name in combinations(artist_names, 2):
            edges.append(
                RelationshipEdge(
                    source_entity=a_name,
                    target_entity=b_name,
                    relationship_type="co_billing",
                    evidence=evidence,
                )
            )

        return edges

    def _extract_venue_artist_edges(
        self,
        entities: ExtractedEntities,
    ) -> list[RelationshipEdge]:
        """Create venue-artist edges linking each artist to the venue.

        Only generated when the flier has a recognised venue.  These
        edges enable venue-centric recommendations ("other artists who
        have played at this venue").
        """
        if entities.venue is None:
            return []

        venue_name = entities.venue.text

        # Build evidence from available flier metadata
        evidence: dict[str, Any] = {}
        if entities.event_name is not None:
            evidence["event_name"] = entities.event_name.text
        if entities.date is not None:
            evidence["date"] = entities.date.text

        edges: list[RelationshipEdge] = []
        for artist_entity in entities.artists:
            edges.append(
                RelationshipEdge(
                    source_entity=artist_entity.text,
                    target_entity=venue_name,
                    relationship_type="venue_artist",
                    evidence=evidence,
                )
            )

        return edges
