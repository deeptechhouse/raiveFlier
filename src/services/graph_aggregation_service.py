"""Graph aggregation service for the combined connection map.

Merges InterconnectionMaps from all stored flier analyses into a single
unified graph (CombinedConnectionMap).  Entity names are deduplicated
via normalization and fuzzy matching so that "Carl Cox" and "DJ Carl Cox"
become one node with appearance_count=2.

Architecture:
    - Called by raiveFeeder API endpoints (connection-map routes).
    - Depends on IFlierHistoryProvider for data retrieval.
    - Uses src/utils/text_normalizer for name normalization and fuzzy matching.

Design patterns:
    - Service (stateless logic, receives dependencies via constructor).
    - TTL cache — the combined map is expensive to rebuild (fetches all
      analyses, normalizes names, fuzzy-matches). A 30-second cache
      ensures that sequential calls to aggregate(), get_node_detail(),
      and get_stats() within the same user interaction reuse one build.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import structlog

from src.interfaces.flier_history_provider import IFlierHistoryProvider
from src.models.analysis import CombinedConnectionMap, CombinedEdge, CombinedNode
from src.utils.text_normalizer import fuzzy_match, normalize_artist_name

logger = structlog.get_logger(logger_name=__name__)

# Fuzzy match thresholds — artists tolerate more variation (OCR errors,
# "DJ" prefixes) than venues/promoters which tend to be more stable.
_ARTIST_FUZZY_THRESHOLD = 0.85
_DEFAULT_FUZZY_THRESHOLD = 0.90

# TTL for the cached aggregate result (seconds).
# 30 seconds is enough to serve a full page load (map + stats + node
# detail clicks) without rebuilding, but short enough that new analyses
# or dismissals appear within half a minute.
_CACHE_TTL_SECONDS = 30


class GraphAggregationService:
    """Aggregates InterconnectionMaps from all analyses into a combined graph.

    Injected with an IFlierHistoryProvider to retrieve stored analyses.
    Maintains a TTL-based cache of the aggregated CombinedConnectionMap
    so that get_node_detail() and get_stats() reuse the same build
    instead of re-aggregating.
    """

    def __init__(self, flier_history: IFlierHistoryProvider) -> None:
        self._flier_history = flier_history
        # TTL cache state — holds the last aggregate result and its build time.
        # Cache is invalidated after _CACHE_TTL_SECONDS or via invalidate_cache().
        self._cached_map: CombinedConnectionMap | None = None
        self._cache_time: float = 0.0

    def invalidate_cache(self) -> None:
        """Force the next aggregate() call to rebuild from the database.

        Called when the caller knows the underlying data has changed
        (e.g. after storing a new analysis or dismissing an edge).
        """
        self._cached_map = None
        self._cache_time = 0.0

    async def aggregate(self) -> CombinedConnectionMap:
        """Build a CombinedConnectionMap from all active analysis snapshots.

        Returns a cached result if the previous build is less than
        _CACHE_TTL_SECONDS old.  Otherwise rebuilds from scratch.

        Steps (on cache miss):
            1. Fetch all active analyses from flier_history.
            2. For each analysis, extract nodes and edges.
            3. Normalize entity names and deduplicate via fuzzy matching.
            4. Merge edges (same source+target+type = single edge).
            5. Apply edge dismissals.
            6. Return the unified graph.
        """
        # ── Cache check ──
        now = time.monotonic()
        if self._cached_map is not None and (now - self._cache_time) < _CACHE_TTL_SECONDS:
            return self._cached_map

        analyses = await self._flier_history.get_all_active_analyses()

        if not analyses:
            result = CombinedConnectionMap(
                nodes=[],
                edges=[],
                total_analyses=0,
                generated_at=datetime.now(timezone.utc),
            )
            self._cached_map = result
            self._cache_time = now
            return result

        # ── Phase 1: Build canonical name registry ──
        # Maps normalized_name → { canonical_name, entity_type, sessions, properties }
        node_registry: dict[str, dict[str, Any]] = {}
        # Maps (canonical_source, canonical_target, rel_type) → edge accumulator
        edge_registry: dict[tuple[str, str, str], dict[str, Any]] = {}
        # Collect all dismissals for filtering
        all_dismissals: set[tuple[str, str, str]] = set()

        for analysis in analyses:
            session_id = analysis["session_id"]
            imap = analysis.get("interconnection_map", {})
            dismissals = analysis.get("dismissals", [])

            # Register dismissals (normalized for matching)
            for d in dismissals:
                d_key = (
                    d["source_entity"].strip().lower(),
                    d["target_entity"].strip().lower(),
                    d["relationship_type"].strip().lower(),
                )
                all_dismissals.add(d_key)

            # Process nodes
            for node in imap.get("nodes", []):
                raw_name = node.get("name", "")
                entity_type = node.get("entity_type", "UNKNOWN")
                canonical = self._resolve_canonical(
                    raw_name, entity_type, node_registry
                )

                if canonical not in node_registry:
                    node_registry[canonical] = {
                        "entity_type": entity_type,
                        "sessions": set(),
                        "properties": {},
                    }

                entry = node_registry[canonical]
                entry["sessions"].add(session_id)
                # Merge properties (later values override earlier ones)
                props = node.get("properties", {})
                if props:
                    entry["properties"].update(props)

            # Process edges
            for edge in imap.get("edges", []):
                if edge.get("dismissed", False):
                    continue

                source_raw = edge.get("source", "")
                target_raw = edge.get("target", "")
                rel_type = edge.get("relationship_type", "related")

                # Resolve source and target to canonical names.
                # Pass "UNKNOWN" as entity_type because edge endpoints
                # may not carry type info — the node registry already
                # has the correct types from node processing above.
                source_canonical = self._resolve_canonical(
                    source_raw, "UNKNOWN", node_registry
                )
                target_canonical = self._resolve_canonical(
                    target_raw, "UNKNOWN", node_registry
                )

                # Check dismissals
                dismissal_key = (
                    source_canonical.strip().lower(),
                    target_canonical.strip().lower(),
                    rel_type.strip().lower(),
                )
                if dismissal_key in all_dismissals:
                    continue

                edge_key = (source_canonical, target_canonical, rel_type)
                if edge_key not in edge_registry:
                    edge_registry[edge_key] = {
                        "confidence_sum": 0.0,
                        "count": 0,
                        "sessions": set(),
                        "details": None,
                    }

                acc = edge_registry[edge_key]
                acc["confidence_sum"] += edge.get("confidence", 0.0)
                acc["count"] += 1
                acc["sessions"].add(session_id)
                # Keep the longest details string (most informative)
                edge_details = edge.get("details")
                if edge_details and (
                    acc["details"] is None
                    or len(edge_details) > len(acc["details"])
                ):
                    acc["details"] = edge_details

        # ── Phase 2: Build output models ──
        combined_nodes = []
        for name, info in node_registry.items():
            combined_nodes.append(CombinedNode(
                entity_type=info["entity_type"],
                name=name,
                appearance_count=len(info["sessions"]),
                source_sessions=sorted(info["sessions"]),
                properties=info["properties"],
            ))

        combined_edges = []
        for (source, target, rel_type), acc in edge_registry.items():
            avg_conf = acc["confidence_sum"] / acc["count"] if acc["count"] > 0 else 0.0
            combined_edges.append(CombinedEdge(
                source=source,
                target=target,
                relationship_type=rel_type,
                avg_confidence=round(avg_conf, 3),
                occurrence_count=acc["count"],
                source_sessions=sorted(acc["sessions"]),
                details=acc["details"],
            ))

        # Sort nodes by appearance count (most connected first), edges by occurrence
        combined_nodes.sort(key=lambda n: n.appearance_count, reverse=True)
        combined_edges.sort(key=lambda e: e.occurrence_count, reverse=True)

        result = CombinedConnectionMap(
            nodes=combined_nodes,
            edges=combined_edges,
            total_analyses=len(analyses),
            generated_at=datetime.now(timezone.utc),
        )

        # ── Update cache ──
        self._cached_map = result
        self._cache_time = now

        logger.info(
            "graph_aggregation_complete",
            total_analyses=len(analyses),
            unique_nodes=len(combined_nodes),
            unique_edges=len(combined_edges),
        )

        return result

    def _resolve_canonical(
        self,
        raw_name: str,
        entity_type: str,
        registry: dict[str, dict[str, Any]],
    ) -> str:
        """Resolve a raw entity name to its canonical form.

        First normalizes the name, then checks existing registry entries
        for fuzzy matches.  If a close match is found, returns the existing
        canonical name; otherwise returns the normalized form as the new
        canonical name.

        The fuzzy threshold is lower for artists (0.85) than for venues
        and promoters (0.90) because artist names have more OCR variation
        and DJ prefix inconsistencies.
        """
        normalized = normalize_artist_name(raw_name)

        if not normalized:
            return raw_name.strip()

        # Check for exact match first (fast path)
        if normalized in registry:
            return normalized

        # Fuzzy match against existing canonical names
        existing_names = list(registry.keys())
        if not existing_names:
            return normalized

        threshold = (
            _ARTIST_FUZZY_THRESHOLD
            if entity_type.upper() == "ARTIST"
            else _DEFAULT_FUZZY_THRESHOLD
        )

        match = fuzzy_match(normalized, existing_names, threshold=threshold)
        if match is not None:
            return match[0]  # Return the existing canonical name

        return normalized

    async def get_node_detail(self, entity_name: str) -> dict[str, Any]:
        """Get detailed information about a specific entity across all analyses.

        Returns all fliers, edges, and metadata for a given entity name.
        Used by the entity detail sidebar in the Connections tab.

        Reuses the cached aggregate instead of rebuilding — this means
        clicking through multiple entities in the sidebar only triggers
        one aggregate() rebuild per TTL window.
        """
        combined = await self.aggregate()

        # Find the node (fuzzy match against combined nodes)
        node_names = [n.name for n in combined.nodes]
        match = fuzzy_match(entity_name, node_names, threshold=0.85)
        canonical_name = match[0] if match else entity_name

        target_node = None
        for node in combined.nodes:
            if node.name == canonical_name:
                target_node = node
                break

        if target_node is None:
            return {"found": False, "name": entity_name}

        # Find all edges involving this entity
        related_edges = []
        for edge in combined.edges:
            if edge.source == canonical_name or edge.target == canonical_name:
                related_edges.append(edge.model_dump())

        return {
            "found": True,
            "name": target_node.name,
            "entity_type": target_node.entity_type,
            "appearance_count": target_node.appearance_count,
            "source_sessions": target_node.source_sessions,
            "properties": target_node.properties,
            "edges": related_edges,
            "total_connections": len(related_edges),
        }

    async def get_stats(self) -> dict[str, Any]:
        """Return summary statistics for the combined connection map.

        Used by the stats header in the Connections tab.
        Reuses the cached aggregate — no redundant rebuild.
        """
        combined = await self.aggregate()

        # Count entity types
        type_counts: dict[str, int] = {}
        for node in combined.nodes:
            t = node.entity_type
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total_analyses": combined.total_analyses,
            "unique_nodes": len(combined.nodes),
            "unique_edges": len(combined.edges),
            "entity_type_counts": type_counts,
            "generated_at": combined.generated_at.isoformat() if combined.generated_at else None,
        }
