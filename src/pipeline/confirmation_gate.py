"""Confirmation gate for the pipeline's user-review pause point.

After Phase 1 (OCR + Entity Extraction), the pipeline pauses so the user
can review, edit, and confirm the extracted entities before research begins.
This module manages the pending session state during that pause.

# ─── HOW THE CONFIRMATION GATE WORKS (Junior Developer Guide) ─────────
#
# The pipeline has a "human-in-the-loop" design:
#
#   Phase 1 (OCR + Extraction) ──→ PAUSE (user reviews) ──→ Phase 2-5 (Research)
#                                       ↑
#                                  ConfirmationGate
#
# Workflow:
#   1. Pipeline Phase 1 completes → state is stored via submit_for_review()
#   2. Frontend displays extracted entities → user can edit/delete/add
#   3. User clicks "Confirm" → API calls confirm() with edited entities
#   4. ConfirmationGate returns updated state → pipeline resumes at Phase 2
#
# The gate also supports cancel() if the user abandons the analysis.
#
# Storage: The pending_store is a MutableMapping (dict-like). In production,
# it's a PersistentSessionStore (SQLite-backed) so pending sessions survive
# container restarts.  In tests, pass a plain dict.
# ──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING

import structlog

from src.models.flier import ExtractedEntities
from src.models.pipeline import PipelinePhase, PipelineState
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.interfaces.feedback_provider import IFeedbackProvider


class ConfirmationGate:
    """Manages pipeline pause at the USER_CONFIRMATION phase for user review.

    Pending sessions are stored in a dict-like store keyed by
    ``session_id``.  When a :class:`PersistentSessionStore` is
    injected, pending sessions survive container restarts.  Falls
    back to a plain in-memory dict when no store is provided.

    When a ``feedback_provider`` is injected, the gate records
    calibration samples on each confirmation — comparing the original
    LLM confidence scores against which entities the user kept or
    removed.  This feeds the progressive confidence calibration system
    (Optimization I) so future extractions show more accurate scores.
    """

    def __init__(
        self,
        pending_store: MutableMapping[str, PipelineState] | None = None,
        feedback_provider: IFeedbackProvider | None = None,
    ) -> None:
        # Accept any dict-like object — this is the "Dependency Inversion"
        # principle: we depend on the MutableMapping abstraction, not on
        # PersistentSessionStore directly.  In tests, pass a plain {}.
        self._pending_sessions: MutableMapping[str, PipelineState] = (
            pending_store if pending_store is not None else {}
        )
        # Optional feedback provider for storing calibration samples.
        # When None, no calibration data is collected (existing behavior).
        self._feedback: IFeedbackProvider | None = feedback_provider
        self._logger: structlog.BoundLogger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit_for_review(self, state: PipelineState) -> str:
        """Store a pipeline state for user review and return its session ID.

        Parameters
        ----------
        state:
            The pipeline state paused at USER_CONFIRMATION phase.

        Returns
        -------
        str
            The ``session_id`` under which the state is stored.
        """
        self._pending_sessions[state.session_id] = state

        self._logger.info(
            "session_submitted_for_review",
            session_id=state.session_id,
            artists=len(state.extracted_entities.artists) if state.extracted_entities else 0,
        )

        return state.session_id

    async def get_pending(self, session_id: str) -> PipelineState | None:
        """Retrieve a pending session state for user review.

        Parameters
        ----------
        session_id:
            The session to retrieve.

        Returns
        -------
        PipelineState or None
            The stored state, or ``None`` if no pending session exists
            for the given ID.
        """
        state = self._pending_sessions.get(session_id)

        if state is None:
            self._logger.debug(
                "pending_session_not_found",
                session_id=session_id,
            )

        return state

    async def confirm(
        self,
        session_id: str,
        confirmed_entities: ExtractedEntities,
    ) -> PipelineState:
        """Confirm a pending session with user-reviewed entities.

        Removes the session from the pending store and returns a new
        :class:`PipelineState` with ``confirmed_entities`` set, ready
        for phases 2–5.

        Parameters
        ----------
        session_id:
            The session to confirm.
        confirmed_entities:
            The user-reviewed (and possibly edited) entities.

        Returns
        -------
        PipelineState
            Updated state with confirmed entities attached.

        Raises
        ------
        KeyError
            If no pending session exists for the given ID.
        """
        state = self._pending_sessions.pop(session_id, None)
        if state is None:
            self._logger.error(
                "confirm_failed_session_not_found",
                session_id=session_id,
            )
            raise KeyError(f"No pending session found for ID: {session_id}")

        # model_copy(update={...}) creates a NEW PipelineState with the
        # confirmed entities and advances the phase to RESEARCH.
        # The original `state` is unchanged (frozen/immutable model).
        confirmed_state = state.model_copy(
            update={
                "confirmed_entities": confirmed_entities,
                "current_phase": PipelinePhase.RESEARCH,
            }
        )

        self._logger.info(
            "session_confirmed",
            session_id=session_id,
            artists=len(confirmed_entities.artists),
            has_venue=confirmed_entities.venue is not None,
            has_date=confirmed_entities.date is not None,
        )

        # -- Calibration sample collection (Optimization I) -----------------
        # Compare original extracted entities (from the LLM) against what
        # the user confirmed.  Each entity that the user kept is ground_truth=1;
        # each entity that was removed is ground_truth=0.  These samples feed
        # the ConfidenceCalibrator so future LLM confidence scores are adjusted
        # to reflect real-world accuracy.
        if self._feedback and state.extracted_entities:
            await self._collect_calibration_samples(
                session_id=session_id,
                original=state.extracted_entities,
                confirmed=confirmed_entities,
            )

        return confirmed_state

    async def cancel(self, session_id: str) -> bool:
        """Cancel a pending session and remove it from the store.

        Parameters
        ----------
        session_id:
            The session to cancel.

        Returns
        -------
        bool
            ``True`` if the session existed and was removed,
            ``False`` if no pending session was found.
        """
        removed = self._pending_sessions.pop(session_id, None)

        if removed is not None:
            self._logger.info("session_cancelled", session_id=session_id)
            return True

        self._logger.debug(
            "cancel_session_not_found",
            session_id=session_id,
        )
        return False

    # ------------------------------------------------------------------
    # Calibration data collection
    # ------------------------------------------------------------------

    async def _collect_calibration_samples(
        self,
        session_id: str,
        original: ExtractedEntities,
        confirmed: ExtractedEntities,
    ) -> None:
        """Compare original and confirmed entities, storing calibration samples.

        For each entity in the original extraction, determine whether the
        user kept it (ground_truth=1) or removed it (ground_truth=0) by
        checking if its text appears in the confirmed set.  Store each
        observation via the feedback provider so the ConfidenceCalibrator
        can learn how raw LLM scores map to actual accuracy.

        Parameters
        ----------
        session_id:
            Pipeline session UUID for traceability.
        original:
            The entities as extracted by the LLM (before user review).
        confirmed:
            The entities after user review (kept, edited, or removed).
        """
        assert self._feedback is not None  # noqa: S101 — guarded by caller

        # Build a set of confirmed entity texts for O(1) membership lookup.
        # We compare by normalized lowercase text to handle minor edits
        # (capitalization changes, whitespace trimming) that still represent
        # the same entity.
        confirmed_artist_texts = {
            a.text.strip().lower() for a in confirmed.artists
        }

        # -- Artists: the most numerous entity type on rave fliers --
        for artist in original.artists:
            kept = artist.text.strip().lower() in confirmed_artist_texts
            try:
                await self._feedback.store_calibration_sample(
                    score_type="entity_confidence",
                    predicted_score=artist.confidence,
                    ground_truth=1 if kept else 0,
                    entity_type="ARTIST",
                    session_id=session_id,
                )
            except Exception:
                # Calibration is non-critical — log and continue so the
                # pipeline is never blocked by a calibration write failure.
                self._logger.warning(
                    "calibration_sample_write_failed",
                    entity_type="ARTIST",
                    session_id=session_id,
                    exc_info=True,
                )

        # -- Singular entities: venue, date, promoter, event_name --
        # Each is either present or absent in the confirmed set.
        singular_pairs = [
            (original.venue, confirmed.venue, "VENUE"),
            (original.date, confirmed.date, "DATE"),
            (original.promoter, confirmed.promoter, "PROMOTER"),
            (original.event_name, confirmed.event_name, "EVENT"),
        ]
        for orig_entity, conf_entity, entity_type in singular_pairs:
            if orig_entity is None:
                continue
            # If the confirmed entity is present and has the same text
            # (case-insensitive), the user kept it.
            kept = (
                conf_entity is not None
                and conf_entity.text.strip().lower()
                == orig_entity.text.strip().lower()
            )
            try:
                await self._feedback.store_calibration_sample(
                    score_type="entity_confidence",
                    predicted_score=orig_entity.confidence,
                    ground_truth=1 if kept else 0,
                    entity_type=entity_type,
                    session_id=session_id,
                )
            except Exception:
                self._logger.warning(
                    "calibration_sample_write_failed",
                    entity_type=entity_type,
                    session_id=session_id,
                    exc_info=True,
                )
