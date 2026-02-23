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

import structlog

from src.models.flier import ExtractedEntities
from src.models.pipeline import PipelinePhase, PipelineState
from src.utils.logging import get_logger


class ConfirmationGate:
    """Manages pipeline pause at the USER_CONFIRMATION phase for user review.

    Pending sessions are stored in a dict-like store keyed by
    ``session_id``.  When a :class:`PersistentSessionStore` is
    injected, pending sessions survive container restarts.  Falls
    back to a plain in-memory dict when no store is provided.
    """

    def __init__(
        self,
        pending_store: MutableMapping[str, PipelineState] | None = None,
    ) -> None:
        # Accept any dict-like object — this is the "Dependency Inversion"
        # principle: we depend on the MutableMapping abstraction, not on
        # PersistentSessionStore directly.  In tests, pass a plain {}.
        self._pending_sessions: MutableMapping[str, PipelineState] = (
            pending_store if pending_store is not None else {}
        )
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
