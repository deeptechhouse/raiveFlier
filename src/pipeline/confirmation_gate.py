"""Confirmation gate for the pipeline's user-review pause point.

After Phase 1 (OCR + Entity Extraction), the pipeline pauses so the user
can review, edit, and confirm the extracted entities before research begins.
This module manages the pending session state during that pause.
"""

from __future__ import annotations

import structlog

from src.models.flier import ExtractedEntities
from src.models.pipeline import PipelinePhase, PipelineState
from src.utils.logging import get_logger


class ConfirmationGate:
    """Manages pipeline pause at the USER_CONFIRMATION phase for user review.

    Pending sessions are stored in an in-memory dictionary keyed by
    ``session_id``.  The gate exposes methods to submit a state for review,
    retrieve it, confirm it (with user-edited entities), or cancel it.
    """

    def __init__(self) -> None:
        self._pending_sessions: dict[str, PipelineState] = {}
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
