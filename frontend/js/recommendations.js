/**
 * recommendations.js — raiveFlier recommendation panel module.
 *
 * ROLE IN THE APPLICATION
 * =======================
 * A bottom-anchored, collapsible floating panel that displays artist
 * recommendations derived from the flier analysis. This is the discovery
 * feature — it helps users find new artists related to the ones on the flier.
 *
 * TWO-PHASE LOADING STRATEGY
 * ==========================
 * Uses progressive enhancement to show results as fast as possible:
 *
 *   Phase 1 (quick) — Label-mate results from Discogs API, fetched
 *     eagerly on init(). No LLM calls. Renders in 1-3 seconds.
 *     Endpoint: GET /api/v1/fliers/{session_id}/recommendations/quick
 *
 *   Phase 2 (full)  — All tiers + LLM-generated explanations, fetched
 *     in background after quick results arrive. Silently replaces the
 *     panel content when ready. A "backfill indicator" (small spinner)
 *     shows at the bottom of the card list while Phase 2 loads.
 *     Endpoint: GET /api/v1/fliers/{session_id}/recommendations
 *
 * Phase 2 failure is non-fatal — quick results remain visible. This
 * ensures the user always sees something even if the LLM is slow/down.
 *
 * THREE-TIER PRIORITY DISPLAY
 * ===========================
 * Recommendations are categorized by how the connection was discovered:
 *   - label_mate: Artists on the same record label (strongest signal)
 *   - shared_flier: Artists who appeared on other fliers with these artists
 *   - shared_lineup: Artists from the same event lineups
 *   - llm_suggestion: LLM-identified connections (weakest signal, most creative)
 *
 * Each tier gets a distinct CSS color via BEM modifier classes.
 *
 * DOM STRUCTURE
 * =============
 * Targets #reco-panel (defined in index.html). Builds its own inner HTML:
 *   .reco-panel__toggle-bar — Collapsed bar with title, count badge, chevron
 *   .reco-panel__body       — Expandable container for recommendation cards
 *
 * MODULE COMMUNICATION
 * ====================
 * - Called by Results.fetchAndDisplayResults() via Recommendations.init()
 * - Uses Rating module for per-recommendation thumbs up/down
 * - No outbound calls to other modules (leaf node in the dependency graph)
 */

"use strict";

const Recommendations = (() => {
  // ------------------------------------------------------------------
  // Private state
  // ------------------------------------------------------------------

  // Panel UI state
  let _isOpen = false;            // Whether the panel body is expanded
  let _sessionId = null;          // Pipeline session UUID for API calls
  let _recommendations = [];      // Array of recommendation objects from the API
  let _error = null;              // Last error message, if any

  // Auto-retry constants — the backend preload may still be running when
  // the frontend first requests recommendations.  Retrying silently avoids
  // showing the error/retry button during the normal preload window.
  const _MAX_QUICK_RETRIES = 2;       // Total retry attempts before showing error
  const _QUICK_RETRY_DELAY_MS = 2500; // Wait between retries (preload needs ~3-5s)

  // Two-phase fetch state — these flags coordinate the progressive loading
  // strategy so the UI knows which phase is active and what to display.
  let _isLoadingQuick = false;    // Phase 1 (label-mates) fetch in flight
  let _isLoadingFull = false;     // Phase 2 (all tiers + LLM) fetch in flight
  let _hasFetchedQuick = false;   // Phase 1 completed successfully
  let _hasFetchedFull = false;    // Phase 2 completed successfully
  let _isPartial = true;          // True until Phase 2 completes — controls "+" suffix on count badge

  // ------------------------------------------------------------------
  // Utility
  // ------------------------------------------------------------------

  /** Escape HTML special characters to prevent XSS. */
  function _esc(str) {
    if (str == null) return "";
    const div = document.createElement("div");
    div.appendChild(document.createTextNode(String(str)));
    return div.innerHTML;
  }

  // ------------------------------------------------------------------
  // DOM references
  // ------------------------------------------------------------------

  function _getPanel() {
    return document.getElementById("reco-panel");
  }

  // ------------------------------------------------------------------
  // Rendering
  // ------------------------------------------------------------------

  /**
   * Render the toggle bar and empty body into #reco-panel.
   * The panel becomes visible but stays collapsed until toggled.
   *
   * The toggle bar acts as an always-visible "peek" element at the bottom
   * of the viewport. It shows a count badge (e.g., "5+ artists") so the
   * user knows recommendations are available without needing to open the panel.
   * Clicking the bar expands the body to reveal the recommendation cards.
   */
  function _renderShell() {
    const panel = _getPanel();
    if (!panel) return;

    panel.innerHTML = `
      <button type="button" class="reco-panel__toggle-bar" id="reco-toggle"
              aria-expanded="false" aria-controls="reco-content">
        <svg class="reco-panel__icon" width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M9 18V5l12-2v13" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <circle cx="6" cy="18" r="3" stroke="currentColor" stroke-width="2"/>
          <circle cx="18" cy="16" r="3" stroke="currentColor" stroke-width="2"/>
        </svg>
        <span class="reco-panel__title">Discover New Artists</span>
        <span class="reco-panel__count" id="reco-count"></span>
        <svg class="reco-panel__chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M18 15l-6-6-6 6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </button>
      <div class="reco-panel__body" id="reco-content" role="region" aria-label="Recommended artists">
      </div>
    `;

    // Show the panel
    panel.classList.add("reco-panel--visible");

    // Attach toggle listener
    const toggle = document.getElementById("reco-toggle");
    if (toggle) {
      toggle.addEventListener("click", togglePanel);
    }
  }

  /** Render a loading spinner inside the panel body. */
  function _renderLoading() {
    const content = document.getElementById("reco-content");
    if (!content) return;
    content.innerHTML = `
      <div class="reco-panel__loading">
        <div class="spinner" aria-label="Loading recommendations"></div>
        <span class="reco-panel__loading-text">Discovering artists&hellip;</span>
      </div>
    `;
  }

  /** Render an error message with a retry button. */
  function _renderError(message) {
    const content = document.getElementById("reco-content");
    if (!content) return;
    content.innerHTML = `
      <div class="reco-panel__error">
        <p class="reco-panel__error-text">${_esc(message)}</p>
        <button type="button" class="reco-panel__retry" id="reco-retry">Retry</button>
      </div>
    `;
    const retryBtn = document.getElementById("reco-retry");
    if (retryBtn) {
      retryBtn.addEventListener("click", () => {
        _hasFetchedQuick = false;
        _hasFetchedFull = false;
        _error = null;
        _fetchQuickRecommendations();
      });
    }
  }

  /**
   * Map a source_tier value to a human-readable label.
   * These labels appear as badge text on each recommendation card's header,
   * telling the user how the recommendation was discovered.
   * @param {string} tier — API value: "label_mate" | "shared_flier" | "shared_lineup" | "llm_suggestion"
   * @returns {string} Display label for the UI
   */
  function _tierLabel(tier) {
    const labels = {
      "label_mate": "Label-Mate",
      "shared_flier": "Shared Flier",
      "shared_lineup": "Shared Lineup",
      "llm_suggestion": "LLM Pick",
    };
    return labels[tier] || tier;
  }

  /**
   * Map a source_tier value to the appropriate CSS modifier class.
   * Each tier gets a distinct visual treatment (color coding) defined
   * in style.css via BEM modifier classes on the tier badge element.
   * @param {string} tier — API value
   * @returns {string} BEM modifier class name for CSS styling
   */
  function _tierClass(tier) {
    const classes = {
      "label_mate": "reco-card__tier--label-mate",
      "shared_flier": "reco-card__tier--shared-flier",
      "shared_lineup": "reco-card__tier--shared-lineup",
      "llm_suggestion": "reco-card__tier--llm",
    };
    return classes[tier] || "";
  }

  /**
   * Update the count badge on the collapsed toggle bar.
   * Shows "5+ artists" when only Phase 1 results are available (_isPartial=true),
   * or "8 artists" when Phase 2 has completed with the final count.
   * The "+" suffix signals to the user that more results may be coming.
   */
  function _updateCountBadge() {
    const countEl = document.getElementById("reco-count");
    if (!countEl) return;

    const count = _recommendations.length;
    if (count > 0) {
      const suffix = _isPartial ? "+" : "";
      countEl.textContent = `${count}${suffix} artists`;
    } else {
      countEl.textContent = "";
    }
  }

  /**
   * Render the recommendation cards into the panel body.
   *
   * This is a full re-render: it builds the entire card list from the
   * _recommendations array, sets innerHTML, then post-processes by
   * injecting Rating widgets into each card's footer placeholder.
   *
   * Called multiple times during the two-phase lifecycle:
   *   1. After Phase 1 completes — shows quick results + backfill spinner
   *   2. While Phase 2 loads — re-renders to show/hide the backfill indicator
   *   3. After Phase 2 completes — final render with all tiers, no spinner
   *
   * Each recommendation card displays:
   *   - Rank number + artist name + tier badge (header)
   *   - Genre tags (optional)
   *   - Reason text explaining the connection
   *   - Connected-to line with label or event context (optional)
   *   - Connection strength bar (0-100% fill)
   *   - Rating widget (thumbs up/down via Rating module)
   */
  function _renderResults() {
    const content = document.getElementById("reco-content");
    if (!content) return;

    // Edge case: no results yet — show appropriate fallback
    if (!_recommendations.length) {
      // No quick results yet — if full fetch is in progress, show spinner
      if (_isLoadingFull) {
        _renderLoading();
        return;
      }
      content.innerHTML = `
        <div class="reco-panel__empty">
          <p>No recommendations available yet.</p>
        </div>
      `;
      return;
    }

    // Build card HTML for each recommendation
    let html = "";
    _recommendations.forEach((rec, i) => {
      const rank = i + 1;

      // Genre tags — rendered as inline chip elements
      const genreTags = (rec.genres || [])
        .map((g) => `<span class="reco-card__genre">${_esc(g)}</span>`)
        .join("");

      // Connected-to — which flier artists this recommendation links to
      const connectedTo = (rec.connected_to || [])
        .map((a) => `<strong>${_esc(a)}</strong>`)
        .join(", ");

      // Connection strength as percentage for the fill bar width
      const strengthPct = Math.round((rec.connection_strength || 0.5) * 100);

      // Context detail — shows "via [label]" for label-mates or "at [event]" for shared events
      let connectionDetail = "";
      if (rec.label_name) {
        connectionDetail = ` via <strong>${_esc(rec.label_name)}</strong>`;
      } else if (rec.event_name) {
        connectionDetail = ` at <strong>${_esc(rec.event_name)}</strong>`;
      }

      // Rating widget placeholder — Rating.renderWidget() HTML will be injected here
      // after innerHTML is set, because Rating needs actual DOM elements to attach to.
      const ratingId = `reco-rating-${rank}`;

      html += `
        <div class="reco-card">
          <div class="reco-card__header">
            <span class="reco-card__rank">${rank}.</span>
            <span class="reco-card__name">${_esc(rec.artist_name)}</span>
            <span class="reco-card__connection-type ${_tierClass(rec.source_tier)}">${_esc(_tierLabel(rec.source_tier))}</span>
          </div>
          ${genreTags ? `<div class="reco-card__genres">${genreTags}</div>` : ""}
          <p class="reco-card__reason">${_esc(rec.reason)}</p>
          ${connectedTo ? `<p class="reco-card__connected-to">Connected to: ${connectedTo}${connectionDetail}</p>` : ""}
          <div class="reco-card__strength">
            <div class="reco-card__strength-fill" style="width: ${strengthPct}%"></div>
          </div>
          <div class="reco-card__footer" id="${ratingId}"></div>
        </div>
      `;
    });

    // Backfill indicator — shown at the bottom of the card list when Phase 1 results
    // are displayed but Phase 2 is still loading. Gives visual feedback that more
    // recommendations are on the way without blocking the existing results.
    if (_isPartial && _isLoadingFull) {
      html += `
        <div class="reco-panel__backfill-indicator">
          <div class="spinner spinner--small" aria-hidden="true"></div>
          <span>Loading additional recommendations&hellip;</span>
        </div>
      `;
    }

    content.innerHTML = html;

    // Post-render: inject Rating widgets into each card's footer placeholder.
    // Uses the same two-step pattern as other modules:
    //   1. Rating.renderWidget() returns an HTML string (inserted via innerHTML)
    //   2. Rating.initWidgets() attaches a single delegated click listener on the container
    if (typeof Rating !== "undefined" && _sessionId) {
      _recommendations.forEach((rec, i) => {
        const container = document.getElementById(`reco-rating-${i + 1}`);
        if (container) {
          container.innerHTML = Rating.renderWidget("RECOMMENDATION", rec.artist_name);
        }
      });
      // Single delegated listener on the panel body covers all rating buttons
      Rating.initWidgets(content, _sessionId);
    }

    _updateCountBadge();
  }

  // ------------------------------------------------------------------
  // API interaction — two-phase fetch
  // ------------------------------------------------------------------

  /**
   * Phase 1: Fetch quick label-mate recommendations (no LLM).
   *
   * Called eagerly from init() — NOT on panel open. This means the data
   * starts loading as soon as the results view renders, so if the user
   * clicks the toggle bar a few seconds later, results are already available.
   *
   * API: GET /api/v1/fliers/{session_id}/recommendations/quick
   * Response: { recommendations: [...], is_partial: true }
   *
   * On success, immediately kicks off Phase 2 (_fetchFullRecommendations)
   * in the background. On failure, retries up to _MAX_QUICK_RETRIES times
   * with a delay (the backend preload may still be running). Only shows
   * the error state after all retries are exhausted.
   *
   * @param {number} attempt — Current retry attempt (0-based, internal use).
   */
  async function _fetchQuickRecommendations(attempt = 0) {
    // Guard against duplicate fetches — idempotent by design
    if (_hasFetchedQuick || _isLoadingQuick || !_sessionId) return;

    _isLoadingQuick = true;
    _error = null;

    // If the user already opened the panel before results arrived, show spinner
    if (_isOpen) {
      _renderLoading();
    }

    try {
      const resp = await fetch(
        `/api/v1/fliers/${encodeURIComponent(_sessionId)}/recommendations/quick`
      );
      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${resp.status}`);
      }
      const data = await resp.json();
      _recommendations = data.recommendations || [];
      _isPartial = data.is_partial !== false;  // Default to partial unless explicitly false
      _hasFetchedQuick = true;
      _isLoadingQuick = false;

      // Update count badge on the collapsed bar (visible even when panel is closed)
      _updateCountBadge();

      // If panel is already open, render quick results immediately
      if (_isOpen) {
        _renderResults();
      }

      // Kick off Phase 2 in background — this will silently upgrade the results
      _fetchFullRecommendations();
    } catch (err) {
      _isLoadingQuick = false;

      // Auto-retry with delay — the backend preload may still be running.
      // This eliminates the need for the user to manually click "Retry".
      if (attempt < _MAX_QUICK_RETRIES) {
        await new Promise((r) => setTimeout(r, _QUICK_RETRY_DELAY_MS));
        return _fetchQuickRecommendations(attempt + 1);
      }

      // All retries exhausted — show error with manual retry button
      _error = err.message || "Failed to load recommendations";
      if (_isOpen) {
        _renderError(_error);
      }
      // Still attempt full fetch — quick failure shouldn't block full
      _fetchFullRecommendations();
    }
  }

  /**
   * Phase 2: Fetch full recommendations (all tiers + LLM explanations).
   *
   * Runs in the background after Phase 1 results are displayed. This is
   * a "silent upgrade" — the user sees quick results immediately, and
   * the full results seamlessly replace them when ready.
   *
   * API: GET /api/v1/fliers/{session_id}/recommendations
   * Response: { recommendations: [...] }  (is_partial is always false here)
   *
   * IMPORTANT: Failure is non-fatal. If this fetch fails, the Phase 1
   * quick results remain visible and the user can still interact with them.
   * Errors are logged to console.warn, not shown to the user.
   * This graceful degradation is a deliberate UX choice — partial results
   * are better than an error message replacing working results.
   */
  async function _fetchFullRecommendations() {
    // Guard against duplicate fetches — idempotent by design
    if (_hasFetchedFull || _isLoadingFull || !_sessionId) return;

    _isLoadingFull = true;

    // If panel is open and we have partial results, re-render to show the
    // backfill indicator (small spinner at the bottom of the card list)
    if (_isOpen && _recommendations.length > 0) {
      _renderResults();
    }

    try {
      const resp = await fetch(
        `/api/v1/fliers/${encodeURIComponent(_sessionId)}/recommendations`
      );
      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        console.warn("Full recommendations fetch failed:", errData.detail || resp.status);
        return;
      }
      const data = await resp.json();

      // Replace the entire recommendations array — full results are a superset
      // of quick results, so this is a complete replacement, not a merge.
      _recommendations = data.recommendations || [];
      _isPartial = false;          // Removes "+" suffix from count badge
      _hasFetchedFull = true;
      _error = null;               // Clear any Phase 1 error since Phase 2 succeeded

      _updateCountBadge();

      if (_isOpen) {
        _renderResults();          // Silent upgrade — replaces quick results with full set
      }
    } catch (err) {
      console.warn("Full recommendations background fetch failed:", err.message);
      // Non-fatal: quick results remain visible — no user-facing error
    } finally {
      _isLoadingFull = false;
      // If panel is open and we were showing the backfill indicator, re-render to remove it
      // (whether the fetch succeeded or failed, the spinner should go away)
      if (_isOpen && _isPartial) {
        _renderResults();
      }
    }
  }

  // ------------------------------------------------------------------
  // Public API
  // ------------------------------------------------------------------

  /**
   * Initialize the recommendation panel for a given session.
   *
   * Called by Results.fetchAndDisplayResults() after the results view renders.
   * This is the entry point for the entire recommendation lifecycle:
   *   1. Resets all private state to clean defaults
   *   2. Renders the collapsed toggle bar (shell) into #reco-panel
   *   3. Eagerly kicks off Phase 1 fetch (label-mates, no LLM)
   *
   * The eager fetch means data starts loading immediately, not when the
   * user opens the panel. By the time they notice and click the toggle bar,
   * Phase 1 results are usually already available.
   *
   * @param {string} sessionId - The pipeline session UUID.
   */
  function init(sessionId) {
    // Full state reset — supports re-initialization for new sessions
    _sessionId = sessionId;
    _isOpen = false;
    _isLoadingQuick = false;
    _isLoadingFull = false;
    _hasFetchedQuick = false;
    _hasFetchedFull = false;
    _recommendations = [];
    _isPartial = true;
    _error = null;
    _renderShell();

    // Eagerly start Phase 1 fetch (label-mates only, no LLM)
    // Phase 2 will be triggered automatically when Phase 1 completes
    _fetchQuickRecommendations();
  }

  /**
   * Expand the panel and render available results.
   * Handles three possible states when opened:
   *   1. Data available (most common) — renders results immediately
   *   2. Phase 1 in flight — shows loading spinner
   *   3. Neither fetched nor loading — triggers fetch (safety fallback)
   */
  function openPanel() {
    const panel = _getPanel();
    if (!panel) return;
    _isOpen = true;
    panel.classList.add("reco-panel--open");
    const toggle = document.getElementById("reco-toggle");
    if (toggle) toggle.setAttribute("aria-expanded", "true");

    if (_hasFetchedQuick || _hasFetchedFull) {
      // Results available — render them
      _renderResults();
    } else if (_isLoadingQuick) {
      // Quick fetch in progress — show spinner
      _renderLoading();
    } else {
      // Edge case: init wasn't called or fetch didn't start
      _fetchQuickRecommendations();
    }
  }

  /** Collapse the panel. */
  function closePanel() {
    const panel = _getPanel();
    if (!panel) return;
    _isOpen = false;
    panel.classList.remove("reco-panel--open");
    const toggle = document.getElementById("reco-toggle");
    if (toggle) toggle.setAttribute("aria-expanded", "false");
  }

  /** Toggle between open and closed states. */
  function togglePanel() {
    if (_isOpen) {
      closePanel();
    } else {
      openPanel();
    }
  }

  /** @returns {boolean} Whether the panel is currently open. */
  function isOpen() {
    return _isOpen;
  }

  // ------------------------------------------------------------------
  // Public API — exposed via the Revealing Module Pattern.
  // Only these 5 methods are accessible outside the IIFE closure.
  // All internal state and helper functions remain private.
  // ------------------------------------------------------------------
  return {
    init,          // Called by Results module to bootstrap the panel
    openPanel,     // Expand the panel body
    closePanel,    // Collapse the panel body
    togglePanel,   // Toggle between open/closed (used by the toggle bar click handler)
    isOpen,        // Query current panel state
  };
})();
