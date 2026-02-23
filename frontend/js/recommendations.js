/**
 * recommendations.js — raiveFlier recommendation panel module.
 *
 * Renders a bottom-anchored, collapsible floating panel for displaying
 * artist recommendations based on flier analysis.
 *
 * Two-phase loading strategy:
 *   Phase 1 (quick) — label-mate results from Discogs API, fetched
 *     eagerly on init(). No LLM calls. Renders in 1-3 seconds.
 *   Phase 2 (full)  — all tiers + LLM explanations, fetched in
 *     background after quick results arrive. Silently replaces the
 *     panel content when ready.
 *
 * Three-tier priority display: label-mates, shared-flier,
 * shared-lineup, LLM picks.
 */

"use strict";

const Recommendations = (() => {
  // ------------------------------------------------------------------
  // Private state
  // ------------------------------------------------------------------

  let _isOpen = false;
  let _sessionId = null;
  let _recommendations = [];
  let _error = null;

  // Two-phase fetch state
  let _isLoadingQuick = false;
  let _isLoadingFull = false;
  let _hasFetchedQuick = false;
  let _hasFetchedFull = false;
  let _isPartial = true;

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
   * @param {string} tier
   * @returns {string}
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
   * @param {string} tier
   * @returns {string}
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

  /** Update the count badge on the collapsed toggle bar. */
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

  /** Render the recommendation cards into the panel body. */
  function _renderResults() {
    const content = document.getElementById("reco-content");
    if (!content) return;

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

    let html = "";
    _recommendations.forEach((rec, i) => {
      const rank = i + 1;
      const genreTags = (rec.genres || [])
        .map((g) => `<span class="reco-card__genre">${_esc(g)}</span>`)
        .join("");

      const connectedTo = (rec.connected_to || [])
        .map((a) => `<strong>${_esc(a)}</strong>`)
        .join(", ");

      const strengthPct = Math.round((rec.connection_strength || 0.5) * 100);

      let connectionDetail = "";
      if (rec.label_name) {
        connectionDetail = ` via <strong>${_esc(rec.label_name)}</strong>`;
      } else if (rec.event_name) {
        connectionDetail = ` at <strong>${_esc(rec.event_name)}</strong>`;
      }

      // Rating widget placeholder
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

    // Backfill indicator when partial results are displayed and full fetch is running
    if (_isPartial && _isLoadingFull) {
      html += `
        <div class="reco-panel__backfill-indicator">
          <div class="spinner spinner--small" aria-hidden="true"></div>
          <span>Loading additional recommendations&hellip;</span>
        </div>
      `;
    }

    content.innerHTML = html;

    // Initialize rating widgets for each recommendation
    if (typeof Rating !== "undefined" && _sessionId) {
      _recommendations.forEach((rec, i) => {
        const container = document.getElementById(`reco-rating-${i + 1}`);
        if (container) {
          container.innerHTML = Rating.renderWidget("RECOMMENDATION", rec.artist_name);
        }
      });
      Rating.initWidgets(content, _sessionId);
    }

    _updateCountBadge();
  }

  // ------------------------------------------------------------------
  // API interaction — two-phase fetch
  // ------------------------------------------------------------------

  /**
   * Phase 1: Fetch quick label-mate recommendations (no LLM).
   * Called eagerly from init(), not on panel open.
   */
  async function _fetchQuickRecommendations() {
    if (_hasFetchedQuick || _isLoadingQuick || !_sessionId) return;

    _isLoadingQuick = true;
    _error = null;

    // If panel is already open, show spinner
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
      _isPartial = data.is_partial !== false;
      _hasFetchedQuick = true;

      // Update count badge on collapsed bar
      _updateCountBadge();

      // If panel is open, render quick results immediately
      if (_isOpen) {
        _renderResults();
      }

      // Kick off background full fetch
      _fetchFullRecommendations();
    } catch (err) {
      _error = err.message || "Failed to load recommendations";
      if (_isOpen) {
        _renderError(_error);
      }
      // Still attempt full fetch — quick failure shouldn't block full
      _fetchFullRecommendations();
    } finally {
      _isLoadingQuick = false;
    }
  }

  /**
   * Phase 2: Fetch full recommendations (all tiers + LLM).
   * Runs in background after quick results are displayed.
   * Failure is non-fatal — quick results remain visible.
   */
  async function _fetchFullRecommendations() {
    if (_hasFetchedFull || _isLoadingFull || !_sessionId) return;

    _isLoadingFull = true;

    // If panel is open and we have partial results, re-render to show backfill indicator
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
      _recommendations = data.recommendations || [];
      _isPartial = false;
      _hasFetchedFull = true;
      _error = null;

      _updateCountBadge();

      if (_isOpen) {
        _renderResults();
      }
    } catch (err) {
      console.warn("Full recommendations background fetch failed:", err.message);
      // Non-fatal: quick results remain visible
    } finally {
      _isLoadingFull = false;
      // If panel is open and we were showing the backfill indicator, re-render to remove it
      if (_isOpen && _isPartial) {
        _renderResults();
      }
    }
  }

  // ------------------------------------------------------------------
  // Public API
  // ------------------------------------------------------------------

  /**
   * Initialise the recommendation panel for a given session.
   * Renders the shell (collapsed toggle bar) and eagerly starts
   * the quick fetch for label-mate results.
   * @param {string} sessionId - The pipeline session UUID.
   */
  function init(sessionId) {
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

    // Eagerly start quick fetch (label-mates only, no LLM)
    _fetchQuickRecommendations();
  }

  /** Expand the panel and render available results. */
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

  return {
    init,
    openPanel,
    closePanel,
    togglePanel,
    isOpen,
  };
})();
