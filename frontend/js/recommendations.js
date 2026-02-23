/**
 * recommendations.js — raiveFlier recommendation panel module.
 *
 * Renders a bottom-anchored, collapsible floating panel for displaying
 * artist recommendations based on flier analysis.
 *
 * Lazy-loads recommendations on first open via
 * GET /api/v1/fliers/{sessionId}/recommendations.
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
  let _isLoading = false;
  let _sessionId = null;
  let _recommendations = [];
  let _hasFetched = false;
  let _error = null;

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
        _hasFetched = false;
        _fetchRecommendations();
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

  /** Render the recommendation cards into the panel body. */
  function _renderResults() {
    const content = document.getElementById("reco-content");
    if (!content) return;

    if (!_recommendations.length) {
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

    // Update count badge
    const countEl = document.getElementById("reco-count");
    if (countEl) {
      countEl.textContent = `${_recommendations.length} artists`;
    }
  }

  // ------------------------------------------------------------------
  // API interaction
  // ------------------------------------------------------------------

  /**
   * Fetch recommendations from the backend. Called on first panel open.
   * Renders loading/error/results states automatically.
   */
  async function _fetchRecommendations() {
    if (_hasFetched || _isLoading || !_sessionId) return;

    _isLoading = true;
    _error = null;
    _renderLoading();

    try {
      const resp = await fetch(
        `/api/v1/fliers/${encodeURIComponent(_sessionId)}/recommendations`
      );
      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${resp.status}`);
      }
      const data = await resp.json();
      _recommendations = data.recommendations || [];
      _hasFetched = true;
      _renderResults();
    } catch (err) {
      _error = err.message || "Failed to load recommendations";
      _renderError(_error);
    } finally {
      _isLoading = false;
    }
  }

  // ------------------------------------------------------------------
  // Public API
  // ------------------------------------------------------------------

  /**
   * Initialise the recommendation panel for a given session.
   * Renders the shell (collapsed toggle bar) but does not fetch data
   * until the panel is opened.
   * @param {string} sessionId - The pipeline session UUID.
   */
  function init(sessionId) {
    _sessionId = sessionId;
    _isOpen = false;
    _isLoading = false;
    _hasFetched = false;
    _recommendations = [];
    _error = null;
    _renderShell();
  }

  /** Expand the panel and trigger a lazy fetch on first open. */
  function openPanel() {
    const panel = _getPanel();
    if (!panel) return;
    _isOpen = true;
    panel.classList.add("reco-panel--open");
    const toggle = document.getElementById("reco-toggle");
    if (toggle) toggle.setAttribute("aria-expanded", "true");

    if (!_hasFetched && !_isLoading) {
      _fetchRecommendations();
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
