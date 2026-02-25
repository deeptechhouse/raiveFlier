/**
 * recommendations.js — raiveFlier recommendation panel module.
 *
 * ROLE IN THE APPLICATION
 * =======================
 * A bottom-anchored, collapsible floating panel that displays artist
 * recommendations derived from the flier analysis. This is the discovery
 * feature — it helps users find new artists related to the ones on the flier.
 *
 * TWO-PHASE FETCH STRATEGY
 * ========================
 * Uses two sequential fetches for progressive loading:
 *
 *   Phase 1 — Quick (~3-5s):
 *     GET /api/v1/fliers/{session_id}/recommendations?mode=quick
 *     Shared-flier DB lookup + simple LLM suggestions.
 *     Returns is_partial=true — displayed immediately as "Quick Picks".
 *
 *   Phase 2 — Deep (~10-60s):
 *     GET /api/v1/fliers/{session_id}/recommendations?mode=full
 *     Full 3-tier discovery (Discogs label-mates, RAG shared-lineup,
 *     shared-flier) + LLM fill + LLM explanation pass.
 *     Returns is_partial=false — replaces quick results with richer data.
 *
 * Both fetches start eagerly on init() — NOT when the user opens the panel.
 * Quick results are typically ready before the user clicks the toggle bar.
 * If the deep fetch fails or times out, quick results remain visible.
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

  let _isOpen = false;            // Whether the panel body is expanded
  let _sessionId = null;          // Pipeline session UUID for API calls
  let _recommendations = [];      // Array of recommendation objects from the API
  let _error = null;              // Last error message, if any
  let _isLoading = false;         // Any fetch in flight (quick or deep)
  let _hasFetched = false;        // At least quick results have arrived
  let _deepDone = false;          // Full results arrived (or failed silently)

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
        _deepDone = false;
        _error = null;
        _fetchQuick();
      });
    }
  }

  /**
   * Map a source_tier value to a human-readable label.
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

  /** Update the count badge on the collapsed toggle bar. */
  function _updateCountBadge() {
    const countEl = document.getElementById("reco-count");
    if (!countEl) return;

    const count = _recommendations.length;
    if (count > 0) {
      // Show "+" suffix when deep fetch is still pending — signals more coming
      const suffix = _deepDone ? "" : "+";
      countEl.textContent = `${count}${suffix} artists`;
    } else {
      countEl.textContent = "";
    }
  }

  /**
   * Render the recommendation cards into the panel body.
   *
   * Each recommendation card displays:
   *   - Rank number + artist name + tier badge (header)
   *   - Genre tags (optional)
   *   - Reason text explaining the connection
   *   - Connected-to line with label or event context (optional)
   *   - Connection strength bar (0-100% fill)
   *   - Rating widget (thumbs up/down via Rating module)
   *
   * When deep results are still loading, appends a small indicator
   * at the bottom so the user knows more results are on the way.
   */
  function _renderResults() {
    const content = document.getElementById("reco-content");
    if (!content) return;

    // No results — show spinner if still loading, otherwise empty message
    if (!_recommendations.length) {
      if (_isLoading) {
        _renderLoading();
        return;
      }
      content.innerHTML = `
        <div class="reco-panel__empty">
          <p>No recommendations available.</p>
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

    // Append deep-loading indicator when quick results are showing
    // but the full pipeline is still running in the background.
    if (!_deepDone) {
      html += `
        <div class="reco-panel__deep-loading" id="reco-deep-loading">
          <div class="spinner spinner--small" aria-hidden="true"></div>
          <span>Discovering deeper connections&hellip;</span>
        </div>
      `;
    }

    content.innerHTML = html;

    // Post-render: inject Rating widgets into each card's footer placeholder.
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

  /** Remove the deep-loading indicator (called when deep fetch completes). */
  function _removeDeepLoading() {
    const indicator = document.getElementById("reco-deep-loading");
    if (indicator) {
      indicator.remove();
    }
  }

  // ------------------------------------------------------------------
  // API interaction — two-phase fetch (quick then deep)
  // ------------------------------------------------------------------

  /**
   * Phase 1 — Fetch quick recommendations.
   *
   * Called eagerly from init(). Uses mode=quick for fast results
   * (SQLite + simple LLM, ~3-5 seconds). On success, renders cards
   * immediately and starts the deep fetch in the background.
   *
   * API: GET /api/v1/fliers/{session_id}/recommendations?mode=quick
   */
  async function _fetchQuick() {
    if (_hasFetched || _isLoading || !_sessionId) return;

    _isLoading = true;
    _error = null;

    // If the user already opened the panel before results arrived, show spinner
    if (_isOpen) {
      _renderLoading();
    }

    try {
      const resp = await fetch(
        `/api/v1/fliers/${encodeURIComponent(_sessionId)}/recommendations?mode=quick`
      );
      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${resp.status}`);
      }
      const data = await resp.json();
      _recommendations = data.recommendations || [];
      _hasFetched = true;

      _updateCountBadge();

      if (_isOpen) {
        _renderResults();
      }

      // Kick off deep fetch in the background — no await, runs independently
      _fetchDeep();
    } catch (err) {
      _error = err.message || "Failed to load recommendations";
      if (_isOpen) {
        _renderError(_error);
      }
    } finally {
      _isLoading = false;
    }
  }

  /**
   * Phase 2 — Fetch deep recommendations.
   *
   * Called automatically after quick fetch succeeds. Uses mode=full
   * for the complete pipeline (Discogs + RAG + full LLM, ~10-60s).
   * On success, replaces quick results with richer data. On failure,
   * quick results remain visible — the user is not interrupted.
   *
   * API: GET /api/v1/fliers/{session_id}/recommendations?mode=full
   */
  async function _fetchDeep() {
    if (_deepDone || !_sessionId) return;

    try {
      const resp = await fetch(
        `/api/v1/fliers/${encodeURIComponent(_sessionId)}/recommendations?mode=full`
      );
      if (!resp.ok) {
        // Deep fetch failed — keep quick results, hide indicator silently
        _deepDone = true;
        _removeDeepLoading();
        _updateCountBadge();
        return;
      }
      const data = await resp.json();
      const deepRecs = data.recommendations || [];

      // Only replace if deep results are non-empty (avoid clearing quick results)
      if (deepRecs.length > 0) {
        _recommendations = deepRecs;
      }

      _deepDone = true;
      _updateCountBadge();

      if (_isOpen) {
        _renderResults();
      }
    } catch {
      // Deep fetch failed silently — quick results stay visible
      _deepDone = true;
      _removeDeepLoading();
      _updateCountBadge();
    }
  }

  // ------------------------------------------------------------------
  // Public API
  // ------------------------------------------------------------------

  /**
   * Initialize the recommendation panel for a given session.
   *
   * Called by Results.fetchAndDisplayResults() after the results view renders.
   * Resets all state, renders the collapsed toggle bar, and eagerly kicks
   * off the quick recommendation fetch in the background.
   *
   * @param {string} sessionId - The pipeline session UUID.
   */
  function init(sessionId) {
    _sessionId = sessionId;
    _isOpen = false;
    _isLoading = false;
    _hasFetched = false;
    _deepDone = false;
    _recommendations = [];
    _error = null;
    _renderShell();

    // Eagerly start quick fetch — results load while user reviews main analysis
    _fetchQuick();
  }

  /**
   * Expand the panel and render available results.
   */
  function openPanel() {
    const panel = _getPanel();
    if (!panel) return;
    _isOpen = true;
    panel.classList.add("reco-panel--open");
    const toggle = document.getElementById("reco-toggle");
    if (toggle) toggle.setAttribute("aria-expanded", "true");

    if (_hasFetched) {
      _renderResults();
    } else if (_isLoading) {
      _renderLoading();
    } else if (_error) {
      _renderError(_error);
    } else {
      _fetchQuick();
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
  // ------------------------------------------------------------------
  return {
    init,          // Called by Results module to bootstrap the panel
    openPanel,     // Expand the panel body
    closePanel,    // Collapse the panel body
    togglePanel,   // Toggle between open/closed
    isOpen,        // Query current panel state
  };
})();
