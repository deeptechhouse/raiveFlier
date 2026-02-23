/**
 * rating.js — raiveFlier rating widget module.
 *
 * ROLE IN THE APPLICATION
 * =======================
 * A utility module that provides reusable thumbs up/down rating functionality.
 * Rating widgets appear on every piece of research output: artist cards, venue
 * cards, releases, labels, relationships, patterns, Q&A answers, corpus search
 * results, and artist recommendations. This user feedback helps assess the
 * quality and accuracy of the AI-generated research.
 *
 * ARCHITECTURE PATTERN
 * ====================
 * This module uses a "render HTML string + event delegation" pattern:
 *   1. renderWidget(itemType, itemKey) returns an HTML string with data attributes
 *   2. The calling module inserts that HTML via innerHTML or insertAdjacentHTML
 *   3. initWidgets(container, sessionId) attaches a single click event listener
 *      on the container using event delegation (not per-button listeners)
 *   4. Clicks on .rating-btn bubble up to the delegated handler
 *
 * Event delegation is used because widgets are frequently created and destroyed
 * (e.g., when results are re-rendered). A single container-level listener avoids
 * the overhead of attaching/detaching per-widget handlers.
 *
 * CACHING & PERSISTENCE
 * =====================
 * - Ratings are cached locally in a Map for instant UI updates
 * - loadRatings(sessionId) fetches existing ratings from the backend and
 *   populates the cache, then refreshes all rendered widget visuals
 * - _submitRating() optimistically updates the UI, then POSTs to the backend.
 *   On failure, it reverts to the previous state (optimistic UI pattern).
 *
 * VISUAL EFFECTS
 * ==============
 * - Thumbs up: green highlight when active
 * - Thumbs down: red highlight when active, AND dims the parent list item
 *   (for releases and labels) with strikethrough text. This "dimming" behavior
 *   is achieved by _applyItemDimming() which adds a CSS class to the parent.
 *
 * API INTERACTION
 * ===============
 * - GET /api/v1/fliers/{session_id}/ratings — Load existing ratings
 * - POST /api/v1/fliers/{session_id}/rate — Submit a new rating
 *   Body: { item_type, item_key, rating: 1|-1 }
 *
 * MODULE COMMUNICATION
 * ====================
 * - Used by results.js, corpus.js, qa.js, recommendations.js
 * - No dependencies on other modules (standalone utility)
 */

"use strict";

const Rating = (() => {
  // Local cache of ratings. Key format: "itemType::itemKey", value: 1 or -1.
  // This allows instant visual updates without waiting for the backend response.
  const _cache = new Map();
  // Tracks whether loadRatings() has been called for the current session.
  let _loaded = false;

  /** Escape HTML special characters to prevent XSS. */
  function _esc(str) {
    if (str == null) return "";
    const div = document.createElement("div");
    div.appendChild(document.createTextNode(String(str)));
    return div.innerHTML;
  }

  /**
   * Generate a simple 32-bit hash of a string, returned as 8-char hex.
   * @param {string} str
   * @returns {string} 8-character hex hash.
   */
  function simpleHash(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const ch = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + ch;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(16).padStart(8, "0");
  }

  /**
   * Generate the HTML for a rating widget.
   * @param {string} itemType - ARTIST, VENUE, PROMOTER, etc.
   * @param {string} itemKey  - Natural key for the item.
   * @returns {string} HTML string for the widget.
   */
  function renderWidget(itemType, itemKey) {
    const cacheKey = itemType + "::" + itemKey;
    const current = _cache.get(cacheKey) || 0;
    const upActive = current === 1 ? " rating-btn--active" : "";
    const downActive = current === -1 ? " rating-btn--active" : "";

    return '<div class="rating-widget" data-item-type="' + _esc(itemType) +
      '" data-item-key="' + _esc(itemKey) + '">' +
      '<button type="button" class="rating-btn rating-btn--up' + upActive +
        '" data-rating="1" aria-label="Thumbs up" title="Accurate result">' +
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">' +
        '<path d="M7 22V11L2 13V22H7ZM10 22H17.5C18.33 22 19.04 21.45 ' +
        '19.22 20.64L21 12.64C21.24 11.55 20.42 10.5 19.3 10.5H14V5.5C14 ' +
        '4.12 12.88 3 11.5 3L10 11V22Z" stroke="currentColor" stroke-width="1.5" ' +
        'stroke-linecap="round" stroke-linejoin="round"/></svg></button>' +
      '<button type="button" class="rating-btn rating-btn--down' + downActive +
        '" data-rating="-1" aria-label="Thumbs down" title="Inaccurate result">' +
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">' +
        '<path d="M17 2V13L22 11V2H17ZM14 2H6.5C5.67 2 4.96 2.55 4.78 ' +
        '3.36L3 11.36C2.76 12.45 3.58 13.5 4.7 13.5H10V18.5C10 19.88 ' +
        '11.12 21 12.5 21L14 13V2Z" stroke="currentColor" stroke-width="1.5" ' +
        'stroke-linecap="round" stroke-linejoin="round"/></svg></button>' +
      '</div>';
  }

  /**
   * Load existing ratings for a session into the local cache.
   * @param {string} sessionId
   */
  async function loadRatings(sessionId) {
    if (!sessionId) return;
    try {
      const resp = await fetch(
        "/api/v1/fliers/" + encodeURIComponent(sessionId) + "/ratings"
      );
      if (!resp.ok) return;
      const data = await resp.json();
      (data.ratings || []).forEach(function (r) {
        _cache.set(r.item_type + "::" + r.item_key, r.rating);
      });
      _loaded = true;

      // Update any already-rendered widgets
      _refreshAllWidgets();
    } catch (err) {
      console.error("[Rating] Failed to load ratings:", err);
    }
  }

  /** Re-apply cached rating state to all rendered widgets. */
  function _refreshAllWidgets() {
    document.querySelectorAll(".rating-widget").forEach(function (widget) {
      var type = widget.dataset.itemType;
      var key = widget.dataset.itemKey;
      var cacheKey = type + "::" + key;
      var rating = _cache.get(cacheKey) || 0;
      _updateWidgetVisuals(widget, rating);
      _applyItemDimming(widget, rating);
    });
  }

  /**
   * Submit a rating to the backend using the optimistic UI pattern:
   *   1. Immediately update the local cache and widget visuals
   *   2. POST to the backend in the background
   *   3. If the POST fails, revert to the previous rating state
   *
   * This makes the UI feel instant even on slow connections.
   *
   * @param {string}      sessionId — Pipeline session UUID
   * @param {string}      itemType  — e.g., "ARTIST", "RELEASE", "CONNECTION"
   * @param {string}      itemKey   — Natural key identifying the rated item
   * @param {number}      rating    — +1 (thumbs up) or -1 (thumbs down)
   * @param {HTMLElement}  widgetEl  — The .rating-widget container DOM element
   */
  async function _submitRating(sessionId, itemType, itemKey, rating, widgetEl) {
    var cacheKey = itemType + "::" + itemKey;
    var current = _cache.get(cacheKey) || 0;

    // Clicking the same rating again is a no-op
    if (current === rating) return;

    _cache.set(cacheKey, rating);
    _updateWidgetVisuals(widgetEl, rating);
    _applyItemDimming(widgetEl, rating);

    try {
      var resp = await fetch(
        "/api/v1/fliers/" + encodeURIComponent(sessionId) + "/rate",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            item_type: itemType,
            item_key: itemKey,
            rating: rating,
          }),
        }
      );
      if (!resp.ok) {
        throw new Error("HTTP " + resp.status);
      }
    } catch (err) {
      console.error("[Rating] Submit failed:", err);
      // Revert to previous state on failure
      if (current === 0) {
        _cache.delete(cacheKey);
      } else {
        _cache.set(cacheKey, current);
      }
      _updateWidgetVisuals(widgetEl, current);
      _applyItemDimming(widgetEl, current);
    }
  }

  /**
   * Toggle visual dimming on the parent list item when thumbs-downed.
   * Applies to release and label items with the --rated modifier class.
   * @param {HTMLElement} widgetEl  The .rating-widget container
   * @param {number}      rating   +1, -1, or 0
   */
  function _applyItemDimming(widgetEl, rating) {
    var listItem = widgetEl.closest(".artist-card__list-item--rated");
    if (!listItem) return;

    if (rating === -1) {
      listItem.classList.add("artist-card__list-item--dimmed");
    } else {
      listItem.classList.remove("artist-card__list-item--dimmed");
    }
  }

  /**
   * Update the visual state of a rating widget's buttons.
   * @param {HTMLElement} widgetEl
   * @param {number}      rating   +1, -1, or 0 (no rating)
   */
  function _updateWidgetVisuals(widgetEl, rating) {
    var upBtn = widgetEl.querySelector(".rating-btn--up");
    var downBtn = widgetEl.querySelector(".rating-btn--down");
    if (upBtn) {
      if (rating === 1) {
        upBtn.classList.add("rating-btn--active");
      } else {
        upBtn.classList.remove("rating-btn--active");
      }
    }
    if (downBtn) {
      if (rating === -1) {
        downBtn.classList.add("rating-btn--active");
      } else {
        downBtn.classList.remove("rating-btn--active");
      }
    }
  }

  /**
   * Attach event delegation for all rating widgets inside a container.
   * Uses a SINGLE click listener on the container element that checks if
   * the click target is (or is inside) a .rating-btn. This is more efficient
   * than attaching individual listeners to every button, especially when
   * the results view can have dozens of rating widgets.
   *
   * @param {HTMLElement} container — Parent element containing rating widgets
   * @param {string}      sessionId — Pipeline session UUID for the API call
   */
  function initWidgets(container, sessionId) {
    container.addEventListener("click", function (e) {
      var btn = e.target.closest(".rating-btn");
      if (!btn) return;

      var widget = btn.closest(".rating-widget");
      if (!widget) return;

      var itemType = widget.dataset.itemType;
      var itemKey = widget.dataset.itemKey;
      var rating = parseInt(btn.dataset.rating, 10);

      _submitRating(sessionId, itemType, itemKey, rating, widget);
    });
  }

  /** @returns {boolean} Whether ratings have been loaded for the current session. */
  function isLoaded() {
    return _loaded;
  }

  /** Clear the cache (e.g., on new session). */
  function clearCache() {
    _cache.clear();
    _loaded = false;
  }

  return {
    renderWidget: renderWidget,
    loadRatings: loadRatings,
    initWidgets: initWidgets,
    isLoaded: isLoaded,
    clearCache: clearCache,
    simpleHash: simpleHash,
  };
})();
