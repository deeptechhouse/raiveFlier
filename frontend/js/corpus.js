/**
 * corpus.js — raiveFlier corpus search sidebar module.
 *
 * Renders a resizable, collapsible left sidebar for semantic search
 * against the RAG knowledge base. Available at any time, independent
 * of analysis sessions.
 */

"use strict";

const Corpus = (() => {
  // ------------------------------------------------------------------
  // Private state
  // ------------------------------------------------------------------

  let _isOpen = false;
  let _isLoading = false;
  let _width = 360;
  let _results = [];
  let _lastQuery = "";
  let _ragAvailable = false;
  let _corpusStats = null;
  let _filtersOpen = false;
  let _debounceTimer = null;

  // Resize state
  let _isResizing = false;
  const _MIN_WIDTH = 280;
  const _MAX_WIDTH = 600;

  // localStorage keys
  const _LS_WIDTH = "corpus_sidebar_width";
  const _LS_COLLAPSED = "corpus_sidebar_collapsed";

  // ------------------------------------------------------------------
  // Utility
  // ------------------------------------------------------------------

  function _esc(str) {
    if (str == null) return "";
    const div = document.createElement("div");
    div.appendChild(document.createTextNode(String(str)));
    return div.innerHTML;
  }

  // ------------------------------------------------------------------
  // DOM references
  // ------------------------------------------------------------------

  function _getSidebar() {
    return document.getElementById("corpus-sidebar");
  }

  function _getToggle() {
    return document.getElementById("corpus-toggle");
  }

  function _getResults() {
    return document.getElementById("corpus-results");
  }

  function _getSearchInput() {
    return document.getElementById("corpus-search-input");
  }

  // ------------------------------------------------------------------
  // Rendering
  // ------------------------------------------------------------------

  function _renderShell() {
    const sidebar = _getSidebar();
    if (!sidebar) return;

    sidebar.innerHTML = `
      <div class="corpus-sidebar__header">
        <div class="corpus-sidebar__title-row">
          <svg class="corpus-sidebar__icon" width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <circle cx="11" cy="11" r="8" stroke="currentColor" stroke-width="2"/>
            <path d="M21 21l-4.35-4.35" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
          </svg>
          <h3 class="corpus-sidebar__title">Corpus Search</h3>
        </div>
        <button type="button" class="corpus-sidebar__close" id="corpus-close-btn" aria-label="Close corpus search">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path d="M4 4L12 12M12 4L4 12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
          </svg>
        </button>
      </div>
      <div class="corpus-sidebar__search-bar">
        <input type="text" id="corpus-search-input" class="corpus-sidebar__search-input"
               placeholder="Search the knowledge base\u2026" maxlength="500" autocomplete="off"
               aria-label="Search the rave culture corpus">
        <button type="button" id="corpus-search-btn" class="corpus-sidebar__search-btn" aria-label="Search">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <circle cx="11" cy="11" r="8" stroke="currentColor" stroke-width="2"/>
            <path d="M21 21l-4.35-4.35" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
          </svg>
        </button>
      </div>
      <div class="corpus-sidebar__filters" id="corpus-filters">
        <button type="button" class="corpus-sidebar__filters-toggle" id="corpus-filters-toggle">
          <svg width="10" height="10" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path d="M4 6l4 4 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
          Filters
        </button>
        <div class="corpus-sidebar__filters-body" id="corpus-filters-body">
          <div class="corpus-sidebar__filter-group">
            <label class="corpus-sidebar__filter-label" for="corpus-filter-source">Source type</label>
            <select id="corpus-filter-source" class="corpus-sidebar__filter-select">
              <option value="">All types</option>
              <option value="book">Book</option>
              <option value="article">Article</option>
              <option value="interview">Interview</option>
              <option value="analysis">Analysis</option>
            </select>
          </div>
          <div class="corpus-sidebar__filter-group">
            <label class="corpus-sidebar__filter-label" for="corpus-filter-entity">Entity tag</label>
            <input type="text" id="corpus-filter-entity" class="corpus-sidebar__filter-input"
                   placeholder="e.g. Carl Cox" maxlength="100">
          </div>
          <div class="corpus-sidebar__filter-group">
            <label class="corpus-sidebar__filter-label" for="corpus-filter-geo">Location</label>
            <input type="text" id="corpus-filter-geo" class="corpus-sidebar__filter-input"
                   placeholder="e.g. Detroit" maxlength="100">
          </div>
        </div>
      </div>
      <div class="corpus-sidebar__results" id="corpus-results" role="list" aria-label="Search results"></div>
      <div class="corpus-sidebar__stats" id="corpus-stats"></div>
      <div class="corpus-sidebar__drag-handle" id="corpus-drag-handle"></div>
    `;

    // Event listeners
    document.getElementById("corpus-close-btn").addEventListener("click", closePanel);
    document.getElementById("corpus-search-btn").addEventListener("click", _handleSearch);
    document.getElementById("corpus-filters-toggle").addEventListener("click", _toggleFilters);

    const input = _getSearchInput();
    if (input) {
      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          _handleSearch();
        }
      });
      input.addEventListener("input", _handleInputDebounce);
    }

    // Filter changes trigger search
    const filterSource = document.getElementById("corpus-filter-source");
    if (filterSource) filterSource.addEventListener("change", _handleSearch);

    const filterEntity = document.getElementById("corpus-filter-entity");
    if (filterEntity) {
      filterEntity.addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); _handleSearch(); }
      });
    }

    const filterGeo = document.getElementById("corpus-filter-geo");
    if (filterGeo) {
      filterGeo.addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); _handleSearch(); }
      });
    }

    // Drag handle for resizing
    _initDragHandle();

    // Apply saved width
    sidebar.style.width = _width + "px";

    // Update stats footer
    _renderStats();
  }

  function _renderResults() {
    const container = _getResults();
    if (!container) return;

    if (_isLoading) {
      container.innerHTML = `
        <div class="corpus-sidebar__loading">
          <span class="qa-loading-dots"><span></span><span></span><span></span></span>
        </div>
      `;
      return;
    }

    if (_results.length === 0 && _lastQuery) {
      container.innerHTML = `
        <div class="corpus-sidebar__empty">
          <p class="corpus-sidebar__empty-title">No results found</p>
          <p class="corpus-sidebar__empty-text">Try a different search term or adjust filters.</p>
        </div>
      `;
      return;
    }

    if (_results.length === 0) {
      container.innerHTML = `
        <div class="corpus-sidebar__empty">
          <p class="corpus-sidebar__empty-title">Search the corpus</p>
          <p class="corpus-sidebar__empty-text">Enter a topic to search the rave culture knowledge base.</p>
        </div>
      `;
      return;
    }

    let html = "";
    _results.forEach((r, idx) => {
      const tierClass = "corpus-result__tier--" + r.citation_tier;
      const scorePercent = Math.round(r.similarity_score * 100);
      const authorLine = r.author
        ? `<div class="corpus-result__author">${_esc(r.author)}${r.page_number ? " \u00B7 " + _esc(r.page_number) : ""}</div>`
        : "";

      // Tags
      let tagsHtml = "";
      const allTags = [];
      if (r.entity_tags) r.entity_tags.forEach((t) => allTags.push({ text: t, cls: "corpus-result__tag--entity" }));
      if (r.geographic_tags) r.geographic_tags.forEach((t) => allTags.push({ text: t, cls: "corpus-result__tag--geo" }));
      if (r.genre_tags) r.genre_tags.forEach((t) => allTags.push({ text: t, cls: "corpus-result__tag--genre" }));

      if (allTags.length > 0) {
        tagsHtml = '<div class="corpus-result__tags">';
        allTags.slice(0, 8).forEach((tag) => {
          tagsHtml += `<span class="corpus-result__tag ${tag.cls}">${_esc(tag.text)}</span>`;
        });
        if (allTags.length > 8) {
          tagsHtml += `<span class="corpus-result__tag">+${allTags.length - 8}</span>`;
        }
        tagsHtml += "</div>";
      }

      let ratingHtml = "";
      if (typeof Rating !== "undefined") {
        const corpusKey = (r.source_title || "") + "::" + Rating.simpleHash(r.text || "");
        ratingHtml = Rating.renderWidget("CORPUS", corpusKey);
      }

      html += `
        <div class="corpus-result" role="listitem" data-index="${idx}">
          <div class="corpus-result__header">
            <div class="corpus-result__source">
              <span class="corpus-result__tier ${tierClass}">T${r.citation_tier}</span>
              <span class="corpus-result__source-title">${_esc(r.source_title)}</span>
            </div>
          </div>
          ${authorLine}
          <div class="corpus-result__text" data-index="${idx}">${_esc(r.text)}</div>
          <div class="corpus-result__score">
            <div class="corpus-result__score-fill" style="width: ${scorePercent}%"></div>
          </div>
          ${tagsHtml}
          ${ratingHtml}
        </div>
      `;
    });

    container.innerHTML = html;

    // Click-to-expand on result text
    container.querySelectorAll(".corpus-result").forEach((card) => {
      card.addEventListener("click", (e) => {
        // Don't toggle expand when clicking rating buttons
        if (e.target.closest(".rating-widget")) return;
        const textEl = card.querySelector(".corpus-result__text");
        if (textEl) {
          textEl.classList.toggle("corpus-result__text--expanded");
        }
      });
    });

    // Rating widgets: attach event delegation
    if (typeof Rating !== "undefined") {
      Rating.initWidgets(container, "global");
    }
  }

  function _renderStats() {
    const el = document.getElementById("corpus-stats");
    if (!el) return;

    if (_corpusStats) {
      el.textContent = `${_corpusStats.total_chunks} chunks \u00B7 ${_corpusStats.total_sources} sources`;
    } else {
      el.textContent = "";
    }
  }

  function _renderUnavailable() {
    const container = _getResults();
    if (!container) return;
    container.innerHTML = `
      <div class="corpus-sidebar__unavailable">
        <p class="corpus-sidebar__unavailable-title">Corpus Not Available</p>
        <p class="corpus-sidebar__unavailable-text">
          No RAG knowledge base is configured. Enable RAG_ENABLED in .env
          and ingest documents to search the corpus.
        </p>
      </div>
    `;

    // Disable search input
    const input = _getSearchInput();
    if (input) input.disabled = true;
  }

  // ------------------------------------------------------------------
  // Filters
  // ------------------------------------------------------------------

  function _toggleFilters() {
    _filtersOpen = !_filtersOpen;
    const body = document.getElementById("corpus-filters-body");
    if (body) {
      body.classList.toggle("corpus-sidebar__filters-body--open", _filtersOpen);
    }
    const toggle = document.getElementById("corpus-filters-toggle");
    if (toggle) {
      const svg = toggle.querySelector("svg");
      if (svg) {
        svg.style.transform = _filtersOpen ? "rotate(180deg)" : "";
      }
    }
  }

  function _getActiveFilters() {
    const filters = {};
    const sourceType = document.getElementById("corpus-filter-source");
    if (sourceType && sourceType.value) {
      filters.source_type = [sourceType.value];
    }
    const entityTag = document.getElementById("corpus-filter-entity");
    if (entityTag && entityTag.value.trim()) {
      filters.entity_tag = entityTag.value.trim();
    }
    const geoTag = document.getElementById("corpus-filter-geo");
    if (geoTag && geoTag.value.trim()) {
      filters.geographic_tag = geoTag.value.trim();
    }
    return filters;
  }

  // ------------------------------------------------------------------
  // Search
  // ------------------------------------------------------------------

  function _handleInputDebounce() {
    if (_debounceTimer) clearTimeout(_debounceTimer);
    _debounceTimer = setTimeout(() => {
      const input = _getSearchInput();
      if (input && input.value.trim().length >= 3) {
        _handleSearch();
      }
    }, 300);
  }

  function _handleSearch() {
    const input = _getSearchInput();
    if (!input) return;
    const query = input.value.trim();
    if (!query || _isLoading) return;
    _submitSearch(query);
  }

  async function _submitSearch(query) {
    _lastQuery = query;
    _isLoading = true;
    _renderResults();

    const filters = _getActiveFilters();
    const body = { query, top_k: 15 };
    if (filters.source_type) body.source_type = filters.source_type;
    if (filters.entity_tag) body.entity_tag = filters.entity_tag;
    if (filters.geographic_tag) body.geographic_tag = filters.geographic_tag;

    try {
      const response = await fetch("/api/v1/corpus/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();
      _results = data.results || [];
    } catch (err) {
      _results = [];
      console.error("[Corpus] Search error:", err.message);
    }

    _isLoading = false;
    _renderResults();
  }

  // ------------------------------------------------------------------
  // RAG availability check
  // ------------------------------------------------------------------

  async function _checkRAGAvailability() {
    try {
      const response = await fetch("/api/v1/corpus/stats");
      if (!response.ok) {
        _ragAvailable = false;
        return;
      }
      const data = await response.json();
      _corpusStats = data;
      _ragAvailable = data.total_chunks > 0;
    } catch {
      _ragAvailable = false;
    }

    // Update toggle button appearance
    const toggle = _getToggle();
    if (toggle) {
      toggle.classList.toggle("corpus-toggle--disabled", !_ragAvailable);
    }
  }

  // ------------------------------------------------------------------
  // Resize (drag handle)
  // ------------------------------------------------------------------

  function _initDragHandle() {
    const handle = document.getElementById("corpus-drag-handle");
    if (!handle) return;

    handle.addEventListener("mousedown", (e) => {
      e.preventDefault();
      _isResizing = true;
      handle.classList.add("corpus-sidebar__drag-handle--active");
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
    });

    document.addEventListener("mousemove", (e) => {
      if (!_isResizing) return;
      const newWidth = Math.max(_MIN_WIDTH, Math.min(_MAX_WIDTH, e.clientX));
      _width = newWidth;
      const sidebar = _getSidebar();
      if (sidebar) sidebar.style.width = newWidth + "px";
    });

    document.addEventListener("mouseup", () => {
      if (!_isResizing) return;
      _isResizing = false;
      const handle = document.getElementById("corpus-drag-handle");
      if (handle) handle.classList.remove("corpus-sidebar__drag-handle--active");
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      _savePreferences();
    });
  }

  // ------------------------------------------------------------------
  // State persistence
  // ------------------------------------------------------------------

  function _loadPreferences() {
    try {
      const savedWidth = localStorage.getItem(_LS_WIDTH);
      if (savedWidth) {
        _width = Math.max(_MIN_WIDTH, Math.min(_MAX_WIDTH, parseInt(savedWidth, 10)));
      }
    } catch {
      // localStorage not available
    }
  }

  function _savePreferences() {
    try {
      localStorage.setItem(_LS_WIDTH, String(_width));
      localStorage.setItem(_LS_COLLAPSED, _isOpen ? "false" : "true");
    } catch {
      // localStorage not available
    }
  }

  // ------------------------------------------------------------------
  // Keyboard handler
  // ------------------------------------------------------------------

  function _handleKeydown(e) {
    if (e.key === "Escape" && _isOpen) {
      closePanel();
    }
  }

  // ------------------------------------------------------------------
  // Public API
  // ------------------------------------------------------------------

  function init() {
    _loadPreferences();
    _renderShell();
    _checkRAGAvailability().then(() => {
      if (!_ragAvailable) {
        _renderUnavailable();
        _renderStats();
      }
    });

    // Pre-load global corpus ratings
    if (typeof Rating !== "undefined") {
      Rating.loadRatings("global");
    }

    // Toggle button click
    const toggle = _getToggle();
    if (toggle) {
      toggle.addEventListener("click", togglePanel);
    }

    // Escape key
    document.addEventListener("keydown", _handleKeydown);
  }

  function openPanel() {
    if (!_ragAvailable) {
      // Still open to show the unavailable message
    }

    const sidebar = _getSidebar();
    if (!sidebar) return;

    sidebar.classList.add("corpus-sidebar--open");
    _isOpen = true;

    // Update toggle button
    const toggle = _getToggle();
    if (toggle) {
      toggle.classList.add("corpus-toggle--hidden");
      toggle.setAttribute("aria-expanded", "true");
    }

    // Focus search input
    const input = _getSearchInput();
    if (input && !input.disabled) {
      setTimeout(() => input.focus(), 300);
    }

    _savePreferences();
  }

  function closePanel() {
    const sidebar = _getSidebar();
    if (sidebar) {
      sidebar.classList.remove("corpus-sidebar--open");
    }
    _isOpen = false;

    // Show toggle button
    const toggle = _getToggle();
    if (toggle) {
      toggle.classList.remove("corpus-toggle--hidden");
      toggle.setAttribute("aria-expanded", "false");
    }

    _savePreferences();
  }

  function togglePanel() {
    if (_isOpen) {
      closePanel();
    } else {
      openPanel();
    }
  }

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
