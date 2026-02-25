/**
 * corpus.js — raiveFlier corpus search sidebar module.
 *
 * ROLE IN THE APPLICATION
 * =======================
 * A standalone sidebar for searching the RAG (Retrieval-Augmented Generation)
 * knowledge base. This is INDEPENDENT of the analysis pipeline — it is always
 * available, even before uploading a flier. It lets users explore rave culture
 * knowledge (books, articles, interviews) via semantic search.
 *
 * FEATURES
 * ========
 * - Resizable sidebar (drag the right edge to adjust width)
 * - Collapsible with toggle button (fixed bottom-left of viewport)
 * - Search with 300ms debounce (auto-searches after 3+ characters)
 * - Filterable by source type, entity tag, and geographic location
 * - Results show: source title, citation tier, excerpt text, similarity score,
 *   entity/geo/genre tags, and rating widgets
 * - Click-to-expand on result text (reveals full excerpt)
 * - Width and collapsed state persisted to localStorage
 *
 * API INTERACTIONS
 * ================
 * - GET /api/v1/corpus/stats — Check if RAG is available and get corpus size
 * - POST /api/v1/corpus/search — Semantic search with query + optional filters
 *   Request: { query, top_k, source_type?, entity_tag?, geographic_tag? }
 *   Response: { results: [{ text, source_title, citation_tier, similarity_score, ... }] }
 *
 * MODULE COMMUNICATION
 * ====================
 * - Initialized by App.initApp() at startup
 * - Uses Rating module for per-result thumbs up/down (optional)
 * - Toggled via the #corpus-toggle button in the HTML
 * - Closes on Escape key
 */

"use strict";

const Corpus = (() => {
  // ------------------------------------------------------------------
  // Private state
  // ------------------------------------------------------------------

  let _isOpen = false;        // Whether the sidebar is currently visible
  let _isLoading = false;     // Whether a search request is in flight
  let _width = 360;           // Current sidebar width in pixels
  let _results = [];          // Array of search result objects from the API
  let _lastQuery = "";        // The most recent search query (for empty-state messaging)
  let _ragAvailable = false;  // Whether the backend has a corpus loaded
  let _corpusStats = null;    // { total_chunks, total_sources } from /corpus/stats
  let _filtersOpen = false;   // Whether the filter panel is expanded
  let _debounceTimer = null;  // Timer ID for search input debouncing
  let _searchError = null;    // Last search error message, if any

  // Pagination state — supports "Load More" button pattern.
  // offset tracks cursor position; hasMore signals whether more pages exist.
  let _currentOffset = 0;     // Current pagination offset
  let _hasMore = false;       // Whether more results are available from backend
  let _totalResults = 0;      // Total result count from backend (for display)

  // Dynamic filter data — populated from /corpus/stats on init
  let _availableGenres = [];  // Genre tags for populating genre filter dropdown
  let _availableEras = [];    // Time periods for populating era filter dropdown

  // Resize state — managed by mouse event listeners on the drag handle
  let _isResizing = false;
  const _MIN_WIDTH = 280;     // Minimum sidebar width (pixels)
  const _MAX_WIDTH = 600;     // Maximum sidebar width (pixels)

  // localStorage keys for persisting user preferences across sessions
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
          <h3 class="corpus-sidebar__title">Explore the History</h3>
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
          <div class="corpus-sidebar__filter-group">
            <label class="corpus-sidebar__filter-label" for="corpus-filter-genre">Genre</label>
            <select id="corpus-filter-genre" class="corpus-sidebar__filter-select" multiple size="3"
                    aria-label="Filter by genre (hold Ctrl/Cmd to select multiple)">
              <option value="">All genres</option>
            </select>
          </div>
          <div class="corpus-sidebar__filter-group">
            <label class="corpus-sidebar__filter-label" for="corpus-filter-era">Era</label>
            <select id="corpus-filter-era" class="corpus-sidebar__filter-select"
                    aria-label="Filter by time period">
              <option value="">Any era</option>
              <option value="1988-1989">Second Summer of Love (1988-89)</option>
              <option value="1988-1992">Early Rave (1988-92)</option>
              <option value="1993-1997">Golden Era (1993-97)</option>
              <option value="1990s">1990s</option>
              <option value="2000s">2000s</option>
              <option value="2010s">2010s</option>
            </select>
          </div>
          <div class="corpus-sidebar__filter-group">
            <label class="corpus-sidebar__filter-label" for="corpus-filter-tier">
              Min quality <span id="corpus-tier-value">Any</span>
            </label>
            <input type="range" id="corpus-filter-tier" class="corpus-sidebar__filter-range"
                   min="1" max="6" value="6" step="1"
                   aria-label="Minimum citation tier quality">
          </div>
          <div class="corpus-sidebar__filter-group">
            <label class="corpus-sidebar__filter-label" for="corpus-filter-minsim">
              Min relevance <span id="corpus-minsim-value">Off</span>
            </label>
            <input type="range" id="corpus-filter-minsim" class="corpus-sidebar__filter-range"
                   min="0" max="100" value="0" step="5"
                   aria-label="Minimum similarity score threshold">
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

    // Genre multi-select — triggers search on selection change
    const filterGenre = document.getElementById("corpus-filter-genre");
    if (filterGenre) filterGenre.addEventListener("change", _handleSearch);

    // Era dropdown — triggers search on selection
    const filterEra = document.getElementById("corpus-filter-era");
    if (filterEra) filterEra.addEventListener("change", _handleSearch);

    // Citation tier range slider — updates label on input, searches on change
    const filterTier = document.getElementById("corpus-filter-tier");
    if (filterTier) {
      filterTier.addEventListener("input", () => {
        const val = parseInt(filterTier.value, 10);
        const label = document.getElementById("corpus-tier-value");
        if (label) label.textContent = val >= 6 ? "Any" : "T" + val + " or better";
      });
      filterTier.addEventListener("change", _handleSearch);
    }

    // Minimum similarity range slider — updates label on input, searches on change
    const filterMinSim = document.getElementById("corpus-filter-minsim");
    if (filterMinSim) {
      filterMinSim.addEventListener("input", () => {
        const val = parseInt(filterMinSim.value, 10);
        const label = document.getElementById("corpus-minsim-value");
        if (label) label.textContent = val === 0 ? "Off" : val + "%";
      });
      filterMinSim.addEventListener("change", _handleSearch);
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
      // Distinguish between "no matches" and "search error" so the user
      // knows whether to rephrase or retry.
      const title = _searchError ? "Search error" : "No results found";
      const text = _searchError
        ? _esc(_searchError) + " — try again or adjust filters."
        : "Try a different search term or adjust filters.";
      container.innerHTML = `
        <div class="corpus-sidebar__empty">
          <p class="corpus-sidebar__empty-title">${title}</p>
          <p class="corpus-sidebar__empty-text">${text}</p>
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

    // Result count header — shows total matches to set expectations
    let html = `<div class="corpus-sidebar__result-count">${_totalResults} result${_totalResults !== 1 ? "s" : ""}</div>`;

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

    // "Load More" button — rendered when the backend signals more pages exist
    if (_hasMore) {
      html += `
        <button type="button" class="corpus-sidebar__load-more" id="corpus-load-more">
          Load more results
        </button>
      `;
    }

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

    // "Load More" click handler — increments offset and appends next page
    const loadMoreBtn = document.getElementById("corpus-load-more");
    if (loadMoreBtn) {
      loadMoreBtn.addEventListener("click", () => {
        _currentOffset += 20;
        _submitSearch(_lastQuery, true);
      });
    }

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
    // Genre multi-select — collect all selected non-empty values
    const genreSelect = document.getElementById("corpus-filter-genre");
    if (genreSelect) {
      const selected = Array.from(genreSelect.selectedOptions)
        .map(o => o.value)
        .filter(v => v !== "");
      if (selected.length > 0) {
        filters.genre_tags = selected;
      }
    }
    // Era select
    const eraSelect = document.getElementById("corpus-filter-era");
    if (eraSelect && eraSelect.value) {
      filters.time_period = eraSelect.value;
    }
    // Citation tier minimum (1-5 means active; 6 means "any")
    const tierRange = document.getElementById("corpus-filter-tier");
    if (tierRange && parseInt(tierRange.value, 10) < 6) {
      filters.min_citation_tier = parseInt(tierRange.value, 10);
    }
    // Minimum similarity score (0 means off)
    const minSimRange = document.getElementById("corpus-filter-minsim");
    if (minSimRange && parseInt(minSimRange.value, 10) > 0) {
      filters.min_similarity = parseInt(minSimRange.value, 10) / 100;
    }
    return filters;
  }

  // ------------------------------------------------------------------
  // Search — Debounced input triggers automatic search after 300ms pause.
  // The debounce prevents firing a request on every keystroke, reducing
  // unnecessary API calls. Minimum 3 characters required before auto-search.
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
    // Reset pagination on every new search or filter change —
    // only "Load More" should increment the offset.
    _currentOffset = 0;
    _results = [];
    _submitSearch(query, false);
  }

  async function _submitSearch(query, isLoadMore = false) {
    _lastQuery = query;
    _isLoading = true;
    // Only show the loading spinner for fresh searches, not load-more
    if (!isLoadMore) _renderResults();

    const filters = _getActiveFilters();
    // Build the request body with pagination support.
    // top_k=50 gives the backend a large candidate pool; page_size=20
    // controls how many results we render per "page" via Load More.
    const body = {
      query,
      top_k: 50,
      page_size: 20,
      offset: _currentOffset,
    };
    if (filters.source_type) body.source_type = filters.source_type;
    if (filters.entity_tag) body.entity_tag = filters.entity_tag;
    if (filters.geographic_tag) body.geographic_tag = filters.geographic_tag;
    if (filters.genre_tags) body.genre_tags = filters.genre_tags;
    if (filters.time_period) body.time_period = filters.time_period;
    if (filters.min_citation_tier) body.min_citation_tier = filters.min_citation_tier;
    if (filters.min_similarity) body.min_similarity = filters.min_similarity;

    _searchError = null;

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
      // Append results on load-more; replace on fresh search
      if (isLoadMore) {
        _results = _results.concat(data.results || []);
      } else {
        _results = data.results || [];
      }
      _hasMore = data.has_more || false;
      _totalResults = data.total_results || 0;
    } catch (err) {
      if (!isLoadMore) _results = [];
      _searchError = err.message || "Search failed";
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
      // Populate filter dropdown data from the backend's stats response.
      // genre_tags and time_periods are sorted lists of distinct values
      // collected from all chunks in the corpus.
      _availableGenres = data.genre_tags || [];
      _availableEras = data.time_periods || [];
      _populateGenreFilter();
    } catch {
      _ragAvailable = false;
    }

    // Update toggle button appearance
    const toggle = _getToggle();
    if (toggle) {
      toggle.classList.toggle("corpus-toggle--disabled", !_ragAvailable);
    }
  }

  /**
   * Dynamically populate the genre multi-select dropdown with genre tags
   * returned by the /corpus/stats endpoint.  Only adds genres that exist
   * in the actual corpus — avoids showing irrelevant filter options.
   */
  function _populateGenreFilter() {
    const select = document.getElementById("corpus-filter-genre");
    if (!select || _availableGenres.length === 0) return;
    // Clear any previously added dynamic options (keep the "All genres" default)
    while (select.options.length > 1) {
      select.remove(1);
    }
    _availableGenres.forEach(genre => {
      const opt = document.createElement("option");
      opt.value = genre;
      opt.textContent = genre;
      select.appendChild(opt);
    });
  }

  // ------------------------------------------------------------------
  // Resize (drag handle) — Lets the user resize the sidebar by dragging
  // the right edge. Uses three mouse events:
  //   mousedown on handle: start resizing, change cursor
  //   mousemove on document: update width based on mouse X position
  //   mouseup on document: stop resizing, save new width to localStorage
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
