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
 *   Response: { results, facets, parsed_filters, has_more, total_results }
 * - POST /api/v1/corpus/parse-query — Extract structured filters from free text
 * - GET /api/v1/corpus/suggest — Autocomplete suggestions for filter fields
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

  // Synthesized NL answer state — populated from the LLM synthesis step
  // on fresh searches (offset=0).  The answer_citations array maps [N]
  // markers in the answer text to their source metadata.
  let _synthesizedAnswer = null;   // String: LLM-generated NL answer, or null
  let _answerCitations = [];       // Array of { index, source_title, author, citation_tier, page_number, excerpt }
  let _sourcesExpanded = false;    // Whether the raw-chunks "Sources" accordion is expanded

  // Pagination state — supports "Load More" button pattern.
  // offset tracks cursor position; hasMore signals whether more pages exist.
  let _currentOffset = 0;     // Current pagination offset
  let _hasMore = false;       // Whether more results are available from backend
  let _totalResults = 0;      // Total result count from backend (for display)

  // Dynamic filter data — populated from /corpus/stats on init
  let _availableGenres = [];  // Genre tags for populating genre filter dropdown
  let _availableEras = [];    // Time periods for populating era filter dropdown

  // Smart search state — supports NL query parsing, autocomplete, and facets.
  // _manualFilters tracks which filters the user explicitly set (vs. auto-detected).
  // _parseTimer debounces the parse-query request separately from search.
  let _manualFilters = {};    // { genre: true, era: true, ... } — set on user interaction
  let _parseTimer = null;     // Timer ID for parse-query debouncing
  let _suggestTimers = {};    // Timer IDs per filter field for autocomplete debouncing
  let _activeDropdown = null; // Currently visible autocomplete dropdown element

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

  /**
   * Highlight query terms in result text by wrapping matches in <mark> tags.
   * HTML-escapes the text first for safety, then applies case-insensitive
   * word-boundary matching to avoid false positives inside unrelated words.
   * Short tokens (< 2 chars) are skipped to prevent noisy highlighting.
   */
  function _highlightText(text, query) {
    if (!query || query.length < 2) return _esc(text);
    var tokens = query.split(/\s+/).filter(function (t) { return t.length >= 2; });
    if (!tokens.length) return _esc(text);
    // Escape regex special characters in each token
    var escaped = tokens.map(function (t) {
      return t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    });
    var pattern = new RegExp("\\b(" + escaped.join("|") + ")", "gi");
    var safe = _esc(text);
    return safe.replace(pattern, "<mark>$1</mark>");
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
      // Track manual changes to suppress auto-fill for this field
      filterEntity.addEventListener("input", () => {
        if (filterEntity.value.trim()) _manualFilters.entity = true;
        else delete _manualFilters.entity;
      });
    }

    const filterGeo = document.getElementById("corpus-filter-geo");
    if (filterGeo) {
      filterGeo.addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); _handleSearch(); }
      });
      filterGeo.addEventListener("input", () => {
        if (filterGeo.value.trim()) _manualFilters.geo = true;
        else delete _manualFilters.geo;
      });
    }

    // Genre multi-select — triggers search on selection change
    const filterGenre = document.getElementById("corpus-filter-genre");
    if (filterGenre) {
      filterGenre.addEventListener("change", () => {
        _manualFilters.genre = true;
        _markAutoDetected("genre", false);
        _handleSearch();
      });
    }

    // Era dropdown — triggers search on selection
    const filterEra = document.getElementById("corpus-filter-era");
    if (filterEra) {
      filterEra.addEventListener("change", () => {
        _manualFilters.era = true;
        _markAutoDetected("era", false);
        _handleSearch();
      });
    }

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

    // Autocomplete for entity and geographic filter text inputs —
    // fetches fuzzy suggestions from /api/v1/corpus/suggest on keystroke.
    _initAutocomplete("corpus-filter-entity", "entity_tag");
    _initAutocomplete("corpus-filter-geo", "geographic_tag");

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

    let html = "";

    // --- Synthesized NL answer section ---
    // When the backend returns a cohesive LLM-generated answer, render it
    // prominently at the top.  Citation markers [N] in the text are linked
    // to source metadata in _answerCitations.
    if (_synthesizedAnswer) {
      // Convert citation markers [N] into clickable badges that show
      // source info on hover.  HTML-escape the answer first, then replace
      // [N] patterns with styled spans.
      let answerHtml = _esc(_synthesizedAnswer);
      // Replace [N] markers with styled citation badges
      answerHtml = answerHtml.replace(/\[(\d+)\]/g, function (match, num) {
        var cit = _answerCitations.find(function (c) { return c.index === parseInt(num, 10); });
        if (cit) {
          var tooltip = _esc(cit.source_title);
          if (cit.author) tooltip += " — " + _esc(cit.author);
          if (cit.page_number) tooltip += ", p. " + _esc(cit.page_number);
          return '<span class="corpus-answer__cite" title="' + tooltip + '" data-cite="' + num + '">[' + num + ']</span>';
        }
        return match;
      });
      // Convert newlines to paragraphs for readability
      answerHtml = answerHtml.split(/\n\n+/).map(function (p) {
        return p.trim() ? "<p>" + p.trim() + "</p>" : "";
      }).join("");
      // Single newlines within paragraphs become <br>
      answerHtml = answerHtml.replace(/\n/g, "<br>");

      html += '<div class="corpus-answer">';
      html += '  <div class="corpus-answer__body">' + answerHtml + '</div>';

      // Citation bibliography — lists all cited sources with tier badges
      if (_answerCitations.length > 0) {
        html += '<div class="corpus-answer__sources">';
        html += '<div class="corpus-answer__sources-label">Sources</div>';
        _answerCitations.forEach(function (cit) {
          var tierClass = "corpus-result__tier--" + cit.citation_tier;
          html += '<div class="corpus-answer__source-item" data-cite="' + cit.index + '">';
          html += '  <span class="corpus-answer__source-num">[' + cit.index + ']</span>';
          html += '  <span class="corpus-result__tier ' + tierClass + '">T' + cit.citation_tier + '</span>';
          html += '  <span class="corpus-answer__source-title">' + _esc(cit.source_title) + '</span>';
          if (cit.author) {
            html += ' <span class="corpus-answer__source-author">— ' + _esc(cit.author) + '</span>';
          }
          if (cit.page_number) {
            html += ' <span class="corpus-answer__source-page">p. ' + _esc(cit.page_number) + '</span>';
          }
          html += '</div>';
        });
        html += '</div>';
      }
      html += '</div>';
    }

    // --- Raw chunks section: collapsible "View passages" accordion ---
    // Shows the individual source chunks that were used to generate the
    // answer.  Users can expand this to read original passages directly.
    var chunksLabel = _totalResults + " passage" + (_totalResults !== 1 ? "s" : "");
    var sourcesToggleText = _sourcesExpanded ? "Hide passages" : "View " + chunksLabel;
    // If there's no synthesized answer, show chunks directly (no accordion)
    var showChunksDirectly = !_synthesizedAnswer;

    if (!showChunksDirectly) {
      html += '<div class="corpus-sources-toggle">';
      html += '  <button type="button" class="corpus-sources-toggle__btn" id="corpus-sources-toggle-btn">';
      html += '    <svg class="corpus-sources-toggle__icon' + (_sourcesExpanded ? ' corpus-sources-toggle__icon--open' : '') + '" width="10" height="10" viewBox="0 0 16 16" fill="none" aria-hidden="true">';
      html += '      <path d="M4 6l4 4 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>';
      html += '    </svg>';
      html += '    ' + sourcesToggleText;
      html += '  </button>';
      html += '</div>';
    }

    // Chunk cards — either always visible (no answer) or in collapsible section
    var chunksContainerClass = showChunksDirectly
      ? "corpus-sources__list"
      : "corpus-sources__list" + (_sourcesExpanded ? "" : " corpus-sources__list--collapsed");
    html += '<div class="' + chunksContainerClass + '" id="corpus-sources-list">';

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
          <div class="corpus-result__text" data-index="${idx}">${_highlightText(r.text, _lastQuery)}</div>
          <div class="corpus-result__score">
            <div class="corpus-result__score-fill" style="width: ${scorePercent}%"></div>
          </div>
          ${tagsHtml}
          ${ratingHtml}
        </div>
      `;
    });

    html += '</div>'; // close corpus-sources__list

    // "Load More" button — rendered when the backend signals more pages exist
    if (_hasMore) {
      html += `
        <button type="button" class="corpus-sidebar__load-more" id="corpus-load-more">
          Load more results
        </button>
      `;
    }

    container.innerHTML = html;

    // --- Event listeners ---

    // Sources accordion toggle
    var sourcesToggleBtn = document.getElementById("corpus-sources-toggle-btn");
    if (sourcesToggleBtn) {
      sourcesToggleBtn.addEventListener("click", function () {
        _sourcesExpanded = !_sourcesExpanded;
        var list = document.getElementById("corpus-sources-list");
        var icon = sourcesToggleBtn.querySelector(".corpus-sources-toggle__icon");
        if (list) {
          list.classList.toggle("corpus-sources__list--collapsed", !_sourcesExpanded);
        }
        if (icon) {
          icon.classList.toggle("corpus-sources-toggle__icon--open", _sourcesExpanded);
        }
        // Update button text
        var newLabel = _sourcesExpanded ? "Hide passages" : "View " + chunksLabel;
        sourcesToggleBtn.innerHTML = '<svg class="corpus-sources-toggle__icon' + (_sourcesExpanded ? ' corpus-sources-toggle__icon--open' : '') + '" width="10" height="10" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M4 6l4 4 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg> ' + newLabel;
      });
    }

    // Citation badge click — scroll to and highlight the cited source in the
    // passages list.  Auto-expands the accordion if collapsed.
    container.querySelectorAll(".corpus-answer__cite").forEach(function (badge) {
      badge.addEventListener("click", function () {
        var citeNum = badge.getAttribute("data-cite");
        // Expand the sources list if collapsed
        if (!_sourcesExpanded && sourcesToggleBtn) {
          sourcesToggleBtn.click();
        }
        // Find the matching citation in the sources list and scroll to it
        var sourceItem = container.querySelector('.corpus-answer__source-item[data-cite="' + citeNum + '"]');
        if (sourceItem) {
          sourceItem.scrollIntoView({ behavior: "smooth", block: "nearest" });
          sourceItem.classList.add("corpus-answer__source-item--highlight");
          setTimeout(function () {
            sourceItem.classList.remove("corpus-answer__source-item--highlight");
          }, 2000);
        }
      });
    });

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
  // Smart Search — NL query parsing, autocomplete, and facet counts.
  // These features augment the existing search flow without replacing it.
  // ------------------------------------------------------------------

  /**
   * Parse the query for structured filter signals (genre, era, location,
   * artist) and auto-fill the corresponding filter controls when the user
   * hasn't manually set them.  Runs on a separate debounce timer so it
   * doesn't block the search itself.
   */
  async function _parseQuery(query) {
    if (query.length < 3) return;
    try {
      var response = await fetch("/api/v1/corpus/parse-query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query }),
      });
      if (!response.ok) return;
      var parsed = await response.json();
      _applyParsedFilters(parsed);
    } catch (e) {
      /* NL parsing is an enhancement — silent failure is fine */
    }
  }

  /**
   * Auto-fill filter controls with values detected by the parse-query
   * endpoint.  Only fills filters that the user hasn't manually changed.
   * Adds small "auto" badges to auto-filled filter labels.
   */
  function _applyParsedFilters(parsed) {
    // Genre auto-fill
    if (parsed.genres && parsed.genres.length > 0) {
      var genreSelect = document.getElementById("corpus-filter-genre");
      if (genreSelect && !_manualFilters.genre) {
        Array.from(genreSelect.options).forEach(function (opt) {
          opt.selected = parsed.genres.indexOf(opt.value) !== -1;
        });
        _markAutoDetected("genre", true);
      }
    }

    // Era / time period auto-fill
    if (parsed.time_period) {
      var eraSelect = document.getElementById("corpus-filter-era");
      if (eraSelect && !_manualFilters.era) {
        // Try exact match first, then find the containing era range
        var matched = false;
        Array.from(eraSelect.options).forEach(function (opt) {
          if (opt.value === parsed.time_period) {
            eraSelect.value = opt.value;
            matched = true;
          }
        });
        if (matched) _markAutoDetected("era", true);
      }
    }

    // Location auto-fill
    if (parsed.geographic_tags && parsed.geographic_tags.length > 0) {
      var geoInput = document.getElementById("corpus-filter-geo");
      if (geoInput && !_manualFilters.geo) {
        geoInput.value = parsed.geographic_tags[0];
        _markAutoDetected("geo", true);
      }
    }

    // Artist / entity auto-fill
    if (parsed.artist_canonical) {
      var entityInput = document.getElementById("corpus-filter-entity");
      if (entityInput && !_manualFilters.entity) {
        entityInput.value = parsed.artist_canonical;
        _markAutoDetected("entity", true);
      }
    }
  }

  /**
   * Toggle a small "auto" badge on a filter label to indicate the value
   * was detected from the query text rather than set by the user.
   */
  function _markAutoDetected(field, detected) {
    var labelMap = {
      genre: "corpus-filter-genre",
      era: "corpus-filter-era",
      geo: "corpus-filter-geo",
      entity: "corpus-filter-entity",
    };
    var inputId = labelMap[field];
    if (!inputId) return;
    var label = document.querySelector('label[for="' + inputId + '"]');
    if (!label) return;
    var badge = label.querySelector(".corpus-filter__auto-badge");
    if (detected && !badge) {
      badge = document.createElement("span");
      badge.className = "corpus-filter__auto-badge";
      badge.textContent = "auto";
      label.appendChild(badge);
    } else if (!detected && badge) {
      badge.remove();
    }
  }

  /**
   * Clear all auto-detected badges and reset manual filter tracking.
   * Called when the search input is cleared.
   */
  function _clearAutoDetected() {
    _manualFilters = {};
    ["genre", "era", "geo", "entity"].forEach(function (f) {
      _markAutoDetected(f, false);
    });
  }

  /**
   * Update filter dropdown option labels with facet counts from the
   * search response.  Shows counts like "Techno (42)" so the user knows
   * which filter values are productive before clicking them.
   * Only runs on fresh searches (not Load More).
   */
  function _updateFacetCounts(facets) {
    if (!facets) return;

    // Source type dropdown — append counts to option labels
    var sourceSelect = document.getElementById("corpus-filter-source");
    if (sourceSelect && facets.source_types) {
      Array.from(sourceSelect.options).forEach(function (opt) {
        if (opt.value === "") return; // "All types" option
        var count = facets.source_types[opt.value] || 0;
        var baseText = opt.value.charAt(0).toUpperCase() + opt.value.slice(1);
        opt.textContent = count > 0 ? baseText + " (" + count + ")" : baseText;
      });
    }

    // Genre multi-select — append counts
    var genreSelect = document.getElementById("corpus-filter-genre");
    if (genreSelect && facets.genre_tags) {
      Array.from(genreSelect.options).forEach(function (opt) {
        if (opt.value === "") return;
        var count = facets.genre_tags[opt.value] || 0;
        // Strip any existing count suffix before re-appending
        var base = opt.value;
        opt.textContent = count > 0 ? base + " (" + count + ")" : base;
      });
    }

    // Era dropdown — append counts (match by option value against time_periods)
    var eraSelect = document.getElementById("corpus-filter-era");
    if (eraSelect && facets.time_periods) {
      Array.from(eraSelect.options).forEach(function (opt) {
        if (opt.value === "") return;
        var count = facets.time_periods[opt.value] || 0;
        // Strip old count suffix using the stored data-base attribute or regex
        var base = opt.textContent.replace(/\s*\(\d+\)$/, "");
        opt.textContent = count > 0 ? base + " (" + count + ")" : base;
      });
    }
  }

  // ------------------------------------------------------------------
  // Autocomplete — fuzzy suggestions for entity and location filter
  // inputs.  Fetches from GET /api/v1/corpus/suggest with 150ms debounce.
  // ------------------------------------------------------------------

  /**
   * Attach autocomplete behavior to a text input filter field.
   * On each keystroke (debounced 150ms), fetches suggestions from the
   * backend and renders a dropdown below the input.
   */
  function _initAutocomplete(inputId, field) {
    var input = document.getElementById(inputId);
    if (!input) return;

    // Make the parent position:relative for absolute dropdown positioning
    input.parentElement.style.position = "relative";

    input.addEventListener("input", function () {
      if (_suggestTimers[field]) clearTimeout(_suggestTimers[field]);
      _suggestTimers[field] = setTimeout(function () {
        _fetchSuggestions(input, field);
      }, 150);
    });

    // Close dropdown on blur (slight delay to allow click events to fire)
    input.addEventListener("blur", function () {
      setTimeout(function () { _closeDropdown(inputId); }, 200);
    });

    // Keyboard navigation in the dropdown
    input.addEventListener("keydown", function (e) {
      _handleDropdownKeyboard(e, inputId, field);
    });
  }

  /**
   * Fetch autocomplete suggestions from the backend suggest endpoint.
   * Renders a dropdown if results are returned; closes it if empty.
   */
  async function _fetchSuggestions(input, field) {
    var prefix = input.value.trim();
    if (prefix.length < 1) { _closeDropdown(input.id); return; }
    try {
      var url = "/api/v1/corpus/suggest?field=" + encodeURIComponent(field) +
                "&prefix=" + encodeURIComponent(prefix);
      var resp = await fetch(url);
      if (!resp.ok) return;
      var data = await resp.json();
      if (data.suggestions && data.suggestions.length > 0) {
        _renderDropdown(input.id, data.suggestions);
      } else {
        _closeDropdown(input.id);
      }
    } catch (e) {
      /* Autocomplete is enhancement — silent failure */
    }
  }

  /**
   * Render an absolutely-positioned dropdown below the input with
   * suggestion items.  Clicking a suggestion fills the input and
   * triggers a search.
   */
  function _renderDropdown(inputId, suggestions) {
    _closeDropdown(inputId);
    var input = document.getElementById(inputId);
    if (!input) return;

    var dropdown = document.createElement("div");
    dropdown.className = "corpus-autocomplete";
    dropdown.id = inputId + "-dropdown";
    dropdown.setAttribute("role", "listbox");

    suggestions.forEach(function (s) {
      var item = document.createElement("div");
      item.className = "corpus-autocomplete__item";
      item.setAttribute("role", "option");
      item.textContent = s;
      item.addEventListener("mousedown", function (e) {
        e.preventDefault(); // Prevent blur from firing before click
        input.value = s;
        _closeDropdown(inputId);
        // Mark this filter as manually set since user chose a suggestion
        if (inputId === "corpus-filter-entity") _manualFilters.entity = true;
        if (inputId === "corpus-filter-geo") _manualFilters.geo = true;
        _handleSearch();
      });
      dropdown.appendChild(item);
    });

    input.parentElement.appendChild(dropdown);
    _activeDropdown = dropdown;
  }

  /**
   * Close/remove the autocomplete dropdown for a given input.
   */
  function _closeDropdown(inputId) {
    var dropdown = document.getElementById(inputId + "-dropdown");
    if (dropdown) dropdown.remove();
    _activeDropdown = null;
  }

  /**
   * Handle keyboard navigation within the autocomplete dropdown.
   * Arrow keys move the active highlight; Enter selects; Escape closes.
   */
  function _handleDropdownKeyboard(e, inputId, field) {
    var dropdown = document.getElementById(inputId + "-dropdown");
    if (!dropdown) return;
    var items = dropdown.querySelectorAll(".corpus-autocomplete__item");
    if (!items.length) return;

    var activeItem = dropdown.querySelector(".corpus-autocomplete__item--active");
    var activeIdx = -1;
    items.forEach(function (item, i) {
      if (item === activeItem) activeIdx = i;
    });

    if (e.key === "ArrowDown") {
      e.preventDefault();
      var nextIdx = activeIdx < items.length - 1 ? activeIdx + 1 : 0;
      if (activeItem) activeItem.classList.remove("corpus-autocomplete__item--active");
      items[nextIdx].classList.add("corpus-autocomplete__item--active");
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      var prevIdx = activeIdx > 0 ? activeIdx - 1 : items.length - 1;
      if (activeItem) activeItem.classList.remove("corpus-autocomplete__item--active");
      items[prevIdx].classList.add("corpus-autocomplete__item--active");
    } else if (e.key === "Enter" && activeItem) {
      e.preventDefault();
      var input = document.getElementById(inputId);
      if (input) input.value = activeItem.textContent;
      _closeDropdown(inputId);
      if (inputId === "corpus-filter-entity") _manualFilters.entity = true;
      if (inputId === "corpus-filter-geo") _manualFilters.geo = true;
      _handleSearch();
    } else if (e.key === "Escape") {
      _closeDropdown(inputId);
    }
  }

  // ------------------------------------------------------------------
  // Search — Debounced input triggers automatic search after 300ms pause.
  // The debounce prevents firing a request on every keystroke, reducing
  // unnecessary API calls. Minimum 3 characters required before auto-search.
  // ------------------------------------------------------------------

  function _handleInputDebounce() {
    if (_debounceTimer) clearTimeout(_debounceTimer);
    // Also fire parse-query on a separate timer so NL filter detection
    // runs in parallel with the search debounce.
    if (_parseTimer) clearTimeout(_parseTimer);

    var input = _getSearchInput();
    var trimmed = input ? input.value.trim() : "";

    // Clear auto-detected state when the search input is emptied
    if (trimmed.length === 0) {
      _clearAutoDetected();
    }

    _debounceTimer = setTimeout(function () {
      if (trimmed.length >= 3) {
        _handleSearch();
      }
    }, 300);

    // Parse-query fires slightly earlier than search to give the backend
    // a head start on detecting structured signals.
    _parseTimer = setTimeout(function () {
      if (trimmed.length >= 3) {
        _parseQuery(trimmed);
      }
    }, 200);
  }

  function _handleSearch() {
    const input = _getSearchInput();
    if (!input) return;
    const query = input.value.trim();
    if (!query || _isLoading) return;
    // Reset pagination and answer state on every new search or filter change —
    // only "Load More" should increment the offset.
    _currentOffset = 0;
    _results = [];
    _synthesizedAnswer = null;
    _answerCitations = [];
    _sourcesExpanded = false;
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
        // Capture synthesized NL answer on fresh searches only —
        // "Load More" pages do not re-synthesize.
        _synthesizedAnswer = data.synthesized_answer || null;
        _answerCitations = data.answer_citations || [];
      }
      _hasMore = data.has_more || false;
      _totalResults = data.total_results || 0;

      // Update facet counts on fresh searches (not Load More) so filter
      // dropdowns show per-value result counts like "Techno (42)".
      if (!isLoadMore && data.facets) {
        _updateFacetCounts(data.facets);
      }
    } catch (err) {
      if (!isLoadMore) _results = [];
      _synthesizedAnswer = null;
      _answerCitations = [];
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
