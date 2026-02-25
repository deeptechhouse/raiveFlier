/**
 * qa.js — raiveFlier interactive Q&A module.
 *
 * ROLE IN THE APPLICATION
 * =======================
 * A slide-in drawer panel on the right side of the viewport that lets users
 * ask follow-up questions about analysis results. Answers come from the backend
 * RAG pipeline (retrieval from the knowledge corpus + LLM generation).
 *
 * OPENING THE DRAWER
 * ==================
 * Opened by clicking "Ask about this" buttons on result cards. Each button
 * carries data-entity-type and data-entity-name attributes, which scope the
 * Q&A context to a specific entity (e.g., "ARTIST" + "Jeff Mills").
 * The results.js module attaches click handlers that call QA.openPanel().
 *
 * CHAT PATTERN
 * ============
 * - Maintains a local conversation history (_history) as an array of
 *   { role: "user"|"assistant", content, citations, suggestions } objects
 * - Each assistant response may include:
 *   - Citations: source references with tier badges
 *   - Related facts: clickable chips that auto-generate follow-up questions
 * - Clicking a fact chip calls _buildFactQuery() to create a contextual query
 *   like "Tell me how this relates to the analysis of Jeff Mills: ..."
 *
 * INITIAL FACT CHIPS
 * ==================
 * When the drawer opens for a specific entity, _getInitialFacts() generates
 * contextual starter chips (e.g., for an artist: "label history", "notable
 * releases", "connected collaborators", "place in the scene"). These give
 * the user quick starting points without needing to type.
 *
 * API INTERACTION
 * ===============
 * POST /api/v1/fliers/{session_id}/ask
 * Body: { question, entity_type?, entity_name? }
 * Response: { answer, citations: [], related_facts: [] }
 *
 * MODULE COMMUNICATION
 * ====================
 * - Called by results.js via QA.openPanel() when "Ask about this" is clicked
 * - Uses Rating module for per-answer thumbs up/down (optional)
 */

"use strict";

const QA = (() => {
  // ------------------------------------------------------------------
  // Private state
  // ------------------------------------------------------------------

  let _sessionId = null;           // Pipeline session UUID for API calls
  let _isOpen = false;             // Whether the drawer is currently visible
  let _isLoading = false;          // Whether an answer request is in flight
  // Conversation history — re-rendered in full on each update via _renderAllMessages().
  // This "re-render everything" approach is simple and avoids DOM diffing complexity.
  let _history = []; // Array of {role: "user"|"assistant", content: string, citations: [], suggestions: []}
  let _currentEntityType = null;   // Scoped entity type (e.g., "ARTIST") or null for general
  let _currentEntityName = null;   // Scoped entity name (e.g., "Jeff Mills") or null

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

  function _getDrawer() {
    return document.getElementById("qa-drawer");
  }

  function _getMessages() {
    return document.getElementById("qa-messages");
  }

  function _getInput() {
    return document.getElementById("qa-input");
  }

  // ------------------------------------------------------------------
  // Rendering
  // ------------------------------------------------------------------

  /** Build the drawer's inner HTML shell. */
  function _renderShell() {
    const drawer = _getDrawer();
    if (!drawer) return;

    drawer.innerHTML = `
      <div class="qa-drawer__header">
        <div class="qa-drawer__title-row">
          <svg class="qa-drawer__icon" width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M12 22C6.477 22 2 17.523 2 12S6.477 2 12 2s10 4.477 10 10-4.477 10-10 10zm-1-7v2h2v-2h-2zm2-1.645A3.502 3.502 0 0012 6.5 3.501 3.501 0 008.5 10h2c0-.827.673-1.5 1.5-1.5s1.5.673 1.5 1.5c0 1.5-2 1.313-2 3h2c0-1.125 2-1.25 2-3 0-1.93-1.57-3.5-3.5-3.5z" fill="currentColor"/>
          </svg>
          <h3 class="qa-drawer__title">Ask a Question</h3>
        </div>
        <button type="button" class="qa-drawer__close" id="qa-close-btn" aria-label="Close Q&A panel">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path d="M4 4L12 12M12 4L4 12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
          </svg>
        </button>
      </div>
      <div class="qa-drawer__context" id="qa-context"></div>
      <div class="qa-drawer__messages" id="qa-messages"></div>
      <div class="qa-drawer__input-bar">
        <input type="text" id="qa-input" class="qa-drawer__input" placeholder="Ask about this..." maxlength="1000" autocomplete="off">
        <button type="button" id="qa-send-btn" class="qa-drawer__send" aria-label="Send question">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path d="M14 2L7 9M14 2L10 14L7 9M14 2L2 6L7 9" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </button>
      </div>
    `;

    // Event listeners
    document.getElementById("qa-close-btn").addEventListener("click", closePanel);
    document.getElementById("qa-send-btn").addEventListener("click", _handleSend);

    const input = _getInput();
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        _handleSend();
      }
    });
  }

  /** Render a user message bubble. */
  function _renderUserMessage(text) {
    return `<div class="qa-message qa-message--user"><p>${_esc(text)}</p></div>`;
  }

  /** Strip [n] citation numbers from text for clean display.
   *  The structured citation data is shown separately as tier badges. */
  function _stripCiteRefs(text) {
    if (!text) return text;
    return text.replace(/\s*\[\d+\]/g, "").replace(/ {2,}/g, " ");
  }

  /** Render an assistant answer with citations and related fact chips. */
  function _renderAssistantMessage(answer, citations, facts, question) {
    let html = '<div class="qa-message qa-message--assistant">';

    // Answer text — strip inline [n] refs (citations shown as tier badges below)
    const cleaned = _stripCiteRefs(answer);
    const paragraphs = cleaned.split("\n\n").filter(Boolean);
    paragraphs.forEach((p) => {
      html += `<p>${_esc(p)}</p>`;
    });

    // Citations
    if (citations && citations.length > 0) {
      html += '<div class="qa-citations">';
      html += '<span class="qa-citations__label">Sources:</span>';
      citations.forEach((c) => {
        const tierCls = c.tier <= 2 ? "qa-citation--high" : c.tier <= 4 ? "qa-citation--mid" : "qa-citation--low";
        html += `<span class="qa-citation ${tierCls}" title="${_esc(c.text || c.source)}">${_esc(c.source || c.text)} <span class="qa-citation__tier">T${c.tier || "?"}</span></span>`;
      });
      html += "</div>";
    }

    // Rating widget
    if (typeof Rating !== "undefined" && question) {
      html += Rating.renderWidget("QA", Rating.simpleHash(question));
    }

    html += "</div>";

    // Related facts
    if (facts && facts.length > 0) {
      html += '<div class="qa-facts">';
      html += '<span class="qa-facts__label">Related facts:</span>';
      facts.forEach((f) => {
        const text = typeof f === "string" ? f : f.text;
        const category = typeof f === "object" ? (f.category || "") : "";
        const eName = typeof f === "object" ? (f.entity_name || "") : "";
        const categoryLabel = category ? `<span class="qa-fact__category">${_esc(category)}</span>` : "";
        html += `<button type="button" class="qa-fact" data-fact="${_esc(text)}" data-category="${_esc(category)}" data-entity-name="${_esc(eName)}">${categoryLabel}${_esc(text)}</button>`;
      });
      html += "</div>";
    }

    return html;
  }

  /** Render a loading indicator. */
  function _renderLoading() {
    return '<div class="qa-message qa-message--loading"><span class="qa-loading-dots"><span></span><span></span><span></span></span></div>';
  }

  /** Build a relational query from a clicked fact chip.
   *  The generated question asks the LLM to explain how the fact relates
   *  to the current analysis context, making the response more relevant
   *  than just answering the fact in isolation. */
  function _buildFactQuery(factText) {
    const contextLabel = _currentEntityName
      ? `the analysis of ${_currentEntityName}`
      : "this flier's analysis";
    return `Tell me how this relates to ${contextLabel}: ${factText}`;
  }

  /** Re-render all messages from history. */
  function _renderAllMessages() {
    const container = _getMessages();
    if (!container) return;

    let html = "";
    let lastQuestion = "";
    _history.forEach((msg) => {
      if (msg.role === "user") {
        lastQuestion = msg.content;
        html += _renderUserMessage(msg.content);
      } else {
        html += _renderAssistantMessage(msg.content, msg.citations, msg.suggestions, lastQuestion);
      }
    });

    if (_isLoading) {
      html += _renderLoading();
    }

    container.innerHTML = html;

    // Attach click handlers to fact chip buttons
    container.querySelectorAll(".qa-fact").forEach((btn) => {
      btn.addEventListener("click", () => {
        const factText = btn.dataset.fact;
        const eName = btn.dataset.entityName || _currentEntityName;
        _currentEntityName = eName || _currentEntityName;
        _submitQuestion(_buildFactQuery(factText));
      });
    });

    // Rating widgets: attach event delegation
    if (typeof Rating !== "undefined" && _sessionId) {
      Rating.initWidgets(container, _sessionId);
    }

    // Scroll to bottom
    container.scrollTop = container.scrollHeight;
  }

  /** Update the context badge at the top. */
  function _updateContext() {
    const ctx = document.getElementById("qa-context");
    if (!ctx) return;

    if (_currentEntityType && _currentEntityName) {
      const typeLabel = _currentEntityType.charAt(0) + _currentEntityType.slice(1).toLowerCase();
      ctx.innerHTML = `<span class="qa-context__badge">${_esc(typeLabel)}: ${_esc(_currentEntityName)}</span>`;
      ctx.hidden = false;
    } else {
      ctx.hidden = true;
    }
  }

  // ------------------------------------------------------------------
  // API interaction
  // ------------------------------------------------------------------

  async function _submitQuestion(question) {
    if (!question || !question.trim() || _isLoading || !_sessionId) return;

    const q = question.trim();

    // Add user message to history
    _history.push({ role: "user", content: q });
    _isLoading = true;
    _renderAllMessages();

    try {
      const response = await fetch(
        `/api/v1/fliers/${encodeURIComponent(_sessionId)}/ask`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: q,
            entity_type: _currentEntityType,
            entity_name: _currentEntityName,
          }),
        }
      );

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();

      _history.push({
        role: "assistant",
        content: data.answer,
        citations: data.citations || [],
        suggestions: data.related_facts || [],
      });
    } catch (err) {
      _history.push({
        role: "assistant",
        content: `Sorry, I couldn't process that question: ${err.message}`,
        citations: [],
        suggestions: [],
      });
    }

    _isLoading = false;
    _renderAllMessages();

    // Clear input
    const input = _getInput();
    if (input) {
      input.value = "";
      input.focus();
    }
  }

  function _handleSend() {
    const input = _getInput();
    if (!input) return;
    _submitQuestion(input.value);
  }

  // ------------------------------------------------------------------
  // Public API
  // ------------------------------------------------------------------

  /**
   * Open the Q&A panel, optionally focused on a specific entity.
   * @param {string} sessionId - The pipeline session UUID.
   * @param {string|null} entityType - "ARTIST", "VENUE", "PROMOTER", "DATE", or null.
   * @param {string|null} entityName - The entity name, or null for general questions.
   */
  function openPanel(sessionId, entityType, entityName) {
    _sessionId = sessionId;
    _currentEntityType = entityType || null;
    _currentEntityName = entityName || null;
    _history = [];
    _isLoading = false;

    const drawer = _getDrawer();
    if (!drawer) return;

    _renderShell();
    _updateContext();
    _renderAllMessages();

    drawer.classList.add("qa-drawer--open");
    _isOpen = true;

    // Focus the input
    const input = _getInput();
    if (input) {
      setTimeout(() => input.focus(), 300); // after transition
    }

    // Add initial fact chips if entity context is provided
    if (entityName) {
      const initialFacts = _getInitialFacts(entityType, entityName);
      if (initialFacts.length > 0) {
        const container = _getMessages();
        if (container) {
          let html = '<div class="qa-facts qa-facts--initial">';
          initialFacts.forEach((f) => {
            const categoryLabel = f.category ? `<span class="qa-fact__category">${_esc(f.category)}</span>` : "";
            html += `<button type="button" class="qa-fact" data-fact="${_esc(f.text)}" data-category="${_esc(f.category || "")}" data-entity-name="${_esc(f.entity_name || "")}">${categoryLabel}${_esc(f.text)}</button>`;
          });
          html += "</div>";
          container.innerHTML = html;

          container.querySelectorAll(".qa-fact").forEach((btn) => {
            btn.addEventListener("click", () => {
              const factText = btn.dataset.fact;
              _submitQuestion(`Tell me about: ${factText}`);
            });
          });
        }
      }
    }
  }

  /** Close the Q&A panel. */
  function closePanel() {
    const drawer = _getDrawer();
    if (drawer) {
      drawer.classList.remove("qa-drawer--open");
    }
    _isOpen = false;
  }

  /** Generate contextual starter fact chips for the drawer opening.
   *  Returns an array of fact objects with text, category, and entity_name.
   *  These give the user pre-built questions to click on, tailored to the
   *  entity type. For example, an ARTIST gets chips about label history,
   *  discography, collaborators, and scene context. */
  function _getInitialFacts(entityType, entityName) {
    if (!entityType || !entityName) return [];

    const name = entityName;
    switch (entityType.toUpperCase()) {
      case "ARTIST":
        return [
          { text: `${name}'s record label history`, category: "LABEL", entity_name: name },
          { text: `${name}'s notable releases and discography`, category: "ARTIST", entity_name: name },
          { text: `Artists and collaborators connected to ${name}`, category: "CONNECTION", entity_name: name },
          { text: `${name}'s place in the scene`, category: "SCENE", entity_name: name },
        ];
      case "VENUE":
        return [
          { text: `History and significance of ${name}`, category: "VENUE", entity_name: name },
          { text: `Notable events held at ${name}`, category: "VENUE", entity_name: name },
          { text: `Artists and scenes associated with ${name}`, category: "CONNECTION", entity_name: name },
        ];
      case "PROMOTER":
        return [
          { text: `Events organized by ${name}`, category: "HISTORY", entity_name: name },
          { text: `Venues and scenes ${name} is associated with`, category: "CONNECTION", entity_name: name },
          { text: `${name}'s role in the local scene`, category: "SCENE", entity_name: name },
        ];
      case "DATE":
        return [
          { text: `Rave scene around ${name}`, category: "HISTORY", entity_name: name },
          { text: `Cultural and political context of ${name}`, category: "HISTORY", entity_name: name },
          { text: `Notable releases and events from this period`, category: "SCENE", entity_name: name },
        ];
      default:
        return [];
    }
  }

  /** Check if the panel is currently open. */
  function isOpen() {
    return _isOpen;
  }

  return {
    openPanel,
    closePanel,
    isOpen,
  };
})();
