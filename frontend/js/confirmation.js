/**
 * confirmation.js — Entity review and editing UI for the Phase 3 gate.
 *
 * Renders editable entity cards from the FlierUploadResponse,
 * allows the user to edit, add, or remove entities, then submits
 * the confirmed entities to kick off the research pipeline.
 */

"use strict";

const Confirmation = (() => {
  /** @type {object|null} Stored upload response for reference */
  let _uploadData = null;

  /** @type {string|null} Object URL for flier preview (revoked on cleanup) */
  let _flierObjectURL = null;

  /**
   * Escape a string for safe HTML insertion.
   * @param {string} str
   * @returns {string}
   */
  function _escapeHTML(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  /**
   * Return the CSS class for a confidence score.
   * @param {number} confidence — 0.0 to 1.0
   * @returns {string}
   */
  function _confidenceClass(confidence) {
    if (confidence >= 0.8) return "confidence-high";
    if (confidence >= 0.5) return "confidence-medium";
    return "confidence-low";
  }

  /**
   * Return the label text for a confidence score.
   * @param {number} confidence — 0.0 to 1.0
   * @returns {string}
   */
  function _confidenceLabel(confidence) {
    if (confidence >= 0.8) return "High";
    if (confidence >= 0.5) return "Medium";
    return "Low";
  }

  /**
   * Build HTML for a single entity card.
   * @param {string} type — display label (e.g. "Artist", "Venue")
   * @param {string} name — entity text value
   * @param {number} confidence — 0.0 to 1.0
   * @param {string} entityType — backend enum value (e.g. "ARTIST")
   * @param {boolean} removable — whether to show a remove button
   * @returns {string}
   */
  function _entityCardHTML(type, name, confidence, entityType, removable) {
    const badgeClass = _confidenceClass(confidence);
    const badgeLabel = _confidenceLabel(confidence);
    const pct = (confidence * 100).toFixed(0);
    const removeBtn = removable
      ? `<button type="button" class="entity-card__remove" aria-label="Remove ${_escapeHTML(type)}: ${_escapeHTML(name)}" onclick="Confirmation.removeEntity(this)">&times;</button>`
      : "";

    return `<div class="entity-card" data-entity-type="${_escapeHTML(entityType)}">
      <div class="entity-card__header">
        <span class="entity-card__type">${_escapeHTML(type)}</span>
        <div class="entity-card__actions">
          <span class="confidence-badge ${badgeClass}" title="Confidence: ${pct}%">${pct}% ${badgeLabel}</span>
          ${removeBtn}
        </div>
      </div>
      <input type="text" class="entity-card__input" value="${_escapeHTML(name)}" aria-label="${_escapeHTML(type)} name" data-entity-type="${_escapeHTML(entityType)}">
    </div>`;
  }

  /**
   * Build HTML for a genre chip.
   * @param {string} genre
   * @returns {string}
   */
  function _genreChipHTML(genre) {
    return `<span class="genre-chip" data-genre="${_escapeHTML(genre)}">
      ${_escapeHTML(genre)}
      <button type="button" class="genre-chip__remove" aria-label="Remove genre: ${_escapeHTML(genre)}" onclick="Confirmation.removeGenre(this)">&times;</button>
    </span>`;
  }

  /**
   * Populate the confirmation view with extracted entities.
   * Called by Upload after a successful upload response.
   * @param {object} uploadResponse — FlierUploadResponse JSON
   */
  function populateConfirmView(uploadResponse) {
    _uploadData = uploadResponse;

    const confirmView = document.getElementById("confirm-view");
    if (!confirmView) return;

    const entities = uploadResponse.extracted_entities || {};
    const artists = entities.artists || [];
    const venue = entities.venue;
    const date = entities.date;
    const promoter = entities.promoter;
    const eventName = entities.event_name;
    const genreTags = entities.genre_tags || [];
    const ticketPrice = entities.ticket_price || "";
    const ocrConfidence = uploadResponse.ocr_confidence || 0;
    const providerUsed = uploadResponse.provider_used || "unknown";
    const pct = (ocrConfidence * 100).toFixed(1);

    // Get the flier preview image from the upload view
    const previewImg = document.getElementById("preview-image");
    const flierSrc = previewImg ? previewImg.src : "";

    let html = "";

    // OCR confidence summary
    html += `<div class="confirm-header">
      <div class="confirm-header__info">
        <h2 class="text-heading">Review Extracted Entities</h2>
        <div class="ocr-summary">
          <span class="ocr-summary__badge confidence-badge ${_confidenceClass(ocrConfidence)}">OCR Confidence: ${pct}%</span>
          <span class="ocr-summary__provider text-caption">via ${_escapeHTML(providerUsed)}</span>
        </div>
      </div>
    </div>`;

    // Main layout: entities + flier reference
    html += `<div class="confirm-layout">`;

    // Left column: entity cards
    html += `<div class="confirm-entities">`;

    // Artists section
    html += `<div class="confirm-section">
      <div class="confirm-section__header">
        <h3 class="confirm-section__title text-caption">Artists</h3>
        <button type="button" class="btn-add" id="add-artist-btn" aria-label="Add artist">
          <span aria-hidden="true">+</span> Add Artist
        </button>
      </div>
      <div id="artist-cards">`;

    artists.forEach((a) => {
      html += _entityCardHTML("Artist", a.text, a.confidence || 0, "ARTIST", true);
    });

    html += `</div></div>`;

    // Venue
    if (venue) {
      html += `<div class="confirm-section">
        <h3 class="confirm-section__title text-caption">Venue</h3>`;
      html += _entityCardHTML("Venue", venue.text, venue.confidence || 0, "VENUE", false);
      html += `</div>`;
    }

    // Date
    if (date) {
      html += `<div class="confirm-section">
        <h3 class="confirm-section__title text-caption">Date</h3>`;
      html += _entityCardHTML("Date", date.text, date.confidence || 0, "DATE", false);
      html += `</div>`;
    }

    // Promoter
    if (promoter) {
      html += `<div class="confirm-section">
        <h3 class="confirm-section__title text-caption">Promoter</h3>`;
      html += _entityCardHTML("Promoter", promoter.text, promoter.confidence || 0, "PROMOTER", false);
      html += `</div>`;
    }

    // Event Name
    if (eventName) {
      html += `<div class="confirm-section">
        <h3 class="confirm-section__title text-caption">Event / Series Name</h3>`;
      html += _entityCardHTML("Event Name", eventName.text, eventName.confidence || 0, "EVENT", false);
      html += `</div>`;
    }

    // Genre tags
    html += `<div class="confirm-section">
      <h3 class="confirm-section__title text-caption">Genre Tags</h3>
      <div id="genre-chips" class="genre-chips">`;

    genreTags.forEach((g) => {
      html += _genreChipHTML(g);
    });

    html += `</div>
      <div class="genre-add">
        <input type="text" id="genre-input" class="genre-add__input" placeholder="Add genre..." aria-label="Add genre tag">
        <button type="button" class="btn-add btn-add--small" id="add-genre-btn" aria-label="Add genre">+</button>
      </div>
    </div>`;

    // Ticket price
    html += `<div class="confirm-section">
      <h3 class="confirm-section__title text-caption">Ticket Price</h3>
      <input type="text" id="ticket-price-input" class="entity-card__input" value="${_escapeHTML(ticketPrice)}" placeholder="e.g. $20, Free, TBA" aria-label="Ticket price">
    </div>`;

    html += `</div>`; // end .confirm-entities

    // Right column: flier reference image
    html += `<div class="confirm-flier-ref">
      <h3 class="confirm-section__title text-caption">Original Flier</h3>
      <div class="confirm-flier-ref__frame">
        <img id="confirm-flier-image" src="${flierSrc}" alt="Original rave flier" class="confirm-flier-ref__img">
      </div>
    </div>`;

    html += `</div>`; // end .confirm-layout

    // Confirm button
    html += `<div class="confirm-actions">
      <button type="button" class="btn-primary" id="confirm-btn">Confirm &amp; Start Research</button>
    </div>`;

    // Error area
    html += `<div class="confirm-error" id="confirm-error" role="alert" hidden></div>`;

    confirmView.innerHTML = html;

    // Bind events
    _bindConfirmEvents();
  }

  /** Bind event listeners for confirm view interactive elements. */
  function _bindConfirmEvents() {
    const addArtistBtn = document.getElementById("add-artist-btn");
    if (addArtistBtn) {
      addArtistBtn.addEventListener("click", addArtistInput);
    }

    const confirmBtn = document.getElementById("confirm-btn");
    if (confirmBtn) {
      confirmBtn.addEventListener("click", handleConfirm);
    }

    const addGenreBtn = document.getElementById("add-genre-btn");
    const genreInput = document.getElementById("genre-input");
    if (addGenreBtn && genreInput) {
      addGenreBtn.addEventListener("click", () => _addGenre());
      genreInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          _addGenre();
        }
      });
    }
  }

  /** Add a genre chip from the genre input. */
  function _addGenre() {
    const input = document.getElementById("genre-input");
    const container = document.getElementById("genre-chips");
    if (!input || !container) return;

    const genre = input.value.trim();
    if (!genre) return;

    container.insertAdjacentHTML("beforeend", _genreChipHTML(genre));
    input.value = "";
    input.focus();
  }

  /**
   * Remove a genre chip from the DOM.
   * @param {HTMLElement} button — the remove button inside the chip
   */
  function removeGenre(button) {
    const chip = button.closest(".genre-chip");
    if (chip) {
      chip.remove();
    }
  }

  /** Add a new empty artist text input card. */
  function addArtistInput() {
    const container = document.getElementById("artist-cards");
    if (!container) return;

    const html = `<div class="entity-card" data-entity-type="ARTIST">
      <div class="entity-card__header">
        <span class="entity-card__type">Artist</span>
        <div class="entity-card__actions">
          <span class="confidence-badge confidence-low" title="Manually added">New</span>
          <button type="button" class="entity-card__remove" aria-label="Remove new artist" onclick="Confirmation.removeEntity(this)">&times;</button>
        </div>
      </div>
      <input type="text" class="entity-card__input" value="" placeholder="Enter artist name..." aria-label="Artist name" data-entity-type="ARTIST" autofocus>
    </div>`;

    container.insertAdjacentHTML("beforeend", html);

    // Focus the new input
    const inputs = container.querySelectorAll(".entity-card__input");
    const last = inputs[inputs.length - 1];
    if (last) last.focus();
  }

  /**
   * Remove an entity card from the DOM.
   * @param {HTMLElement} button — the remove button inside the card
   */
  function removeEntity(button) {
    const card = button.closest(".entity-card");
    if (card) {
      card.remove();
    }
  }

  /**
   * Collect confirmed entities and POST to the confirm endpoint.
   * On success, switch to progress view and open WebSocket.
   */
  async function handleConfirm() {
    const sessionId = App.getSessionId();
    if (!sessionId) return;

    const confirmBtn = document.getElementById("confirm-btn");
    const errorEl = document.getElementById("confirm-error");
    if (confirmBtn) confirmBtn.disabled = true;
    if (errorEl) errorEl.hidden = true;

    // Collect artists
    const artistCards = document.querySelectorAll('#artist-cards .entity-card[data-entity-type="ARTIST"]');
    const artists = [];
    artistCards.forEach((card) => {
      const input = card.querySelector(".entity-card__input");
      const name = input ? input.value.trim() : "";
      if (name) {
        artists.push({ name: name, entity_type: "ARTIST" });
      }
    });

    // Collect venue
    const venueInput = document.querySelector('.entity-card[data-entity-type="VENUE"] .entity-card__input');
    const venue = venueInput && venueInput.value.trim()
      ? { name: venueInput.value.trim(), entity_type: "VENUE" }
      : null;

    // Collect date
    const dateInput = document.querySelector('.entity-card[data-entity-type="DATE"] .entity-card__input');
    const date = dateInput && dateInput.value.trim()
      ? { name: dateInput.value.trim(), entity_type: "DATE" }
      : null;

    // Collect promoter
    const promoterInput = document.querySelector('.entity-card[data-entity-type="PROMOTER"] .entity-card__input');
    const promoter = promoterInput && promoterInput.value.trim()
      ? { name: promoterInput.value.trim(), entity_type: "PROMOTER" }
      : null;

    // Collect event name
    const eventNameInput = document.querySelector('.entity-card[data-entity-type="EVENT"] .entity-card__input');
    const eventName = eventNameInput && eventNameInput.value.trim()
      ? { name: eventNameInput.value.trim(), entity_type: "EVENT" }
      : null;

    // Collect genre tags
    const genreChips = document.querySelectorAll("#genre-chips .genre-chip");
    const genreTags = [];
    genreChips.forEach((chip) => {
      const genre = chip.dataset.genre;
      if (genre) genreTags.push(genre);
    });

    // Collect ticket price
    const priceInput = document.getElementById("ticket-price-input");
    const ticketPrice = priceInput ? priceInput.value.trim() || null : null;

    // Build ConfirmEntitiesRequest
    const body = {
      artists: artists,
      venue: venue,
      date: date,
      promoter: promoter,
      event_name: eventName,
      genre_tags: genreTags,
      ticket_price: ticketPrice,
    };

    try {
      const response = await fetch(`/api/v1/fliers/${encodeURIComponent(sessionId)}/confirm`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        const detail = errorData?.detail || `Confirmation failed (HTTP ${response.status})`;
        throw new Error(detail);
      }

      // Switch to progress view and open WebSocket
      App.showView("progress");

      if (typeof Progress !== "undefined" && Progress.connectProgress) {
        Progress.connectProgress(sessionId);
      }
    } catch (err) {
      if (errorEl) {
        errorEl.textContent = err.message || "Confirmation failed. Please try again.";
        errorEl.hidden = false;
      }
      if (confirmBtn) confirmBtn.disabled = false;
    }
  }

  return {
    populateConfirmView,
    handleConfirm,
    addArtistInput,
    removeEntity,
    removeGenre,
  };
})();
