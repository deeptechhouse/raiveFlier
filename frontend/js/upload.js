/**
 * upload.js — Drag-and-drop file upload for raiveFlier.
 *
 * Handles drag-over visual feedback, file selection (drop + input),
 * client-side validation (type, size), image preview, and
 * POST to /api/v1/fliers/upload.
 */

"use strict";

const Upload = (() => {
  const _ALLOWED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
  const _MAX_SIZE = 10 * 1024 * 1024; // 10 MB

  let _selectedFile = null;

  // DOM references (resolved once during init)
  let _dropZone = null;
  let _dropContent = null;
  let _previewArea = null;
  let _previewImage = null;
  let _clearBtn = null;
  let _fileInput = null;
  let _submitBtn = null;
  let _errorEl = null;
  let _loadingOverlay = null;

  /** Bind all DOM references. */
  function _cacheDom() {
    _dropZone = document.getElementById("drop-zone");
    _dropContent = _dropZone.querySelector(".drop-zone__content");
    _previewArea = document.getElementById("preview-area");
    _previewImage = document.getElementById("preview-image");
    _clearBtn = document.getElementById("clear-btn");
    _fileInput = document.getElementById("file-input");
    _submitBtn = document.getElementById("submit-btn");
    _errorEl = document.getElementById("upload-error");
    _loadingOverlay = document.getElementById("loading-overlay");
  }

  /** Set up all event listeners. */
  function _bindEvents() {
    // Drag-and-drop
    _dropZone.addEventListener("dragover", _handleDragOver);
    _dropZone.addEventListener("dragleave", _handleDragLeave);
    _dropZone.addEventListener("drop", _handleDrop);

    // Click / keyboard on the drop zone opens file picker
    _dropZone.addEventListener("click", () => _fileInput.click());
    _dropZone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        _fileInput.click();
      }
    });

    // File input change
    _fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        _handleFileSelect(e.target.files[0]);
      }
    });

    // Clear preview
    _clearBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      _resetUpload();
    });

    // Submit
    _submitBtn.addEventListener("click", () => {
      if (_selectedFile) {
        _submitFlier(_selectedFile);
      }
    });
  }

  /**
   * @param {DragEvent} e
   */
  function _handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    _dropZone.classList.add("drag-over");
  }

  /**
   * @param {DragEvent} e
   */
  function _handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    _dropZone.classList.remove("drag-over");
  }

  /**
   * @param {DragEvent} e
   */
  function _handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    _dropZone.classList.remove("drag-over");

    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) {
      _handleFileSelect(file);
    }
  }

  /**
   * Validate the chosen file and show a preview thumbnail.
   * @param {File} file
   */
  function _handleFileSelect(file) {
    _hideError();

    // Validate type
    if (!_ALLOWED_TYPES.has(file.type)) {
      _showError(
        `Unsupported file type: ${file.type || "unknown"}. Use JPEG, PNG, or WEBP.`
      );
      return;
    }

    // Validate size
    if (file.size > _MAX_SIZE) {
      const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
      _showError(`File too large (${sizeMB} MB). Maximum is 10 MB.`);
      return;
    }

    _selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (ev) => {
      _previewImage.src = ev.target.result;
      _dropContent.hidden = true;
      _previewArea.hidden = false;
    };
    reader.readAsDataURL(file);

    // Enable submit
    _submitBtn.disabled = false;
  }

  /**
   * POST the file to the upload endpoint.
   * @param {File} file
   */
  async function _submitFlier(file) {
    _hideError();
    _loadingOverlay.hidden = false;
    _submitBtn.disabled = true;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/api/v1/fliers/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        const detail =
          errorData?.detail || `Upload failed (HTTP ${response.status})`;
        throw new Error(detail);
      }

      const data = await response.json();

      // Store session ID
      App.setSessionId(data.session_id);

      // Populate confirm view (stub — G2 will implement)
      _populateConfirmView(data);

      // Switch to confirm view
      App.showView("confirm");
    } catch (err) {
      _showError(err.message || "Upload failed. Please try again.");
    } finally {
      _loadingOverlay.hidden = true;
      _submitBtn.disabled = _selectedFile === null;
    }
  }

  /**
   * Minimal confirm-view population (G2 will replace this).
   * @param {object} data — FlierUploadResponse
   */
  function _populateConfirmView(data) {
    const confirmView = document.getElementById("confirm-view");
    if (!confirmView) return;

    const entities = data.extracted_entities || {};
    const artists = entities.artists || [];
    const venue = entities.venue;
    const date = entities.date;
    const promoter = entities.promoter;

    let html = `<h2 class="text-heading">Extracted Entities</h2>`;
    html += `<p class="text-caption" style="margin:var(--space-3) 0 var(--space-5)">OCR confidence: ${(data.ocr_confidence * 100).toFixed(1)}% &middot; Provider: ${data.provider_used}</p>`;

    artists.forEach((a) => {
      html += _entityCardHTML("Artist", a.text);
    });
    if (venue) html += _entityCardHTML("Venue", venue.text);
    if (date) html += _entityCardHTML("Date", date.text);
    if (promoter) html += _entityCardHTML("Promoter", promoter.text);

    html += `<div class="upload-actions"><button class="btn-primary" disabled>Confirm &amp; Research</button></div>`;
    html += `<p class="text-caption" style="margin-top:var(--space-3)">Full confirmation UI coming in G2</p>`;

    confirmView.innerHTML = html;
  }

  /**
   * @param {string} type
   * @param {string} name
   * @returns {string}
   */
  function _entityCardHTML(type, name) {
    return `<div class="entity-card">
      <div class="entity-card__header">
        <span class="entity-card__type">${_escapeHTML(type)}</span>
      </div>
      <p class="entity-card__name">${_escapeHTML(name)}</p>
    </div>`;
  }

  /** Reset the upload area to its initial state. */
  function _resetUpload() {
    _selectedFile = null;
    _fileInput.value = "";
    _previewArea.hidden = true;
    _previewImage.src = "";
    _dropContent.hidden = false;
    _submitBtn.disabled = true;
    _hideError();
  }

  /**
   * @param {string} message
   */
  function _showError(message) {
    _errorEl.textContent = message;
    _errorEl.hidden = false;
  }

  function _hideError() {
    _errorEl.hidden = true;
    _errorEl.textContent = "";
  }

  /**
   * @param {string} str
   * @returns {string}
   */
  function _escapeHTML(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  /** Public initialiser — called by App.initApp(). */
  function initUpload() {
    _cacheDom();
    _bindEvents();
  }

  return { initUpload };
})();
