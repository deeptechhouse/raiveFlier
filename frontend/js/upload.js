/**
 * upload.js — Drag-and-drop file upload for raiveFlier.
 *
 * ROLE IN THE APPLICATION
 * =======================
 * This module handles the first step of the pipeline: getting a flier image from
 * the user and sending it to the backend for OCR processing. It covers:
 *   - Drag-and-drop with visual feedback (CSS class toggling)
 *   - Click-to-browse file selection (delegating to a hidden <input type="file">)
 *   - Client-side validation (MIME type and file size checks)
 *   - Image preview via FileReader API
 *   - POST to /api/v1/fliers/upload with the image as multipart FormData
 *   - Duplicate detection handling (perceptual hash match from backend)
 *   - Navigation to the confirm view on success
 *
 * DATA FLOW
 * =========
 * 1. User drops/selects file -> _handleFileSelect() validates and previews it
 * 2. User clicks "Analyze Flier" -> _submitFlier() POSTs to /api/v1/fliers/upload
 * 3. Backend returns FlierUploadResponse JSON with:
 *    - session_id: UUID for this analysis pipeline
 *    - extracted_entities: OCR results (artists, venue, date, etc.)
 *    - duplicate_match: optional match if a similar flier was seen before
 * 4. If duplicate detected: show warning, user can proceed or cancel
 * 5. On proceed: App.setSessionId(id), delegate to Confirmation module, switch view
 *
 * MODULE COMMUNICATION
 * ====================
 * - Calls App.setSessionId() and App.showView() for navigation
 * - Calls Confirmation.populateConfirmView() to pass upload data to next step
 * - Called by App.initApp() to bootstrap
 */

"use strict";

const Upload = (() => {
  // Client-side validation constants — must match backend accepted types
  const _ALLOWED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
  const _MAX_SIZE = 10 * 1024 * 1024; // 10 MB limit

  // The currently selected File object, or null if nothing is selected
  let _selectedFile = null;

  // DOM references — resolved once during _cacheDom() at init time.
  // Caching avoids repeated getElementById calls during event handlers.
  let _dropZone = null;
  let _dropContent = null;
  let _previewArea = null;
  let _previewImage = null;
  let _clearBtn = null;
  let _fileInput = null;
  let _submitBtn = null;
  let _errorEl = null;
  let _loadingOverlay = null;
  let _duplicateWarning = null;
  let _duplicateText = null;
  let _duplicateMeta = null;
  let _duplicateAnalyzeBtn = null;
  let _duplicateCancelBtn = null;

  // When the backend detects a duplicate flier, we store the full upload
  // response here so the user can choose "Analyze Anyway" to proceed.
  let _pendingDuplicateData = null;

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
    _duplicateWarning = document.getElementById("duplicate-warning");
    _duplicateText = document.getElementById("duplicate-warning-text");
    _duplicateMeta = document.getElementById("duplicate-warning-meta");
    _duplicateAnalyzeBtn = document.getElementById("duplicate-analyze-btn");
    _duplicateCancelBtn = document.getElementById("duplicate-cancel-btn");
  }

  /** Set up all event listeners.
   *  The drop zone supports three input methods:
   *  1. Drag and drop (dragover/dragleave/drop events)
   *  2. Click to open file picker (click event -> triggers hidden input)
   *  3. Keyboard activation (Enter/Space -> triggers hidden input)
   */
  function _bindEvents() {
    // Drag-and-drop — three events handle the full D&D lifecycle
    _dropZone.addEventListener("dragover", _handleDragOver);
    _dropZone.addEventListener("dragleave", _handleDragLeave);
    _dropZone.addEventListener("drop", _handleDrop);

    // Click / keyboard on the drop zone opens file picker.
    // The drop zone has tabindex="0" in the HTML, making it focusable.
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

    // Duplicate warning: "Analyze Anyway" — proceed to confirm view
    _duplicateAnalyzeBtn.addEventListener("click", () => {
      if (_pendingDuplicateData) {
        _hideDuplicateWarning();
        _proceedToConfirm(_pendingDuplicateData);
      }
    });

    // Duplicate warning: "Cancel" — reset the upload
    _duplicateCancelBtn.addEventListener("click", () => {
      _hideDuplicateWarning();
      _resetUpload();
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
   * POST the file to the backend upload endpoint via multipart FormData.
   *
   * API: POST /api/v1/fliers/upload
   * Request: multipart/form-data with a single "file" field
   * Response: FlierUploadResponse JSON containing:
   *   - session_id: UUID for the pipeline session
   *   - extracted_entities: { artists: [], venue: {}, date: {}, ... }
   *   - ocr_confidence: float 0-1
   *   - duplicate_match: optional object if perceptual hash matches
   *
   * Flow after response:
   *   - If duplicate_match present -> show warning, pause for user decision
   *   - Otherwise -> proceed directly to confirm view
   *
   * @param {File} file
   */
  async function _submitFlier(file) {
    _hideError();
    _hideDuplicateWarning();
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

      // Check for duplicate match — show warning and pause before proceeding
      if (data.duplicate_match) {
        _pendingDuplicateData = data;
        _showDuplicateWarning(data.duplicate_match);
        return;
      }

      _proceedToConfirm(data);
    } catch (err) {
      _showError(err.message || "Upload failed. Please try again.");
    } finally {
      _loadingOverlay.hidden = true;
      _submitBtn.disabled = _selectedFile === null;
    }
  }

  /**
   * Continue to the confirm view with the upload response data.
   * This is the main success path — called after a clean upload or after the
   * user clicks "Analyze Anyway" on a duplicate warning.
   *
   * Three things happen:
   * 1. Store the session ID globally so all modules can reference it
   * 2. Delegate to Confirmation module to build the entity review UI
   * 3. Switch the visible view from upload to confirm
   *
   * @param {object} data — FlierUploadResponse JSON from the API
   */
  function _proceedToConfirm(data) {
    App.setSessionId(data.session_id);
    _populateConfirmView(data);
    App.showView("confirm");
  }

  /**
   * Display the duplicate flier warning with match details.
   * @param {object} match — DuplicateMatch from the API response
   */
  function _showDuplicateWarning(match) {
    const pct = Math.round(match.similarity * 100);
    const dateStr = match.analyzed_at
      ? new Date(match.analyzed_at).toLocaleDateString()
      : "unknown date";

    _duplicateText.textContent =
      `This flier appears ${pct}% visually similar to one analyzed on ${dateStr}. ` +
      "You can analyze it again or cancel.";

    // Build metadata tags
    _duplicateMeta.innerHTML = "";
    const tags = [];
    if (match.artists && match.artists.length > 0) {
      match.artists.forEach((a) => tags.push(a));
    }
    if (match.venue) tags.push(match.venue);
    if (match.event_name) tags.push(match.event_name);
    if (match.event_date) tags.push(match.event_date);

    tags.forEach((tag) => {
      const el = document.createElement("span");
      el.className = "duplicate-warning__tag";
      el.textContent = tag;
      _duplicateMeta.appendChild(el);
    });

    _duplicateWarning.hidden = false;
  }

  /** Hide the duplicate warning and clear pending data. */
  function _hideDuplicateWarning() {
    if (_duplicateWarning) {
      _duplicateWarning.hidden = true;
      _pendingDuplicateData = null;
    }
  }

  /**
   * Delegate to the Confirmation module for full entity review UI.
   * @param {object} data — FlierUploadResponse
   */
  function _populateConfirmView(data) {
    if (typeof Confirmation !== "undefined" && Confirmation.populateConfirmView) {
      Confirmation.populateConfirmView(data);
    }
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
    _hideDuplicateWarning();
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

  /** Public initialiser — called by App.initApp(). */
  function initUpload() {
    _cacheDom();
    _bindEvents();
  }

  return { initUpload };
})();
