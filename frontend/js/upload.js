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
 * 4. If duplicate detected: show non-blocking notice, user clicks Analyze again
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
  let _duplicateNotice = null;
  let _duplicateText = null;
  let _analysisCountBadge = null;

  // When the backend detects a duplicate, we store the full upload response
  // here. The next click of "Analyze Flier" uses this data instead of
  // re-uploading, then clears it.
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
    _duplicateNotice = document.getElementById("duplicate-warning");
    _duplicateText = document.getElementById("duplicate-warning-text");
    _analysisCountBadge = document.getElementById("analysis-count-badge");
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
   * If the user already saw a duplicate notification and clicked "Analyze Flier"
   * again, we skip the re-upload and proceed with the stored response data.
   *
   * API: POST /api/v1/fliers/upload
   * Request: multipart/form-data with a single "file" field
   * Response: FlierUploadResponse JSON containing:
   *   - session_id, extracted_entities, ocr_confidence, duplicate_match, times_analyzed
   *
   * Flow after response:
   *   - If duplicate_match present -> show non-blocking notice, keep submit enabled
   *   - Next "Analyze Flier" click -> dismiss notice, proceed to confirm
   *   - No duplicate -> proceed directly to confirm view
   *
   * @param {File} file
   */
  async function _submitFlier(file) {
    // If duplicate was already acknowledged, proceed with stored data
    if (_pendingDuplicateData) {
      const data = _pendingDuplicateData;
      _pendingDuplicateData = null;
      _hideDuplicateNotice();
      _proceedToConfirm(data);
      return;
    }

    _hideError();
    _hideDuplicateNotice();
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

      // Duplicate detected — show non-blocking notice, store data for next click
      if (data.duplicate_match) {
        _pendingDuplicateData = data;
        _showDuplicateNotice(data.duplicate_match, data.times_analyzed || 1);
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
    _updateAnalysisCount(data.times_analyzed || 1);
    _populateConfirmView(data);
    App.showView("confirm");
  }

  /**
   * Show a non-blocking duplicate notice. The user can click "Analyze Flier"
   * again to dismiss and proceed.
   * @param {object} match — DuplicateMatch from the API response
   * @param {number} timesAnalyzed — total previous analyses of this image
   */
  function _showDuplicateNotice(match, timesAnalyzed) {
    const dateStr = match.analyzed_at
      ? new Date(match.analyzed_at).toLocaleDateString()
      : "unknown date";
    const countLabel = timesAnalyzed === 1 ? "once" : `${timesAnalyzed} times`;

    _duplicateText.textContent =
      `Previously analyzed ${countLabel} (last: ${dateStr}). ` +
      "Click Analyze Flier to continue.";

    _duplicateNotice.hidden = false;
  }

  /** Hide the duplicate notice. */
  function _hideDuplicateNotice() {
    if (_duplicateNotice) {
      _duplicateNotice.hidden = true;
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
    _hideDuplicateNotice();
    _pendingDuplicateData = null;
    _updateAnalysisCount(0);
  }

  /**
   * Show or hide the analysis count badge in the site header.
   * Only visible when a flier has been analyzed more than once.
   * @param {number} count — total times this flier has been analyzed
   */
  function _updateAnalysisCount(count) {
    if (!_analysisCountBadge) return;
    if (count > 1) {
      _analysisCountBadge.textContent = `analysis #${count}`;
      _analysisCountBadge.hidden = false;
    } else {
      _analysisCountBadge.hidden = true;
      _analysisCountBadge.textContent = "";
    }
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
