/**
 * websocket.js — Real-time pipeline progress tracking via WebSocket.
 *
 * Connects to the server's WebSocket endpoint to receive progress
 * updates and renders a 5-phase pipeline visualization with an
 * animated progress bar and status messages.
 */

"use strict";

const Progress = (() => {
  /** @type {WebSocket|null} */
  let _socket = null;

  /** @type {number} Current retry count */
  let _retryCount = 0;

  /** @type {number} Maximum retries before showing manual refresh */
  const _MAX_RETRIES = 3;

  /** @type {number} Retry delay in milliseconds */
  const _RETRY_DELAY = 2000;

  /** @type {string|null} Current session ID */
  let _sessionId = null;

  /** Pipeline phase metadata — ordered sequence */
  const _PHASES = [
    { key: "OCR",                 label: "OCR",              description: "Extracting text from flier" },
    { key: "ENTITY_EXTRACTION",   label: "Entities",         description: "Identifying artists, venue, date..." },
    { key: "RESEARCH",            label: "Research",         description: "Researching entities..." },
    { key: "INTERCONNECTION",     label: "Connections",      description: "Analyzing connections between entities..." },
    { key: "OUTPUT",              label: "Output",           description: "Compiling results with citations..." },
  ];

  /**
   * Build the progress view HTML structure.
   * Called once when switching to the progress view.
   */
  function _buildProgressView() {
    const progressView = document.getElementById("progress-view");
    if (!progressView) return;

    let html = "";

    // Header
    html += `<h2 class="text-heading">Analyzing Flier</h2>`;

    // Phase pipeline visualization
    html += `<div class="phase-pipeline" role="list" aria-label="Pipeline phases">`;
    _PHASES.forEach((phase, i) => {
      const connector = i < _PHASES.length - 1
        ? `<div class="phase-connector" id="connector-${i}"></div>`
        : "";
      html += `<div class="phase-step" id="phase-${phase.key}" role="listitem" aria-label="${phase.label}">
        <div class="phase-step__number">${i + 1}</div>
        <div class="phase-step__label">${phase.label}</div>
      </div>${connector}`;
    });
    html += `</div>`;

    // Progress bar
    html += `<div class="progress-section">
      <div class="progress-bar" role="progressbar" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">
        <div class="progress-bar__fill" id="progress-fill" style="width: 0%"></div>
      </div>
      <div class="progress-info">
        <span class="progress-pct text-caption" id="progress-pct">0%</span>
      </div>
    </div>`;

    // Phase description
    html += `<div class="progress-status">
      <p class="progress-status__phase text-caption" id="progress-phase-text"></p>
      <p class="progress-status__message" id="progress-message">Waiting for pipeline to start...</p>
    </div>`;

    // Error / reconnection area
    html += `<div class="progress-error" id="progress-error" role="alert" hidden></div>`;

    progressView.innerHTML = html;
  }

  /**
   * Determine the WebSocket URL from the current page location.
   * @param {string} sessionId
   * @returns {string}
   */
  function _getWSUrl(sessionId) {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${window.location.host}/ws/progress/${encodeURIComponent(sessionId)}`;
  }

  /**
   * Open a WebSocket connection to receive progress updates.
   * @param {string} sessionId
   */
  function connectProgress(sessionId) {
    _sessionId = sessionId;
    _retryCount = 0;

    _buildProgressView();
    _openSocket(sessionId);
  }

  /**
   * Internal: create and configure the WebSocket.
   * @param {string} sessionId
   */
  function _openSocket(sessionId) {
    const url = _getWSUrl(sessionId);

    try {
      _socket = new WebSocket(url);
    } catch (err) {
      _handleWebSocketError(err);
      return;
    }

    _socket.onopen = () => {
      _retryCount = 0;
      _hideError();
    };

    _socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        updateProgressUI(data);
      } catch (err) {
        // Ignore non-JSON messages (keep-alive pings)
      }
    };

    _socket.onerror = () => {
      _handleWebSocketError(new Error("WebSocket connection error"));
    };

    _socket.onclose = (event) => {
      // Only retry if not a clean close and not at 100%
      if (!event.wasClean && _retryCount < _MAX_RETRIES) {
        _handleWebSocketError(new Error("Connection lost"));
      }
    };
  }

  /**
   * Update the progress UI with data from a WebSocket message.
   * @param {object} data — { phase: string, progress: float, message: string }
   */
  function updateProgressUI(data) {
    const phase = data.phase || "";
    const progress = data.progress || 0;
    const message = data.message || "";

    // Update phase pipeline steps
    _updatePhaseSteps(phase);

    // Update progress bar
    const fill = document.getElementById("progress-fill");
    const pctEl = document.getElementById("progress-pct");
    const progressBar = fill ? fill.closest(".progress-bar") : null;
    const pctRounded = Math.round(progress);

    if (fill) fill.style.width = `${pctRounded}%`;
    if (pctEl) pctEl.textContent = `${pctRounded}%`;
    if (progressBar) {
      progressBar.setAttribute("aria-valuenow", String(pctRounded));
    }

    // Update status text
    const phaseTextEl = document.getElementById("progress-phase-text");
    const messageEl = document.getElementById("progress-message");

    const phaseInfo = _PHASES.find((p) => p.key === phase);
    if (phaseTextEl && phaseInfo) {
      phaseTextEl.textContent = phaseInfo.description;
    }
    if (messageEl && message) {
      messageEl.textContent = message;
    }

    // On completion
    if (progress >= 100) {
      _onComplete();
    }
  }

  /**
   * Update the phase step indicators.
   * @param {string} currentPhase — the current pipeline phase key
   */
  function _updatePhaseSteps(currentPhase) {
    const currentIdx = _PHASES.findIndex((p) => p.key === currentPhase);

    _PHASES.forEach((phase, i) => {
      const stepEl = document.getElementById(`phase-${phase.key}`);
      if (!stepEl) return;

      // Reset classes
      stepEl.classList.remove("phase-step--active", "phase-step--completed", "phase-step--pending");

      if (i < currentIdx) {
        stepEl.classList.add("phase-step--completed");
      } else if (i === currentIdx) {
        stepEl.classList.add("phase-step--active");
      } else {
        stepEl.classList.add("phase-step--pending");
      }

      // Update connectors
      const connector = document.getElementById(`connector-${i}`);
      if (connector) {
        connector.classList.remove("phase-connector--completed", "phase-connector--active");
        if (i < currentIdx) {
          connector.classList.add("phase-connector--completed");
        } else if (i === currentIdx) {
          connector.classList.add("phase-connector--active");
        }
      }
    });
  }

  /** Handle pipeline completion: close socket, fetch results, switch view. */
  async function _onComplete() {
    disconnectProgress();

    const sessionId = App.getSessionId();
    if (!sessionId) return;

    // Brief delay so the user sees 100%
    await new Promise((resolve) => setTimeout(resolve, 800));

    try {
      const response = await fetch(`/api/v1/fliers/${encodeURIComponent(sessionId)}/results`);
      if (response.ok) {
        const results = await response.json();
        // Results view (G3) will consume this — store on App for now
        if (typeof App._resultsData !== "undefined" || true) {
          App._resultsData = results;
        }
      }
    } catch (err) {
      // Results fetch failed — user can still navigate
    }

    App.showView("results");

    // Populate results view if the Results module exists (G3)
    if (typeof Results !== "undefined" && Results.populateResultsView) {
      Results.populateResultsView(App._resultsData);
    } else {
      // Temporary placeholder until G3
      const resultsView = document.getElementById("results-view");
      if (resultsView) {
        resultsView.innerHTML = `<h2 class="text-heading">Analysis Complete</h2>
          <p class="text-body" style="margin-top: var(--space-4)">Results view will be implemented in G3.</p>
          <p class="text-caption" style="margin-top: var(--space-3)">Session: ${_escapeHTML(sessionId)}</p>`;
      }
    }
  }

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
   * Handle WebSocket errors with retry logic.
   * @param {Error} err
   */
  function _handleWebSocketError(err) {
    _retryCount++;

    const errorEl = document.getElementById("progress-error");
    if (!errorEl) return;

    if (_retryCount <= _MAX_RETRIES) {
      errorEl.textContent = `Connection lost. Reconnecting... (attempt ${_retryCount}/${_MAX_RETRIES})`;
      errorEl.hidden = false;

      setTimeout(() => {
        if (_sessionId) {
          _openSocket(_sessionId);
        }
      }, _RETRY_DELAY);
    } else {
      errorEl.innerHTML = `<span>Connection failed after ${_MAX_RETRIES} attempts.</span>
        <button type="button" class="btn-secondary" id="retry-ws-btn" style="margin-left: var(--space-3)">Retry</button>`;
      errorEl.hidden = false;

      const retryBtn = document.getElementById("retry-ws-btn");
      if (retryBtn) {
        retryBtn.addEventListener("click", () => {
          _retryCount = 0;
          errorEl.hidden = true;
          if (_sessionId) _openSocket(_sessionId);
        });
      }
    }
  }

  /** Hide the progress error element. */
  function _hideError() {
    const errorEl = document.getElementById("progress-error");
    if (errorEl) errorEl.hidden = true;
  }

  /** Cleanly close the WebSocket connection. */
  function disconnectProgress() {
    if (_socket) {
      _socket.onclose = null; // prevent retry on intentional close
      _socket.onerror = null;
      _socket.close();
      _socket = null;
    }
  }

  return {
    connectProgress,
    updateProgressUI,
    disconnectProgress,
  };
})();
