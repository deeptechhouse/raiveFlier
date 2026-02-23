/**
 * app.js — raiveFlier SPA controller (main entry point).
 *
 * ROLE IN THE APPLICATION
 * =======================
 * This is the central coordinator for the single-page application. It manages:
 *   1. View switching — hiding/showing the four SPA sections
 *   2. Global shared state — the pipeline session ID
 *   3. Application bootstrap — initialising modules on DOMContentLoaded
 *
 * ARCHITECTURE PATTERN
 * ====================
 * Uses the Revealing Module Pattern (IIFE returning a public API object).
 * This is the same pattern used by all other frontend modules. The IIFE creates
 * a private closure for internal state (_currentView, _sessionId), and the
 * returned object exposes only the public API methods.
 *
 * HOW MODULES COMMUNICATE
 * =======================
 * Other modules call App's public methods directly:
 *   - Upload calls App.setSessionId(id) after successful upload
 *   - Upload calls App.showView("confirm") to navigate forward
 *   - Confirmation calls App.getSessionId() and App.showView("progress")
 *   - WebSocket (Progress) calls App.showView("results") on completion
 *   - Results calls App.getSessionId() to fetch analysis data
 *
 * VIEW SWITCHING MECHANISM
 * ========================
 * All four views exist simultaneously in the DOM as <section> elements.
 * showView() sets the HTML `hidden` attribute on all views except the target.
 * CSS ensures .view[hidden] { display: none; }. This is simpler than a
 * client-side router and works well for a linear 4-step flow.
 */

"use strict";

const App = (() => {
  // The four sequential views in the application pipeline.
  // Each name corresponds to a DOM element with id="${name}-view".
  const _VIEW_NAMES = ["upload", "confirm", "progress", "results"];

  // Tracks which view is currently visible. Starts on "upload".
  let _currentView = "upload";

  // The UUID session ID returned by the /api/v1/fliers/upload endpoint.
  // Null until a flier is successfully uploaded. All subsequent API calls
  // (confirm, progress WebSocket, results, Q&A, ratings) use this ID.
  let _sessionId = null;

  /**
   * Switch the visible view by toggling the `hidden` attribute on each section.
   * This is the core navigation mechanism — all view transitions go through here.
   * @param {string} viewName - One of "upload", "confirm", "progress", "results".
   */
  function showView(viewName) {
    if (!_VIEW_NAMES.includes(viewName)) {
      console.error(`[App] Unknown view: ${viewName}`);
      return;
    }

    // Iterate all views: hide everything except the target view
    _VIEW_NAMES.forEach((name) => {
      const el = document.getElementById(`${name}-view`);
      if (el) {
        el.hidden = name !== viewName;
      }
    });

    _currentView = viewName;
  }

  /** @returns {string} The currently visible view name. */
  function getCurrentView() {
    return _currentView;
  }

  /**
   * Store the pipeline session ID returned by the upload endpoint.
   * Called by Upload._proceedToConfirm() immediately after a successful upload.
   * @param {string} id - UUID session identifier from the backend.
   */
  function setSessionId(id) {
    _sessionId = id;
  }

  /** @returns {string|null} The current session ID, or null before upload. */
  function getSessionId() {
    return _sessionId;
  }

  /**
   * Bootstrap the application on page load.
   * Shows the upload view (the starting point) and initialises modules that
   * need setup before user interaction. Uses typeof checks for graceful
   * degradation — if a module script fails to load, the app still starts.
   */
  function initApp() {
    showView("upload");

    // Initialise the upload module (sets up drag-and-drop listeners, etc.)
    if (typeof Upload !== "undefined" && Upload.initUpload) {
      Upload.initUpload();
    }

    // Initialise the corpus sidebar (renders shell, checks RAG availability)
    if (typeof Corpus !== "undefined" && Corpus.init) {
      Corpus.init();
    }
  }

  // Public API — only these methods are accessible from other modules
  return {
    showView,
    getCurrentView,
    setSessionId,
    getSessionId,
    initApp,
  };
})();

// Entry point: wait for the DOM to be ready, then bootstrap the app.
// This listener fires after the HTML is parsed but before images/stylesheets finish loading.
document.addEventListener("DOMContentLoaded", () => {
  App.initApp();
});
