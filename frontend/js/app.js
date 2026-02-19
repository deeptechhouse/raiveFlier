/**
 * app.js — raiveFlier SPA controller.
 *
 * Manages view switching, global state (sessionId, current view),
 * and application initialisation.
 */

"use strict";

const App = (() => {
  const _VIEW_NAMES = ["upload", "confirm", "progress", "results"];
  let _currentView = "upload";
  let _sessionId = null;

  /**
   * Hide all view sections, then show the one matching `viewName`.
   * @param {string} viewName - One of "upload", "confirm", "progress", "results".
   */
  function showView(viewName) {
    if (!_VIEW_NAMES.includes(viewName)) {
      console.error(`[App] Unknown view: ${viewName}`);
      return;
    }

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
   * @param {string} id - UUID session identifier.
   */
  function setSessionId(id) {
    _sessionId = id;
  }

  /** @returns {string|null} The current session ID, or null before upload. */
  function getSessionId() {
    return _sessionId;
  }

  /**
   * Bootstrap the application: show the upload view and
   * initialise the upload module.
   */
  function initApp() {
    showView("upload");

    if (typeof Upload !== "undefined" && Upload.initUpload) {
      Upload.initUpload();
    }
  }

  return {
    showView,
    getCurrentView,
    setSessionId,
    getSessionId,
    initApp,
  };
})();

document.addEventListener("DOMContentLoaded", () => {
  App.initApp();
});
