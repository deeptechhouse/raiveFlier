/**
 * raiveFeeder — Main Application Controller
 *
 * Manages tab navigation and shared state.  All other modules depend
 * on FeederApp for view switching and API base URL.
 *
 * Pattern: Module (IIFE returning public API).
 */
const FeederApp = (() => {
  'use strict';

  const API_BASE = '/api/v1';
  let _activeTab = 'documents';

  // ─── Tab switching ──────────────────────────────────────────────────

  function _initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        if (tab) showTab(tab);
      });
    });
  }

  function showTab(tabName) {
    _activeTab = tabName;

    // Update tab buttons.
    document.querySelectorAll('.tab-btn').forEach(btn => {
      const isActive = btn.dataset.tab === tabName;
      btn.classList.toggle('tab-btn--active', isActive);
      btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });

    // Show/hide tab panels.
    document.querySelectorAll('.tab-panel').forEach(panel => {
      panel.hidden = panel.id !== `tab-${tabName}`;
    });

    // Trigger tab-specific initialization.
    if (tabName === 'corpus' && typeof FeederCorpus !== 'undefined') {
      FeederCorpus.refresh();
    }
  }

  // ─── Shared API helper ──────────────────────────────────────────────

  async function apiFetch(path, options = {}) {
    const url = `${API_BASE}${path}`;
    const resp = await fetch(url, options);
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`API ${resp.status}: ${text}`);
    }
    return resp.json();
  }

  async function apiUpload(path, formData) {
    const url = `${API_BASE}${path}`;
    const resp = await fetch(url, { method: 'POST', body: formData });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`API ${resp.status}: ${text}`);
    }
    return resp.json();
  }

  // ─── Utility ────────────────────────────────────────────────────────

  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  // ─── Init ───────────────────────────────────────────────────────────

  document.addEventListener('DOMContentLoaded', () => {
    _initTabs();
  });

  return {
    showTab,
    apiFetch,
    apiUpload,
    formatBytes,
    get activeTab() { return _activeTab; },
    API_BASE,
  };
})();
