/* ─── Rave Stories — App Controller ─────────────────────────────────
 *
 * Core SPA controller: tab switching, API fetch helper, initialization.
 * Exposes the global ``StoriesApp`` object that all tab modules depend on.
 *
 * Pattern: IIFE (Immediately Invoked Function Expression) that attaches
 * a single global object.  Each tab module (submit.js, browse.js, etc.)
 * reads from StoriesApp but never mutates it — one-directional dependency.
 *
 * API base path auto-detects whether the app is mounted at /stories/
 * (sub-app of raiveFlier) or running standalone at the root.
 * ─────────────────────────────────────────────────────────────────── */
"use strict";

const StoriesApp = (function () {
    // Auto-detect path prefix for API calls.
    // When mounted at /stories/ the API is at /stories/api/v1/stories/...
    const pathPrefix = window.location.pathname.startsWith('/stories')
        ? '/stories'
        : '';
    const API_BASE = pathPrefix + '/api/v1/stories';

    // ── API Fetch Helper ──────────────────────────────────────────
    async function apiFetch(path, options = {}) {
        const url = API_BASE + path;
        const defaults = {
            headers: { 'Content-Type': 'application/json' },
        };

        // Merge headers, allowing overrides (e.g. multipart/form-data).
        const merged = { ...defaults, ...options };
        if (options.headers) {
            merged.headers = { ...defaults.headers, ...options.headers };
        }
        // Remove Content-Type for FormData (browser sets boundary automatically).
        if (options.body instanceof FormData) {
            delete merged.headers['Content-Type'];
        }

        const response = await fetch(url, merged);
        if (!response.ok) {
            const errorBody = await response.text();
            throw new Error(`API error ${response.status}: ${errorBody}`);
        }
        return response.json();
    }

    // ── Tab Switching ─────────────────────────────────────────────
    const tabButtons = document.querySelectorAll('.tab-btn[data-tab]');
    const tabPanels = document.querySelectorAll('.tab-panel');

    // Callbacks invoked when a tab is first activated.
    const tabInitCallbacks = {};
    // Track which tabs have been initialized.
    const tabInitialized = {};

    function switchTab(tabName) {
        tabButtons.forEach(btn => {
            const isActive = btn.dataset.tab === tabName;
            btn.classList.toggle('tab-btn--active', isActive);
            btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
        });
        tabPanels.forEach(panel => {
            panel.hidden = panel.id !== `tab-${tabName}`;
        });

        // Fire init callback on first activation.
        if (!tabInitialized[tabName] && tabInitCallbacks[tabName]) {
            tabInitCallbacks[tabName]();
            tabInitialized[tabName] = true;
        }
    }

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });

    // Register a callback to run when a tab is first shown.
    function onTabInit(tabName, callback) {
        tabInitCallbacks[tabName] = callback;
    }

    // ── Text Escaping (XSS prevention) ────────────────────────────
    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    // ── Tag Pill Renderer ─────────────────────────────────────────
    function renderTag(value, type) {
        const cssClass = type ? `tag--${type}` : '';
        return `<span class="tag ${cssClass}">${escapeHtml(value)}</span>`;
    }

    // ── Story Card Renderer ───────────────────────────────────────
    function renderStoryCard(story) {
        const tags = [];
        (story.genre_tags || []).forEach(t => tags.push(renderTag(t, 'genre')));
        (story.geographic_tags || []).forEach(t => tags.push(renderTag(t, 'city')));
        (story.entity_tags || []).forEach(t => tags.push(renderTag(t, 'entity')));

        const meta = story.metadata || {};
        if (meta.event_name) tags.unshift(renderTag(meta.event_name, 'promoter'));

        return `
            <article class="story-card">
                <div class="story-card__text">${escapeHtml(story.text)}</div>
                <div class="story-card__meta">
                    ${tags.join('')}
                    <span class="story-card__word-count">${story.word_count} words</span>
                    <span class="story-card__date">${escapeHtml(story.created_at)}</span>
                </div>
            </article>
        `;
    }

    // ── Public API ────────────────────────────────────────────────
    return {
        apiFetch,
        onTabInit,
        switchTab,
        escapeHtml,
        renderTag,
        renderStoryCard,
        API_BASE,
    };
})();
