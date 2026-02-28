/* ─── Rave Stories — Search Module ──────────────────────────────────
 *
 * Semantic search across stories via ChromaDB vector store.
 * Sends natural language queries and displays results with relevance
 * scores and tag metadata.
 *
 * Depends on: StoriesApp (app.js)
 * ─────────────────────────────────────────────────────────────────── */
"use strict";

(function () {
    const searchInput = document.getElementById('search-input');
    const searchBtn = document.getElementById('search-btn');
    const resultsContainer = document.getElementById('search-results');

    // ── Perform Search ────────────────────────────────────────────
    async function performSearch() {
        const query = searchInput.value.trim();
        if (!query || query.length < 3) {
            resultsContainer.innerHTML = '<div class="message">Enter at least 3 characters to search.</div>';
            return;
        }

        searchBtn.disabled = true;
        resultsContainer.innerHTML = '<div class="loading">Searching stories...</div>';

        try {
            const results = await StoriesApp.apiFetch('/search', {
                method: 'POST',
                body: JSON.stringify({ query, limit: 20 }),
            });

            if (!results.length) {
                resultsContainer.innerHTML = '<div class="empty-state"><div class="empty-state__title">No matching stories</div><p>Try different keywords or a broader query.</p></div>';
                return;
            }

            resultsContainer.innerHTML = results.map(r => {
                const score = (r.similarity_score * 100).toFixed(1);
                const tags = [];
                (r.genre_tags || []).forEach(t => tags.push(StoriesApp.renderTag(t, 'genre')));
                (r.geographic_tags || []).forEach(t => tags.push(StoriesApp.renderTag(t, 'city')));
                (r.entity_tags || []).forEach(t => tags.push(StoriesApp.renderTag(t, 'entity')));
                if (r.time_period) tags.push(StoriesApp.renderTag(r.time_period, 'promoter'));

                return `
                    <article class="story-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--space-2);">
                            <span class="search-result__score">${score}% match</span>
                        </div>
                        <div class="story-card__text">${StoriesApp.escapeHtml(r.text_excerpt)}</div>
                        <div class="story-card__meta">${tags.join('')}</div>
                    </article>
                `;
            }).join('');
        } catch (err) {
            resultsContainer.innerHTML = `<div class="message message--error">Search failed: ${StoriesApp.escapeHtml(err.message)}</div>`;
        }

        searchBtn.disabled = false;
    }

    // ── Event Listeners ───────────────────────────────────────────
    searchBtn.addEventListener('click', performSearch);
    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') performSearch();
    });
})();
