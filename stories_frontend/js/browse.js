/* ─── Rave Stories — Browse Module ──────────────────────────────────
 *
 * Story browsing with tag filters and pagination.
 * Loads stories on tab activation, supports filtering by tag type/value.
 *
 * Depends on: StoriesApp (app.js)
 * ─────────────────────────────────────────────────────────────────── */
"use strict";

(function () {
    const storiesContainer = document.getElementById('browse-stories');
    const paginationContainer = document.getElementById('browse-pagination');
    const filterTagType = document.getElementById('filter-tag-type');
    const filterTagValue = document.getElementById('filter-tag-value');
    const clearFiltersBtn = document.getElementById('clear-filters-btn');

    const PAGE_SIZE = 20;
    let currentOffset = 0;
    let currentTagType = '';
    let currentTagValue = '';

    // ── Load Stories ──────────────────────────────────────────────
    async function loadStories() {
        storiesContainer.innerHTML = '<div class="loading">Loading stories...</div>';

        try {
            let queryParams = `?limit=${PAGE_SIZE}&offset=${currentOffset}`;
            if (currentTagType && currentTagValue) {
                queryParams += `&tag_type=${encodeURIComponent(currentTagType)}&tag_value=${encodeURIComponent(currentTagValue)}`;
            }

            const stories = await StoriesApp.apiFetch('/' + queryParams);

            if (!stories.length) {
                storiesContainer.innerHTML = '<div class="empty-state"><div class="empty-state__title">No stories yet</div><p>Be the first to share a rave experience!</p></div>';
                paginationContainer.innerHTML = '';
                return;
            }

            storiesContainer.innerHTML = stories.map(s => StoriesApp.renderStoryCard(s)).join('');
            renderPagination(stories.length);
        } catch (err) {
            storiesContainer.innerHTML = `<div class="message message--error">Failed to load stories: ${StoriesApp.escapeHtml(err.message)}</div>`;
        }
    }

    // ── Pagination ────────────────────────────────────────────────
    function renderPagination(resultCount) {
        let html = '';
        if (currentOffset > 0) {
            html += '<button class="btn btn--secondary btn--small" id="page-prev">Previous</button>';
        }
        if (resultCount >= PAGE_SIZE) {
            html += '<button class="btn btn--secondary btn--small" id="page-next">Next</button>';
        }
        paginationContainer.innerHTML = html;

        const prevBtn = document.getElementById('page-prev');
        const nextBtn = document.getElementById('page-next');
        if (prevBtn) prevBtn.addEventListener('click', () => { currentOffset = Math.max(0, currentOffset - PAGE_SIZE); loadStories(); });
        if (nextBtn) nextBtn.addEventListener('click', () => { currentOffset += PAGE_SIZE; loadStories(); });
    }

    // ── Tag Filters ───────────────────────────────────────────────
    filterTagType.addEventListener('change', async () => {
        currentTagType = filterTagType.value;
        currentTagValue = '';
        filterTagValue.innerHTML = '<option value="">Select tag...</option>';

        if (currentTagType) {
            filterTagValue.disabled = false;
            try {
                const tags = await StoriesApp.apiFetch(`/tags/${currentTagType}`);
                tags.forEach(tag => {
                    const opt = document.createElement('option');
                    opt.value = tag;
                    opt.textContent = tag;
                    filterTagValue.appendChild(opt);
                });
            } catch (err) {
                // Silently fail — filter just won't have values.
            }
        } else {
            filterTagValue.disabled = true;
            currentOffset = 0;
            loadStories();
        }
    });

    filterTagValue.addEventListener('change', () => {
        currentTagValue = filterTagValue.value;
        currentOffset = 0;
        loadStories();
    });

    clearFiltersBtn.addEventListener('click', () => {
        filterTagType.value = '';
        filterTagValue.value = '';
        filterTagValue.disabled = true;
        currentTagType = '';
        currentTagValue = '';
        currentOffset = 0;
        loadStories();
    });

    // ── Initialize on tab activation ──────────────────────────────
    StoriesApp.onTabInit('browse', loadStories);
})();
