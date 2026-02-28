/* ─── Rave Stories — Events Module ──────────────────────────────────
 *
 * Event collection pages: grid of event cards with story counts,
 * event detail view with collective narrative and story timeline.
 *
 * Depends on: StoriesApp (app.js)
 * ─────────────────────────────────────────────────────────────────── */
"use strict";

(function () {
    const eventsList = document.getElementById('events-list');
    const eventDetail = document.getElementById('event-detail');
    const eventHeader = document.getElementById('event-header');
    const eventNarrative = document.getElementById('event-narrative');
    const eventStories = document.getElementById('event-stories');
    const backBtn = document.getElementById('back-to-events');

    // ── Load Event List ───────────────────────────────────────────
    async function loadEvents() {
        eventsList.innerHTML = '<div class="loading">Loading events...</div>';
        eventDetail.hidden = true;
        eventsList.hidden = false;

        try {
            const events = await StoriesApp.apiFetch('/events');

            if (!events.length) {
                eventsList.innerHTML = '<div class="empty-state"><div class="empty-state__title">No events yet</div><p>Submit stories to create event collections.</p></div>';
                return;
            }

            eventsList.innerHTML = events.map(e => {
                const yearCity = [e.event_year, e.city].filter(Boolean).join(' / ');
                return `
                    <div class="event-card" data-event="${StoriesApp.escapeHtml(e.event_name)}" data-year="${e.event_year || ''}">
                        <div class="event-card__name">${StoriesApp.escapeHtml(e.event_name)}</div>
                        ${yearCity ? `<div class="event-card__info">${StoriesApp.escapeHtml(yearCity)}</div>` : ''}
                        <div class="event-card__count">${e.story_count} ${e.story_count === 1 ? 'story' : 'stories'}</div>
                    </div>
                `;
            }).join('');

            // Add click handlers.
            eventsList.querySelectorAll('.event-card').forEach(card => {
                card.addEventListener('click', () => {
                    const name = card.dataset.event;
                    const year = card.dataset.year ? parseInt(card.dataset.year) : null;
                    loadEventDetail(name, year);
                });
            });
        } catch (err) {
            eventsList.innerHTML = `<div class="message message--error">Failed to load events: ${StoriesApp.escapeHtml(err.message)}</div>`;
        }
    }

    // ── Load Event Detail ─────────────────────────────────────────
    async function loadEventDetail(eventName, eventYear) {
        eventsList.hidden = true;
        eventDetail.hidden = false;
        eventHeader.innerHTML = '<div class="loading">Loading event...</div>';
        eventNarrative.innerHTML = '';
        eventStories.innerHTML = '';

        try {
            let queryParams = '';
            if (eventYear) queryParams = `?event_year=${eventYear}`;

            const collection = await StoriesApp.apiFetch(`/events/${encodeURIComponent(eventName)}${queryParams}`);

            // Render header.
            const yearCity = [collection.event_year, collection.city].filter(Boolean).join(' / ');
            eventHeader.innerHTML = `
                <div class="event-header">
                    <h2>${StoriesApp.escapeHtml(collection.event_name)}</h2>
                    <div class="event-header__details">
                        ${yearCity ? `<span>${StoriesApp.escapeHtml(yearCity)}</span>` : ''}
                        <span>${collection.story_count} ${collection.story_count === 1 ? 'story' : 'stories'}</span>
                    </div>
                </div>
            `;

            // Render narrative (if available).
            if (collection.narrative) {
                const themeTags = (collection.themes || []).map(t => StoriesApp.renderTag(t, 'genre')).join('');
                eventNarrative.innerHTML = `
                    <div class="narrative">
                        <div class="narrative__label">Collective Narrative</div>
                        ${StoriesApp.escapeHtml(collection.narrative)}
                        ${themeTags ? `<div class="narrative__themes">${themeTags}</div>` : ''}
                    </div>
                `;
            } else if (collection.story_count >= 3) {
                // Offer to generate narrative.
                eventNarrative.innerHTML = `
                    <div class="card" style="text-align: center;">
                        <p style="color: var(--color-text-secondary); margin-bottom: var(--space-3);">
                            ${collection.story_count} stories available — generate a collective narrative?
                        </p>
                        <button class="btn btn--primary btn--small" id="generate-narrative-btn">Generate Narrative</button>
                    </div>
                `;
                document.getElementById('generate-narrative-btn').addEventListener('click', async () => {
                    await generateNarrative(eventName, eventYear);
                });
            }

            // Render stories.
            if (collection.stories && collection.stories.length) {
                eventStories.innerHTML = collection.stories.map(s => StoriesApp.renderStoryCard(s)).join('');
            }
        } catch (err) {
            eventHeader.innerHTML = `<div class="message message--error">Failed to load event: ${StoriesApp.escapeHtml(err.message)}</div>`;
        }
    }

    // ── Generate Narrative ────────────────────────────────────────
    async function generateNarrative(eventName, eventYear) {
        eventNarrative.innerHTML = '<div class="loading">Generating collective narrative...</div>';

        try {
            let queryParams = '';
            if (eventYear) queryParams = `?event_year=${eventYear}`;

            const result = await StoriesApp.apiFetch(`/events/${encodeURIComponent(eventName)}/narrative${queryParams}`);

            if (result.error) {
                eventNarrative.innerHTML = `<div class="message message--error">${StoriesApp.escapeHtml(result.error)}</div>`;
            } else if (result.narrative) {
                const themeTags = (result.themes || []).map(t => StoriesApp.renderTag(t, 'genre')).join('');
                eventNarrative.innerHTML = `
                    <div class="narrative">
                        <div class="narrative__label">Collective Narrative</div>
                        ${StoriesApp.escapeHtml(result.narrative)}
                        ${themeTags ? `<div class="narrative__themes">${themeTags}</div>` : ''}
                    </div>
                `;
            }
        } catch (err) {
            eventNarrative.innerHTML = `<div class="message message--error">Failed to generate narrative: ${StoriesApp.escapeHtml(err.message)}</div>`;
        }
    }

    // ── Back Button ───────────────────────────────────────────────
    backBtn.addEventListener('click', () => {
        eventDetail.hidden = true;
        eventsList.hidden = false;
    });

    // ── Initialize on tab activation ──────────────────────────────
    StoriesApp.onTabInit('events', loadEvents);
})();
