/* ─── Rave Stories — Submit Module ──────────────────────────────────
 *
 * Handles text story submission: word count tracking, metadata validation,
 * form submission via API, and status feedback.
 *
 * Also manages the text/audio mode toggle (audio recording is handled
 * by recorder.js — this module just toggles visibility).
 *
 * Depends on: StoriesApp (app.js)
 * ─────────────────────────────────────────────────────────────────── */
"use strict";

(function () {
    const textArea = document.getElementById('story-text');
    const wordCounter = document.getElementById('word-counter');
    const submitBtn = document.getElementById('submit-text-btn');
    const messageDiv = document.getElementById('submit-message');
    const textForm = document.getElementById('text-form');
    const audioForm = document.getElementById('audio-form');
    const modeTextBtn = document.getElementById('mode-text-btn');
    const modeAudioBtn = document.getElementById('mode-audio-btn');

    const MAX_WORDS = 2000;
    const MIN_WORDS = 10;

    // ── Word Counter ──────────────────────────────────────────────
    function updateWordCount() {
        const text = textArea.value.trim();
        const count = text ? text.split(/\s+/).length : 0;
        wordCounter.textContent = `${count.toLocaleString()} / ${MAX_WORDS.toLocaleString()} words`;

        wordCounter.className = 'word-counter';
        if (count > MAX_WORDS) {
            wordCounter.classList.add('word-counter--error');
        } else if (count > MAX_WORDS * 0.9) {
            wordCounter.classList.add('word-counter--warning');
        }

        // Enable submit if we have enough words and at least one metadata field.
        submitBtn.disabled = count < MIN_WORDS || count > MAX_WORDS || !hasMetadata();
    }

    function hasMetadata() {
        return !!(
            document.getElementById('meta-event').value.trim() ||
            document.getElementById('meta-year').value.trim() ||
            document.getElementById('meta-city').value.trim() ||
            document.getElementById('meta-genre').value.trim() ||
            document.getElementById('meta-promoter').value.trim() ||
            document.getElementById('meta-artist').value.trim()
        );
    }

    textArea.addEventListener('input', updateWordCount);
    // Also re-check when metadata fields change.
    document.querySelectorAll('.metadata-grid input').forEach(input => {
        input.addEventListener('input', updateWordCount);
    });

    // ── Mode Toggle ───────────────────────────────────────────────
    modeTextBtn.addEventListener('click', () => {
        textForm.hidden = false;
        audioForm.hidden = true;
        modeTextBtn.classList.add('btn--active');
        modeAudioBtn.classList.remove('btn--active');
    });

    modeAudioBtn.addEventListener('click', () => {
        textForm.hidden = true;
        audioForm.hidden = false;
        modeAudioBtn.classList.add('btn--active');
        modeTextBtn.classList.remove('btn--active');
    });

    // ── Submit Text Story ─────────────────────────────────────────
    submitBtn.addEventListener('click', async () => {
        submitBtn.disabled = true;
        messageDiv.innerHTML = '<div class="message">Submitting your story...</div>';

        try {
            const result = await StoriesApp.apiFetch('/submit', {
                method: 'POST',
                body: JSON.stringify({
                    text: textArea.value.trim(),
                    metadata: {
                        event_name: document.getElementById('meta-event').value.trim() || null,
                        event_year: parseInt(document.getElementById('meta-year').value) || null,
                        city: document.getElementById('meta-city').value.trim() || null,
                        genre: document.getElementById('meta-genre').value.trim() || null,
                        promoter: document.getElementById('meta-promoter').value.trim() || null,
                        artist: document.getElementById('meta-artist').value.trim() || null,
                    },
                }),
            });

            if (result.error) {
                messageDiv.innerHTML = `<div class="message message--error">${StoriesApp.escapeHtml(result.error)}</div>`;
            } else if (result.status === 'APPROVED') {
                messageDiv.innerHTML = '<div class="message message--success">Story submitted and approved! It\'s now live in the Browse section.</div>';
                // Clear the form.
                textArea.value = '';
                document.querySelectorAll('.metadata-grid input').forEach(i => { i.value = ''; });
                updateWordCount();
            } else if (result.status === 'REJECTED') {
                const reason = result.moderation_reason || 'Content did not pass moderation.';
                messageDiv.innerHTML = `<div class="message message--error">Story rejected: ${StoriesApp.escapeHtml(reason)}</div>`;
            } else {
                messageDiv.innerHTML = `<div class="message">Story submitted (status: ${StoriesApp.escapeHtml(result.status)})</div>`;
            }
        } catch (err) {
            messageDiv.innerHTML = `<div class="message message--error">Submission failed: ${StoriesApp.escapeHtml(err.message)}</div>`;
        }

        submitBtn.disabled = false;
        updateWordCount();
    });
})();
