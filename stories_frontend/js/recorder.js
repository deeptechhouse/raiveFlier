/* ─── Rave Stories — Audio Recorder Module ─────────────────────────
 *
 * Web Audio API recorder using MediaRecorder → WebM format.
 * Renders a live waveform visualization during recording.
 * Handles both recording and file upload submission.
 *
 * Anonymity: No audio is stored client-side after submission.
 * The recorded blob is sent to the server and discarded.
 *
 * Depends on: StoriesApp (app.js)
 * ─────────────────────────────────────────────────────────────────── */
"use strict";

(function () {
    const recordBtn = document.getElementById('record-btn');
    const recordTime = document.getElementById('record-time');
    const waveformCanvas = document.getElementById('waveform-canvas');
    const submitAudioBtn = document.getElementById('submit-audio-btn');
    const audioUploadInput = document.getElementById('audio-upload');
    const uploadAudioBtn = document.getElementById('upload-audio-btn');
    const messageDiv = document.getElementById('submit-message');

    let mediaRecorder = null;
    let audioChunks = [];
    let audioBlob = null;
    let isRecording = false;
    let startTime = 0;
    let timerInterval = null;
    let analyser = null;
    let animationFrame = null;
    let audioContext = null;

    const MAX_DURATION_MS = 5 * 60 * 1000; // 5 minutes

    // ── Waveform Visualization ────────────────────────────────────
    function drawWaveform() {
        if (!analyser || !isRecording) return;

        const canvas = waveformCanvas;
        const ctx = canvas.getContext('2d');
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        analyser.getByteTimeDomainData(dataArray);

        canvas.width = canvas.parentElement.clientWidth;
        canvas.height = canvas.parentElement.clientHeight;

        ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--color-bg').trim();
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.lineWidth = 2;
        ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--color-accent-text').trim();
        ctx.beginPath();

        const sliceWidth = canvas.width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = (v * canvas.height) / 2;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
            x += sliceWidth;
        }

        ctx.lineTo(canvas.width, canvas.height / 2);
        ctx.stroke();

        animationFrame = requestAnimationFrame(drawWaveform);
    }

    // ── Timer ─────────────────────────────────────────────────────
    function updateTimer() {
        const elapsed = Date.now() - startTime;
        const seconds = Math.floor(elapsed / 1000);
        const minutes = Math.floor(seconds / 60);
        const secs = seconds % 60;
        recordTime.textContent = `${minutes}:${secs.toString().padStart(2, '0')}`;

        // Auto-stop at max duration.
        if (elapsed >= MAX_DURATION_MS) {
            stopRecording();
        }
    }

    // ── Start Recording ───────────────────────────────────────────
    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioContext.createMediaStreamSource(stream);
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 2048;
            source.connect(analyser);

            mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
            audioChunks = [];

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) audioChunks.push(e.data);
            };

            mediaRecorder.onstop = () => {
                audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                submitAudioBtn.disabled = !hasAudioMetadata();
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start(250); // Collect data every 250ms.
            isRecording = true;
            startTime = Date.now();
            recordBtn.classList.add('recording');
            timerInterval = setInterval(updateTimer, 1000);
            drawWaveform();
        } catch (err) {
            messageDiv.innerHTML = `<div class="message message--error">Microphone access denied: ${StoriesApp.escapeHtml(err.message)}</div>`;
        }
    }

    // ── Stop Recording ────────────────────────────────────────────
    function stopRecording() {
        if (mediaRecorder && isRecording) {
            mediaRecorder.stop();
            isRecording = false;
            recordBtn.classList.remove('recording');
            clearInterval(timerInterval);
            cancelAnimationFrame(animationFrame);
            if (audioContext) audioContext.close();
        }
    }

    // ── Record Button Toggle ──────────────────────────────────────
    recordBtn.addEventListener('click', () => {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    });

    // ── File Upload ───────────────────────────────────────────────
    uploadAudioBtn.addEventListener('click', () => audioUploadInput.click());

    audioUploadInput.addEventListener('change', () => {
        const file = audioUploadInput.files[0];
        if (!file) return;

        // Validate size (25MB).
        if (file.size > 25 * 1024 * 1024) {
            messageDiv.innerHTML = '<div class="message message--error">Audio file exceeds 25MB limit.</div>';
            return;
        }

        audioBlob = file;
        recordTime.textContent = 'File selected';
        submitAudioBtn.disabled = !hasAudioMetadata();
    });

    // ── Metadata Check ────────────────────────────────────────────
    function hasAudioMetadata() {
        return !!(
            document.getElementById('audio-meta-event').value.trim() ||
            document.getElementById('audio-meta-year').value.trim() ||
            document.getElementById('audio-meta-city').value.trim() ||
            document.getElementById('audio-meta-genre').value.trim()
        );
    }

    // Re-check submit button when audio metadata changes.
    document.querySelectorAll('#audio-form .metadata-grid input').forEach(input => {
        input.addEventListener('input', () => {
            if (audioBlob) submitAudioBtn.disabled = !hasAudioMetadata();
        });
    });

    // ── Submit Audio ──────────────────────────────────────────────
    submitAudioBtn.addEventListener('click', async () => {
        if (!audioBlob) return;

        submitAudioBtn.disabled = true;
        messageDiv.innerHTML = '<div class="message">Uploading and transcribing audio...</div>';

        try {
            const formData = new FormData();
            const filename = audioBlob instanceof File ? audioBlob.name : 'recording.webm';
            formData.append('audio', audioBlob, filename);

            // Append metadata fields.
            const eventName = document.getElementById('audio-meta-event').value.trim();
            const year = document.getElementById('audio-meta-year').value.trim();
            const city = document.getElementById('audio-meta-city').value.trim();
            const genre = document.getElementById('audio-meta-genre').value.trim();

            if (eventName) formData.append('event_name', eventName);
            if (year) formData.append('event_year', year);
            if (city) formData.append('city', city);
            if (genre) formData.append('genre', genre);

            const result = await StoriesApp.apiFetch('/submit-audio', {
                method: 'POST',
                body: formData,
            });

            if (result.error) {
                messageDiv.innerHTML = `<div class="message message--error">${StoriesApp.escapeHtml(result.error)}</div>`;
            } else if (result.status === 'APPROVED') {
                messageDiv.innerHTML = '<div class="message message--success">Audio story transcribed and approved!</div>';
                audioBlob = null;
                recordTime.textContent = '0:00';
            } else if (result.status === 'REJECTED') {
                const reason = result.moderation_reason || 'Content did not pass moderation.';
                messageDiv.innerHTML = `<div class="message message--error">Story rejected: ${StoriesApp.escapeHtml(reason)}</div>`;
            }
        } catch (err) {
            messageDiv.innerHTML = `<div class="message message--error">Upload failed: ${StoriesApp.escapeHtml(err.message)}</div>`;
        }

        submitAudioBtn.disabled = false;
    });
})();
