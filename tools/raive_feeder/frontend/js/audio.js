/**
 * raiveFeeder — Audio Transcription Module (Tab 2)
 *
 * Handles audio file upload, provider selection, transcription,
 * transcript editing, and ingestion.
 */
const FeederAudio = (() => {
  'use strict';

  let _selectedFile = null;
  let _selectedProvider = 'whisper_local';

  function _init() {
    const dropZone = document.getElementById('audio-drop-zone');
    const fileInput = document.getElementById('audio-file-input');
    const transcribeBtn = document.getElementById('audio-transcribe-btn');
    const ingestBtn = document.getElementById('audio-ingest-btn');

    if (!dropZone || !fileInput) return;

    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => { e.preventDefault(); dropZone.classList.remove('drag-over'); _handleFile(e.dataTransfer.files[0]); });
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', () => _handleFile(fileInput.files[0]));

    // Provider toggle.
    document.querySelectorAll('#audio-provider-toggle .toggle-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        _selectedProvider = btn.dataset.provider;
        document.querySelectorAll('#audio-provider-toggle .toggle-btn').forEach(b => {
          b.classList.toggle('toggle-btn--active', b.dataset.provider === _selectedProvider);
        });
      });
    });

    if (transcribeBtn) transcribeBtn.addEventListener('click', _transcribe);
    if (ingestBtn) ingestBtn.addEventListener('click', _ingestTranscript);
  }

  function _handleFile(file) {
    if (!file) return;
    _selectedFile = file;
    const toggle = document.getElementById('audio-provider-toggle');
    if (toggle) toggle.hidden = false;
  }

  async function _transcribe() {
    if (!_selectedFile) return;

    const progress = document.getElementById('audio-progress');
    const progressFill = document.getElementById('audio-progress-fill');
    const progressText = document.getElementById('audio-progress-text');

    if (progress) progress.hidden = false;
    if (progressFill) progressFill.style.width = '30%';
    if (progressText) progressText.textContent = 'Transcribing...';

    const language = document.getElementById('audio-language')?.value || '';
    const formData = new FormData();
    formData.append('file', _selectedFile);
    formData.append('provider', _selectedProvider);
    if (language) formData.append('language', language);

    try {
      const resp = await FeederApp.apiUpload('/ingest/audio', formData);

      if (progressFill) progressFill.style.width = '100%';
      if (progressText) progressText.textContent = `Transcribed (${resp.provider_used})`;

      // Show transcript for editing.
      const transcriptArea = document.getElementById('audio-transcript');
      const textArea = document.getElementById('audio-transcript-text');
      if (transcriptArea) transcriptArea.hidden = false;
      if (textArea) textArea.value = resp.transcript;

      // Auto-fill title.
      const titleInput = document.getElementById('audio-title');
      if (titleInput && !titleInput.value) {
        titleInput.value = _selectedFile.name.replace(/\.[^/.]+$/, '');
      }
    } catch (err) {
      if (progressText) progressText.textContent = `Error: ${err.message}`;
    }
  }

  async function _ingestTranscript() {
    const text = document.getElementById('audio-transcript-text')?.value;
    if (!text) return;

    const title = document.getElementById('audio-title')?.value || 'Audio transcript';
    const sourceType = document.getElementById('audio-type')?.value || 'interview';

    // Create a text file from the edited transcript and ingest it.
    const blob = new Blob([text], { type: 'text/plain' });
    const file = new File([blob], `${title}.txt`, { type: 'text/plain' });

    const formData = new FormData();
    formData.append('file', file);
    formData.append('title', title);
    formData.append('source_type', sourceType);
    formData.append('citation_tier', '3');

    try {
      const resp = await FeederApp.apiUpload('/ingest/document', formData);
      const resultText = `Ingested: ${resp.chunks_created} chunks created`;
      const progressText = document.getElementById('audio-progress-text');
      if (progressText) progressText.textContent = resultText;
    } catch (err) {
      const progressText = document.getElementById('audio-progress-text');
      if (progressText) progressText.textContent = `Ingest error: ${err.message}`;
    }
  }

  document.addEventListener('DOMContentLoaded', _init);

  return {};
})();
