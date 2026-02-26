/**
 * raiveFeeder — Image OCR Ingestion Module (Tab 3)
 *
 * Handles image upload, mode selection (single flier vs. multi-page scan),
 * image preview grid, OCR, and ingestion.
 */
const FeederImages = (() => {
  'use strict';

  let _selectedFiles = [];
  let _mode = 'single_flier';

  function _init() {
    const dropZone = document.getElementById('image-drop-zone');
    const fileInput = document.getElementById('image-file-input');
    const ingestBtn = document.getElementById('image-ingest-btn');

    if (!dropZone || !fileInput) return;

    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => { e.preventDefault(); dropZone.classList.remove('drag-over'); _handleFiles(e.dataTransfer.files); });
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', () => _handleFiles(fileInput.files));

    // Mode selector.
    document.querySelectorAll('.mode-selector .toggle-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        _mode = btn.dataset.mode;
        document.querySelectorAll('.mode-selector .toggle-btn').forEach(b => {
          b.classList.toggle('toggle-btn--active', b.dataset.mode === _mode);
        });
      });
    });

    if (ingestBtn) ingestBtn.addEventListener('click', _ingestImages);
  }

  function _handleFiles(fileList) {
    _selectedFiles = Array.from(fileList);
    _showPreviewGrid();
    _runOCR();
  }

  function _showPreviewGrid() {
    const grid = document.getElementById('image-preview-grid');
    if (!grid) return;
    grid.hidden = false;
    grid.innerHTML = '';

    _selectedFiles.forEach((file, idx) => {
      const item = document.createElement('div');
      item.className = 'image-grid__item';

      const img = document.createElement('img');
      img.src = URL.createObjectURL(file);
      img.alt = file.name;

      const num = document.createElement('span');
      num.className = 'image-grid__item-number';
      num.textContent = idx + 1;

      item.appendChild(img);
      item.appendChild(num);
      grid.appendChild(item);
    });
  }

  async function _runOCR() {
    if (_selectedFiles.length === 0) return;

    const progress = document.getElementById('image-progress');
    const progressFill = document.getElementById('image-progress-fill');
    const progressText = document.getElementById('image-progress-text');

    if (progress) progress.hidden = false;
    if (progressFill) progressFill.style.width = '30%';
    if (progressText) progressText.textContent = 'Running OCR...';

    const formData = new FormData();
    _selectedFiles.forEach(f => formData.append('files', f));
    formData.append('mode', _mode);

    try {
      const resp = await FeederApp.apiUpload('/ingest/images', formData);

      if (progressFill) progressFill.style.width = '100%';
      if (progressText) progressText.textContent = 'OCR complete';

      // Show OCR result for review.
      const preview = document.getElementById('image-ocr-preview');
      const textArea = document.getElementById('image-ocr-text');
      if (preview) preview.hidden = false;
      if (textArea) textArea.value = resp.ocr_text || '';
    } catch (err) {
      if (progressText) progressText.textContent = `OCR error: ${err.message}`;
    }
  }

  async function _ingestImages() {
    const text = document.getElementById('image-ocr-text')?.value;
    if (!text) return;

    const title = document.getElementById('image-title')?.value || 'Image OCR';
    const author = document.getElementById('image-author')?.value || '';

    const blob = new Blob([text], { type: 'text/plain' });
    const file = new File([blob], `${title}.txt`, { type: 'text/plain' });

    const formData = new FormData();
    formData.append('file', file);
    formData.append('title', title);
    formData.append('author', author);
    formData.append('source_type', _mode === 'single_flier' ? 'flier' : 'book');

    try {
      const resp = await FeederApp.apiUpload('/ingest/document', formData);
      const progressText = document.getElementById('image-progress-text');
      if (progressText) progressText.textContent = `Ingested: ${resp.chunks_created} chunks`;
    } catch (err) {
      const progressText = document.getElementById('image-progress-text');
      if (progressText) progressText.textContent = `Ingest error: ${err.message}`;
    }
  }

  document.addEventListener('DOMContentLoaded', _init);

  return {};
})();
