/**
 * raiveFeeder — Document Upload Module (Tab 1)
 *
 * Handles drag-and-drop document upload, metadata form, and ingestion.
 * Supports PDF, EPUB, TXT, DOCX, RTF, MOBI, DJVU formats.
 */
const FeederUpload = (() => {
  'use strict';

  let _selectedFiles = [];

  function _init() {
    const dropZone = document.getElementById('doc-drop-zone');
    const fileInput = document.getElementById('doc-file-input');
    const submitBtn = document.getElementById('doc-submit-btn');

    if (!dropZone || !fileInput) return;

    // Drag-and-drop events.
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => { e.preventDefault(); dropZone.classList.remove('drag-over'); _handleFiles(e.dataTransfer.files); });
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', () => _handleFiles(fileInput.files));

    if (submitBtn) submitBtn.addEventListener('click', _submitDocuments);
  }

  function _handleFiles(fileList) {
    _selectedFiles = Array.from(fileList);
    const metaForm = document.getElementById('doc-meta-form');
    const fileListEl = document.getElementById('doc-file-list');

    if (_selectedFiles.length === 0) return;

    // Show metadata form.
    if (metaForm) metaForm.hidden = false;

    // Auto-fill title from first filename.
    const titleInput = document.getElementById('doc-title');
    if (titleInput && !titleInput.value) {
      titleInput.value = _selectedFiles[0].name.replace(/\.[^/.]+$/, '');
    }

    // Show file list.
    if (fileListEl) {
      fileListEl.hidden = false;
      fileListEl.innerHTML = _selectedFiles.map((f, i) => `
        <div class="file-item">
          <span class="file-item__name">${f.name}</span>
          <span class="file-item__size">${FeederApp.formatBytes(f.size)}</span>
          <button class="file-item__remove" data-idx="${i}" aria-label="Remove">&times;</button>
        </div>
      `).join('');

      fileListEl.querySelectorAll('.file-item__remove').forEach(btn => {
        btn.addEventListener('click', () => {
          _selectedFiles.splice(parseInt(btn.dataset.idx), 1);
          _handleFiles(_selectedFiles);
        });
      });
    }
  }

  async function _submitDocuments() {
    if (_selectedFiles.length === 0) return;

    const progressArea = document.getElementById('doc-progress');
    const progressFill = document.getElementById('doc-progress-fill');
    const progressText = document.getElementById('doc-progress-text');
    const resultArea = document.getElementById('doc-result');

    if (progressArea) progressArea.hidden = false;
    if (resultArea) resultArea.hidden = true;

    const title = document.getElementById('doc-title')?.value || '';
    const author = document.getElementById('doc-author')?.value || '';
    const year = parseInt(document.getElementById('doc-year')?.value) || 0;
    const sourceType = document.getElementById('doc-type')?.value || 'book';
    const tier = parseInt(document.getElementById('doc-tier')?.value) || 2;

    const results = [];
    for (let i = 0; i < _selectedFiles.length; i++) {
      const pct = ((i / _selectedFiles.length) * 100).toFixed(0);
      if (progressFill) progressFill.style.width = `${pct}%`;
      if (progressText) progressText.textContent = `Processing ${_selectedFiles[i].name}...`;

      const formData = new FormData();
      formData.append('file', _selectedFiles[i]);
      formData.append('title', _selectedFiles.length === 1 ? title : _selectedFiles[i].name.replace(/\.[^/.]+$/, ''));
      formData.append('author', author);
      formData.append('year', year.toString());
      formData.append('source_type', sourceType);
      formData.append('citation_tier', tier.toString());

      try {
        const resp = await FeederApp.apiUpload('/ingest/document', formData);
        results.push(resp);
      } catch (err) {
        results.push({ status: 'failed', error: err.message, source_title: _selectedFiles[i].name });
      }
    }

    if (progressFill) progressFill.style.width = '100%';
    if (progressText) progressText.textContent = 'Complete';

    // Show results.
    if (resultArea) {
      resultArea.hidden = false;
      const successCount = results.filter(r => r.status === 'completed').length;
      const totalChunks = results.reduce((s, r) => s + (r.chunks_created || 0), 0);
      resultArea.innerHTML = `
        <p>${successCount}/${results.length} files ingested successfully</p>
        <p>Total chunks created: ${totalChunks}</p>
        ${results.filter(r => r.error).map(r => `<p style="color: var(--color-error-text);">${r.source_title}: ${r.error}</p>`).join('')}
      `;
    }

    _selectedFiles = [];
  }

  document.addEventListener('DOMContentLoaded', _init);

  return { handleFiles: _handleFiles };
})();
