/**
 * raiveFeeder — Corpus Management Module (Tab 5)
 *
 * Dashboard showing corpus statistics, source table, semantic search,
 * and CRUD operations (delete, export, import).
 */
const FeederCorpus = (() => {
  'use strict';

  let _sources = [];

  function _init() {
    const searchBtn = document.getElementById('corpus-search-btn');
    const exportBtn = document.getElementById('corpus-export-btn');
    const importInput = document.getElementById('corpus-import-input');

    if (searchBtn) searchBtn.addEventListener('click', _search);
    if (exportBtn) exportBtn.addEventListener('click', _exportCorpus);
    if (importInput) importInput.addEventListener('change', _importCorpus);

    // Enter key in search input.
    const searchInput = document.getElementById('corpus-search-input');
    if (searchInput) {
      searchInput.addEventListener('keydown', e => {
        if (e.key === 'Enter') _search();
      });
    }
  }

  async function refresh() {
    await Promise.all([_loadStats(), _loadSources()]);
  }

  async function _loadStats() {
    try {
      const stats = await FeederApp.apiFetch('/corpus/stats');
      document.getElementById('stat-chunks').textContent = stats.total_chunks.toLocaleString();
      document.getElementById('stat-sources').textContent = stats.total_sources.toLocaleString();
      document.getElementById('stat-entities').textContent = stats.entity_tag_count.toLocaleString();
      document.getElementById('stat-genres').textContent = stats.genre_tag_count.toLocaleString();

      // Type breakdown chips.
      const breakdown = document.getElementById('type-breakdown');
      if (breakdown && stats.sources_by_type) {
        breakdown.innerHTML = Object.entries(stats.sources_by_type)
          .sort((a, b) => b[1] - a[1])
          .map(([type, count]) => `<span class="type-chip">${type}: ${count}</span>`)
          .join('');
      }
    } catch {
      // Stats unavailable.
    }
  }

  async function _loadSources() {
    try {
      _sources = await FeederApp.apiFetch('/corpus/sources');
      _renderSourceTable();
    } catch {
      // Sources unavailable.
    }
  }

  function _renderSourceTable() {
    const table = document.getElementById('corpus-source-table');
    if (!table) return;

    table.innerHTML = _sources.map(s => `
      <div class="source-row">
        <span class="source-row__title">${s.source_title || s.source_id.substring(0, 12)}</span>
        <span class="source-row__type">${s.source_type}</span>
        <span class="source-row__chunks">${s.chunk_count}</span>
        <span class="source-row__actions">
          <button class="btn-icon" data-action="delete" data-source-id="${s.source_id}" title="Delete">&times;</button>
        </span>
      </div>
    `).join('');

    // Attach delete handlers.
    table.querySelectorAll('[data-action="delete"]').forEach(btn => {
      btn.addEventListener('click', async () => {
        const sid = btn.dataset.sourceId;
        if (!confirm('Delete this source and all its chunks?')) return;
        try {
          await FeederApp.apiFetch(`/corpus/sources/${encodeURIComponent(sid)}`, { method: 'DELETE' });
          await refresh();
        } catch (err) {
          alert(`Delete failed: ${err.message}`);
        }
      });
    });
  }

  async function _search() {
    const input = document.getElementById('corpus-search-input');
    const query = input?.value?.trim();
    if (!query) return;

    try {
      const results = await FeederApp.apiFetch('/corpus/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: 20 }),
      });

      const container = document.getElementById('corpus-search-results');
      if (!container) return;
      container.hidden = false;
      container.innerHTML = results.map(r => `
        <div class="search-result">
          <div class="search-result__citation">${r.formatted_citation}</div>
          <div class="search-result__text">${r.text.substring(0, 400)}${r.text.length > 400 ? '...' : ''}</div>
          <div class="search-result__score">Similarity: ${(r.similarity_score * 100).toFixed(1)}%</div>
        </div>
      `).join('') || '<p style="color:var(--color-text-muted)">No results found</p>';
    } catch (err) {
      alert(`Search failed: ${err.message}`);
    }
  }

  async function _exportCorpus() {
    try {
      const resp = await FeederApp.apiFetch('/corpus/export', { method: 'POST' });
      alert(`Corpus exported to: ${resp.tarball_path}`);
    } catch (err) {
      alert(`Export failed: ${err.message}`);
    }
  }

  async function _importCorpus(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      await FeederApp.apiUpload('/corpus/import', formData);
      alert('Corpus imported successfully. Refresh to see changes.');
      await refresh();
    } catch (err) {
      alert(`Import failed: ${err.message}`);
    }
  }

  document.addEventListener('DOMContentLoaded', _init);

  return { refresh };
})();
