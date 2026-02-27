/**
 * raiveFeeder — Corpus Management Module (Tab 5)
 *
 * Dashboard showing corpus statistics, source table, semantic search,
 * and CRUD operations (delete, export, import).
 */
const FeederCorpus = (() => {
  'use strict';

  let _sources = [];
  let _pendingItems = [];

  function _init() {
    const searchBtn = document.getElementById('corpus-search-btn');
    const exportBtn = document.getElementById('corpus-export-btn');
    const importInput = document.getElementById('corpus-import-input');
    const publishBtn = document.getElementById('corpus-publish-btn');

    if (searchBtn) searchBtn.addEventListener('click', _search);
    if (exportBtn) exportBtn.addEventListener('click', _exportCorpus);
    if (importInput) importInput.addEventListener('change', _importCorpus);
    if (publishBtn) publishBtn.addEventListener('click', _publishCorpus);

    // Approval queue bulk action buttons.
    const bulkApproveBtn = document.getElementById('approval-bulk-approve-btn');
    const bulkRejectBtn = document.getElementById('approval-bulk-reject-btn');
    if (bulkApproveBtn) bulkApproveBtn.addEventListener('click', _bulkApprove);
    if (bulkRejectBtn) bulkRejectBtn.addEventListener('click', _bulkReject);

    // Enter key in search input.
    const searchInput = document.getElementById('corpus-search-input');
    if (searchInput) {
      searchInput.addEventListener('keydown', e => {
        if (e.key === 'Enter') _search();
      });
    }
  }

  async function refresh() {
    await Promise.all([_loadStats(), _loadSources(), _loadPublishStatus(), _loadPending()]);
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

  async function _loadPublishStatus() {
    try {
      const status = await FeederApp.apiFetch('/corpus/publish/status');
      const container = document.getElementById('publish-status');
      const tagEl = document.getElementById('publish-latest-tag');
      const deployEl = document.getElementById('publish-deploy-indicator');
      const publishBtn = document.getElementById('corpus-publish-btn');

      if (!container) return;
      container.hidden = false;

      if (tagEl) tagEl.textContent = status.latest_tag || 'none';
      if (deployEl) {
        deployEl.textContent = status.deploy_hook_configured ? 'Auto-deploy enabled' : 'Manual deploy';
        deployEl.className = 'publish-status__deploy' + (status.deploy_hook_configured ? ' publish-status__deploy--active' : '');
      }
      if (publishBtn) {
        publishBtn.disabled = !status.github_token_set;
        if (!status.github_token_set) publishBtn.title = 'Set GITHUB_TOKEN to enable publishing';
      }
    } catch {
      // Publish status unavailable.
    }
  }

  async function _publishCorpus() {
    const latestTag = document.getElementById('publish-latest-tag')?.textContent || 'v1.0.0';
    const suggestedTag = _incrementTag(latestTag);
    const tag = prompt('Enter release tag:', suggestedTag);
    if (!tag) return;

    const publishBtn = document.getElementById('corpus-publish-btn');
    if (publishBtn) {
      publishBtn.disabled = true;
      publishBtn.textContent = 'Publishing...';
    }

    try {
      const result = await FeederApp.apiFetch('/corpus/publish', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tag }),
      });

      let msg = `Published ${result.tag} (${result.size_mb} MB)`;
      if (result.deploy_triggered) msg += '\nRender deploy triggered.';
      alert(msg);
      await _loadPublishStatus();
    } catch (err) {
      alert(`Publish failed: ${err.message}`);
    } finally {
      if (publishBtn) {
        publishBtn.disabled = false;
        publishBtn.textContent = 'Publish to GitHub';
      }
    }
  }

  function _incrementTag(tag) {
    // Auto-increment: v1.0.1 → v1.0.2
    const match = tag.match(/^(v?\d+\.\d+\.)(\d+)$/);
    if (match) return match[1] + (parseInt(match[2], 10) + 1);
    return tag;
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

  // ─── Approval Queue Methods ────────────────────────────────────────

  async function _loadPending() {
    try {
      _pendingItems = await FeederApp.apiFetch('/approval/pending');
      const section = document.getElementById('approval-section');
      const badge = document.getElementById('approval-count-badge');

      if (!section) return;

      if (_pendingItems.length > 0) {
        section.hidden = false;
        if (badge) badge.textContent = _pendingItems.length;
        _renderPendingTable();
      } else {
        section.hidden = true;
      }
    } catch {
      // Approval queue not available (no passcode set, local dev mode).
    }
  }

  function _renderPendingTable() {
    const table = document.getElementById('approval-table');
    if (!table) return;

    table.innerHTML = `
      <div class="approval-row approval-row--header">
        <span class="approval-row__check"><input type="checkbox" id="approval-select-all" title="Select all"></span>
        <span class="approval-row__title">Title</span>
        <span class="approval-row__type">Type</span>
        <span class="approval-row__content">Content</span>
        <span class="approval-row__date">Submitted</span>
        <span class="approval-row__actions">Actions</span>
      </div>
    ` + _pendingItems.map(item => `
      <div class="approval-row" data-id="${item.id}">
        <span class="approval-row__check"><input type="checkbox" class="approval-checkbox" value="${item.id}"></span>
        <span class="approval-row__title">${_escapeHtml(item.title)}</span>
        <span class="approval-row__type">${item.source_type}</span>
        <span class="approval-row__content">${item.content_type === 'url' ? _escapeHtml(item.content_data) : item.content_type}</span>
        <span class="approval-row__date">${new Date(item.submitted_at).toLocaleDateString()}</span>
        <span class="approval-row__actions">
          <button class="btn-sm btn-approve" data-action="approve" data-id="${item.id}">Approve</button>
          <button class="btn-sm btn-reject" data-action="reject" data-id="${item.id}">Reject</button>
        </span>
      </div>
    `).join('');

    // Select-all checkbox handler.
    const selectAll = document.getElementById('approval-select-all');
    if (selectAll) selectAll.addEventListener('change', _toggleSelectAll);

    // Individual checkbox handlers — update bulk button state.
    table.querySelectorAll('.approval-checkbox').forEach(cb => {
      cb.addEventListener('change', _updateBulkButtonState);
    });

    // Individual approve/reject handlers.
    table.querySelectorAll('[data-action="approve"]').forEach(btn => {
      btn.addEventListener('click', () => _approveItem(btn.dataset.id));
    });
    table.querySelectorAll('[data-action="reject"]').forEach(btn => {
      btn.addEventListener('click', () => _rejectItem(btn.dataset.id));
    });
  }

  function _toggleSelectAll() {
    const selectAll = document.getElementById('approval-select-all');
    const checked = selectAll ? selectAll.checked : false;
    document.querySelectorAll('.approval-checkbox').forEach(cb => { cb.checked = checked; });
    _updateBulkButtonState();
  }

  function _getSelectedIds() {
    return Array.from(document.querySelectorAll('.approval-checkbox:checked')).map(cb => cb.value);
  }

  function _updateBulkButtonState() {
    const selected = _getSelectedIds();
    const approveBtn = document.getElementById('approval-bulk-approve-btn');
    const rejectBtn = document.getElementById('approval-bulk-reject-btn');
    if (approveBtn) approveBtn.disabled = selected.length === 0;
    if (rejectBtn) rejectBtn.disabled = selected.length === 0;
  }

  async function _approveItem(id) {
    try {
      await FeederApp.apiFetch(`/approval/${encodeURIComponent(id)}/approve`, { method: 'POST' });
      await refresh();
    } catch (err) {
      alert(`Approve failed: ${err.message}`);
    }
  }

  async function _rejectItem(id) {
    const reason = prompt('Rejection reason (optional):') || '';
    try {
      await FeederApp.apiFetch(`/approval/${encodeURIComponent(id)}/reject`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason }),
      });
      await refresh();
    } catch (err) {
      alert(`Reject failed: ${err.message}`);
    }
  }

  async function _bulkApprove() {
    const ids = _getSelectedIds();
    if (ids.length === 0) return;
    if (!confirm(`Approve ${ids.length} item(s)?`)) return;
    try {
      await FeederApp.apiFetch('/approval/bulk-approve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ids }),
      });
      await refresh();
    } catch (err) {
      alert(`Bulk approve failed: ${err.message}`);
    }
  }

  async function _bulkReject() {
    const ids = _getSelectedIds();
    if (ids.length === 0) return;
    const reason = prompt('Rejection reason (optional):') || '';
    try {
      await FeederApp.apiFetch('/approval/bulk-reject', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ids, reason }),
      });
      await refresh();
    } catch (err) {
      alert(`Bulk reject failed: ${err.message}`);
    }
  }

  function _escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  document.addEventListener('DOMContentLoaded', _init);

  return { refresh };
})();
