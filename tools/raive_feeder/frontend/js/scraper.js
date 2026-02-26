/**
 * raiveFeeder — URL Scraping Module (Tab 4)
 *
 * Handles URL input, depth/pages sliders, NL query, scraping, and
 * selective page ingestion.
 */
const FeederScraper = (() => {
  'use strict';

  let _scrapedPages = [];

  function _init() {
    const scrapeBtn = document.getElementById('url-scrape-btn');
    const ingestBtn = document.getElementById('url-ingest-btn');
    const depthSlider = document.getElementById('url-depth');
    const pagesSlider = document.getElementById('url-pages');

    if (scrapeBtn) scrapeBtn.addEventListener('click', _scrape);
    if (ingestBtn) ingestBtn.addEventListener('click', _ingestSelected);

    // Slider value display.
    if (depthSlider) {
      depthSlider.addEventListener('input', () => {
        document.getElementById('url-depth-value').textContent = depthSlider.value;
      });
    }
    if (pagesSlider) {
      pagesSlider.addEventListener('input', () => {
        document.getElementById('url-pages-value').textContent = pagesSlider.value;
      });
    }
  }

  async function _scrape() {
    const url = document.getElementById('url-input')?.value;
    if (!url) return;

    const depth = parseInt(document.getElementById('url-depth')?.value) || 0;
    const maxPages = parseInt(document.getElementById('url-pages')?.value) || 1;
    const nlQuery = document.getElementById('url-query')?.value || null;

    const progress = document.getElementById('url-progress');
    const progressFill = document.getElementById('url-progress-fill');
    const progressText = document.getElementById('url-progress-text');

    if (progress) progress.hidden = false;
    if (progressFill) progressFill.style.width = '30%';
    if (progressText) progressText.textContent = 'Scraping...';

    try {
      const body = { url, max_depth: depth, max_pages: maxPages };
      if (nlQuery) body.nl_query = nlQuery;

      const resp = await FeederApp.apiFetch('/ingest/url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      _scrapedPages = resp.pages || [];

      if (progressFill) progressFill.style.width = '100%';
      if (progressText) progressText.textContent = `Found ${_scrapedPages.length} pages`;

      _renderPages();
    } catch (err) {
      if (progressText) progressText.textContent = `Error: ${err.message}`;
    }
  }

  function _renderPages() {
    const results = document.getElementById('url-results');
    const list = document.getElementById('url-page-list');
    if (!results || !list) return;

    results.hidden = false;
    list.innerHTML = _scrapedPages.map((p, i) => `
      <div class="crawl-item">
        <input type="checkbox" class="crawl-item__check" data-idx="${i}" checked>
        <div style="flex:1">
          <div class="crawl-item__title">${p.title || 'Untitled'}</div>
          <div class="crawl-item__url">${p.url}</div>
          <div class="crawl-item__preview">${(p.text_preview || '').substring(0, 200)}...</div>
        </div>
        ${p.relevance_score != null ? `<span class="crawl-item__score">${p.relevance_score.toFixed(1)}/10</span>` : ''}
      </div>
    `).join('');
  }

  async function _ingestSelected() {
    const checkboxes = document.querySelectorAll('.crawl-item__check:checked');
    const selectedUrls = Array.from(checkboxes).map(cb => _scrapedPages[parseInt(cb.dataset.idx)]?.url).filter(Boolean);

    const progressText = document.getElementById('url-progress-text');
    if (progressText) progressText.textContent = `Ingesting ${selectedUrls.length} pages...`;

    let successCount = 0;
    for (const url of selectedUrls) {
      try {
        await FeederApp.apiFetch('/ingest/url', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url, max_depth: 0, max_pages: 1, auto_ingest: true }),
        });
        successCount++;
      } catch {
        // Continue on failure.
      }
    }

    if (progressText) progressText.textContent = `Ingested ${successCount}/${selectedUrls.length} pages`;
  }

  document.addEventListener('DOMContentLoaded', _init);

  return {};
})();
