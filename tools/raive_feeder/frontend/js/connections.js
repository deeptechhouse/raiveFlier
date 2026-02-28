/**
 * raiveFeeder — Connections Tab Controller (Skeleton)
 *
 * Manages the combined connection map view: a force-directed graph of
 * entities (artists, venues, promoters) aggregated from all analyzed
 * fliers.  Uses vis-network for interactive graph visualization.
 *
 * Architecture:
 *   - Fetches combined graph data from GET /api/v1/connection-map
 *   - Renders via vis-network (loaded as vendor dependency)
 *   - Stats header shows aggregate counts
 *   - Search input highlights matching nodes
 *   - Click handler opens entity detail sidebar
 *
 * Entity type → color mapping (matches raiveFlier's graph colors):
 *   ARTIST   → Amethyst (#9b59b6)
 *   VENUE    → Teal     (#1abc9c)
 *   PROMOTER → Amber    (#f39c12)
 *   DEFAULT  → Muted    (#6a6a70)
 *
 * Pattern: Module (IIFE returning public API).
 * Depends on: FeederApp (app.js), vis-network (vendor).
 */
const FeederConnections = (() => {
  'use strict';

  // ─── Entity type → vis-network node color ─────────────────────────
  const TYPE_COLORS = {
    ARTIST:   { background: '#9b59b6', border: '#c39bd3', highlight: { background: '#af7ac5', border: '#d2b4de' } },
    VENUE:    { background: '#1abc9c', border: '#76d7c4', highlight: { background: '#48c9b0', border: '#a3e4d7' } },
    PROMOTER: { background: '#f39c12', border: '#f9e79f', highlight: { background: '#f5b041', border: '#fdebd0' } },
    DEFAULT:  { background: '#6a6a70', border: '#9a9a9e', highlight: { background: '#8a8a90', border: '#bababe' } },
  };

  // ─── State ────────────────────────────────────────────────────────
  let _network = null;       // vis-network instance
  let _graphData = null;     // Raw API response (CombinedConnectionMap)
  let _nodesDataset = null;  // vis-network DataSet for nodes
  let _edgesDataset = null;  // vis-network DataSet for edges
  let _initialized = false;

  // ─── DOM references (cached on first refresh) ─────────────────────
  let _els = {};

  function _cacheElements() {
    _els = {
      stats:         document.getElementById('connections-stats'),
      statFliers:    document.getElementById('conn-stat-fliers'),
      statEntities:  document.getElementById('conn-stat-entities'),
      statEdges:     document.getElementById('conn-stat-edges'),
      search:        document.getElementById('connections-search'),
      graph:         document.getElementById('connections-graph'),
      empty:         document.getElementById('connections-empty'),
      sidebar:       document.getElementById('connections-sidebar'),
      sidebarClose:  document.getElementById('connections-sidebar-close'),
      sidebarName:   document.getElementById('sidebar-entity-name'),
      sidebarType:   document.getElementById('sidebar-entity-type'),
      sidebarStats:  document.getElementById('sidebar-entity-stats'),
      sidebarEdges:  document.getElementById('sidebar-entity-edges'),
      sidebarFliers: document.getElementById('sidebar-entity-fliers'),
      layout:        document.querySelector('.connections-layout'),
    };
  }

  // ─── API calls ────────────────────────────────────────────────────

  async function _fetchCombinedMap() {
    try {
      return await FeederApp.apiFetch('/connection-map');
    } catch (err) {
      console.warn('Failed to fetch connection map:', err.message);
      return null;
    }
  }

  async function _fetchNodeDetail(name) {
    try {
      return await FeederApp.apiFetch(`/connection-map/node/${encodeURIComponent(name)}`);
    } catch (err) {
      console.warn('Failed to fetch node detail:', err.message);
      return null;
    }
  }

  // ─── Stats header ────────────────────────────────────────────────

  function _updateStats(data) {
    if (!data) return;
    _els.statFliers.textContent   = data.total_analyses || 0;
    _els.statEntities.textContent = (data.nodes || []).length;
    _els.statEdges.textContent    = (data.edges || []).length;
  }

  // ─── Graph rendering ─────────────────────────────────────────────

  function _getNodeColor(entityType) {
    return TYPE_COLORS[entityType] || TYPE_COLORS.DEFAULT;
  }

  function _buildVisData(data) {
    const nodes = (data.nodes || []).map((n, i) => ({
      id: i,
      label: n.name,
      title: `${n.entity_type}: ${n.name}\nAppearances: ${n.appearance_count}`,
      // Node size scales with appearance count (min 15, max 50)
      size: Math.min(50, Math.max(15, 10 + n.appearance_count * 5)),
      color: _getNodeColor(n.entity_type),
      font: { color: '#d8d5cd', face: 'Inter, sans-serif', size: 12 },
      // Store original data for click handler
      _data: n,
    }));

    // Build name → node id lookup for edge resolution
    const nameToId = {};
    (data.nodes || []).forEach((n, i) => { nameToId[n.name] = i; });

    const edges = (data.edges || []).filter(e =>
      nameToId[e.source] !== undefined && nameToId[e.target] !== undefined
    ).map(e => ({
      from: nameToId[e.source],
      to: nameToId[e.target],
      title: `${e.relationship_type}\nConfidence: ${(e.avg_confidence * 100).toFixed(0)}%\nSeen ${e.occurrence_count}x`,
      // Edge thickness scales with confidence × occurrence
      width: Math.max(1, Math.min(5, e.avg_confidence * e.occurrence_count * 2)),
      color: { color: 'rgba(154, 154, 158, 0.4)', highlight: 'rgba(80, 192, 128, 0.7)' },
      smooth: { type: 'continuous' },
      _data: e,
    }));

    return { nodes, edges };
  }

  function _renderGraph(data) {
    if (!data || !data.nodes || data.nodes.length === 0) {
      _els.empty.hidden = false;
      return;
    }

    _els.empty.hidden = true;

    const visData = _buildVisData(data);
    _nodesDataset = new vis.DataSet(visData.nodes);
    _edgesDataset = new vis.DataSet(visData.edges);

    const options = {
      physics: {
        solver: 'forceAtlas2Based',
        forceAtlas2Based: {
          gravitationalConstant: -80,
          centralGravity: 0.01,
          springLength: 120,
          springConstant: 0.05,
          damping: 0.4,
        },
        stabilization: { iterations: 150 },
      },
      interaction: {
        hover: true,
        tooltipDelay: 200,
        zoomView: true,
        dragView: true,
      },
      nodes: {
        shape: 'dot',
        borderWidth: 2,
        shadow: { enabled: true, size: 4, x: 0, y: 2 },
      },
      edges: {
        arrows: { to: { enabled: false } },
        selectionWidth: 2,
      },
    };

    // Destroy previous network if it exists
    if (_network) {
      _network.destroy();
      _network = null;
    }

    _network = new vis.Network(
      _els.graph,
      { nodes: _nodesDataset, edges: _edgesDataset },
      options,
    );

    // Click handler → entity detail sidebar
    _network.on('click', (params) => {
      if (params.nodes.length > 0) {
        const nodeId = params.nodes[0];
        const nodeData = _nodesDataset.get(nodeId);
        if (nodeData && nodeData._data) {
          _showEntityDetail(nodeData._data);
        }
      } else {
        _hideSidebar();
      }
    });
  }

  // ─── Entity detail sidebar ───────────────────────────────────────

  async function _showEntityDetail(nodeData) {
    // Show sidebar and update layout
    _els.sidebar.hidden = false;
    _els.layout.classList.add('has-sidebar');

    _els.sidebarName.textContent = nodeData.name;

    // Set type badge with color class
    const typeLower = (nodeData.entity_type || 'default').toLowerCase();
    _els.sidebarType.textContent = nodeData.entity_type;
    _els.sidebarType.className = `connections-sidebar__type connections-sidebar__type--${typeLower}`;

    // Stats
    _els.sidebarStats.innerHTML = `
      <p>Appears on <strong>${nodeData.appearance_count}</strong> flier(s)</p>
    `;

    // Fetch full detail from API
    const detail = await _fetchNodeDetail(nodeData.name);
    if (detail && detail.found) {
      // Render edges
      const edgesHtml = (detail.edges || []).map(e => {
        const other = e.source === nodeData.name ? e.target : e.source;
        return `<div class="sidebar-edge-item">
          <span class="sidebar-edge-item__type">${e.relationship_type}</span>
          <span class="sidebar-edge-item__target">${other}</span>
          <span class="sidebar-edge-item__confidence">${(e.avg_confidence * 100).toFixed(0)}%</span>
        </div>`;
      }).join('');
      _els.sidebarEdges.innerHTML = edgesHtml
        ? `<h4>Connections</h4>${edgesHtml}`
        : '<h4>Connections</h4><p style="color:var(--color-text-muted);font-size:var(--text-xs)">No connections found.</p>';

      // Render flier sessions
      const fliersHtml = (detail.source_sessions || []).map(sid =>
        `<div class="sidebar-flier-item">${sid.slice(0, 8)}...</div>`
      ).join('');
      _els.sidebarFliers.innerHTML = fliersHtml
        ? `<h4>Fliers</h4>${fliersHtml}`
        : '';
    }
  }

  function _hideSidebar() {
    _els.sidebar.hidden = true;
    _els.layout.classList.remove('has-sidebar');
  }

  // ─── Search / highlight ──────────────────────────────────────────

  function _initSearch() {
    if (!_els.search) return;

    let debounceTimer = null;
    _els.search.addEventListener('input', () => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        _handleSearch(_els.search.value.trim());
      }, 250);
    });
  }

  function _handleSearch(query) {
    if (!_network || !_nodesDataset || !_graphData) return;

    if (!query) {
      // Reset all nodes to original appearance
      _network.unselectAll();
      return;
    }

    const lowerQuery = query.toLowerCase();
    const matchingIds = [];

    _nodesDataset.forEach((node) => {
      if (node.label && node.label.toLowerCase().includes(lowerQuery)) {
        matchingIds.push(node.id);
      }
    });

    if (matchingIds.length > 0) {
      _network.selectNodes(matchingIds);
      if (matchingIds.length === 1) {
        _network.focus(matchingIds[0], { scale: 1.2, animation: true });
      }
    }
  }

  // ─── Sidebar close button ────────────────────────────────────────

  function _initSidebarClose() {
    if (_els.sidebarClose) {
      _els.sidebarClose.addEventListener('click', _hideSidebar);
    }
  }

  // ─── Public API ──────────────────────────────────────────────────

  async function refresh() {
    if (!_initialized) {
      _cacheElements();
      _initSearch();
      _initSidebarClose();
      _initialized = true;
    }

    _graphData = await _fetchCombinedMap();
    _updateStats(_graphData);
    _renderGraph(_graphData);
  }

  return {
    refresh,
  };
})();
