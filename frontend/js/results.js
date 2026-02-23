/**
 * results.js — raiveFlier results display module.
 *
 * ROLE IN THE APPLICATION
 * =======================
 * This is the largest and most complex frontend module. It renders the final
 * analysis results after the pipeline completes. Responsibilities:
 *
 *   1. Fetch completed results from /api/v1/fliers/{session_id}/results
 *   2. Normalize the API response (handles two possible JSON shapes)
 *   3. Render six content sections via HTML string building:
 *      - Event Summary: header with flier thumbnail and metadata tags
 *      - Artist Cards: expandable cards with releases, labels, external links
 *      - Venue & Promoter: expandable research cards
 *      - Event History: past instances of the same event series
 *      - Date & Context: scene, city, and cultural context panels
 *      - Interconnections: narrative prose + relationship list + SVG graph
 *   4. Initialize interactive behaviors after rendering:
 *      - Expandable card toggles (accordion pattern)
 *      - SVG graph hover/focus highlighting
 *      - Connection dismiss buttons (POST to dismiss-connection endpoint)
 *      - Q&A trigger buttons (open the QA drawer)
 *      - Rating widgets (thumbs up/down)
 *      - Recommendations panel initialization
 *      - JSON export functionality
 *
 * DATA NORMALIZATION
 * ==================
 * The backend can return data in two shapes:
 *   1. FlierAnalysisResponse (raw API format with research_results array)
 *   2. OutputFormatter shape (pre-normalized with research.artists array)
 * The _normalizeApiResponse() function detects which shape it received and
 * transforms it into a consistent display-ready format.
 *
 * DOM MANIPULATION PATTERN
 * ========================
 * All HTML is built as strings via template literals, then set via innerHTML.
 * After innerHTML, separate init functions attach event listeners using
 * addEventListener (not inline onclick) for better separation of concerns.
 * The exception is "Ask about this" buttons, which use the QA module directly.
 *
 * MODULE COMMUNICATION
 * ====================
 * - Called by Progress._onComplete() after pipeline finishes
 * - Calls App.showView("results") and App.getSessionId()
 * - Calls QA.openPanel() when "Ask about this" buttons are clicked
 * - Calls Rating.renderWidget() and Rating.initWidgets() for feedback UI
 * - Calls Recommendations.init() to start artist recommendation fetching
 */

"use strict";

const Results = (() => {
  // ------------------------------------------------------------------
  // Constants
  // ------------------------------------------------------------------

  // Human-readable names for the 6-tier citation hierarchy.
  // Tier 1 (published books) is most authoritative; tier 6 (community archives) is least.
  const TIER_NAMES = {
    1: "Published Books",
    2: "Press & Magazines",
    3: "Event Postings",
    4: "Database Records",
    5: "Web Articles",
    6: "Community Archives",
  };

  // Color mapping for SVG graph nodes — each entity type gets a distinct color
  // from the design system. These reference CSS custom properties for consistency.
  const NODE_COLORS = {
    ARTIST: "var(--color-accent)",       // Amethyst purple
    VENUE: "var(--color-verdigris)",      // Teal green
    PROMOTER: "var(--color-amber)",      // Gold amber
    LABEL: "var(--color-info)",          // Blue
    DATE: "var(--color-warning-text)",   // Amber text
  };

  // ------------------------------------------------------------------
  // Private state
  // ------------------------------------------------------------------

  // The raw (un-normalized) API response, kept for JSON export
  let _rawData = null;
  // The current session ID, used for API calls (dismiss, Q&A, ratings)
  let _sessionId = null;

  // ------------------------------------------------------------------
  // Utility helpers
  // ------------------------------------------------------------------

  /** Escape HTML special characters to prevent XSS. */
  const _ESC_MAP = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  const _ESC_RE = /[&<>"']/g;

  function _esc(str) {
    if (str == null) return "";
    return String(str).replace(_ESC_RE, (ch) => _ESC_MAP[ch]);
  }

  /** Map a numeric confidence score to a CSS class suffix. */
  function _confidenceClass(score) {
    if (score >= 0.8) return "high";
    if (score >= 0.5) return "medium";
    return "low";
  }

  /** Map a numeric confidence score to a human label. */
  function _confidenceLabel(score) {
    if (score >= 0.8) return "HIGH";
    if (score >= 0.5) return "MEDIUM";
    if (score >= 0.3) return "LOW";
    return "UNCERTAIN";
  }

  /** Get the CSS class for a citation tier. */
  function _tierClass(tier) {
    if (tier === 1) return "tier-gold";
    if (tier === 2) return "tier-silver";
    return "tier-plain";
  }

  /**
   * Format narrative text: escape HTML, convert [n] citation refs to
   * styled superscript marks, and split into paragraphs.
   */
  function _formatNarrative(text) {
    if (!text) return "";
    // Escape HTML first
    let safe = _esc(text);
    // Convert [n] references to superscript citation marks
    safe = safe.replace(
      /\[(\d+)\]/g,
      '<sup class="cite-ref" title="Source [$1]">[$1]</sup>'
    );
    // Split into paragraphs on double newlines
    const paragraphs = safe.split(/\n\s*\n/).filter((p) => p.trim());
    if (paragraphs.length <= 1) return `<p>${safe}</p>`;
    return paragraphs.map((p) => `<p>${p.trim()}</p>`).join("\n");
  }

  // ------------------------------------------------------------------
  // Data normaliser
  // ------------------------------------------------------------------

  /**
   * Transform the API response into a consistent display-ready format.
   *
   * WHY: The backend can return data in two different shapes depending on
   * whether the OutputFormatter service was used or the raw research results
   * are returned directly. This function detects which shape was received
   * and normalizes it so the rendering functions always work with the same
   * predictable structure.
   *
   * Output shape:
   *   { session_id, entities, research: { artists, venue, promoter, date_context },
   *     interconnections: { relationships, patterns, narrative, nodes },
   *     citations, completed_at }
   */
  function _normalizeApiResponse(raw) {
    // If already in OutputFormatter shape, augment and return
    if (raw.research && Array.isArray(raw.research.artists)) {
      return _augmentOutputFormatterShape(raw);
    }

    // Transform FlierAnalysisResponse to display format
    const artists = [];
    let venue = null;
    let promoter = null;
    let dateContext = null;

    let eventHistory = null;

    if (raw.research_results) {
      for (const result of raw.research_results) {
        if (result.artist) {
          artists.push(_normalizeArtist(result));
        }
        if (result.venue) {
          venue = _normalizeVenue(result.venue);
        }
        if (result.promoter) {
          promoter = _normalizePromoter(result.promoter);
        }
        if (result.date_context) {
          dateContext = _normalizeDateContext(result.date_context);
        }
        if (result.event_history) {
          eventHistory = _normalizeEventHistory(result.event_history);
        }
      }
    }

    // Normalise interconnection_map
    const imap = raw.interconnection_map || {};
    const relationships = (imap.edges || []).map((e) => ({
      source: e.source,
      target: e.target,
      type: e.relationship_type,
      details: e.details,
      citation:
        e.citations && e.citations.length > 0 ? e.citations[0].text : null,
      confidence: e.confidence || 0,
      dismissed: e.dismissed || false,
    }));

    const patterns = (imap.patterns || []).map((p) => ({
      type: p.pattern_type,
      description: p.description,
      entities: p.involved_entities || [],
    }));

    const nodes = (imap.nodes || []).map((n) => ({
      name: n.name,
      entity_type: n.entity_type,
    }));

    const citations = (imap.citations || []).map((c) => ({
      text: c.text,
      source: c.source_name,
      url: c.source_url,
      tier: c.tier || 6,
      accessible: null,
    }));

    // Build entities from extracted_entities
    const ent = raw.extracted_entities || {};

    return {
      session_id: raw.session_id,
      entities: {
        artists: (ent.artists || []).map((a) => ({
          name: a.text || a.name,
          confidence: a.confidence,
        })),
        venue: ent.venue
          ? { name: ent.venue.text || ent.venue.name }
          : null,
        date: ent.date ? { text: ent.date.text } : null,
        promoter: ent.promoter
          ? { name: ent.promoter.text || ent.promoter.name }
          : null,
        event_name: ent.event_name
          ? { name: ent.event_name.text || ent.event_name.name }
          : null,
      },
      research: { artists, venue, promoter, date_context: dateContext, event_history: eventHistory },
      interconnections: {
        relationships,
        patterns,
        narrative: imap.narrative || null,
        nodes,
      },
      citations,
      completed_at: raw.completed_at,
    };
  }

  function _normalizeArtist(result) {
    const a = result.artist;
    const labels = [
      ...new Set((a.labels || []).map((l) => (typeof l === "string" ? l : l.name))),
    ];

    const labelObjects = (a.labels || []).map((l) => {
      if (typeof l === "string") return { name: l, discogs_url: null };
      return {
        name: l.name,
        discogs_url:
          l.discogs_url ||
          (l.discogs_id
            ? `https://www.discogs.com/label/${l.discogs_id}`
            : null),
      };
    });

    return {
      name: a.name,
      profile_summary: a.profile_summary || null,
      discogs_url: a.discogs_id
        ? `https://www.discogs.com/artist/${a.discogs_id}`
        : null,
      musicbrainz_url: a.musicbrainz_id
        ? `https://musicbrainz.org/artist/${a.musicbrainz_id}`
        : null,
      bandcamp_url: a.bandcamp_url || null,
      beatport_url: a.beatport_url || null,
      releases_count: (a.releases || []).length,
      releases: (a.releases || []).map((r) => ({
        title: r.title,
        label: r.label,
        year: r.year,
        format: r.format,
        discogs_url: r.discogs_url,
        bandcamp_url: r.bandcamp_url || null,
        beatport_url: r.beatport_url || null,
      })),
      labels,
      label_objects: labelObjects,
      confidence: result.confidence || a.confidence || 0,
    };
  }

  function _normalizeVenue(v) {
    return {
      name: v.name,
      history: v.history,
      notable_events: v.notable_events || [],
      cultural_significance: v.cultural_significance,
      articles: (v.articles || []).map((ar) => ({
        title: ar.title,
        source: ar.source,
        url: ar.url,
      })),
    };
  }

  function _normalizePromoter(p) {
    return {
      name: p.name,
      event_history: p.event_history || [],
      affiliated_artists: p.affiliated_artists || [],
      affiliated_venues: p.affiliated_venues || [],
      articles: (p.articles || []).map((ar) => ({
        title: ar.title,
        source: ar.source,
        url: ar.url,
      })),
    };
  }

  function _normalizeDateContext(dc) {
    return {
      scene: dc.scene_context || dc.scene || null,
      city: dc.city_context || dc.city || null,
      cultural: dc.cultural_context || dc.cultural || null,
      nearby_events: dc.nearby_events || [],
    };
  }

  function _normalizeEventHistory(eh) {
    return {
      event_name: eh.event_name || "Unknown",
      instances: (eh.instances || []).map((i) => ({
        event_name: i.event_name,
        promoter: i.promoter,
        venue: i.venue,
        city: i.city,
        date: i.date,
        source_url: i.source_url,
      })),
      promoter_groups: eh.promoter_groups || {},
      promoter_name_changes: eh.promoter_name_changes || [],
      total_found: eh.total_found || 0,
      articles: (eh.articles || []).map((ar) => ({
        title: ar.title,
        source: ar.source,
        url: ar.url,
        tier: ar.citation_tier || 6,
      })),
    };
  }

  /** Augment OutputFormatter-shaped data with derived nodes. */
  function _augmentOutputFormatterShape(data) {
    const ic = data.interconnections || {};
    if (!ic.nodes || ic.nodes.length === 0) {
      ic.nodes = _deriveNodes(
        ic.relationships || [],
        data.entities,
        data.research
      );
      data.interconnections = ic;
    }
    // Ensure date_context has nearby_events
    if (data.research && data.research.date_context) {
      const dc = data.research.date_context;
      dc.nearby_events = dc.nearby_events || [];
    }
    return data;
  }

  /** Derive graph nodes from entities and relationships. */
  function _deriveNodes(relationships, entities, research) {
    const nodeMap = {};

    if (research && research.artists) {
      research.artists.forEach((a) => {
        nodeMap[a.name] = { name: a.name, entity_type: "ARTIST" };
      });
    }
    if (research && research.venue) {
      nodeMap[research.venue.name] = {
        name: research.venue.name,
        entity_type: "VENUE",
      };
    }
    if (research && research.promoter) {
      nodeMap[research.promoter.name] = {
        name: research.promoter.name,
        entity_type: "PROMOTER",
      };
    }

    relationships.forEach((r) => {
      if (!nodeMap[r.source]) {
        nodeMap[r.source] = { name: r.source, entity_type: "UNKNOWN" };
      }
      if (!nodeMap[r.target]) {
        nodeMap[r.target] = { name: r.target, entity_type: "UNKNOWN" };
      }
    });

    return Object.values(nodeMap);
  }

  // ------------------------------------------------------------------
  // Q&A Trigger button helper
  // ------------------------------------------------------------------

  /** Render an "Ask about this" button for the Q&A drawer. */
  function _renderQATrigger(entityType, entityName) {
    return `<button type="button" class="qa-trigger" data-entity-type="${_esc(entityType)}" data-entity-name="${_esc(entityName)}">` +
      '<svg class="qa-trigger__icon" width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">' +
      '<path d="M7 1C3.686 1 1 3.686 1 7c0 1.21.36 2.34.976 3.282L1 13l2.718-.976A5.975 5.975 0 007 13c3.314 0 6-2.686 6-6s-2.686-6-6-6z" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/>' +
      '<path d="M5.5 5.5a1.5 1.5 0 112.565 1.06L7 7.5V8" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/>' +
      '<circle cx="7" cy="10" r="0.5" fill="currentColor"/>' +
      "</svg> Ask about this</button>";
  }

  // ------------------------------------------------------------------
  // Section 1: Event Summary Header
  // ------------------------------------------------------------------

  function _renderEventSummary(data) {
    const entities = data.entities || {};
    const research = data.research || {};
    const artistCount = (research.artists || []).length;
    const citationCount = (data.citations || []).length;
    const venueName = entities.venue ? _esc(entities.venue.name) : "Unknown venue";
    const promoterName = entities.promoter ? _esc(entities.promoter.name) : null;
    const eventName = entities.event_name ? _esc(entities.event_name.name || entities.event_name.text) : null;
    const dateText = entities.date ? _esc(entities.date.text) : null;

    let html = '<div class="results-summary">';

    // Flier thumbnail (if available from upload)
    const previewImg = document.getElementById("preview-image");
    if (previewImg && previewImg.src) {
      html += `<div class="results-summary__thumb">
        <img src="${_esc(previewImg.src)}" alt="Flier thumbnail" class="results-summary__thumb-img">
      </div>`;
    }

    html += '<div class="results-summary__info">';
    html += '<h2 class="text-heading results-summary__title">Analysis Complete</h2>';
    html += '<div class="results-summary__meta">';

    if (eventName) {
      html += `<span class="results-summary__tag"><span class="results-summary__tag-label">Event</span> ${eventName}</span>`;
    }
    if (dateText) {
      html += `<span class="results-summary__tag"><span class="results-summary__tag-label">Date</span> ${dateText}</span>`;
    }
    html += `<span class="results-summary__tag"><span class="results-summary__tag-label">Venue</span> ${venueName}</span>`;
    if (promoterName) {
      html += `<span class="results-summary__tag"><span class="results-summary__tag-label">Promoter</span> ${promoterName}</span>`;
    }

    html += "</div>"; // .meta
    html += '<div class="results-summary__stats">';
    html += `<span class="results-summary__stat">${artistCount} artist${artistCount !== 1 ? "s" : ""} researched</span>`;
    // Citation count logged, not displayed
    if (citationCount > 0) {
      console.log("[Results] %d citation(s) found", citationCount);
    }
    html += "</div>";
    html += "</div>"; // .info
    html += "</div>"; // .results-summary

    return html;
  }

  // ------------------------------------------------------------------
  // Section 2: Artist Cards
  // ------------------------------------------------------------------

  function _renderArtistCards(artists) {
    if (!artists || artists.length === 0) {
      return '<p class="text-caption">No artist research data available.</p>';
    }

    let html =
      '<div class="results-section" id="results-artists">' +
      '<h2 class="results-section__title text-heading">Artists</h2>' +
      '<div class="artist-cards">';

    artists.forEach((artist) => {
      html += _renderArtistCard(artist);
    });

    html += "</div></div>";
    return html;
  }

  function _renderArtistCard(artist) {
    const confClass = _confidenceClass(artist.confidence);
    const confLabel = _confidenceLabel(artist.confidence);
    const releaseCount = artist.releases_count || (artist.releases || []).length;
    const labelCount = (artist.labels || []).length;
    let html = '<article class="artist-card expandable">';

    // Trigger / header
    html += '<button class="expandable__trigger" type="button" aria-expanded="false">';
    html += '<div class="artist-card__title-row">';
    html += `<h3 class="artist-card__name">${_esc(artist.name)}</h3>`;
    html += `<span class="confidence-badge confidence-${confClass}">${confLabel}</span>`;
    html += "</div>";
    html += '<div class="artist-card__meta">';
    html += `<span class="artist-card__stat">${releaseCount} release${releaseCount !== 1 ? "s" : ""}</span>`;
    html += `<span class="artist-card__stat">${labelCount} label${labelCount !== 1 ? "s" : ""}</span>`;
    html += "</div>";
    html += '<svg class="expandable__chevron" width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">';
    html += '<path d="M4 6L8 10L12 6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>';
    html += "</svg>";
    html += "</button>";

    // Expandable content
    html += '<div class="expandable__content">';

    // External links
    if (artist.discogs_url || artist.musicbrainz_url || artist.bandcamp_url || artist.beatport_url) {
      html += '<div class="artist-card__links">';
      if (artist.discogs_url) {
        html += `<a href="${_esc(artist.discogs_url)}" target="_blank" rel="noopener noreferrer" class="artist-card__link">View on Discogs &rarr;</a>`;
      }
      if (artist.musicbrainz_url) {
        html += `<a href="${_esc(artist.musicbrainz_url)}" target="_blank" rel="noopener noreferrer" class="artist-card__link">View on MusicBrainz &rarr;</a>`;
      }
      if (artist.bandcamp_url) {
        html += `<a href="${_esc(artist.bandcamp_url)}" target="_blank" rel="noopener noreferrer" class="artist-card__link">View on Bandcamp &rarr;</a>`;
      }
      if (artist.beatport_url) {
        html += `<a href="${_esc(artist.beatport_url)}" target="_blank" rel="noopener noreferrer" class="artist-card__link">View on Beatport &rarr;</a>`;
      }
      html += "</div>";
    }

    // Profile summary
    if (artist.profile_summary) {
      html += `<div class="artist-card__profile"><p>${_esc(artist.profile_summary)}</p></div>`;
    }

    // Releases subsection
    html += _renderArtistReleases(artist);

    // Labels subsection
    html += _renderArtistLabels(artist);


    // Q&A trigger + rating
    html += _renderQATrigger("ARTIST", artist.name);
    if (typeof Rating !== "undefined") {
      html += Rating.renderWidget("ARTIST", artist.name);
    }

    html += "</div>"; // .expandable__content
    html += "</article>";

    return html;
  }

  function _renderArtistReleases(artist) {
    const releases = artist.releases || [];
    const count = artist.releases_count || releases.length;

    if (count === 0) return "";

    let html = '<div class="artist-card__subsection">';
    html += `<h4 class="artist-card__subsection-title">Releases (${count})</h4>`;

    if (releases.length > 0) {
      html += '<ul class="artist-card__list">';
      releases.forEach((r) => {
        const parts = [_esc(r.title)];
        if (r.label) parts.push(_esc(r.label));
        if (r.year) parts.push(String(r.year));
        if (r.format) parts.push(_esc(r.format));

        const itemKey = artist.name + "::release::" + r.title;

        let li = `<li class="artist-card__list-item artist-card__list-item--rated">`;
        li += `<span class="artist-card__list-item-text">${parts.join(" &mdash; ")}`;
        if (r.discogs_url) {
          li += ` <a href="${_esc(r.discogs_url)}" target="_blank" rel="noopener noreferrer" class="artist-card__ext-link" aria-label="View on Discogs">&#x2197;</a>`;
        }
        if (r.bandcamp_url) {
          li += ` <a href="${_esc(r.bandcamp_url)}" target="_blank" rel="noopener noreferrer" class="artist-card__ext-link" aria-label="View on Bandcamp">BC</a>`;
        }
        if (r.beatport_url) {
          li += ` <a href="${_esc(r.beatport_url)}" target="_blank" rel="noopener noreferrer" class="artist-card__ext-link" aria-label="View on Beatport">BP</a>`;
        }
        li += "</span>";
        if (typeof Rating !== "undefined") {
          li += Rating.renderWidget("RELEASE", itemKey);
        }
        li += "</li>";
        html += li;
      });
      html += "</ul>";
    } else {
      html += `<p class="text-caption">${count} release${count !== 1 ? "s" : ""} found (details not loaded)</p>`;
    }

    html += "</div>";
    return html;
  }

  function _renderArtistLabels(artist) {
    const labels = artist.label_objects || [];
    const labelNames = artist.labels || [];

    if (labelNames.length === 0 && labels.length === 0) return "";

    let html = '<div class="artist-card__subsection">';
    html += '<h4 class="artist-card__subsection-title">Labels</h4>';
    html += '<ul class="artist-card__list">';

    if (labels.length > 0) {
      labels.forEach((l) => {
        const itemKey = artist.name + "::label::" + l.name;

        let li = `<li class="artist-card__list-item artist-card__list-item--rated">`;
        li += `<span class="artist-card__list-item-text">${_esc(l.name)}`;
        if (l.discogs_url) {
          li += ` <a href="${_esc(l.discogs_url)}" target="_blank" rel="noopener noreferrer" class="artist-card__ext-link" aria-label="View ${_esc(l.name)} on Discogs">&#x2197;</a>`;
        }
        li += "</span>";
        if (typeof Rating !== "undefined") {
          li += Rating.renderWidget("LABEL", itemKey);
        }
        li += "</li>";
        html += li;
      });
    } else {
      labelNames.forEach((name) => {
        const itemKey = artist.name + "::label::" + name;

        let li = `<li class="artist-card__list-item artist-card__list-item--rated">`;
        li += `<span class="artist-card__list-item-text">${_esc(name)}</span>`;
        if (typeof Rating !== "undefined") {
          li += Rating.renderWidget("LABEL", itemKey);
        }
        li += "</li>";
        html += li;
      });
    }

    html += "</ul></div>";
    return html;
  }

  function _renderArtistAppearances(artist) {
    const appearances = artist.appearances || [];
    if (appearances.length === 0) return "";

    let html = '<div class="artist-card__subsection">';
    html += `<h4 class="artist-card__subsection-title">Past Appearances (${appearances.length})</h4>`;
    html += '<ul class="artist-card__list">';

    appearances.forEach((a) => {
      const parts = [];
      if (a.event) parts.push(_esc(a.event));
      if (a.venue) parts.push(_esc(a.venue));
      if (a.date) parts.push(_esc(a.date));
      html += `<li class="artist-card__list-item">${parts.join(" &mdash; ") || "Unknown event"}</li>`;
    });

    html += "</ul></div>";
    return html;
  }

  function _renderArtistArticles(artist) {
    const articles = artist.articles || [];
    if (articles.length === 0) return "";

    let html = '<div class="artist-card__subsection">';
    html += `<h4 class="artist-card__subsection-title">Press &amp; Articles (${articles.length})</h4>`;
    html += '<ul class="artist-card__article-list">';

    articles.forEach((a) => {
      const tierCls = _tierClass(a.tier);
      const tierName = TIER_NAMES[a.tier] || "Source";
      let li = `<li class="artist-card__article-item ${tierCls}">`;
      li += `<span class="citation-tier-badge citation-tier-${a.tier}">${_esc(tierName)}</span> `;
      if (a.url) {
        li += `<a href="${_esc(a.url)}" target="_blank" rel="noopener noreferrer">${_esc(a.title || a.source)}</a>`;
      } else {
        li += _esc(a.title || a.source);
      }
      if (a.source && a.title) {
        li += ` <span class="text-caption">${_esc(a.source)}</span>`;
      }
      li += "</li>";
      html += li;
    });

    html += "</ul></div>";
    return html;
  }

  // ------------------------------------------------------------------
  // Section 3: Venue & Promoter
  // ------------------------------------------------------------------

  function _renderVenuePromoter(venue, promoter) {
    if (!venue && !promoter) return "";

    let html = '<div class="results-section" id="results-venue-promoter">';
    html += '<h2 class="results-section__title text-heading">Venue &amp; Promoter</h2>';
    html += '<div class="venue-promoter-grid">';

    if (venue) {
      html += '<article class="venue-card expandable">';
      html += '<button class="expandable__trigger" type="button" aria-expanded="false">';
      html += `<h3 class="venue-card__name">${_esc(venue.name)}</h3>`;
      html += '<svg class="expandable__chevron" width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">';
      html += '<path d="M4 6L8 10L12 6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>';
      html += "</svg>";
      html += "</button>";
      html += '<div class="expandable__content">';

      if (venue.history) {
        html += `<div class="venue-card__section"><h4>History</h4><p>${_esc(venue.history)}</p></div>`;
      }
      if (venue.notable_events && venue.notable_events.length > 0) {
        html += '<div class="venue-card__section"><h4>Notable Events</h4><ul>';
        venue.notable_events.forEach((e) => {
          html += `<li>${_esc(e)}</li>`;
        });
        html += "</ul></div>";
      }
      if (venue.cultural_significance) {
        html += `<div class="venue-card__section"><h4>Cultural Significance</h4><p>${_esc(venue.cultural_significance)}</p></div>`;
      }
      if (venue.articles && venue.articles.length > 0) {
        html += '<div class="venue-card__section"><h4>Articles</h4><ul>';
        venue.articles.forEach((a) => {
          if (a.url) {
            html += `<li><a href="${_esc(a.url)}" target="_blank" rel="noopener noreferrer">${_esc(a.title)}</a> <span class="text-caption">${_esc(a.source)}</span></li>`;
          } else {
            html += `<li>${_esc(a.title)} <span class="text-caption">${_esc(a.source)}</span></li>`;
          }
        });
        html += "</ul></div>";
      }

      // Q&A trigger + rating
      html += _renderQATrigger("VENUE", venue.name);
      if (typeof Rating !== "undefined") {
        html += Rating.renderWidget("VENUE", venue.name);
      }

      html += "</div></article>"; // expandable__content + venue-card
    }

    if (promoter) {
      html += '<article class="promoter-card expandable">';
      html += '<button class="expandable__trigger" type="button" aria-expanded="false">';
      html += `<h3 class="promoter-card__name">${_esc(promoter.name)}</h3>`;
      html += '<svg class="expandable__chevron" width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">';
      html += '<path d="M4 6L8 10L12 6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>';
      html += "</svg>";
      html += "</button>";
      html += '<div class="expandable__content">';

      if (promoter.event_history && promoter.event_history.length > 0) {
        html += '<div class="promoter-card__section"><h4>Event History</h4><ul>';
        promoter.event_history.forEach((e) => {
          html += `<li>${_esc(e)}</li>`;
        });
        html += "</ul></div>";
      }
      if (promoter.affiliated_artists && promoter.affiliated_artists.length > 0) {
        html += '<div class="promoter-card__section"><h4>Affiliated Artists</h4><ul>';
        promoter.affiliated_artists.forEach((a) => {
          html += `<li>${_esc(a)}</li>`;
        });
        html += "</ul></div>";
      }
      if (promoter.affiliated_venues && promoter.affiliated_venues.length > 0) {
        html += '<div class="promoter-card__section"><h4>Affiliated Venues</h4><ul>';
        promoter.affiliated_venues.forEach((v) => {
          html += `<li>${_esc(v)}</li>`;
        });
        html += "</ul></div>";
      }
      if (promoter.articles && promoter.articles.length > 0) {
        html += '<div class="promoter-card__section"><h4>Articles</h4><ul>';
        promoter.articles.forEach((a) => {
          if (a.url) {
            html += `<li><a href="${_esc(a.url)}" target="_blank" rel="noopener noreferrer">${_esc(a.title)}</a> <span class="text-caption">${_esc(a.source)}</span></li>`;
          } else {
            html += `<li>${_esc(a.title)} <span class="text-caption">${_esc(a.source)}</span></li>`;
          }
        });
        html += "</ul></div>";
      }

      // Q&A trigger + rating
      html += _renderQATrigger("PROMOTER", promoter.name);
      if (typeof Rating !== "undefined") {
        html += Rating.renderWidget("PROMOTER", promoter.name);
      }

      html += "</div></article>"; // expandable__content + promoter-card
    }

    html += "</div></div>"; // venue-promoter-grid + results-section
    return html;
  }

  // ------------------------------------------------------------------
  // Section 3b: Event History
  // ------------------------------------------------------------------

  function _renderEventHistory(eventHistory) {
    if (!eventHistory || eventHistory.total_found === 0) return "";

    let html = '<div class="results-section" id="results-event-history">';
    html += '<h2 class="results-section__title text-heading">Event History</h2>';

    html += `<div class="event-history__header">`;
    html += `<h3 class="event-history__name">${_esc(eventHistory.event_name)}</h3>`;
    html += `<span class="event-history__count text-caption">${eventHistory.total_found} instance${eventHistory.total_found !== 1 ? "s" : ""} found</span>`;
    if (typeof Rating !== "undefined") {
      html += Rating.renderWidget("EVENT", eventHistory.event_name);
    }
    html += "</div>";

    // Promoter name change callout
    if (eventHistory.promoter_name_changes && eventHistory.promoter_name_changes.length > 0) {
      html += '<div class="event-history__callout">';
      html += '<h4 class="event-history__callout-title">Promoter Name Changes Detected</h4>';
      html += "<ul>";
      eventHistory.promoter_name_changes.forEach((change) => {
        html += `<li>${_esc(change)}</li>`;
      });
      html += "</ul></div>";
    }

    // Promoter groups
    const groups = eventHistory.promoter_groups || {};
    const groupNames = Object.keys(groups);

    if (groupNames.length > 0) {
      html += '<div class="event-history__groups">';
      html += `<h4 class="results-subsection__title">By Promoter (${groupNames.length} promoter${groupNames.length !== 1 ? "s" : ""})</h4>`;

      groupNames.forEach((pname) => {
        const instances = groups[pname] || [];
        html += '<article class="event-history__group expandable">';
        html += '<button class="expandable__trigger" type="button" aria-expanded="false">';
        html += `<span class="event-history__group-name">${_esc(pname)}</span>`;
        html += `<span class="event-history__group-count text-caption">${instances.length} event${instances.length !== 1 ? "s" : ""}</span>`;
        html += '<svg class="expandable__chevron" width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">';
        html += '<path d="M4 6L8 10L12 6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>';
        html += "</svg>";
        html += "</button>";
        html += '<div class="expandable__content">';
        html += '<ul class="event-history__instance-list">';

        instances.forEach((inst) => {
          const parts = [];
          if (inst.venue) parts.push(_esc(inst.venue));
          if (inst.city) parts.push(_esc(inst.city));
          if (inst.date) parts.push(_esc(inst.date));
          html += `<li class="event-history__instance">${parts.join(" &mdash; ") || "Details unknown"}</li>`;
        });

        html += "</ul></div></article>";
      });

      html += "</div>";
    }

    // Articles
    if (eventHistory.articles && eventHistory.articles.length > 0) {
      html += '<div class="event-history__articles">';
      html += `<h4 class="results-subsection__title">Sources (${eventHistory.articles.length})</h4>`;
      html += '<ul class="artist-card__article-list">';

      eventHistory.articles.forEach((a) => {
        const tierCls = _tierClass(a.tier);
        const tierName = TIER_NAMES[a.tier] || "Source";
        let li = `<li class="artist-card__article-item ${tierCls}">`;
        li += `<span class="citation-tier-badge citation-tier-${a.tier}">${_esc(tierName)}</span> `;
        if (a.url) {
          li += `<a href="${_esc(a.url)}" target="_blank" rel="noopener noreferrer">${_esc(a.title || a.source)}</a>`;
        } else {
          li += _esc(a.title || a.source);
        }
        if (a.source && a.title) {
          li += ` <span class="text-caption">${_esc(a.source)}</span>`;
        }
        li += "</li>";
        html += li;
      });

      html += "</ul></div>";
    }

    html += "</div>"; // results-section
    return html;
  }

  // ------------------------------------------------------------------
  // Section 4: Date Context
  // ------------------------------------------------------------------

  function _renderDateContext(dateContext) {
    if (!dateContext) return "";

    const hasContent =
      dateContext.scene ||
      dateContext.city ||
      dateContext.cultural ||
      (dateContext.nearby_events && dateContext.nearby_events.length > 0);

    if (!hasContent) return "";

    let html = '<div class="results-section" id="results-date-context">';
    html += '<h2 class="results-section__title text-heading">Date &amp; Context</h2>';
    html += '<div class="context-panels">';

    if (dateContext.scene) {
      html += '<div class="context-panel">';
      html += '<h4 class="context-panel__title">Scene Context</h4>';
      html += `<p>${_esc(dateContext.scene)}</p>`;
      html += "</div>";
    }

    if (dateContext.city) {
      html += '<div class="context-panel">';
      html += '<h4 class="context-panel__title">City Context</h4>';
      html += `<p>${_esc(dateContext.city)}</p>`;
      html += "</div>";
    }

    if (dateContext.cultural) {
      html += '<div class="context-panel">';
      html += '<h4 class="context-panel__title">Cultural Context</h4>';
      html += `<p>${_esc(dateContext.cultural)}</p>`;
      html += "</div>";
    }

    if (dateContext.nearby_events && dateContext.nearby_events.length > 0) {
      html += '<div class="context-panel">';
      html += '<h4 class="context-panel__title">Nearby Events</h4>';
      html += "<ul>";
      dateContext.nearby_events.forEach((e) => {
        html += `<li>${_esc(e)}</li>`;
      });
      html += "</ul></div>";
    }

    // Q&A trigger + rating for date context
    html += _renderQATrigger("DATE", "this event's date and context");
    if (typeof Rating !== "undefined") {
      html += Rating.renderWidget("DATE", "date_context");
    }

    html += "</div></div>"; // context-panels + results-section
    return html;
  }

  // ------------------------------------------------------------------
  // Section 5: Interconnections
  // ------------------------------------------------------------------

  function _renderInterconnections(data) {
    const ic = data.interconnections || {};
    const relationships = ic.relationships || [];
    const patterns = ic.patterns || [];
    const narrative = ic.narrative;
    const nodes = ic.nodes || [];

    if (!narrative && relationships.length === 0 && patterns.length === 0) {
      return "";
    }

    let html = '<div class="results-section" id="results-interconnections">';
    html += '<h2 class="results-section__title text-heading">Interconnections</h2>';

    // Factual chronicle
    if (narrative) {
      html += '<div class="narrative-prose">';
      html += _formatNarrative(narrative);
      html += "</div>";
    }

    // Relationship list
    if (relationships.length > 0) {
      html += '<div class="interconnections__relationships">';
      html += `<h3 class="results-subsection__title">Relationships (${relationships.length})</h3>`;
      html += '<ul class="relationship-list">';

      relationships.forEach((rel) => {
        if (rel.dismissed) return;
        const confPct = Math.round(rel.confidence * 100);
        html += `<li class="relationship-item" data-source="${_esc(rel.source)}" data-target="${_esc(rel.target)}" data-type="${_esc(rel.type)}">`;
        html += `<span class="relationship-item__edge">`;
        html += `<strong>${_esc(rel.source)}</strong>`;
        html += ` <span class="relationship-item__type">&mdash;[${_esc(rel.type)}]&rarr;</span> `;
        html += `<strong>${_esc(rel.target)}</strong>`;
        html += "</span>";
        html += `<span class="confidence-badge confidence-${_confidenceClass(rel.confidence)}">${confPct}%</span>`;
        html += `<button type="button" class="relationship-item__dismiss" aria-label="Dismiss connection: ${_esc(rel.source)} to ${_esc(rel.target)}" title="Dismiss incorrect connection">&times;</button>`;
        if (typeof Rating !== "undefined") {
          html += Rating.renderWidget("CONNECTION", rel.source + "|" + rel.target + "|" + rel.type);
        }
        if (rel.citation) {
          html += ` <span class="relationship-item__citation text-caption">${_esc(rel.citation)}</span>`;
        }
        html += "</li>";
      });

      html += "</ul></div>";
    }

    // Pattern insights
    if (patterns.length > 0) {
      html += '<div class="interconnections__patterns">';
      html += '<h3 class="results-subsection__title">Pattern Insights</h3>';
      html += '<div class="pattern-cards">';

      patterns.forEach((p) => {
        html += '<div class="pattern-card">';
        html += `<span class="pattern-card__type text-caption">${_esc(p.type)}</span>`;
        html += `<p class="pattern-card__desc">${_esc(p.description)}</p>`;
        if (p.entities && p.entities.length > 0) {
          html += '<div class="pattern-card__entities">';
          p.entities.forEach((e) => {
            html += `<span class="pattern-card__entity">${_esc(e)}</span>`;
          });
          html += "</div>";
        }
        if (typeof Rating !== "undefined") {
          const descKey = (p.type || "") + "::" + (p.description || "").substring(0, 60);
          html += Rating.renderWidget("PATTERN", descKey);
        }
        html += "</div>";
      });

      html += "</div></div>";
    }

    // Relationship graph (SVG)
    if (nodes.length > 0 && relationships.length > 0) {
      html += '<div class="interconnections__graph">';
      html += '<h3 class="results-subsection__title">Relationship Map</h3>';
      html += renderRelationshipGraph(relationships, nodes);
      html += "</div>";
    }

    html += "</div>"; // results-section
    return html;
  }

  // ------------------------------------------------------------------
  // Relationship graph (SVG)
  // ------------------------------------------------------------------

  /**
   * Render an entity relationship graph as an SVG element.
   *
   * LAYOUT: Nodes are placed in a circular arrangement (not force-directed,
   * despite the comment — this is a simpler circular layout). Each node is
   * positioned at equal angular intervals around a circle centered in the SVG.
   *
   * VISUAL ENCODING:
   *   - Node color = entity type (artist=amethyst, venue=teal, promoter=amber)
   *   - Edge thickness = confidence score (higher confidence = thicker line)
   *   - Labels are truncated to 14 characters to prevent overlap
   *
   * INTERACTIVITY: After rendering, _initGraphInteractions() adds hover/focus
   * handlers that highlight connected edges and dim unrelated ones.
   */
  function renderRelationshipGraph(relationships, nodes) {
    if (!nodes || nodes.length === 0) return "";

    const width = 700;
    const height = 500;
    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.min(cx, cy) - 80;

    // Position nodes in a circle
    const positions = {};
    nodes.forEach((node, i) => {
      const angle = (2 * Math.PI * i) / nodes.length - Math.PI / 2;
      positions[node.name] = {
        x: cx + radius * Math.cos(angle),
        y: cy + radius * Math.sin(angle),
      };
    });

    let svg = `<svg class="relationship-graph__svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Entity relationship graph">`;

    // Draw edges
    relationships.forEach((rel) => {
      const from = positions[rel.source];
      const to = positions[rel.target];
      if (from && to) {
        const thickness = Math.max(1, (rel.confidence || 0.5) * 3);
        svg += `<line class="graph-edge" x1="${from.x}" y1="${from.y}" x2="${to.x}" y2="${to.y}" stroke-width="${thickness}" data-source="${_esc(rel.source)}" data-target="${_esc(rel.target)}"/>`;
      }
    });

    // Draw nodes
    nodes.forEach((node) => {
      const pos = positions[node.name];
      if (!pos) return;

      const fillColor = NODE_COLORS[node.entity_type] || NODE_COLORS.ARTIST;
      const nodeRadius = 22;

      svg += `<g class="graph-node-group" data-entity="${_esc(node.name)}" tabindex="0" role="button" aria-label="${_esc(node.name)}">`;
      svg += `<circle class="graph-node" cx="${pos.x}" cy="${pos.y}" r="${nodeRadius}" fill="${fillColor}" />`;
      // Truncate label if too long
      const label = node.name.length > 14 ? node.name.substring(0, 12) + "\u2026" : node.name;
      svg += `<text class="graph-label" x="${pos.x}" y="${pos.y + nodeRadius + 16}" text-anchor="middle">${_esc(label)}</text>`;
      svg += "</g>";
    });

    svg += "</svg>";

    return `<div class="relationship-graph" id="relationship-graph">${svg}</div>`;
  }

  /** Attach hover/focus handlers to the relationship graph. */
  function _initGraphInteractions() {
    const graphEl = document.getElementById("relationship-graph");
    if (!graphEl) return;

    const svgEl = graphEl.querySelector("svg");
    if (!svgEl) return;

    const nodeGroups = svgEl.querySelectorAll(".graph-node-group");
    const edges = svgEl.querySelectorAll(".graph-edge");

    function highlightEntity(entityName) {
      nodeGroups.forEach((g) => {
        const isMatch = g.dataset.entity === entityName;
        g.classList.toggle("graph-node-group--highlighted", isMatch);
      });
      edges.forEach((e) => {
        const isConnected =
          e.dataset.source === entityName || e.dataset.target === entityName;
        e.classList.toggle("graph-edge--highlighted", isConnected);
        e.classList.toggle("graph-edge--dimmed", !isConnected);
      });
    }

    function clearHighlights() {
      nodeGroups.forEach((g) =>
        g.classList.remove("graph-node-group--highlighted")
      );
      edges.forEach((e) => {
        e.classList.remove("graph-edge--highlighted");
        e.classList.remove("graph-edge--dimmed");
      });
    }

    nodeGroups.forEach((group) => {
      group.addEventListener("mouseenter", () =>
        highlightEntity(group.dataset.entity)
      );
      group.addEventListener("mouseleave", clearHighlights);
      group.addEventListener("focus", () =>
        highlightEntity(group.dataset.entity)
      );
      group.addEventListener("blur", clearHighlights);
    });
  }

  // ------------------------------------------------------------------
  // Dismiss connection handler
  // ------------------------------------------------------------------

  /**
   * Send a dismiss request to the backend for an incorrect relationship.
   * The backend marks the connection as dismissed so it is excluded from
   * future outputs. On success, the list item fades out and is removed.
   *
   * API: POST /api/v1/fliers/{session_id}/dismiss-connection
   * Body: { source, target, relationship_type }
   */
  async function _dismissConnection(source, target, type, listItemEl) {
    const sessionId = App.getSessionId();
    if (!sessionId) return;

    try {
      const resp = await fetch(
        `/api/v1/fliers/${sessionId}/dismiss-connection`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            source: source,
            target: target,
            relationship_type: type,
          }),
        }
      );

      if (resp.ok) {
        listItemEl.classList.add("relationship-item--dismissed");
        setTimeout(() => listItemEl.remove(), 300);
      }
    } catch (err) {
      console.error("[Results] Dismiss connection failed:", err);
    }
  }

  function _initDismissHandlers() {
    const container = document.getElementById("results-interconnections");
    if (!container) return;

    container.addEventListener("click", (e) => {
      const btn = e.target.closest(".relationship-item__dismiss");
      if (!btn) return;

      const li = btn.closest(".relationship-item");
      if (!li) return;

      _dismissConnection(li.dataset.source, li.dataset.target, li.dataset.type, li);
    });
  }

  // ------------------------------------------------------------------
  // Section 6: Citations
  // ------------------------------------------------------------------

  function _renderCitations(citations) {
    if (!citations || citations.length === 0) return "";

    // Group by tier
    const grouped = {};
    citations.forEach((c) => {
      const tier = c.tier || 6;
      if (!grouped[tier]) grouped[tier] = [];
      grouped[tier].push(c);
    });

    let html = '<div class="results-section" id="results-citations">';
    html += '<h2 class="results-section__title text-heading">All Citations</h2>';

    // Render tier groups in order
    for (let tier = 1; tier <= 6; tier++) {
      const items = grouped[tier];
      if (!items || items.length === 0) continue;

      const tierName = TIER_NAMES[tier] || `Tier ${tier}`;
      html += `<div class="citation-tier-group">`;
      html += `<h3 class="citation-tier-group__header"><span class="citation-tier-badge citation-tier-${tier}">${_esc(tierName)}</span> <span class="text-caption">(${items.length})</span></h3>`;
      html += '<ul class="citation-list">';

      items.forEach((c) => {
        html += `<li class="citation-item citation-item--tier-${tier}">`;

        // Accessibility status icon
        if (c.accessible === true) {
          html += '<span class="citation-item__status citation-item__status--ok" aria-label="Accessible" title="Link accessible">&#x2713;</span>';
        } else if (c.accessible === false) {
          html += '<span class="citation-item__status citation-item__status--fail" aria-label="Not accessible" title="Link not accessible">&#x2717;</span>';
        } else {
          html += '<span class="citation-item__status citation-item__status--unknown" aria-label="Status unknown" title="Status unknown">&mdash;</span>';
        }

        html += `<span class="citation-item__text">${_esc(c.text)}</span>`;

        if (c.source) {
          html += ` <span class="citation-item__source text-caption">${_esc(c.source)}</span>`;
        }

        if (c.url) {
          html += ` <a href="${_esc(c.url)}" target="_blank" rel="noopener noreferrer" class="citation-item__link">View source &rarr;</a>`;
        }

        html += "</li>";
      });

      html += "</ul></div>";
    }

    // Export JSON button
    html += '<div class="citation-export">';
    html += '<button type="button" class="btn-secondary" id="export-json-btn">Export JSON</button>';
    html += "</div>";

    html += "</div>"; // results-section
    return html;
  }

  // ------------------------------------------------------------------
  // Export — Download the raw analysis data as a JSON file
  // ------------------------------------------------------------------

  /** Create a downloadable JSON file from the raw API response.
   *  Uses the Blob API + URL.createObjectURL pattern to trigger a download
   *  without any server interaction. The temporary link is cleaned up after click. */
  function _exportJson() {
    if (!_rawData) return;

    const blob = new Blob([JSON.stringify(_rawData, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `raiveflier-analysis-${_rawData.session_id || "export"}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  // ------------------------------------------------------------------
  // Expandable card behaviour
  // ------------------------------------------------------------------

  /** Attach click handlers to all .qa-trigger buttons. */
  function _initQATriggers(container) {
    const triggers = container.querySelectorAll(".qa-trigger");
    triggers.forEach((btn) => {
      btn.addEventListener("click", () => {
        if (typeof QA !== "undefined" && QA.openPanel) {
          QA.openPanel(
            _sessionId,
            btn.dataset.entityType || null,
            btn.dataset.entityName || null
          );
        }
      });
    });
  }

  /** Initialise expand/collapse for all .expandable elements inside a container. */
  function _initExpandables(container) {
    const triggers = container.querySelectorAll(".expandable__trigger");

    triggers.forEach((trigger) => {
      trigger.addEventListener("click", () => {
        const card = trigger.closest(".expandable");
        const isOpen = card.classList.contains("expandable--open");

        card.classList.toggle("expandable--open", !isOpen);
        trigger.setAttribute("aria-expanded", String(!isOpen));
      });
    });
  }

  // ------------------------------------------------------------------
  // Public API
  // ------------------------------------------------------------------

  /**
   * Fetch completed analysis results and render them.
   * @param {string} sessionId — The pipeline session UUID.
   */
  async function fetchAndDisplayResults(sessionId) {
    const resultsView = document.getElementById("results-view");
    if (!resultsView) return;

    // Show a loading state
    resultsView.innerHTML =
      '<div class="results-loading"><div class="spinner"></div><p class="loading-text">Loading results&hellip;</p></div>';
    resultsView.hidden = false;

    try {
      const response = await fetch(`/api/v1/fliers/${encodeURIComponent(sessionId)}/results`);

      if (!response.ok) {
        throw new Error(`Failed to fetch results (HTTP ${response.status})`);
      }

      const raw = await response.json();

      if (raw.status && raw.status !== "completed") {
        resultsView.innerHTML =
          '<div class="results-loading"><div class="spinner"></div>' +
          `<p class="loading-text">Analysis still in progress (${_esc(raw.status)})&hellip;</p></div>`;
        return;
      }

      _rawData = raw;
      const data = _normalizeApiResponse(raw);
      renderResults(data);
    } catch (err) {
      resultsView.innerHTML =
        `<div class="results-error"><p>Failed to load results: ${_esc(err.message)}</p>` +
        '<button type="button" class="btn-secondary" onclick="Results.fetchAndDisplayResults(\'' +
        _esc(sessionId) +
        "')\">Retry</button></div>";
    }
  }

  /**
   * Render the full results view from normalised data.
   *
   * This is the main rendering orchestrator. It calls each section renderer
   * in order, concatenates their HTML strings, sets innerHTML once (for
   * performance — single DOM write), then initializes all interactive behaviors.
   *
   * The initialization sequence after innerHTML:
   *   1. _initExpandables() — accordion click handlers
   *   2. _initGraphInteractions() — SVG hover/focus highlighting
   *   3. _initDismissHandlers() — connection dismiss buttons
   *   4. _initQATriggers() — "Ask about this" buttons
   *   5. Rating.loadRatings() — fetch cached ratings from backend
   *   6. Rating.initWidgets() — event delegation for thumbs up/down
   *   7. Recommendations.init() — start background recommendation fetch
   *   8. Export button click handler
   *
   * @param {Object} data — Normalised analysis data.
   */
  function renderResults(data) {
    _rawData = _rawData || data;
    _sessionId = data.session_id || null;
    const resultsView = document.getElementById("results-view");
    if (!resultsView) return;

    const research = data.research || {};

    let html = "";

    // Section 1: Event Summary
    html += _renderEventSummary(data);

    // Section 2: Artist Cards
    html += _renderArtistCards(research.artists);

    // Section 3: Venue & Promoter
    html += _renderVenuePromoter(research.venue, research.promoter);

    // Section 3b: Event History
    html += _renderEventHistory(research.event_history);

    // Section 4: Date Context
    html += _renderDateContext(research.date_context);

    // Section 5: Interconnections
    html += _renderInterconnections(data);

    // Section 6: Citations — logged but not displayed
    if (data.citations && data.citations.length > 0) {
      console.log("[Results] All citations (%d):", data.citations.length, data.citations);
    }

    // Export JSON button (standalone, not inside citations)
    html += '<div class="citation-export">';
    html += '<button type="button" class="btn-secondary" id="export-json-btn">Export JSON</button>';
    html += "</div>";

    resultsView.innerHTML = html;

    // Initialise interactive behaviours
    _initExpandables(resultsView);
    _initGraphInteractions();
    _initDismissHandlers();
    _initQATriggers(resultsView);

    // Rating widgets: load cached state and attach event delegation
    if (typeof Rating !== "undefined" && _sessionId) {
      Rating.loadRatings(_sessionId);
      Rating.initWidgets(resultsView, _sessionId);
    }

    // Initialize recommendations panel (lazy — fetches on first open)
    if (typeof Recommendations !== "undefined" && Recommendations.init) {
      Recommendations.init(_sessionId);
    }

    // Export button
    const exportBtn = document.getElementById("export-json-btn");
    if (exportBtn) {
      exportBtn.addEventListener("click", _exportJson);
    }

    // Show the results view
    if (typeof App !== "undefined" && App.showView) {
      App.showView("results");
    }
  }

  return {
    fetchAndDisplayResults,
    renderResults,
    renderRelationshipGraph,
  };
})();
