# =============================================================================
# src/services/rbma_scrape_service.py — RBMA Article Discovery & Scrape Service
# =============================================================================
#
# Orchestrates the full Red Bull Music Academy article ingestion pipeline:
#   1. URL Discovery   — Uses the Wayback Machine CDX API to enumerate all
#                        RBMA Daily article URLs and RBMA lecture URLs
#   2. Genre Filtering — Two-pass filter: URL slug keywords (pre-fetch) and
#                        content body keywords (post-fetch)
#   3. Article Scraping — Fetches article HTML with a live-site-first,
#                         Wayback-fallback strategy; extracts text via trafilatura
#   4. Corpus Generation — Groups articles by genre into .txt files matching
#                          the existing reference_corpus format
#
# Design Pattern: Service class with checkpoint/resume — follows the same
# pattern as RAScrapeService (src/services/ra_scrape_service.py). All state
# is serialised to JSON in data/rbma_scrape/ so scraping can be interrupted
# and resumed without loss.
#
# Dependencies (all already in the project):
#   - httpx          — async HTTP client for CDX API + article fetching
#   - trafilatura    — HTML-to-text extraction with metadata
#   - structlog      — structured JSON logging
#   - pydantic v2    — frozen data models
#
# This service is LOCAL-ONLY. It is not run in production (512 MB RAM
# constraint on Render). After scraping locally, the generated corpus
# .txt files are committed and deployed; the existing auto-ingest on
# startup handles embedding them into ChromaDB.
# =============================================================================

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx
import structlog
import trafilatura
from pydantic import BaseModel, ConfigDict, Field

logger = structlog.get_logger(logger_name=__name__)

# ─── Default directories ───
_CHECKPOINT_DIR = "data/rbma_scrape"
_CORPUS_DIR = "data/reference_corpus"

# ─── Rate limiting ───
# 2-second delay between HTTP requests. RBMA is an archived/legacy site
# so aggressive scraping is unnecessary and disrespectful.
_DEFAULT_SCRAPE_DELAY = 2.0

# ─── HTTP client settings ───
_DEFAULT_TIMEOUT = 15.0
_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; raiveFlier/0.1; "
        "+https://github.com/raiveFlier)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ─── CDX API endpoints ───
# The Wayback Machine CDX API returns URL inventories for archived domains.
# collapse=urlkey deduplicates by URL; fl= selects output fields.
_CDX_ARTICLE_URL = (
    "https://web.archive.org/cdx/search/cdx"
    "?url=daily.redbullmusicacademy.com/*"
    "&output=json&fl=original,timestamp"
    "&filter=statuscode:200&filter=mimetype:text/html"
    "&collapse=urlkey&limit=50000"
)
_CDX_LECTURE_URL = (
    "https://web.archive.org/cdx/search/cdx"
    "?url=www.redbullmusicacademy.com/lectures/*"
    "&output=json&fl=original,timestamp"
    "&filter=statuscode:200&filter=mimetype:text/html"
    "&collapse=urlkey&limit=10000"
)

# ─── Genre keyword sets ───
# Used for two-pass genre relevance filtering. Pass 1 matches these
# against URL slugs (hyphenated); Pass 2 matches against article body text.
GENRE_KEYWORDS: set[str] = {
    "house", "techno", "rave", "drum", "bass", "jungle", "detroit",
    "disco", "acid", "garage", "dubstep", "breakbeat", "chicago",
    "ambient", "trance", "gabber", "hardcore", "industrial", "electro",
    "minimal", "dub", "deep", "electronic", "dance", "club", "warehouse",
    "underground", "303", "808", "909", "synth", "remix", "turntable",
    "vinyl", "bpm", "nightclub", "soundsystem", "pirate-radio",
    "free-party",
}

# Minimum distinct genre keyword matches required for content-based
# filtering (Pass 2). Articles must contain at least this many different
# genre keywords in their body text to be considered genre-relevant.
_MIN_CONTENT_KEYWORD_MATCHES = 3

# ─── URL pattern regexes ───
# RBMA Daily articles follow: /{4-digit-year}/{2-digit-month}/{slug}
_ARTICLE_URL_PATTERN = re.compile(
    r"daily\.redbullmusicacademy\.com/\d{4}/\d{2}/[\w-]+/?$"
)
# RBMA Lectures follow: /lectures/{slug}
_LECTURE_URL_PATTERN = re.compile(
    r"www\.redbullmusicacademy\.com/lectures/[\w-]+/?$"
)
# Pages to exclude from discovery — index pages, search, author pages,
# pagination, and static assets that match the HTML mimetype filter.
_EXCLUDE_PATTERNS = re.compile(
    r"(/search/|/author/|/page/|\?|\.css|\.js|\.xml|\.json|/tag/|/category/|/feed/)"
)

# ─── Genre groupings for corpus file generation ───
# Maps corpus file names to the genre keywords that route articles into
# that file. Articles are placed into their PRIMARY group (first match).
GENRE_GROUPS: dict[str, list[str]] = {
    "rbma_house_chicago": ["house", "chicago", "deep"],
    "rbma_techno_detroit": ["techno", "detroit", "minimal"],
    "rbma_drum_bass_jungle": ["drum", "bass", "jungle", "breakbeat"],
    "rbma_disco": ["disco"],
    "rbma_rave_culture": [
        "rave", "acid", "free-party", "warehouse", "underground",
        "pirate-radio", "soundsystem", "gabber", "hardcore",
    ],
    "rbma_electronic_general": [
        "electronic", "ambient", "trance", "electro", "industrial",
        "synth", "dance", "303", "808", "909",
    ],
    "rbma_dj_culture": [
        "dub", "garage", "dubstep", "club", "nightclub", "remix",
        "turntable", "vinyl", "bpm",
    ],
}


# ─── Data Model ───

class RBMAArticle(BaseModel):
    """A single article or lecture scraped from RBMA.

    Frozen (immutable) per project convention — state changes use
    model_copy(update={...}).
    """

    model_config = ConfigDict(frozen=True)

    url: str = Field(description="Original article URL on RBMA Daily or lectures site.")
    title: str = Field(description="Article title extracted by trafilatura.")
    author: str | None = Field(
        default=None, description="Author name if available."
    )
    pub_date: date | None = Field(
        default=None, description="Publication date if available."
    )
    text: str = Field(description="Extracted article body text.")
    word_count: int = Field(description="Word count of the extracted text.")
    genre_tags: list[str] = Field(
        default_factory=list,
        description="Genre keywords matched in URL slug or body text.",
    )
    source: str = Field(
        description='Fetch source: "live" if from the live site, "wayback" if from archive.'
    )
    content_type: str = Field(
        default="article",
        description='Content type: "article" or "lecture".',
    )
    scraped_at: str = Field(
        description="ISO 8601 timestamp of when the article was scraped."
    )


# ─── Service Class ───

class RBMAScrapeService:
    """Orchestrates RBMA article discovery, scraping, and corpus generation.

    Follows the same checkpoint/resume pattern as RAScrapeService:
      - Discovered URLs are saved to JSON after each CDX query
      - Scrape progress is checkpointed after each article
      - All operations are idempotent and resume-safe

    Parameters
    ----------
    http_client:
        An httpx.AsyncClient for all HTTP requests (CDX API, article fetching).
    checkpoint_dir:
        Directory for intermediate JSON files (default: data/rbma_scrape/).
    corpus_dir:
        Directory for generated corpus .txt files (default: data/reference_corpus/).
    scrape_delay:
        Seconds between HTTP requests (default: 2.0).
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        checkpoint_dir: str = _CHECKPOINT_DIR,
        corpus_dir: str = _CORPUS_DIR,
        scrape_delay: float = _DEFAULT_SCRAPE_DELAY,
    ) -> None:
        self._client = http_client
        self._checkpoint_dir = Path(checkpoint_dir)
        self._corpus_dir = Path(corpus_dir)
        self._scrape_delay = scrape_delay
        self._logger = logger

    # ------------------------------------------------------------------
    # URL Discovery
    # ------------------------------------------------------------------

    async def discover_urls(self) -> dict[str, list[dict]]:
        """Query the Wayback Machine CDX API for all RBMA article and lecture URLs.

        Returns a dict with keys "articles" and "lectures", each containing a
        list of dicts with "url" and "timestamp" fields.

        Results are saved to checkpoint files for resume capability.
        """
        articles = await self._discover_article_urls()
        lectures = await self._discover_lecture_urls()

        result = {"articles": articles, "lectures": lectures}

        self._logger.info(
            "rbma_discovery_complete",
            article_urls=len(articles),
            lecture_urls=len(lectures),
        )
        return result

    async def _discover_article_urls(self) -> list[dict]:
        """Fetch all RBMA Daily article URLs from the CDX API.

        Filters results to match the article URL pattern (/{year}/{month}/{slug})
        and excludes non-article pages (search, author, pagination, assets).
        """
        self._logger.info("rbma_discovering_articles", cdx_url=_CDX_ARTICLE_URL)

        try:
            response = await self._client.get(
                _CDX_ARTICLE_URL, timeout=60.0
            )
            response.raise_for_status()
            rows = response.json()
        except Exception as exc:
            self._logger.error("rbma_cdx_article_failed", error=str(exc))
            return []

        # CDX JSON output: first row is header ["original", "timestamp"],
        # subsequent rows are data.
        if not rows or len(rows) < 2:
            self._logger.warning("rbma_cdx_article_empty")
            return []

        urls: list[dict] = []
        seen: set[str] = set()
        for row in rows[1:]:
            url, timestamp = row[0], row[1]

            # Normalize: strip trailing slashes for dedup
            normalized = url.rstrip("/")
            if normalized in seen:
                continue

            # Filter: must match article pattern, must not match exclusions
            if not _ARTICLE_URL_PATTERN.search(url):
                continue
            if _EXCLUDE_PATTERNS.search(url):
                continue

            seen.add(normalized)
            urls.append({"url": url, "timestamp": timestamp})

        self._save_discovered_urls("articles", urls)

        self._logger.info("rbma_articles_discovered", count=len(urls))
        return urls

    async def _discover_lecture_urls(self) -> list[dict]:
        """Fetch all RBMA lecture URLs from the CDX API.

        Filters results to match the lecture URL pattern (/lectures/{slug})
        and excludes index/pagination pages.
        """
        self._logger.info("rbma_discovering_lectures", cdx_url=_CDX_LECTURE_URL)

        try:
            response = await self._client.get(
                _CDX_LECTURE_URL, timeout=60.0
            )
            response.raise_for_status()
            rows = response.json()
        except Exception as exc:
            self._logger.error("rbma_cdx_lecture_failed", error=str(exc))
            return []

        if not rows or len(rows) < 2:
            self._logger.warning("rbma_cdx_lecture_empty")
            return []

        urls: list[dict] = []
        seen: set[str] = set()
        for row in rows[1:]:
            url, timestamp = row[0], row[1]

            normalized = url.rstrip("/")
            if normalized in seen:
                continue

            if not _LECTURE_URL_PATTERN.search(url):
                continue
            if _EXCLUDE_PATTERNS.search(url):
                continue

            seen.add(normalized)
            urls.append({"url": url, "timestamp": timestamp})

        self._save_discovered_urls("lectures", urls)

        self._logger.info("rbma_lectures_discovered", count=len(urls))
        return urls

    # ------------------------------------------------------------------
    # Genre Filtering
    # ------------------------------------------------------------------

    def filter_by_genre(
        self,
        urls: list[dict],
        genres: set[str] | None = None,
    ) -> list[dict]:
        """Pass 1 — filter discovered URLs by genre keywords in the URL slug.

        Parameters
        ----------
        urls:
            List of URL dicts from discovery (each has "url" and "timestamp").
        genres:
            Optional subset of GENRE_KEYWORDS to filter by. If None, uses
            the full keyword set.

        Returns
        -------
        list[dict]
            URLs whose slug contains at least one genre keyword, with a
            "matched_genres" key added to each dict.
        """
        keywords = genres or GENRE_KEYWORDS
        filtered: list[dict] = []

        for entry in urls:
            url = entry["url"]
            # Extract the slug portion of the URL for keyword matching.
            # For articles: the last path segment; for lectures: the slug after /lectures/
            parsed = urlparse(url)
            slug = parsed.path.rstrip("/").split("/")[-1].lower()

            matched = {kw for kw in keywords if kw in slug}
            if matched:
                filtered.append({
                    **entry,
                    "matched_genres": sorted(matched),
                })

        self._logger.info(
            "rbma_genre_filter_pass1",
            input_count=len(urls),
            filtered_count=len(filtered),
        )
        return filtered

    def filter_article_by_content(
        self, article: RBMAArticle, genres: set[str] | None = None,
    ) -> list[str]:
        """Pass 2 — check article body text for genre keyword relevance.

        Returns the list of matched genre keywords if the article meets the
        minimum threshold (_MIN_CONTENT_KEYWORD_MATCHES), otherwise returns
        an empty list.
        """
        keywords = genres or GENRE_KEYWORDS
        text_lower = article.text.lower()

        matched = [kw for kw in keywords if kw in text_lower]
        if len(matched) >= _MIN_CONTENT_KEYWORD_MATCHES:
            return sorted(set(matched))
        return []

    # ------------------------------------------------------------------
    # Article Scraping
    # ------------------------------------------------------------------

    async def scrape_articles(
        self,
        urls: list[dict],
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> list[RBMAArticle]:
        """Fetch and extract articles from a list of discovered URLs.

        Uses a live-site-first, Wayback-fallback strategy. Checkpoints
        progress after each article for resume capability.

        Parameters
        ----------
        urls:
            List of URL dicts (must have "url" and "timestamp" keys).
        on_progress:
            Callback(current_index, total, url) invoked after each article.

        Returns
        -------
        list[RBMAArticle]
            All successfully scraped articles.
        """
        # Load existing articles and progress for resume
        existing_articles = self._load_articles()
        progress = self._load_progress()
        start_index = progress.get("last_index", 0) if progress else 0

        # Build a set of already-scraped URLs to skip
        scraped_urls: set[str] = {a.url for a in existing_articles}
        articles = list(existing_articles)

        total = len(urls)
        self._logger.info(
            "rbma_scrape_start",
            total_urls=total,
            resume_from=start_index,
            already_scraped=len(scraped_urls),
        )

        for i in range(start_index, total):
            entry = urls[i]
            url = entry["url"]

            if url in scraped_urls:
                if on_progress:
                    on_progress(i + 1, total, url)
                continue

            # Determine content type from URL
            content_type = "lecture" if "/lectures/" in url else "article"

            article = await self._fetch_article(
                url=url,
                wayback_ts=entry["timestamp"],
                matched_genres=entry.get("matched_genres", []),
                content_type=content_type,
            )

            if article is not None:
                articles.append(article)
                scraped_urls.add(url)

            # Checkpoint after each article
            self._save_articles(articles)
            self._save_progress({
                "last_index": i + 1,
                "total_urls": total,
                "articles_scraped": len(articles),
                "updated_at": datetime.now().isoformat(),
            })

            if on_progress:
                on_progress(i + 1, total, url)

            # Rate limit — respectful delay between requests
            await asyncio.sleep(self._scrape_delay)

        self._logger.info(
            "rbma_scrape_complete",
            total_articles=len(articles),
        )
        return articles

    async def _fetch_article(
        self,
        url: str,
        wayback_ts: str,
        matched_genres: list[str] | None = None,
        content_type: str = "article",
    ) -> RBMAArticle | None:
        """Fetch a single article with live-site-first, Wayback-fallback.

        The live site (daily.redbullmusicacademy.com) may still serve content.
        If the live fetch fails or returns empty text, falls back to the
        Wayback Machine archive using the timestamp from CDX discovery.

        Returns None if both sources fail to produce usable text.
        """
        # Try live site first
        result = await self._fetch_live(url)

        # Fallback to Wayback if live failed
        if result is None:
            result = await self._fetch_wayback(url, wayback_ts)

        if result is None:
            self._logger.debug("rbma_article_failed_both", url=url)
            return None

        text, metadata, source = result

        # Parse publication date from metadata
        pub_date: date | None = None
        raw_date = metadata.get("date")
        if raw_date:
            with contextlib.suppress(ValueError):
                pub_date = datetime.strptime(raw_date, "%Y-%m-%d").date()

        genre_tags = matched_genres or []

        article = RBMAArticle(
            url=url,
            title=metadata.get("title", ""),
            author=metadata.get("author") or None,
            pub_date=pub_date,
            text=text,
            word_count=len(text.split()),
            genre_tags=genre_tags,
            source=source,
            content_type=content_type,
            scraped_at=datetime.now().isoformat(),
        )

        self._logger.debug(
            "rbma_article_scraped",
            url=url,
            source=source,
            word_count=article.word_count,
            title=article.title[:60],
        )
        return article

    async def _fetch_live(self, url: str) -> tuple[str, dict, str] | None:
        """Attempt to fetch article text from the live RBMA site.

        Returns (text, metadata_dict, "live") on success, None on failure.
        """
        try:
            response = await self._client.get(url, timeout=_DEFAULT_TIMEOUT)
            response.raise_for_status()
            html = response.text

            text = trafilatura.extract(
                html, include_comments=False, include_tables=True
            )
            if not text or len(text.strip()) < 100:
                return None

            # Extract metadata (title, author, date) via trafilatura
            metadata = self._extract_metadata(html)
            return (text, metadata, "live")

        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            self._logger.debug("rbma_live_fetch_failed", url=url, error=str(exc))
            return None

    async def _fetch_wayback(
        self, url: str, timestamp: str
    ) -> tuple[str, dict, str] | None:
        """Fetch article text from the Wayback Machine archive.

        Uses the id_ flag to get the original page without the Wayback toolbar.
        Returns (text, metadata_dict, "wayback") on success, None on failure.
        """
        # The id_ flag tells Wayback to return raw original content
        wayback_url = f"https://web.archive.org/web/{timestamp}id_/{url}"

        try:
            response = await self._client.get(
                wayback_url, timeout=_DEFAULT_TIMEOUT
            )
            response.raise_for_status()
            html = response.text

            text = trafilatura.extract(
                html, include_comments=False, include_tables=True
            )
            if not text or len(text.strip()) < 100:
                return None

            metadata = self._extract_metadata(html)
            return (text, metadata, "wayback")

        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            self._logger.debug(
                "rbma_wayback_fetch_failed",
                url=url,
                wayback_url=wayback_url,
                error=str(exc),
            )
            return None

    @staticmethod
    def _extract_metadata(html: str) -> dict:
        """Extract title, author, and date from HTML via trafilatura.

        Uses trafilatura's JSON output mode with metadata extraction enabled.
        Returns a dict with keys: title, author, date (all optional strings).
        """
        metadata: dict = {"title": "", "author": None, "date": None}
        try:
            meta_json = trafilatura.extract(
                html,
                include_comments=False,
                output_format="json",
                with_metadata=True,
            )
            if meta_json:
                meta_dict = json.loads(meta_json)
                metadata["title"] = meta_dict.get("title", "")
                metadata["author"] = meta_dict.get("author") or None
                metadata["date"] = meta_dict.get("date") or None
        except (json.JSONDecodeError, ValueError):
            pass
        return metadata

    # ------------------------------------------------------------------
    # Corpus File Generation
    # ------------------------------------------------------------------

    def generate_corpus_files(
        self, articles: list[RBMAArticle] | None = None,
    ) -> list[Path]:
        """Generate .txt corpus files grouped by genre from scraped articles.

        Each genre grouping (defined in GENRE_GROUPS) gets its own file.
        Lectures get a dedicated file regardless of genre.
        Articles appearing in multiple genres go into their PRIMARY group
        (first match in GENRE_GROUPS iteration order).

        Parameters
        ----------
        articles:
            Articles to process. If None, loads from checkpoint.

        Returns
        -------
        list[Path]
            Paths to all generated corpus files.
        """
        if articles is None:
            articles = self._load_articles()

        if not articles:
            self._logger.warning("rbma_corpus_no_articles")
            return []

        # Separate lectures from articles
        lecture_articles = [a for a in articles if a.content_type == "lecture"]
        daily_articles = [a for a in articles if a.content_type == "article"]

        # Route daily articles into genre groups
        # Track which articles have been assigned to prevent duplicates
        assigned_urls: set[str] = set()
        genre_buckets: dict[str, list[RBMAArticle]] = {
            group: [] for group in GENRE_GROUPS
        }

        for article in daily_articles:
            placed = False
            for group_name, group_keywords in GENRE_GROUPS.items():
                if article.url in assigned_urls:
                    break
                # Check if any of the article's genre tags match this group
                if any(tag in group_keywords for tag in article.genre_tags):
                    genre_buckets[group_name].append(article)
                    assigned_urls.add(article.url)
                    placed = True
                    break

            # Articles with no genre group match go to electronic_general
            if not placed and article.url not in assigned_urls:
                genre_buckets["rbma_electronic_general"].append(article)
                assigned_urls.add(article.url)

        # Generate corpus files
        self._corpus_dir.mkdir(parents=True, exist_ok=True)
        generated: list[Path] = []

        # Genre group files
        for group_name, group_articles in genre_buckets.items():
            if not group_articles:
                continue
            path = self._write_corpus_file(
                filename=f"{group_name}.txt",
                category=self._group_display_name(group_name),
                articles=group_articles,
            )
            generated.append(path)

        # Lecture files — split into chunks of ~50 lectures each to avoid
        # OOM during embedding ingestion (201 lectures at ~9.6MB total would
        # generate ~6,600 chunks which exceeds memory limits on some machines).
        if lecture_articles:
            _LECTURES_PER_FILE = 50
            for part_idx in range(0, len(lecture_articles), _LECTURES_PER_FILE):
                part_num = (part_idx // _LECTURES_PER_FILE) + 1
                part_articles = lecture_articles[part_idx:part_idx + _LECTURES_PER_FILE]
                path = self._write_corpus_file(
                    filename=f"rbma_lectures_part{part_num}.txt",
                    category=f"Lectures & Artist Interviews (Part {part_num})",
                    articles=part_articles,
                )
                generated.append(path)

        self._logger.info(
            "rbma_corpus_generated",
            files=len(generated),
            total_articles=len(articles),
        )
        return generated

    def _write_corpus_file(
        self,
        filename: str,
        category: str,
        articles: list[RBMAArticle],
    ) -> Path:
        """Write a single corpus .txt file in the standard format.

        Format matches existing corpus files for consistency:
        header block → separator → article blocks with metadata.
        """
        # Sort articles chronologically (undated articles go last)
        sorted_articles = sorted(
            articles,
            key=lambda a: a.pub_date or date(9999, 12, 31),
        )

        lines: list[str] = []

        # Header block — matches existing corpus file style
        lines.append(f"Red Bull Music Academy — {category}")
        lines.append("")
        lines.append("Source: Red Bull Music Academy Daily (daily.redbullmusicacademy.com)")
        lines.append("Citation Tier: 2 (established music publication)")
        lines.append(f"Articles: {len(sorted_articles)}")
        lines.append("")

        # Article blocks
        for article in sorted_articles:
            lines.append("---")
            lines.append("")
            lines.append(f"Title: {article.title}")
            if article.author:
                lines.append(f"Author: {article.author}")
            if article.pub_date:
                lines.append(f"Date: {article.pub_date.isoformat()}")
            lines.append(f"URL: {article.url}")
            lines.append("")
            lines.append(article.text)
            lines.append("")

        path = self._corpus_dir / filename
        path.write_text("\n".join(lines), encoding="utf-8")

        self._logger.info(
            "rbma_corpus_file_written",
            filename=filename,
            articles=len(sorted_articles),
            path=str(path),
        )
        return path

    @staticmethod
    def _group_display_name(group_key: str) -> str:
        """Convert a genre group key to a human-readable display name.

        'rbma_house_chicago' → 'House & Chicago House'
        """
        display_map = {
            "rbma_house_chicago": "House & Chicago House",
            "rbma_techno_detroit": "Techno & Detroit Techno",
            "rbma_drum_bass_jungle": "Drum & Bass / Jungle",
            "rbma_disco": "Disco",
            "rbma_rave_culture": "Rave Culture & Acid House",
            "rbma_electronic_general": "Electronic Music (General)",
            "rbma_dj_culture": "DJ Culture & Sound Systems",
        }
        return display_map.get(group_key, group_key.replace("rbma_", "").replace("_", " ").title())

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_scrape_status(self) -> dict:
        """Return a summary of discovery and scrape progress.

        Reads checkpoint files from disk. No network requests.

        Returns
        -------
        dict
            Keys: discovered_articles, discovered_lectures, filtered_articles,
            scraped_articles, scrape_progress, corpus_files.
        """
        # Count discovered URLs
        article_urls = self._load_discovered_urls("articles")
        lecture_urls = self._load_discovered_urls("lectures")

        # Count scraped articles
        articles = self._load_articles()

        # Scrape progress
        progress = self._load_progress()

        # Count corpus files
        corpus_files = sorted(self._corpus_dir.glob("rbma_*.txt"))

        return {
            "discovered_articles": len(article_urls),
            "discovered_lectures": len(lecture_urls),
            "scraped_articles": len(articles),
            "scrape_progress": progress or {},
            "corpus_files": [p.name for p in corpus_files],
            "corpus_file_count": len(corpus_files),
        }

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _ensure_dir(self) -> None:
        """Create checkpoint directory if it doesn't exist."""
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _save_discovered_urls(self, content_type: str, urls: list[dict]) -> None:
        """Save discovered URLs to a checkpoint JSON file.

        content_type is "articles" or "lectures".
        """
        self._ensure_dir()
        path = self._checkpoint_dir / f"discovered_{content_type}_urls.json"
        path.write_text(
            json.dumps(urls, indent=2), encoding="utf-8"
        )

    def _load_discovered_urls(self, content_type: str) -> list[dict]:
        """Load previously discovered URLs from checkpoint."""
        path = self._checkpoint_dir / f"discovered_{content_type}_urls.json"
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._logger.warning(
                "rbma_url_checkpoint_load_failed", content_type=content_type
            )
            return []

    def _save_articles(self, articles: list[RBMAArticle]) -> None:
        """Save all scraped articles to the checkpoint JSON file."""
        self._ensure_dir()
        path = self._checkpoint_dir / "articles.json"
        data = [a.model_dump(mode="json") for a in articles]
        path.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )

    def _load_articles(self) -> list[RBMAArticle]:
        """Load previously scraped articles from checkpoint.

        Deduplicates by URL to handle any checkpoint corruption.
        """
        path = self._checkpoint_dir / "articles.json"
        if not path.exists():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            seen: set[str] = set()
            articles: list[RBMAArticle] = []
            for item in raw:
                article = RBMAArticle.model_validate(item)
                if article.url not in seen:
                    seen.add(article.url)
                    articles.append(article)
            return articles
        except Exception:
            self._logger.warning("rbma_article_checkpoint_load_failed")
            return []

    def _save_progress(self, progress: dict) -> None:
        """Save scrape progress to checkpoint file."""
        self._ensure_dir()
        path = self._checkpoint_dir / "rbma_progress.json"
        path.write_text(
            json.dumps(progress, indent=2), encoding="utf-8"
        )

    def _load_progress(self) -> dict | None:
        """Load scrape progress from checkpoint file."""
        path = self._checkpoint_dir / "rbma_progress.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
