"""Article extraction providers.

Two implementations of IArticleProvider:
    1. WebScraperProvider — extracts readable content from live web pages
       using trafilatura/beautifulsoup4. Used to pull article text from
       URLs found during web search, enabling citation and analysis.
    2. WaybackProvider — falls back to the Internet Archive's Wayback Machine
       when live URLs return 404/errors. Valuable for rave history research
       since many early-2000s event pages and forums are no longer online.
"""

from src.providers.article.wayback_provider import WaybackProvider
from src.providers.article.web_scraper_provider import WebScraperProvider

__all__ = ["WebScraperProvider", "WaybackProvider"]
