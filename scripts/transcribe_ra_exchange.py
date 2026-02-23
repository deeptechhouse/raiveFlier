#!/usr/bin/env python3
# =============================================================================
# scripts/transcribe_ra_exchange.py — RA Exchange Podcast Transcription
# =============================================================================
#
# Transcribes RA Exchange podcast episodes (Resident Advisor's long-form
# interview series with DJs, producers, and scene figures) and ingests the
# transcripts into the ChromaDB vector store for RAG retrieval.
#
# RA Exchange is one of the most important interview series in electronic
# music. These transcripts provide rich first-person context about artists,
# labels, scenes, and cultural movements — invaluable for enriching flier
# analysis with quotes and biographical detail.
#
# Architecture:
#   EpisodeDiscovery    — Discovers episodes from SoundCloud via yt-dlp
#   TranscriptionService — Sends audio to TogetherAI's Whisper API
#   ProgressTracker     — Tracks completed/failed/ingested state (JSON file)
#   DBIngestor          — Chunks, tags, embeds, and stores transcripts
#   RAExchangeTranscriber — Orchestrates the full pipeline in batches
#
# Pipeline per batch:
#   1. Discover all episodes from SoundCloud (yt-dlp --flat-playlist)
#   2. Filter out already-completed episodes (via _progress.json)
#   3. For each episode in the batch:
#      a. Download audio to temp dir (yt-dlp -x --audio-format mp3)
#      b. Send audio to TogetherAI Whisper for transcription
#      c. Save transcript as .txt file
#      d. Auto-delete the audio file (temp dir cleanup)
#   4. Ingest all batch transcripts into ChromaDB
#
# Cost: ~$0.0015/min of audio via TogetherAI Whisper. A typical episode
# is ~50 minutes, so ~$0.075/episode. Full corpus (~500 episodes) ~$37.50.
#
# External Dependencies:
#   - yt-dlp: CLI tool for downloading audio from SoundCloud
#   - TogetherAI API: OpenAI-compatible Whisper endpoint for transcription
#   - Project .env: Embedding provider keys for ChromaDB ingestion
#
# Usage:
#   python scripts/transcribe_ra_exchange.py --dry-run      # Cost estimate only
#   TOGETHER_API_KEY=key python scripts/transcribe_ra_exchange.py
#   TOGETHER_API_KEY=key python scripts/transcribe_ra_exchange.py --batch-size 10
#   TOGETHER_API_KEY=key python scripts/transcribe_ra_exchange.py --skip-ingest
#   TOGETHER_API_KEY=key python scripts/transcribe_ra_exchange.py --retry-failed
# =============================================================================

"""Transcribe all RA Exchange podcast episodes via TogetherAI.

Downloads audio from SoundCloud using yt-dlp, sends to TogetherAI's
Whisper API for transcription, saves the text transcript locally,
and ingests each batch into the ChromaDB vector store.

Processes episodes in configurable batches (default 5). After each
batch is transcribed, the transcripts are chunked, tagged, embedded,
and stored in ChromaDB before the next batch begins.

Requires:
    - yt-dlp (installed globally or in PATH)
    - openai Python package (already in project venv)
    - TOGETHER_API_KEY environment variable
    - Project .env configured for embedding provider (for DB ingestion)

Usage:
    # Discover episodes and show cost estimate (no transcription):
    python scripts/transcribe_ra_exchange.py --dry-run

    # Transcribe all episodes in batches of 5, ingesting after each batch:
    TOGETHER_API_KEY=your-key python scripts/transcribe_ra_exchange.py

    # Custom batch size:
    TOGETHER_API_KEY=your-key python scripts/transcribe_ra_exchange.py --batch-size 10

    # Transcribe only, skip DB ingestion:
    TOGETHER_API_KEY=your-key python scripts/transcribe_ra_exchange.py --skip-ingest

    # Retry only previously failed episodes:
    TOGETHER_API_KEY=your-key python scripts/transcribe_ra_exchange.py --retry-failed
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Add project root to sys.path so we can import from src/ when running
# this script directly (not as a module).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# We use the OpenAI SDK to talk to TogetherAI's Whisper endpoint because
# TogetherAI provides an OpenAI-compatible API. This means we just change
# the base_url and api_key — no new SDK needed.
from openai import OpenAI


class EpisodeDiscovery:
    """Discovers RA Exchange episodes from SoundCloud via yt-dlp.

    Uses yt-dlp's --flat-playlist mode to list episodes without downloading
    any audio. This is fast (~10-30 seconds) and gives us URLs, titles, and
    durations for all episodes.

    The primary source is the main RA Exchange SoundCloud account. Optional
    playlist sources can pick up episodes that were posted under the main
    Resident Advisor account instead.
    """

    # Primary SoundCloud account for RA Exchange episodes.
    PRIMARY_SOURCE = "https://soundcloud.com/ra-exchange"

    # Additional playlists that may contain episodes not on the main account.
    # These are only scanned if --include-playlists is passed.
    PLAYLIST_SOURCES = [
        "https://soundcloud.com/resident-advisor/sets/ra-exchange",
        "https://soundcloud.com/resident-advisor/sets/ra-exchange-podcast",
    ]

    def discover(self, include_playlists: bool = False) -> list[dict]:
        """Return deduplicated list of episodes with url, title, duration.

        Episodes are deduplicated by normalized URL (lowercase, no query params,
        no trailing slash) to prevent duplicate transcriptions when the same
        episode appears in both the main account and playlists.

        Results are sorted by episode number (EX.NNN) when available, then
        alphabetically for episodes without standard numbering.
        """
        episodes: dict[str, dict] = {}

        print(f"  Scanning: {self.PRIMARY_SOURCE}")
        for ep in self._scan_source(self.PRIMARY_SOURCE):
            normalized = self._normalize_url(ep["url"])
            if normalized:
                episodes[normalized] = ep

        primary_count = len(episodes)
        print(f"  Found {primary_count} episodes from main account.")

        if include_playlists:
            for source in self.PLAYLIST_SOURCES:
                print(f"  Scanning: {source}")
                for ep in self._scan_source(source):
                    normalized = self._normalize_url(ep["url"])
                    if normalized and normalized not in episodes:
                        if not ep["title"] or ep["title"] == "unknown":
                            ep["title"] = self._title_from_url(normalized)
                        episodes[normalized] = ep

            new_from_playlists = len(episodes) - primary_count
            if new_from_playlists:
                print(f"  Found {new_from_playlists} additional episodes from playlists.")

        return sorted(episodes.values(), key=self._sort_key)

    def _scan_source(self, url: str) -> list[dict]:
        """Use yt-dlp --flat-playlist to list episodes without downloading.

        yt-dlp outputs one JSON object per line, each containing the episode
        URL, title, and duration. We parse each line individually to handle
        partial output (some lines may be malformed).
        """
        result = subprocess.run(
            ["yt-dlp", "--flat-playlist", "--dump-json", "--no-warnings", url],
            capture_output=True,
            text=True,
            timeout=600,
        )

        episodes = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            ep_url = data.get("url") or data.get("webpage_url") or ""
            if not ep_url:
                continue

            if not ep_url.startswith("http"):
                ep_url = f"https://soundcloud.com{ep_url}" if ep_url.startswith("/") else ep_url

            episodes.append(
                {
                    "url": ep_url,
                    "title": data.get("title") or "unknown",
                    "duration": data.get("duration"),
                }
            )

        return episodes

    @staticmethod
    def _normalize_url(url: str) -> str:
        url = url.split("?")[0].rstrip("/").lower()
        return url

    @staticmethod
    def _title_from_url(url: str) -> str:
        slug = url.rstrip("/").split("/")[-1]
        return slug.replace("-", " ").title()

    @staticmethod
    def _sort_key(ep: dict) -> tuple:
        match = re.search(r"EX\.?(\d+)", ep["title"], re.IGNORECASE)
        if match:
            return (0, int(match.group(1)))
        return (1, ep["title"])


class TranscriptionService:
    """Transcribes audio files via TogetherAI's Whisper API.

    Uses the OpenAI SDK with a custom base_url pointing to TogetherAI.
    The model is Whisper Large V3, which provides high-quality English
    transcription at $0.0015/minute (~$0.075 per 50-minute episode).
    """

    MODEL = "openai/whisper-large-v3"  # TogetherAI model identifier
    COST_PER_MINUTE = 0.0015           # USD per minute of audio

    def __init__(self, api_key: str):
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
        )

    def transcribe_file(self, audio_path: Path) -> str:
        """Send an audio file to TogetherAI and return the transcript text."""
        with open(audio_path, "rb") as f:
            response = self._client.audio.transcriptions.create(
                file=f,
                model=self.MODEL,
                language="en",
            )
        return response.text


class ProgressTracker:
    """Tracks which episodes have been transcribed and ingested.

    Persists state to a JSON file (_progress.json) in the transcript output
    directory. This enables resumable operation — if the script is interrupted,
    re-running it will skip already-completed episodes.

    State categories:
      - completed: Transcription finished, transcript file saved
      - failed: Transcription or download failed (with error message)
      - ingested: Transcript has been chunked and stored in ChromaDB
    """

    def __init__(self, progress_file: Path):
        self._file = progress_file
        self._data = self._load()

    def _load(self) -> dict:
        if self._file.exists():
            return json.loads(self._file.read_text())
        return {"completed": {}, "failed": {}, "ingested": {}}

    def save(self) -> None:
        self._file.write_text(json.dumps(self._data, indent=2))

    def is_completed(self, url: str) -> bool:
        return url in self._data["completed"]

    def is_ingested(self, url: str) -> bool:
        return url in self._data.get("ingested", {})

    def mark_completed(self, url: str, title: str, transcript_file: str) -> None:
        self._data["completed"][url] = {"title": title, "file": transcript_file}
        self._data["failed"].pop(url, None)
        self.save()

    def mark_ingested(self, url: str, chunks: int, tokens: int) -> None:
        self._data.setdefault("ingested", {})[url] = {
            "chunks": chunks,
            "tokens": tokens,
        }
        self.save()

    def mark_failed(self, url: str, title: str, error: str) -> None:
        self._data["failed"][url] = {"title": title, "error": error}
        self.save()

    @property
    def completed_count(self) -> int:
        return len(self._data["completed"])

    @property
    def ingested_count(self) -> int:
        return len(self._data.get("ingested", {}))

    @property
    def failed_count(self) -> int:
        return len(self._data["failed"])

    @property
    def failed_urls(self) -> set[str]:
        return set(self._data["failed"].keys())


class DBIngestor:
    """Ingests transcript text files into the ChromaDB vector store.

    Lazy-initializes the full ingestion pipeline (embedding provider, LLM,
    chunker, vector store) on first use. This avoids loading heavy
    dependencies if --skip-ingest is used.

    Transcripts are ingested as source_type="interview" with citation_tier=3
    (primary source — first-person accounts from artists and scene figures).
    """

    def __init__(self):
        self._service = None
        self._initialized = False

    def _init_service(self) -> None:
        """Lazy-initialize the ingestion service (requires project deps)."""
        if self._initialized:
            return

        from src.config.settings import Settings
        from src.providers.vector_store.chromadb_provider import ChromaDBProvider
        from src.services.ingestion.chunker import TextChunker
        from src.services.ingestion.metadata_extractor import MetadataExtractor

        settings = Settings()

        # Embedding provider
        embedding_provider = None
        if settings.openai_api_key:
            from src.providers.embedding.openai_embedding_provider import (
                OpenAIEmbeddingProvider,
            )
            provider = OpenAIEmbeddingProvider(settings=settings)
            if provider.is_available():
                embedding_provider = provider

        if embedding_provider is None:
            from src.providers.embedding.nomic_embedding_provider import (
                NomicEmbeddingProvider,
            )
            provider = NomicEmbeddingProvider(settings=settings)
            if provider.is_available():
                embedding_provider = provider

        if embedding_provider is None:
            raise RuntimeError("No embedding provider available. Check .env config.")

        # LLM provider for metadata extraction
        llm_provider = None
        if settings.anthropic_api_key:
            from src.providers.llm.anthropic_provider import AnthropicLLMProvider
            llm_provider = AnthropicLLMProvider(settings=settings)
        elif settings.openai_api_key:
            from src.providers.llm.openai_provider import OpenAILLMProvider
            llm_provider = OpenAILLMProvider(settings=settings)
        else:
            from src.providers.llm.ollama_provider import OllamaLLMProvider
            llm_provider = OllamaLLMProvider(settings=settings)

        vector_store = ChromaDBProvider(
            embedding_provider=embedding_provider,
            persist_directory=settings.chromadb_persist_dir,
            collection_name=settings.chromadb_collection,
        )

        from src.services.ingestion.ingestion_service import IngestionService

        self._service = IngestionService(
            chunker=TextChunker(),
            metadata_extractor=MetadataExtractor(llm=llm_provider),
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )
        self._embedding_name = embedding_provider.get_provider_name()
        self._initialized = True
        print(f"  DB ingestion initialized (embedding: {self._embedding_name})")

    async def ingest_transcript(self, file_path: str, title: str) -> tuple[int, int]:
        """Ingest a single transcript file into ChromaDB.

        Returns (chunks_created, total_tokens).
        """
        self._init_service()

        from src.services.ingestion.source_processors.article_processor import (
            ArticleProcessor,
        )
        from src.services.ingestion.chunker import TextChunker
        from src.models.rag import DocumentChunk

        # Process file into initial chunks (source_type=interview, tier=3)
        processor = ArticleProcessor()
        raw_chunks = processor.process_file(
            file_path, source_type="interview", tier=3
        )
        if not raw_chunks:
            return 0, 0

        source_id = raw_chunks[0].source_id

        # Re-chunk into embedding-sized windows
        chunker = TextChunker()
        all_chunks: list[DocumentChunk] = []
        for rc in raw_chunks:
            metadata = {
                "source_id": rc.source_id,
                "source_title": title,
                "source_type": "interview",
                "citation_tier": 3,
                "author": "Resident Advisor",
            }
            chunks = chunker.chunk(rc.text, metadata)
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0, 0

        # Tag, embed, store via the ingestion service
        result = await self._service._tag_embed_store(
            all_chunks, source_id=source_id, title=title, start=time.monotonic()
        )

        return result.chunks_created, result.total_tokens


class RAExchangeTranscriber:
    """Orchestrates the full RA Exchange transcription + ingestion pipeline.

    This is the top-level coordinator that runs the 3-phase pipeline:
      Phase 1: Discover episodes from SoundCloud
      Phase 2: Estimate cost and display summary
      Phase 3: Batch transcribe + ingest (configurable batch size)

    Each batch follows the pattern: transcribe N episodes -> ingest N transcripts.
    This batching strategy provides regular progress updates and limits the
    blast radius of failures (a bad batch does not lose prior work).
    """

    def __init__(
        self,
        api_key: str,
        output_dir: str = "transcripts/ra_exchange",
        batch_size: int = 5,
        skip_ingest: bool = False,
    ):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._batch_size = batch_size
        self._skip_ingest = skip_ingest

        self._discovery = EpisodeDiscovery()
        self._transcription = TranscriptionService(api_key)
        self._progress = ProgressTracker(self._output_dir / "_progress.json")
        self._ingestor = DBIngestor() if not skip_ingest else None

    def run(
        self,
        dry_run: bool = False,
        retry_failed: bool = False,
        include_playlists: bool = False,
        max_batches: int = 0,
    ) -> None:
        """Main pipeline: discover -> estimate -> batch(transcribe -> ingest)."""
        print("=" * 60)
        print("RA EXCHANGE PODCAST TRANSCRIBER")
        limit_str = f" | Max batches: {max_batches}" if max_batches else ""
        print(f"  Batch size: {self._batch_size} | DB ingest: {'OFF' if self._skip_ingest else 'ON'}{limit_str}")
        print("=" * 60)
        print()

        # Phase 1: Discover episodes
        print("Phase 1: Discovering episodes from SoundCloud...")
        episodes = self._discovery.discover(include_playlists=include_playlists)
        print(f"\n  Total unique episodes found: {len(episodes)}")
        print(f"  Already transcribed: {self._progress.completed_count}")
        print(f"  Already ingested:    {self._progress.ingested_count}")
        print(f"  Previously failed:   {self._progress.failed_count}")
        print()

        # Filter
        if retry_failed:
            failed_urls = self._progress.failed_urls
            episodes = [ep for ep in episodes if ep["url"] in failed_urls]
            print(f"  Retrying {len(episodes)} failed episodes.")
        else:
            episodes = [
                ep for ep in episodes
                if not self._progress.is_completed(ep["url"])
            ]
            print(f"  Episodes to transcribe: {len(episodes)}")

        if not episodes:
            print("\n  Nothing to do. All episodes already transcribed.")
            return

        # Phase 2: Cost estimate
        avg_duration_sec = 50 * 60
        total_minutes = sum(
            (ep["duration"] or avg_duration_sec) / 60 for ep in episodes
        )
        known_count = sum(1 for ep in episodes if ep["duration"])
        estimated_cost = total_minutes * TranscriptionService.COST_PER_MINUTE
        num_batches = (len(episodes) + self._batch_size - 1) // self._batch_size
        if max_batches > 0:
            num_batches = min(num_batches, max_batches)
            episodes = episodes[: num_batches * self._batch_size]
            # Recalculate cost for limited run
            total_minutes = sum(
                (ep["duration"] or avg_duration_sec) / 60 for ep in episodes
            )
            estimated_cost = total_minutes * TranscriptionService.COST_PER_MINUTE

        print()
        print(f"  Estimated total audio: {total_minutes:.0f} min ({total_minutes / 60:.1f} hrs)")
        if known_count < len(episodes):
            print(f"    ({known_count} known durations, {len(episodes) - known_count} estimated at 50min avg)")
        print(f"  Estimated cost:        ${estimated_cost:.2f}")
        print(f"  Rate:                  ${TranscriptionService.COST_PER_MINUTE}/min")
        print(f"  Batches:               {num_batches} x {self._batch_size}")
        print()

        if dry_run:
            print("DRY RUN — no transcriptions performed.")
            print("\nEpisode list:")
            for i, ep in enumerate(episodes, 1):
                dur = f"{ep['duration'] / 60:.0f}min" if ep["duration"] else "??min"
                batch_num = (i - 1) // self._batch_size + 1
                if (i - 1) % self._batch_size == 0:
                    print(f"\n  --- Batch {batch_num}/{num_batches} ---")
                print(f"  {i:4d}. {ep['title']} ({dur})")
            return

        # Phase 3: Batch transcribe + ingest
        print("Phase 3: Batch transcription + ingestion...")
        print("=" * 60)

        for batch_idx in range(num_batches):
            batch_start = batch_idx * self._batch_size
            batch_end = min(batch_start + self._batch_size, len(episodes))
            batch = episodes[batch_start:batch_end]

            print(f"\n{'='*60}")
            print(f"BATCH {batch_idx + 1}/{num_batches} — Episodes {batch_start + 1}-{batch_end}")
            print(f"{'='*60}")

            # Step A: Transcribe this batch
            batch_transcripts: list[dict] = []  # {"url", "title", "file_path", "filename"}

            for i, ep in enumerate(batch, 1):
                title = ep["title"]
                global_idx = batch_start + i
                print(f"\n  [{global_idx}/{len(episodes)}] {title}")

                try:
                    transcript = self._download_and_transcribe(ep)
                    filename = self._save_transcript(title, transcript)
                    file_path = str(self._output_dir / filename)
                    self._progress.mark_completed(ep["url"], title, filename)
                    print(f"    TRANSCRIBED -> {filename}")

                    batch_transcripts.append({
                        "url": ep["url"],
                        "title": title,
                        "file_path": file_path,
                        "filename": filename,
                    })

                except Exception as exc:
                    error_msg = str(exc)[:300]
                    self._progress.mark_failed(ep["url"], title, error_msg)
                    print(f"    FAIL: {error_msg}")

                time.sleep(1)

            # Step B: Ingest this batch into DB
            if batch_transcripts and self._ingestor:
                print(f"\n  Ingesting {len(batch_transcripts)} transcripts into ChromaDB...")

                for t in batch_transcripts:
                    if self._progress.is_ingested(t["url"]):
                        print(f"    SKIP (already ingested): {t['title']}")
                        continue

                    try:
                        chunks, tokens = asyncio.run(
                            self._ingestor.ingest_transcript(t["file_path"], t["title"])
                        )
                        self._progress.mark_ingested(t["url"], chunks, tokens)
                        print(f"    INGESTED: {t['title']} ({chunks} chunks, {tokens} tokens)")

                    except Exception as exc:
                        print(f"    INGEST FAIL: {t['title']} — {str(exc)[:200]}")

            # Batch summary
            print(f"\n  Batch {batch_idx + 1} complete.")
            print(f"  Running totals: {self._progress.completed_count} transcribed, "
                  f"{self._progress.ingested_count} ingested, "
                  f"{self._progress.failed_count} failed")

        # Final summary
        print()
        print("=" * 60)
        print("ALL BATCHES COMPLETE")
        print(f"  Transcribed: {self._progress.completed_count}")
        print(f"  Ingested:    {self._progress.ingested_count}")
        print(f"  Failed:      {self._progress.failed_count}")
        print(f"  Output dir:  {self._output_dir.resolve()}")
        print("=" * 60)

    def _download_and_transcribe(self, episode: dict) -> str:
        """Download audio to temp dir, transcribe, auto-delete audio.

        Uses a TemporaryDirectory so the downloaded audio file is automatically
        cleaned up after transcription completes, preventing disk space
        accumulation when processing hundreds of episodes.

        yt-dlp flags:
          -x: Extract audio only (no video)
          --audio-format mp3: Convert to mp3 (Whisper compatible)
          --audio-quality 5: Medium quality (smaller file, faster upload)
          --no-playlist: Download single track only
        """
        with tempfile.TemporaryDirectory(prefix="ra_exchange_") as tmpdir:
            tmp_path = Path(tmpdir)

            result = subprocess.run(
                [
                    "yt-dlp",
                    "-x",
                    "--audio-format", "mp3",
                    "--audio-quality", "5",
                    "--no-playlist",
                    "--no-warnings",
                    "-o", str(tmp_path / "audio.%(ext)s"),
                    episode["url"],
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                raise RuntimeError(f"yt-dlp failed: {result.stderr[:200]}")

            audio_files = list(tmp_path.glob("audio.*"))
            if not audio_files:
                raise RuntimeError("yt-dlp produced no output file")

            audio_path = audio_files[0]
            size_mb = audio_path.stat().st_size / (1024 * 1024)
            print(f"    Downloaded: {size_mb:.1f} MB -> temp")

            print("    Transcribing via TogetherAI...")
            transcript = self._transcription.transcribe_file(audio_path)

            return transcript

    def _save_transcript(self, title: str, text: str) -> str:
        """Save transcript text to a .txt file. Returns the filename."""
        safe_name = re.sub(r"[^\w\s\-.]", "_", title)
        safe_name = re.sub(r"\s+", "_", safe_name).strip("_")
        filename = f"{safe_name}.txt"
        filepath = self._output_dir / filename
        filepath.write_text(text, encoding="utf-8")
        return filename


def main() -> None:
    """CLI entry point: parse arguments, resolve API key, run the pipeline.

    The API key resolution chain:
      1. TOGETHER_API_KEY environment variable
      2. OPENAI_API_KEY environment variable
      3. TOGETHER_API_KEY or OPENAI_API_KEY from .env file
    The key is required for transcription but not for --dry-run mode.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Transcribe RA Exchange podcasts via TogetherAI and ingest into ChromaDB"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover episodes and show cost estimate only",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of episodes per batch (default: 5)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Stop after N batches (0 = unlimited, default: 0)",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Transcribe only — do not ingest into ChromaDB",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry only previously failed episodes",
    )
    parser.add_argument(
        "--include-playlists",
        action="store_true",
        help="Also scan RA SoundCloud playlists for additional episodes",
    )
    parser.add_argument(
        "--output-dir",
        default="transcripts/ra_exchange",
        help="Directory to save transcripts (default: transcripts/ra_exchange)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("TOGETHER_API_KEY") or os.environ.get("OPENAI_API_KEY")

    # Also try loading from .env file if not in environment
    if not api_key:
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("OPENAI_API_KEY=") and "together" not in line.lower():
                    # Only use if OPENAI_BASE_URL points to TogetherAI
                    continue
                if line.startswith("TOGETHER_API_KEY=") or line.startswith("OPENAI_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val and not val.startswith("your-"):
                        api_key = val
                        break

    if not api_key and not args.dry_run:
        print("Error: Set TOGETHER_API_KEY environment variable")
        print("  (or OPENAI_API_KEY in .env with OPENAI_BASE_URL=https://api.together.xyz/v1)")
        sys.exit(1)

    transcriber = RAExchangeTranscriber(
        api_key=api_key or "dry-run",
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        skip_ingest=args.skip_ingest,
    )
    transcriber.run(
        dry_run=args.dry_run,
        retry_failed=args.retry_failed,
        include_playlists=args.include_playlists,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
