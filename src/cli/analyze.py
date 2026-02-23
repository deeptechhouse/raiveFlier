# =============================================================================
# src/cli/analyze.py — CLI Analyze Command (Upload + Run Pipeline)
# =============================================================================
#
# Standalone CLI tool for analyzing a rave/concert flier image from the
# command line. This bypasses the web frontend and API server entirely,
# running the full 5-phase pipeline directly:
#
#   Phase 1: OCR         — Extract text from the flier image
#   Phase 2: Entities    — Identify artists, venues, dates, promoters
#   Phase 3: Research    — Look up entities in music databases and web search
#   Phase 4: Interconnect — Map relationships between entities (graph analysis)
#   Phase 5: Output      — Format and present results
#
# Typical usage:
#   python -m src.cli.analyze /path/to/flier.jpg          # Human-readable text
#   python -m src.cli.analyze /path/to/flier.png --json    # Machine-readable JSON
#   python -m src.cli.analyze flier.webp -o results.json   # Write to file
#
# Input constraints:
#   - Supported formats: JPEG, PNG, WEBP
#   - Maximum file size: 10 MB
#   - Image is SHA-256 hashed for deduplication tracking
#
# Output modes:
#   - Text (default): Formatted report with sections for OCR, entities,
#     research, interconnections, and narrative
#   - JSON (--json): Structured data suitable for programmatic consumption
#
# The --quiet flag (auto-enabled with --json) redirects all structlog and
# stdlib logging to stderr at WARNING+ level, keeping stdout clean for
# the analysis output only.
# =============================================================================

"""Standalone CLI for running the full raiveFlier analysis pipeline.

Usage::

    python -m src.cli.analyze /path/to/flier.jpg
    python -m src.cli.analyze /path/to/flier.png --json
    python -m src.cli.analyze /path/to/flier.webp --output results.json

Accepts JPEG, PNG, or WEBP images up to 10 MB.  Runs all five pipeline
phases (OCR, entity extraction, research, interconnection, output) and
prints a formatted summary or JSON to stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path
from uuid import uuid4

# FlierImage is the Pydantic model representing an uploaded flier.
# It holds metadata (filename, content type, hash) but not the raw bytes
# directly — those are stored in a private Pydantic field (_image_data).
from src.models.flier import FlierImage


# -- File validation constants --
# These guard against unsupported or oversized uploads before the pipeline runs.
_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
_CONTENT_TYPE_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB — matches the API upload limit


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_text_output(state) -> str:  # noqa: ANN001
    """Format the pipeline state as a human-readable text report.

    Walks through each section of the PipelineState and builds a
    multi-line string report. Sections are only included if the
    pipeline produced data for them (graceful degradation if a
    phase failed or was skipped).
    """
    lines: list[str] = []
    sep = "=" * 60

    lines.append(sep)
    lines.append("  raiveFlier — Analysis Report")
    lines.append(sep)
    lines.append("")

    # -- OCR results section --
    # Shows which OCR provider was used and its confidence score.
    ocr = state.ocr_result
    if ocr:
        lines.append(f"OCR Provider: {ocr.provider_used}  |  Confidence: {ocr.confidence:.0%}")
        lines.append(f"Processing time: {ocr.processing_time:.2f}s")
        lines.append("")

    # -- Entities section --
    # Prefers confirmed_entities (post-research validation) over raw extracted_entities.
    # This is because Phase 3 (research) may correct or refine entities from Phase 2.
    entities = state.confirmed_entities or state.extracted_entities
    if entities:
        lines.append("EXTRACTED ENTITIES")
        lines.append("-" * 40)
        if entities.artists:
            artist_names = [a.text for a in entities.artists]
            lines.append(f"  Artists:   {', '.join(artist_names)}")
        if entities.venue:
            lines.append(f"  Venue:     {entities.venue.text}")
        if entities.date:
            lines.append(f"  Date:      {entities.date.text}")
        if entities.promoter:
            lines.append(f"  Promoter:  {entities.promoter.text}")
        if entities.event_name:
            lines.append(f"  Event:     {entities.event_name.text}")
        if entities.genre_tags:
            lines.append(f"  Genres:    {', '.join(entities.genre_tags)}")
        if entities.ticket_price:
            lines.append(f"  Price:     {entities.ticket_price}")
        lines.append("")

    # -- Research Results section --
    # Each research result corresponds to one entity (artist, venue, promoter, date)
    # and contains detailed background info from music databases and web search.
    if state.research_results:
        lines.append("RESEARCH RESULTS")
        lines.append("-" * 40)
        for r in state.research_results:
            lines.append(f"\n  [{r.entity_type.value}] {r.entity_name}")
            lines.append(f"  Confidence: {r.confidence:.0%}  |  Sources: {len(r.sources_consulted)}")

            if r.artist:
                a = r.artist
                if a.profile_summary:
                    bio = a.profile_summary[:200] + ("..." if len(a.profile_summary) > 200 else "")
                    lines.append(f"  Bio: {bio}")
                if a.labels:
                    label_names = [lb.name for lb in a.labels[:8]]
                    lines.append(f"  Labels: {', '.join(label_names)}")
                if a.releases:
                    lines.append(f"  Releases: {len(a.releases)} found")
                    for rel in a.releases[:5]:
                        year_str = f" ({rel.year})" if rel.year else ""
                        lines.append(f"    - {rel.title} [{rel.label}]{year_str}")
                    if len(a.releases) > 5:
                        lines.append(f"    ... and {len(a.releases) - 5} more")

            if r.venue:
                v = r.venue
                if v.history:
                    desc = v.history[:200] + ("..." if len(v.history) > 200 else "")
                    lines.append(f"  History: {desc}")
                if v.city:
                    lines.append(f"  City: {v.city}")

            if r.promoter:
                p = r.promoter
                if p.event_history:
                    lines.append(f"  Events: {', '.join(p.event_history[:5])}")

            if r.date_context:
                dc = r.date_context
                if dc.scene_context:
                    lines.append(f"  Scene: {dc.scene_context[:200]}")
                if dc.cultural_context:
                    lines.append(f"  Culture: {dc.cultural_context[:200]}")

            if r.warnings:
                for w in r.warnings:
                    lines.append(f"  ⚠ {w}")

        lines.append("")

    # -- Interconnection Map section --
    # The interconnection map is a graph of relationships between entities:
    # nodes = entities, edges = relationships (e.g., "DJ plays at venue"),
    # patterns = higher-level observations (e.g., "Berlin techno scene connection").
    imap = state.interconnection_map
    if imap:
        lines.append("INTERCONNECTIONS")
        lines.append("-" * 40)
        lines.append(f"  Nodes: {len(imap.nodes)}  |  Edges: {len(imap.edges)}  |  Patterns: {len(imap.patterns)}")

        if imap.edges:
            lines.append("")
            lines.append("  Relationships:")
            for edge in imap.edges[:10]:
                lines.append(f"    {edge.source} --[{edge.relationship_type}]--> {edge.target}")
                if edge.details:
                    lines.append(f"      {edge.details[:120]}")

        if imap.patterns:
            lines.append("")
            lines.append("  Patterns:")
            for pat in imap.patterns[:5]:
                lines.append(f"    [{pat.pattern_type}] {pat.description[:120]}")

        if imap.narrative:
            lines.append("")
            lines.append("  Narrative:")
            for para in imap.narrative.split("\n\n"):
                wrapped = para.strip()
                if wrapped:
                    lines.append(f"    {wrapped[:300]}")

        lines.append("")

    # -- Completion --
    if state.completed_at:
        lines.append(sep)
        lines.append(f"  Completed: {state.completed_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(sep)

    return "\n".join(lines)


def _format_json_output(state) -> str:  # noqa: ANN001
    """Serialize the pipeline state to JSON.

    Produces a flat JSON document with top-level keys for each pipeline
    phase. Uses Pydantic's model_dump(mode="json") for proper serialization
    of nested models, enums, datetimes, etc. The default=str fallback
    handles any types that json.dumps cannot serialize natively.
    """
    output: dict = {
        "session_id": state.session_id,
    }

    if state.ocr_result:
        output["ocr"] = {
            "raw_text": state.ocr_result.raw_text,
            "confidence": state.ocr_result.confidence,
            "provider_used": state.ocr_result.provider_used,
            "processing_time": state.ocr_result.processing_time,
        }

    entities = state.confirmed_entities or state.extracted_entities
    if entities:
        output["entities"] = entities.model_dump(mode="json")

    if state.research_results:
        output["research_results"] = [r.model_dump(mode="json") for r in state.research_results]

    if state.interconnection_map:
        output["interconnection_map"] = state.interconnection_map.model_dump(mode="json")

    if state.completed_at:
        output["completed_at"] = state.completed_at.isoformat()

    if state.errors:
        output["errors"] = [
            {"phase": e.phase.value, "message": e.message, "recoverable": e.recoverable}
            for e in state.errors
        ]

    return json.dumps(output, indent=2, default=str)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def _suppress_logs() -> None:
    """Redirect all structlog and stdlib logging to stderr at WARNING+ level.

    Must be called BEFORE importing src.main so loggers are configured
    before they get cached. This is critical because structlog caches
    logger instances on first use — if we configure after import, the
    cached loggers will still use the old (verbose) configuration.

    This function is called when --quiet or --json flags are used, ensuring
    that stdout contains only the analysis output (no interleaved log lines).
    """
    import logging
    import os

    import structlog

    # Force WARNING level via environment variable so any code that reads
    # LOG_LEVEL at import time will also respect the quiet setting.
    os.environ["LOG_LEVEL"] = "WARNING"

    # Configure structlog to output plain text to stderr (no JSON in CLI mode).
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.WARNING)
    root.addHandler(handler)
    root.setLevel(logging.WARNING)

    # Suppress chatty HTTP client and database libraries that log at INFO/DEBUG.
    # Without this, every API request would generate multiple log lines.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


async def _run(image_path: Path, json_output: bool, output_file: str | None, quiet: bool) -> int:
    """Load image and run the full pipeline.

    This is the core async function that:
      1. Validates the input file (exists, correct extension, within size limit)
      2. Constructs a FlierImage model instance with the image bytes
      3. Invokes run_pipeline() which executes all 5 analysis phases
      4. Formats and outputs the results (text or JSON, stdout or file)

    Returns 0 on success, 1 on validation error.
    """
    # Deferred import: src.main bootstraps the full application (FastAPI app,
    # providers, services). We import it here so the CLI can validate the
    # input file quickly before spending time on heavy initialization.
    from src.main import run_pipeline

    # -- Validate file --
    if not image_path.exists():
        print(f"Error: File not found: {image_path}", file=sys.stderr)
        return 1

    suffix = image_path.suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        print(
            f"Error: Unsupported file type: {suffix}. "
            f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}",
            file=sys.stderr,
        )
        return 1

    image_data = image_path.read_bytes()
    if len(image_data) > _MAX_FILE_SIZE:
        print(
            f"Error: File too large: {len(image_data):,} bytes. Maximum: {_MAX_FILE_SIZE:,} bytes.",
            file=sys.stderr,
        )
        return 1

    # -- Build FlierImage model --
    # FlierImage is a Pydantic model that the pipeline expects as input.
    # We construct it manually here (in the API route, FastAPI does this
    # from the multipart upload). The SHA-256 hash enables duplicate detection
    # via perceptual hashing — if the same flier is analyzed twice, the
    # system can recognize it.
    content_type = _CONTENT_TYPE_MAP.get(suffix, "image/jpeg")
    image_hash = hashlib.sha256(image_data).hexdigest()
    session_id = str(uuid4())  # Unique ID for this analysis session

    flier_image = FlierImage(
        id=session_id,
        filename=image_path.name,
        content_type=content_type,
        file_size=len(image_data),
        image_hash=image_hash,
    )
    # Inject the raw image bytes into the Pydantic model's private field.
    # This is a workaround because Pydantic private fields are not settable
    # via the constructor — we must write directly to __pydantic_private__.
    # The pipeline reads _image_data when sending the image to OCR providers.
    flier_image.__pydantic_private__["_image_data"] = image_data

    # -- Run the 5-phase analysis pipeline --
    # Progress messages go to stderr so they don't interfere with stdout output.
    print(f"Analyzing: {image_path.name} ({len(image_data):,} bytes)", file=sys.stderr)
    start = time.monotonic()

    state = await run_pipeline(flier_image)

    elapsed = time.monotonic() - start
    print(f"Done in {elapsed:.1f}s", file=sys.stderr)

    # -- Output --
    if json_output:
        text = _format_json_output(state)
    else:
        text = _format_text_output(state)

    if output_file:
        Path(output_file).write_text(text, encoding="utf-8")
        print(f"Results written to: {output_file}", file=sys.stderr)
    else:
        print(text)

    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the analyze CLI.

    Arguments:
      image (positional) — Path to the flier image file (JPEG, PNG, or WEBP)
      --json             — Output results as JSON instead of formatted text
      --output / -o      — Write results to a file instead of stdout
      --quiet / -q       — Suppress all log output (auto-enabled with --json)
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.analyze",
        description=(
            "Analyze a rave flier image from the command line. "
            "Runs OCR, entity extraction, research, and interconnection analysis."
        ),
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to the flier image (JPEG, PNG, or WEBP).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON instead of formatted text.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Write results to a file instead of stdout.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress log output (useful with --json for clean stdout).",
    )
    return parser


def main() -> None:
    """CLI entry point for the analyze tool.

    Parses arguments, configures logging, resolves the image path to an
    absolute path, and runs the async pipeline via asyncio.run(). The
    process exits with code 0 on success or 1 on any validation error.
    """
    parser = _build_parser()
    args = parser.parse_args()

    # JSON mode implies quiet — we never want log lines mixed into JSON output.
    quiet = args.quiet or args.json_output
    if quiet:
        # Must suppress logs BEFORE any src.main imports happen (see docstring).
        _suppress_logs()

    # Resolve to absolute path so the pipeline can find the file regardless
    # of the current working directory.
    image_path = Path(args.image).resolve()
    exit_code = asyncio.run(_run(image_path, args.json_output, args.output, quiet))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
