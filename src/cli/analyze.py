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

from src.models.flier import FlierImage


_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
_CONTENT_TYPE_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_text_output(state) -> str:  # noqa: ANN001
    """Format the pipeline state as a human-readable text report."""
    lines: list[str] = []
    sep = "=" * 60

    lines.append(sep)
    lines.append("  raiveFlier — Analysis Report")
    lines.append(sep)
    lines.append("")

    # -- OCR --
    ocr = state.ocr_result
    if ocr:
        lines.append(f"OCR Provider: {ocr.provider_used}  |  Confidence: {ocr.confidence:.0%}")
        lines.append(f"Processing time: {ocr.processing_time:.2f}s")
        lines.append("")

    # -- Entities --
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

    # -- Research Results --
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

    # -- Interconnection Map --
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
    """Serialize the pipeline state to JSON."""
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
    before they get cached.
    """
    import logging
    import os

    import structlog

    os.environ["LOG_LEVEL"] = "WARNING"

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

    # Also suppress httpx/httpcore logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


async def _run(image_path: Path, json_output: bool, output_file: str | None, quiet: bool) -> int:
    """Load image and run the full pipeline."""
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

    # -- Build FlierImage --
    content_type = _CONTENT_TYPE_MAP.get(suffix, "image/jpeg")
    image_hash = hashlib.sha256(image_data).hexdigest()
    session_id = str(uuid4())

    flier_image = FlierImage(
        id=session_id,
        filename=image_path.name,
        content_type=content_type,
        file_size=len(image_data),
        image_hash=image_hash,
    )
    flier_image.__pydantic_private__["_image_data"] = image_data

    # -- Run pipeline --
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
    """Build the argparse parser."""
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
    """CLI entry point for the analyze tool."""
    parser = _build_parser()
    args = parser.parse_args()

    quiet = args.quiet or args.json_output  # Always quiet in JSON mode
    if quiet:
        _suppress_logs()

    image_path = Path(args.image).resolve()
    exit_code = asyncio.run(_run(image_path, args.json_output, args.output, quiet))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
