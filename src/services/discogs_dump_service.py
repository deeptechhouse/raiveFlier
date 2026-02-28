# =============================================================================
# src/services/discogs_dump_service.py — Discogs Data Dump Streaming Parser
# =============================================================================
#
# Streaming XML parser for Discogs monthly data dumps (~90+ GB uncompressed).
# Extracts artists, labels, and master releases tagged with Electronic genre
# and target styles (house, techno, electro, drum & bass), then converts the
# structured data into prose-format text files suitable for raiveFlier's
# RAG ingestion pipeline.
#
# Architecture:
#   This service sits outside the normal layered pipeline — it's a data
#   preparation tool that runs locally (not on the 512 MB Render instance).
#   Its output is plain text files in data/discogs_corpus/ that get ingested
#   via the existing IngestionService.ingest_directory() method.
#
# Design decisions:
#   - Uses xml.etree.ElementTree.iterparse (SAX-style) to process one
#     <artist>/<label>/<master> element at a time. Peak RAM stays ~100-200 MB
#     even for 90+ GB XML files because each element is cleared after processing.
#   - Supports gzip-compressed input directly (no need to decompress first).
#   - Saves intermediate parsed data as JSON Lines (.jsonl) checkpoints so
#     the expensive XML parse only needs to run once. The prose conversion
#     step reads from these checkpoints.
#   - Prose output matches the narrative style of existing corpus files
#     (e.g., genres_subgenres.txt, major_djs_artists.txt) for embedding
#     consistency.
#
# Data flow:
#   Discogs .xml.gz → iterparse filter → JSONL checkpoint → prose .txt files
#                                                           → ingest_directory()
#
# Why not use the Discogs API instead?
#   The API is rate-limited to 60 req/min (authenticated). Fetching ~2M
#   electronic releases would take ~23 days of continuous polling. The data
#   dump approach processes everything in ~30-60 minutes locally.
# =============================================================================

from __future__ import annotations

import gzip
import json
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import IO, Iterator, TextIO

import structlog

logger = structlog.get_logger(logger_name=__name__)


# ─── TARGET GENRE / STYLE CONFIGURATION ───────────────────────────────────────
# Discogs uses a two-level taxonomy: Genre (top level) and Style (sub-genre).
# We filter by Genre = "Electronic" and then group by style clusters.
# Each cluster maps a human-readable name to the set of Discogs style values
# that belong to it. A single release can have multiple styles.

TARGET_GENRE = "Electronic"

# Maps cluster name → frozenset of Discogs style strings (case-sensitive,
# matching exactly how Discogs stores them in the XML dumps).
STYLE_CLUSTERS: dict[str, frozenset[str]] = {
    "house": frozenset({
        "House", "Deep House", "Acid House", "Progressive House",
        "Tech House", "Tribal House", "Funky House", "Garage House",
        "Disco House", "Italo House", "Minimal", "Microhouse",
        "Electro House", "Ghetto House", "Hip-House", "Hard House",
        "Euro House", "Ambient House",
    }),
    "techno": frozenset({
        "Techno", "Minimal Techno", "Dub Techno", "Industrial",
        "Acid", "Detroit Techno", "Hard Techno", "Schranz",
        "Nortec", "Goa Trance", "Ambient",
    }),
    "dnb": frozenset({
        "Drum n Bass", "Jungle", "Breakbeat", "Liquid Funk",
        "Darkstep", "Neurofunk", "Ragga Jungle", "Hardstep",
        "Intelligent", "Breakcore",
    }),
    "electro": frozenset({
        "Electro", "Electro Funk", "Miami Bass",
    }),
}

# Flattened set for fast membership testing during XML parsing.
ALL_TARGET_STYLES: frozenset[str] = frozenset().union(*STYLE_CLUSTERS.values())


# ─── PARSED DATA MODELS ──────────────────────────────────────────────────────
# Simple dataclasses for the intermediate JSONL checkpoint format.
# These hold only the fields we need for prose conversion — not the full
# Discogs schema (which includes barcodes, formats, images, etc.).

@dataclass
class ParsedArtist:
    """Artist record extracted from discogs_*_artists.xml."""
    discogs_id: int
    name: str
    real_name: str = ""
    profile: str = ""
    aliases: list[str] = field(default_factory=list)
    name_variations: list[str] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    members: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    # Styles are not on artist records directly — they come from
    # cross-referencing with releases. We populate this during the
    # optional enrichment step or leave it empty.
    styles: list[str] = field(default_factory=list)


@dataclass
class ParsedLabel:
    """Label record extracted from discogs_*_labels.xml."""
    discogs_id: int
    name: str
    profile: str = ""
    contact_info: str = ""
    parent_label: str = ""
    sublabels: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    styles: list[str] = field(default_factory=list)


@dataclass
class ParsedMaster:
    """Master release record extracted from discogs_*_masters.xml."""
    discogs_id: int
    title: str
    year: int = 0
    artists: list[str] = field(default_factory=list)
    genres: list[str] = field(default_factory=list)
    styles: list[str] = field(default_factory=list)
    notes: str = ""
    # Data quality field from Discogs (Correct, Needs Vote, etc.)
    data_quality: str = ""


@dataclass
class ParsedRelease:
    """Release record extracted from discogs_*_releases.xml.

    Used for cross-referencing artists/labels with styles when the
    artist/label XML files don't carry style information directly.
    Only key fields are retained to keep memory bounded.
    """
    discogs_id: int
    title: str
    year: int = 0
    artists: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    genres: list[str] = field(default_factory=list)
    styles: list[str] = field(default_factory=list)
    notes: str = ""
    master_id: int = 0
    country: str = ""
    data_quality: str = ""


# ─── STREAMING XML PARSER ────────────────────────────────────────────────────

class DiscogsDumpParser:
    """Streaming parser for Discogs monthly XML data dumps.

    Uses iterparse to process one element at a time, keeping peak memory
    at ~100-200 MB regardless of the input file size (which can exceed
    90 GB uncompressed for the releases dump).

    The parser supports both plain XML and gzip-compressed (.gz) input
    files. Compressed files are decompressed on-the-fly during parsing.

    Parameters
    ----------
    output_dir:
        Directory for JSONL checkpoint files and final corpus text.
        Created automatically if it doesn't exist.
    """

    def __init__(self, output_dir: str = "./data/discogs_corpus") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        # Subdirectory for intermediate JSONL checkpoints.
        self._checkpoint_dir = self._output_dir / "checkpoints"
        self._checkpoint_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # File I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _open_xml(file_path: str) -> IO[bytes]:
        """Open an XML file, transparently decompressing .gz files.

        Returns a binary file handle suitable for iterparse. For .gz files,
        uses gzip.open which streams decompression — the full uncompressed
        content is never held in memory.
        """
        if file_path.endswith(".gz"):
            return gzip.open(file_path, "rb")
        return open(file_path, "rb")  # noqa: SIM115

    # ------------------------------------------------------------------
    # Artist parsing
    # ------------------------------------------------------------------

    def parse_artists(
        self,
        xml_path: str,
        artist_style_map: dict[str, set[str]] | None = None,
    ) -> int:
        """Parse the artists XML dump, filtering for electronic music artists.

        Since the artists XML doesn't contain genre/style information,
        filtering requires a pre-built map of artist_name → styles from
        the releases dump. If no map is provided, ALL artists with non-empty
        profiles are extracted (to be filtered during prose generation).

        Parameters
        ----------
        xml_path:
            Path to discogs_*_artists.xml or .xml.gz file.
        artist_style_map:
            Optional dict mapping lowercased artist names to their associated
            Discogs styles (built from releases/masters). When provided, only
            artists appearing in this map are saved.

        Returns
        -------
        int
            Number of artists written to the checkpoint file.
        """
        checkpoint = self._checkpoint_dir / "artists.jsonl"
        count = 0
        skipped = 0
        start = time.monotonic()

        logger.info("parse_artists_start", xml_path=xml_path)

        with self._open_xml(xml_path) as fh, open(checkpoint, "w") as out:
            # iterparse with "end" events fires when the closing tag is
            # reached — at that point, the full element subtree is available.
            context = ET.iterparse(fh, events=("end",))

            for event, elem in context:
                if elem.tag != "artist":
                    continue

                artist = self._extract_artist(elem)

                # Free the element's memory immediately. Without this,
                # iterparse accumulates all parsed elements in the tree
                # root, eventually consuming all available RAM.
                elem.clear()

                if artist is None:
                    skipped += 1
                    continue

                # Filter by style map if provided.
                if artist_style_map is not None:
                    key = artist.name.lower().strip()
                    if key not in artist_style_map:
                        skipped += 1
                        continue
                    artist.styles = sorted(artist_style_map[key])
                elif not artist.profile.strip():
                    # Without a style map, skip artists with empty profiles
                    # since they won't produce useful prose.
                    skipped += 1
                    continue

                out.write(json.dumps(asdict(artist)) + "\n")
                count += 1

                if count % 10000 == 0:
                    logger.info(
                        "parse_artists_progress",
                        extracted=count,
                        skipped=skipped,
                    )

        elapsed = time.monotonic() - start
        logger.info(
            "parse_artists_complete",
            extracted=count,
            skipped=skipped,
            elapsed_s=round(elapsed, 1),
            checkpoint=str(checkpoint),
        )
        return count

    @staticmethod
    def _extract_artist(elem: ET.Element) -> ParsedArtist | None:
        """Extract a ParsedArtist from an <artist> XML element.

        Returns None if the element lacks an ID or name (corrupt/incomplete
        records exist in some Discogs dumps).
        """
        id_elem = elem.find("id")
        name_elem = elem.find("name")
        if id_elem is None or name_elem is None:
            return None
        if not id_elem.text or not name_elem.text:
            return None

        # Aliases are stored as <aliases><name>Alias1</name>...</aliases>
        aliases = [
            n.text for n in elem.findall(".//aliases/name")
            if n.text
        ]
        # Name variations (alternate spellings, transliterations)
        namevariations = [
            n.text for n in elem.findall(".//namevariations/name")
            if n.text
        ]
        # Groups this artist is a member of
        groups = [
            n.text for n in elem.findall(".//groups/name")
            if n.text
        ]
        # Members (if this is a group/band)
        members = [
            n.text for n in elem.findall(".//members/name")
            if n.text
        ]
        urls = [
            u.text for u in elem.findall(".//urls/url")
            if u.text
        ]

        profile_elem = elem.find("profile")
        realname_elem = elem.find("realname")

        return ParsedArtist(
            discogs_id=int(id_elem.text),
            name=name_elem.text.strip(),
            real_name=(realname_elem.text or "").strip() if realname_elem is not None else "",
            profile=(profile_elem.text or "").strip() if profile_elem is not None else "",
            aliases=aliases,
            name_variations=namevariations,
            groups=groups,
            members=members,
            urls=urls,
        )

    # ------------------------------------------------------------------
    # Label parsing
    # ------------------------------------------------------------------

    def parse_labels(
        self,
        xml_path: str,
        label_style_map: dict[str, set[str]] | None = None,
    ) -> int:
        """Parse the labels XML dump, filtering for electronic music labels.

        Like artists, labels don't carry genre/style in their own XML records.
        A pre-built label_style_map from the releases dump is used for filtering.

        Parameters
        ----------
        xml_path:
            Path to discogs_*_labels.xml or .xml.gz file.
        label_style_map:
            Optional dict mapping lowercased label names to their associated
            styles. When provided, only labels in this map are saved.

        Returns
        -------
        int
            Number of labels written to the checkpoint file.
        """
        checkpoint = self._checkpoint_dir / "labels.jsonl"
        count = 0
        skipped = 0
        start = time.monotonic()

        logger.info("parse_labels_start", xml_path=xml_path)

        with self._open_xml(xml_path) as fh, open(checkpoint, "w") as out:
            context = ET.iterparse(fh, events=("end",))

            for event, elem in context:
                if elem.tag != "label":
                    continue

                label = self._extract_label(elem)
                elem.clear()

                if label is None:
                    skipped += 1
                    continue

                if label_style_map is not None:
                    key = label.name.lower().strip()
                    if key not in label_style_map:
                        skipped += 1
                        continue
                    label.styles = sorted(label_style_map[key])
                elif not label.profile.strip():
                    skipped += 1
                    continue

                out.write(json.dumps(asdict(label)) + "\n")
                count += 1

                if count % 5000 == 0:
                    logger.info(
                        "parse_labels_progress",
                        extracted=count,
                        skipped=skipped,
                    )

        elapsed = time.monotonic() - start
        logger.info(
            "parse_labels_complete",
            extracted=count,
            skipped=skipped,
            elapsed_s=round(elapsed, 1),
            checkpoint=str(checkpoint),
        )
        return count

    @staticmethod
    def _extract_label(elem: ET.Element) -> ParsedLabel | None:
        """Extract a ParsedLabel from a <label> XML element."""
        id_elem = elem.find("id")
        name_elem = elem.find("name")
        if id_elem is None or name_elem is None:
            return None
        if not id_elem.text or not name_elem.text:
            return None

        profile_elem = elem.find("profile")
        contact_elem = elem.find("contactinfo")
        parent_elem = elem.find("parentLabel")

        sublabels = [
            s.text for s in elem.findall(".//sublabels/label")
            if s.text
        ]
        urls = [
            u.text for u in elem.findall(".//urls/url")
            if u.text
        ]

        return ParsedLabel(
            discogs_id=int(id_elem.text),
            name=name_elem.text.strip(),
            profile=(profile_elem.text or "").strip() if profile_elem is not None else "",
            contact_info=(contact_elem.text or "").strip() if contact_elem is not None else "",
            parent_label=(parent_elem.text or "").strip() if parent_elem is not None else "",
            sublabels=sublabels,
            urls=urls,
        )

    # ------------------------------------------------------------------
    # Master release parsing
    # ------------------------------------------------------------------

    def parse_masters(self, xml_path: str) -> int:
        """Parse the masters XML dump, filtering for target electronic styles.

        Master releases carry genre and style information directly in the XML,
        so no cross-reference map is needed. Only masters with Genre=Electronic
        and at least one target style are extracted.

        Parameters
        ----------
        xml_path:
            Path to discogs_*_masters.xml or .xml.gz file.

        Returns
        -------
        int
            Number of master releases written to the checkpoint file.
        """
        checkpoint = self._checkpoint_dir / "masters.jsonl"
        count = 0
        skipped = 0
        start = time.monotonic()

        logger.info("parse_masters_start", xml_path=xml_path)

        with self._open_xml(xml_path) as fh, open(checkpoint, "w") as out:
            context = ET.iterparse(fh, events=("end",))

            for event, elem in context:
                if elem.tag != "master":
                    continue

                master = self._extract_master(elem)
                elem.clear()

                if master is None:
                    skipped += 1
                    continue

                # Filter: must have Electronic genre and at least one
                # target style from our style clusters.
                if TARGET_GENRE not in master.genres:
                    skipped += 1
                    continue
                if not any(s in ALL_TARGET_STYLES for s in master.styles):
                    skipped += 1
                    continue

                out.write(json.dumps(asdict(master)) + "\n")
                count += 1

                if count % 25000 == 0:
                    logger.info(
                        "parse_masters_progress",
                        extracted=count,
                        skipped=skipped,
                    )

        elapsed = time.monotonic() - start
        logger.info(
            "parse_masters_complete",
            extracted=count,
            skipped=skipped,
            elapsed_s=round(elapsed, 1),
            checkpoint=str(checkpoint),
        )
        return count

    @staticmethod
    def _extract_master(elem: ET.Element) -> ParsedMaster | None:
        """Extract a ParsedMaster from a <master> XML element."""
        master_id = elem.get("id")
        if not master_id:
            return None

        title_elem = elem.find("title")
        year_elem = elem.find("year")
        notes_elem = elem.find("notes")
        dq_elem = elem.find("data_quality")

        artists = [
            a.text for a in elem.findall(".//artists/artist/name")
            if a.text
        ]
        genres = [
            g.text for g in elem.findall(".//genres/genre")
            if g.text
        ]
        styles = [
            s.text for s in elem.findall(".//styles/style")
            if s.text
        ]

        return ParsedMaster(
            discogs_id=int(master_id),
            title=(title_elem.text or "").strip() if title_elem is not None else "",
            year=int(year_elem.text) if year_elem is not None and year_elem.text else 0,
            artists=artists,
            genres=genres,
            styles=styles,
            notes=(notes_elem.text or "").strip() if notes_elem is not None else "",
            data_quality=(dq_elem.text or "").strip() if dq_elem is not None else "",
        )

    # ------------------------------------------------------------------
    # Release parsing (for building artist/label → style cross-reference)
    # ------------------------------------------------------------------

    def build_style_maps(self, xml_path: str) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        """Scan the releases XML to build artist_name → styles and label_name → styles maps.

        This is a prerequisite for parse_artists() and parse_labels() because
        the artist and label XML dumps don't carry genre/style information.
        The only way to know that "Derrick May" is a techno artist is to find
        his releases tagged with Genre=Electronic, Style=Techno.

        This method streams through the entire releases dump (~90+ GB
        uncompressed) but only retains two in-memory dicts (artist names
        and label names to their style sets). Peak memory is ~500 MB for
        the full Discogs database.

        Parameters
        ----------
        xml_path:
            Path to discogs_*_releases.xml or .xml.gz file.

        Returns
        -------
        tuple[dict, dict]
            (artist_style_map, label_style_map) where keys are lowercased
            names and values are sets of Discogs style strings.
        """
        artist_styles: dict[str, set[str]] = {}
        label_styles: dict[str, set[str]] = {}
        releases_seen = 0
        matched = 0
        start = time.monotonic()

        logger.info("build_style_maps_start", xml_path=xml_path)

        with self._open_xml(xml_path) as fh:
            context = ET.iterparse(fh, events=("end",))

            for event, elem in context:
                if elem.tag != "release":
                    continue

                releases_seen += 1

                # Quick genre check before extracting full details.
                genres = [
                    g.text for g in elem.findall(".//genres/genre")
                    if g.text
                ]
                if TARGET_GENRE not in genres:
                    elem.clear()
                    continue

                styles = [
                    s.text for s in elem.findall(".//styles/style")
                    if s.text
                ]
                matching_styles = {s for s in styles if s in ALL_TARGET_STYLES}
                if not matching_styles:
                    elem.clear()
                    continue

                matched += 1

                # Map each artist on this release to the matching styles.
                for artist_elem in elem.findall(".//artists/artist/name"):
                    if artist_elem.text:
                        key = artist_elem.text.lower().strip()
                        if key not in artist_styles:
                            artist_styles[key] = set()
                        artist_styles[key].update(matching_styles)

                # Map each label on this release to the matching styles.
                for label_elem in elem.findall(".//labels/label"):
                    label_name = label_elem.get("name")
                    if label_name:
                        key = label_name.lower().strip()
                        if key not in label_styles:
                            label_styles[key] = set()
                        label_styles[key].update(matching_styles)

                elem.clear()

                if releases_seen % 500000 == 0:
                    logger.info(
                        "build_style_maps_progress",
                        releases_scanned=releases_seen,
                        matched=matched,
                        unique_artists=len(artist_styles),
                        unique_labels=len(label_styles),
                    )

        elapsed = time.monotonic() - start
        logger.info(
            "build_style_maps_complete",
            releases_scanned=releases_seen,
            matched=matched,
            unique_artists=len(artist_styles),
            unique_labels=len(label_styles),
            elapsed_s=round(elapsed, 1),
        )

        # Save style maps to checkpoint so they don't need to be rebuilt.
        self._save_style_map(artist_styles, "artist_style_map.json")
        self._save_style_map(label_styles, "label_style_map.json")

        return artist_styles, label_styles

    def load_style_maps(self) -> tuple[dict[str, set[str]], dict[str, set[str]]] | None:
        """Load previously saved style maps from checkpoint files.

        Returns None if checkpoint files don't exist (need to run
        build_style_maps first).
        """
        artist_path = self._checkpoint_dir / "artist_style_map.json"
        label_path = self._checkpoint_dir / "label_style_map.json"

        if not artist_path.exists() or not label_path.exists():
            return None

        artist_map = self._load_style_map(artist_path)
        label_map = self._load_style_map(label_path)
        logger.info(
            "style_maps_loaded",
            artists=len(artist_map),
            labels=len(label_map),
        )
        return artist_map, label_map

    def _save_style_map(self, style_map: dict[str, set[str]], filename: str) -> None:
        """Serialize a style map to JSON (sets → sorted lists)."""
        path = self._checkpoint_dir / filename
        # Convert sets to sorted lists for JSON serialization.
        serializable = {k: sorted(v) for k, v in style_map.items()}
        with open(path, "w") as f:
            json.dump(serializable, f)
        logger.info("style_map_saved", path=str(path), entries=len(style_map))

    @staticmethod
    def _load_style_map(path: Path) -> dict[str, set[str]]:
        """Deserialize a style map from JSON (lists → sets)."""
        with open(path) as f:
            data = json.load(f)
        return {k: set(v) for k, v in data.items()}

    # ------------------------------------------------------------------
    # Prose corpus generation
    # ------------------------------------------------------------------

    def generate_corpus(self) -> dict[str, int]:
        """Convert JSONL checkpoints into prose-format text files.

        Reads the artists, labels, and masters checkpoint files and produces
        one text file per data-type per style cluster:

            data/discogs_corpus/artists_house.txt
            data/discogs_corpus/artists_techno.txt
            data/discogs_corpus/labels_dnb.txt
            data/discogs_corpus/masters_electro.txt
            ...

        Returns a dict mapping output filename to the number of entries written.

        The prose format matches the existing reference corpus style so that
        embeddings are consistent with the rest of the RAG knowledge base.
        """
        results: dict[str, int] = {}

        # Generate artist prose files grouped by style cluster.
        artist_checkpoint = self._checkpoint_dir / "artists.jsonl"
        if artist_checkpoint.exists():
            counts = self._generate_artist_prose(artist_checkpoint)
            results.update(counts)

        # Generate label prose files grouped by style cluster.
        label_checkpoint = self._checkpoint_dir / "labels.jsonl"
        if label_checkpoint.exists():
            counts = self._generate_label_prose(label_checkpoint)
            results.update(counts)

        # Generate master release prose files grouped by style cluster.
        master_checkpoint = self._checkpoint_dir / "masters.jsonl"
        if master_checkpoint.exists():
            counts = self._generate_master_prose(master_checkpoint)
            results.update(counts)

        logger.info(
            "corpus_generation_complete",
            files=len(results),
            total_entries=sum(results.values()),
        )
        return results

    def _generate_artist_prose(self, checkpoint: Path) -> dict[str, int]:
        """Convert artist JSONL checkpoint to prose text files per style cluster.

        Each artist entry is converted to a prose paragraph containing their
        name, real name, profile/bio, aliases, group memberships, and
        associated styles. This narrative format produces better embeddings
        than raw structured fields.
        """
        # Open one output file handle per style cluster.
        writers: dict[str, TextIO] = {}
        counts: dict[str, int] = {}

        for cluster in STYLE_CLUSTERS:
            path = self._output_dir / f"artists_{cluster}.txt"
            writers[cluster] = open(path, "w")
            counts[f"artists_{cluster}.txt"] = 0
            # Write a header matching the existing corpus file style.
            writers[cluster].write(
                f"Discogs Artist Profiles — {cluster.upper().replace('DNB', 'DRUM & BASS')}\n"
                f"{'=' * 60}\n\n"
                f"Artist profiles extracted from the Discogs database, filtered\n"
                f"for electronic music artists associated with "
                f"{cluster.replace('dnb', 'drum & bass')} styles.\n\n"
            )

        try:
            with open(checkpoint) as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    artist = ParsedArtist(**data)

                    # Determine which style cluster(s) this artist belongs to.
                    matched_clusters = self._match_clusters(artist.styles)
                    if not matched_clusters:
                        # No style info — write to all clusters as uncategorized
                        # only if the profile is substantial enough.
                        if len(artist.profile) > 100:
                            matched_clusters = list(STYLE_CLUSTERS.keys())
                        else:
                            continue

                    prose = self._artist_to_prose(artist)
                    if not prose:
                        continue

                    for cluster in matched_clusters:
                        writers[cluster].write(prose + "\n\n")
                        counts[f"artists_{cluster}.txt"] += 1

        finally:
            for w in writers.values():
                w.close()

        return counts

    def _generate_label_prose(self, checkpoint: Path) -> dict[str, int]:
        """Convert label JSONL checkpoint to prose text files per style cluster."""
        writers: dict[str, TextIO] = {}
        counts: dict[str, int] = {}

        for cluster in STYLE_CLUSTERS:
            path = self._output_dir / f"labels_{cluster}.txt"
            writers[cluster] = open(path, "w")
            counts[f"labels_{cluster}.txt"] = 0
            writers[cluster].write(
                f"Discogs Label Profiles — {cluster.upper().replace('DNB', 'DRUM & BASS')}\n"
                f"{'=' * 60}\n\n"
                f"Label profiles extracted from the Discogs database, filtered\n"
                f"for electronic music labels associated with "
                f"{cluster.replace('dnb', 'drum & bass')} styles.\n\n"
            )

        try:
            with open(checkpoint) as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    label = ParsedLabel(**data)

                    matched_clusters = self._match_clusters(label.styles)
                    if not matched_clusters:
                        if len(label.profile) > 100:
                            matched_clusters = list(STYLE_CLUSTERS.keys())
                        else:
                            continue

                    prose = self._label_to_prose(label)
                    if not prose:
                        continue

                    for cluster in matched_clusters:
                        writers[cluster].write(prose + "\n\n")
                        counts[f"labels_{cluster}.txt"] += 1

        finally:
            for w in writers.values():
                w.close()

        return counts

    def _generate_master_prose(self, checkpoint: Path) -> dict[str, int]:
        """Convert master release JSONL checkpoint to prose text files per style cluster.

        Only masters with non-empty notes fields produce useful prose. Masters
        without notes are reduced to a one-line entry (artist - title, year,
        styles) which still helps entity resolution during flier analysis.
        """
        writers: dict[str, TextIO] = {}
        counts: dict[str, int] = {}

        for cluster in STYLE_CLUSTERS:
            path = self._output_dir / f"masters_{cluster}.txt"
            writers[cluster] = open(path, "w")
            counts[f"masters_{cluster}.txt"] = 0
            writers[cluster].write(
                f"Discogs Master Releases — {cluster.upper().replace('DNB', 'DRUM & BASS')}\n"
                f"{'=' * 60}\n\n"
                f"Key releases extracted from the Discogs database, filtered\n"
                f"for electronic music releases in "
                f"{cluster.replace('dnb', 'drum & bass')} styles.\n\n"
            )

        try:
            with open(checkpoint) as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    master = ParsedMaster(**data)

                    matched_clusters = self._match_clusters(master.styles)
                    if not matched_clusters:
                        continue

                    prose = self._master_to_prose(master)
                    if not prose:
                        continue

                    for cluster in matched_clusters:
                        writers[cluster].write(prose + "\n\n")
                        counts[f"masters_{cluster}.txt"] += 1

        finally:
            for w in writers.values():
                w.close()

        return counts

    # ------------------------------------------------------------------
    # Prose formatters
    # ------------------------------------------------------------------

    @staticmethod
    def _artist_to_prose(artist: ParsedArtist) -> str:
        """Convert a ParsedArtist to a prose paragraph for embedding.

        Output style matches the existing corpus format (narrative sentences,
        not key-value pairs) because embedding models produce better retrieval
        results from natural language than from structured fields.
        """
        parts: list[str] = []

        # Opening line: name and real name.
        if artist.real_name:
            parts.append(f"{artist.name} (real name: {artist.real_name})")
        else:
            parts.append(artist.name)

        # Profile text — the main biographical content.
        if artist.profile:
            # Clean up common Discogs markup artifacts.
            profile = _clean_discogs_markup(artist.profile)
            if profile:
                parts.append(profile)

        # Aliases provide important cross-referencing for flier analysis
        # (a flier might list "Rhythim Is Rhythim" instead of "Derrick May").
        if artist.aliases:
            parts.append(f"Also known as: {', '.join(artist.aliases)}.")

        # Group memberships connect individual artists to collaborative projects.
        if artist.groups:
            parts.append(f"Member of: {', '.join(artist.groups)}.")

        # Members (if this is a group/project).
        if artist.members:
            parts.append(f"Members: {', '.join(artist.members)}.")

        # Style tags — explicit genre association for retrieval filtering.
        if artist.styles:
            parts.append(f"Styles: {', '.join(artist.styles)}.")

        return " ".join(parts)

    @staticmethod
    def _label_to_prose(label: ParsedLabel) -> str:
        """Convert a ParsedLabel to a prose paragraph for embedding."""
        parts: list[str] = []

        parts.append(f"{label.name} is a record label.")

        if label.profile:
            profile = _clean_discogs_markup(label.profile)
            if profile:
                parts.append(profile)

        if label.parent_label:
            parts.append(f"Parent label: {label.parent_label}.")

        if label.sublabels:
            # Limit sublabel listing to avoid excessively long entries.
            subs = label.sublabels[:20]
            parts.append(f"Sublabels: {', '.join(subs)}.")
            if len(label.sublabels) > 20:
                parts.append(f"({len(label.sublabels)} sublabels total.)")

        if label.styles:
            parts.append(f"Styles: {', '.join(label.styles)}.")

        return " ".join(parts)

    @staticmethod
    def _master_to_prose(master: ParsedMaster) -> str:
        """Convert a ParsedMaster to a prose paragraph for embedding.

        Masters with notes get a full prose treatment. Masters without notes
        produce a condensed one-line entry that still helps with entity
        resolution (artist name + release title + year + styles).
        """
        artists_str = ", ".join(master.artists) if master.artists else "Unknown Artist"
        year_str = f" ({master.year})" if master.year else ""
        styles_str = ", ".join(master.styles) if master.styles else ""

        if master.notes:
            notes = _clean_discogs_markup(master.notes)
            parts = [
                f'"{master.title}" by {artists_str}{year_str}.',
            ]
            if styles_str:
                parts.append(f"Styles: {styles_str}.")
            parts.append(notes)
            return " ".join(parts)

        # Condensed format for releases without notes.
        line = f'{artists_str} — "{master.title}"{year_str}'
        if styles_str:
            line += f". Styles: {styles_str}"
        return line + "."

    # ------------------------------------------------------------------
    # Cluster matching
    # ------------------------------------------------------------------

    @staticmethod
    def _match_clusters(styles: list[str]) -> list[str]:
        """Return the style cluster names that match the given styles.

        A style list like ["House", "Deep House", "Techno"] would match
        both "house" and "techno" clusters, so the entry appears in both
        output files.
        """
        matched: list[str] = []
        style_set = set(styles)
        for cluster_name, cluster_styles in STYLE_CLUSTERS.items():
            if style_set & cluster_styles:
                matched.append(cluster_name)
        return matched

    # ------------------------------------------------------------------
    # Status / progress reporting
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, object]:
        """Return a summary of checkpoint and corpus file status.

        Used by the CLI status subcommand to show what parsing steps
        have been completed and what corpus files are available.
        """
        status: dict[str, object] = {
            "output_dir": str(self._output_dir),
            "checkpoint_dir": str(self._checkpoint_dir),
        }

        # Check JSONL checkpoints.
        for name in ("artists", "labels", "masters"):
            path = self._checkpoint_dir / f"{name}.jsonl"
            if path.exists():
                # Count lines (= records) without loading into memory.
                count = sum(1 for _ in open(path))
                size_mb = path.stat().st_size / (1024 * 1024)
                status[f"{name}_checkpoint"] = {
                    "records": count,
                    "size_mb": round(size_mb, 1),
                }
            else:
                status[f"{name}_checkpoint"] = None

        # Check style map checkpoints.
        for name in ("artist_style_map", "label_style_map"):
            path = self._checkpoint_dir / f"{name}.json"
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                status[f"{name}"] = {"entries": len(data)}
            else:
                status[f"{name}"] = None

        # Check corpus output files.
        corpus_files: dict[str, dict[str, object]] = {}
        for txt_file in sorted(self._output_dir.glob("*.txt")):
            size_mb = txt_file.stat().st_size / (1024 * 1024)
            line_count = sum(1 for _ in open(txt_file))
            corpus_files[txt_file.name] = {
                "size_mb": round(size_mb, 1),
                "lines": line_count,
            }
        status["corpus_files"] = corpus_files

        return status


# ─── UTILITY FUNCTIONS ────────────────────────────────────────────────────────

def _clean_discogs_markup(text: str) -> str:
    """Remove Discogs-specific markup from profile/notes text.

    Discogs uses a BBCode-like markup system in profile and notes fields:
      [a=Artist Name]  → links to an artist page
      [l=Label Name]   → links to a label page
      [r=12345]        → links to a release
      [m=12345]        → links to a master release
      [url=...]...[/url] → external links
      [b]...[/b]       → bold text
      [i]...[/i]       → italic text

    We strip the markup tags but preserve the readable text content,
    since the prose is destined for embedding, not display.
    """
    import re

    if not text:
        return ""

    # [a=Name], [l=Name] → just "Name"
    text = re.sub(r"\[a=([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[l=([^\]]+)\]", r"\1", text)

    # [r=12345] and [m=12345] → remove entirely (numeric IDs aren't useful)
    text = re.sub(r"\[[rm]=\d+\]", "", text)

    # [url=...]Text[/url] → just "Text"
    text = re.sub(r"\[url=[^\]]*\](.*?)\[/url\]", r"\1", text)

    # [b], [/b], [i], [/i] and other simple formatting tags → remove
    text = re.sub(r"\[/?[biu]\]", "", text)

    # Collapse multiple whitespace and trim.
    text = re.sub(r"\s+", " ", text).strip()

    return text
