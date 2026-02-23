"""Flier image and OCR extraction models for the raiveFlier pipeline.

Defines Pydantic v2 models for flier images, text regions, OCR results,
and extracted entities. All models use frozen config to enforce immutability.

These models represent the earliest stages of the pipeline:
    1. A user uploads a rave flier image  → FlierImage
    2. OCR reads the image               → OCRResult (with TextRegion boxes)
    3. An LLM parses the OCR text         → ExtractedEntity / ExtractedEntities

Downstream, the extracted entities feed into the Research phase (see
src/services/research_service.py) and ultimately into the Interconnection
phase (see src/services/interconnection_service.py).
"""

# Enables PEP 604 union syntax (X | Y) for type hints on Python 3.9.
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

# Pydantic v2 imports:
#   BaseModel     — base class for all data models; provides validation & serialization
#   ConfigDict    — typed dict for model-level configuration (replaces inner class Config)
#   Field         — annotates individual fields with defaults, constraints, descriptions
#   PrivateAttr   — declares private attributes excluded from serialization/validation
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

# EntityType enum (ARTIST, VENUE, PROMOTER, DATE, etc.) used to tag each
# extracted entity so downstream code knows how to research it.
from src.models.entities import EntityType


# ---------------------------------------------------------------------------
# FlierImage — the very first model created when a user uploads a file.
# ---------------------------------------------------------------------------
class FlierImage(BaseModel):
    """Represents an uploaded rave flier image.

    Created by the upload endpoint (src/api/routes.py) and stored on
    PipelineState. Carries metadata but NOT the raw bytes in serialized form
    (image data is held in a private attribute to keep JSON payloads small).
    """

    # frozen=True makes every instance immutable after creation — any attempt
    # to set an attribute will raise a ValidationError. This follows the
    # immutability-by-default pattern used throughout raiveFlier models.
    model_config = ConfigDict(frozen=True)

    # Auto-generated UUID string — uniquely identifies this flier upload.
    id: str = Field(default_factory=lambda: str(uuid4()))
    # Original filename from the user's upload (e.g. "rave_flier_2003.jpg").
    filename: str
    # MIME type validated at upload (image/jpeg, image/png, image/webp).
    content_type: str
    # File size in bytes — the upload endpoint enforces a 10 MB maximum.
    file_size: int
    # SHA-256 hex digest of the raw image bytes — used for exact-duplicate detection.
    image_hash: str
    # Perceptual hash (pHash via imagehash library) — a short hex string that
    # lets us detect *visually similar* images even if they differ in resolution,
    # cropping, or compression. Two images with a small Hamming distance between
    # their pHash values are likely the same flier. See src/providers/flier_history/.
    image_phash: str | None = None
    # Timestamp of when the upload was received (always UTC).
    upload_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)  # noqa: UP017
    )
    # PrivateAttr stores the raw image bytes in memory but EXCLUDES them from
    # Pydantic serialization (.model_dump(), .model_dump_json()). This keeps
    # JSON responses lightweight while still making the bytes accessible via
    # the .image_data property for OCR providers that need them.
    _image_data: bytes | None = PrivateAttr(default=None)

    @property
    def image_data(self) -> bytes | None:
        """Return the raw image data (excluded from serialization).

        OCR providers (src/providers/ocr/) access this property to get the
        original bytes for text extraction. Returns None if the image was
        loaded from a serialized state (e.g. session restore) without bytes.
        """
        return self._image_data


# ---------------------------------------------------------------------------
# TextRegion — a single bounding box of detected text within the flier image.
# ---------------------------------------------------------------------------
class TextRegion(BaseModel):
    """A region of text detected in a flier image with bounding box coordinates.

    Some OCR providers (EasyOCR, Tesseract) return per-region bounding boxes.
    LLM Vision providers typically do NOT populate these, returning only raw_text
    on OCRResult instead. Downstream code should not assume bounding_boxes exist.
    """

    model_config = ConfigDict(frozen=True)

    # The text string detected in this region.
    text: str
    # How confident the OCR engine is in this detection (0.0 = no confidence,
    # 1.0 = fully confident). The ge/le constraints enforce the valid range
    # at validation time — Pydantic will reject values outside [0, 1].
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # Bounding box coordinates (top-left origin, pixel units).
    x: int
    y: int
    width: int
    height: int


# ---------------------------------------------------------------------------
# OCRResult — the complete output of one OCR processing run.
# ---------------------------------------------------------------------------
class OCRResult(BaseModel):
    """The result of OCR processing on a flier image.

    Produced by src/services/ocr_service.py which tries providers in priority
    order (LLM Vision → EasyOCR → Tesseract) and returns the first successful
    result that meets the minimum confidence threshold from config.yaml.
    """

    model_config = ConfigDict(frozen=True)

    # The full concatenated text extracted from the image.
    raw_text: str
    # Overall confidence score for the OCR run (provider-dependent).
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # Name of the provider that produced this result (e.g. "openai_vision").
    provider_used: str
    # Wall-clock seconds the OCR provider took to process the image.
    processing_time: float
    # Optional per-region bounding boxes — empty list if the provider doesn't
    # support them (e.g. LLM Vision only returns raw_text).
    bounding_boxes: list[TextRegion] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# ExtractedEntity — a single entity (artist name, venue, date, etc.) found
# in the OCR text by the LLM entity extractor.
# ---------------------------------------------------------------------------
class ExtractedEntity(BaseModel):
    """A single entity extracted from OCR text.

    Created by src/services/entity_extractor.py which sends the OCR raw_text
    to an LLM with a structured prompt asking it to identify artists, venues,
    dates, promoters, and genre tags from the flier.
    """

    model_config = ConfigDict(frozen=True)

    # The entity's text as it appears on the flier (e.g. "DJ Shadow").
    text: str
    # What kind of entity this is — determines which researcher service handles it.
    entity_type: EntityType
    # How confident the LLM is that this text is actually this entity type.
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # If bounding boxes were available, links this entity to a specific
    # region on the flier image. Often None when using LLM Vision OCR.
    source_region: TextRegion | None = None


# ---------------------------------------------------------------------------
# ExtractedEntities — the complete set of entities parsed from one flier.
# ---------------------------------------------------------------------------
class ExtractedEntities(BaseModel):
    """All entities extracted from a flier's OCR output.

    This is the "contract" between the Entity Extraction phase and the
    Research phase. The user gets a chance to review/edit these entities
    at the confirmation gate (src/pipeline/confirmation_gate.py) before
    research begins.
    """

    model_config = ConfigDict(frozen=True)

    # Multiple artists can appear on one flier (headliners, openers, etc.).
    artists: list[ExtractedEntity] = Field(default_factory=list)
    # Typically one venue per flier, but may be None if the LLM couldn't find one.
    venue: ExtractedEntity | None = None
    # The event date — may be None for undated fliers.
    date: ExtractedEntity | None = None
    # The event promoter/organizer — often None on fliers.
    promoter: ExtractedEntity | None = None
    # Named event/party series (e.g. "Bugged Out!", "BLOC") — if present.
    event_name: ExtractedEntity | None = None
    # Genre tags the LLM inferred from the flier (e.g. ["techno", "house"]).
    genre_tags: list[str] = Field(default_factory=list)
    # Ticket price if detected on the flier (free-text string like "$15 adv").
    ticket_price: str | None = None
    # Any additional text snippets that didn't fit into the structured fields
    # above but may still be useful for research context.
    supporting_text: list[str] = Field(default_factory=list)
    # The original OCR result is kept here so downstream code can reference
    # the raw text and the provider that was used.
    raw_ocr: OCRResult
