"""Flier image and OCR extraction models for the raiveFlier pipeline.

Defines Pydantic v2 models for flier images, text regions, OCR results,
and extracted entities. All models use frozen config to enforce immutability.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from src.models.entities import EntityType


class FlierImage(BaseModel):
    """Represents an uploaded rave flier image."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    content_type: str
    file_size: int
    image_hash: str
    upload_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)  # noqa: UP017
    )
    _image_data: bytes | None = PrivateAttr(default=None)

    @property
    def image_data(self) -> bytes | None:
        """Return the raw image data (excluded from serialization)."""
        return self._image_data


class TextRegion(BaseModel):
    """A region of text detected in a flier image with bounding box coordinates."""

    model_config = ConfigDict(frozen=True)

    text: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    x: int
    y: int
    width: int
    height: int


class OCRResult(BaseModel):
    """The result of OCR processing on a flier image."""

    model_config = ConfigDict(frozen=True)

    raw_text: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    provider_used: str
    processing_time: float
    bounding_boxes: list[TextRegion] = Field(default_factory=list)


class ExtractedEntity(BaseModel):
    """A single entity extracted from OCR text."""

    model_config = ConfigDict(frozen=True)

    text: str
    entity_type: EntityType
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_region: TextRegion | None = None


class ExtractedEntities(BaseModel):
    """All entities extracted from a flier's OCR output."""

    model_config = ConfigDict(frozen=True)

    artists: list[ExtractedEntity] = Field(default_factory=list)
    venue: ExtractedEntity | None = None
    date: ExtractedEntity | None = None
    promoter: ExtractedEntity | None = None
    genre_tags: list[str] = Field(default_factory=list)
    ticket_price: str | None = None
    supporting_text: list[str] = Field(default_factory=list)
    raw_ocr: OCRResult
