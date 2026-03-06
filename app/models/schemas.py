"""Data models and schemas."""
from dataclasses import dataclass
from typing import Any


@dataclass
class BBox:
    """Bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Width."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Height."""
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        """Center X."""
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        """Center Y."""
        return (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        """Area."""
        return self.width * self.height

    def to_dict(self) -> dict[str, float]:
        """Convert to dict."""
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass
class DetectionResult:
    """Detection result."""

    dog_bbox: BBox
    head_bbox: BBox | None
    confidence: float
    orientation: str | None = None  # "front", "side", "back", etc.

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict."""
        return {
            "dog_bbox": self.dog_bbox.to_dict(),
            "head_bbox": self.head_bbox.to_dict() if self.head_bbox else None,
            "confidence": self.confidence,
            "orientation": self.orientation,
        }


@dataclass
class ValidationResult:
    """Input validation result."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    image_width: int
    image_height: int
    format: str
    file_size_mb: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "format": self.format,
            "file_size_mb": self.file_size_mb,
        }


@dataclass
class QualityScore:
    """Quality assessment score."""

    overall: float  # 0.0 - 1.0
    head_visible: bool
    head_size_ok: bool
    edges_ok: bool
    composition_ok: bool
    artifacts_detected: bool
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict."""
        return {
            "overall": self.overall,
            "head_visible": self.head_visible,
            "head_size_ok": self.head_size_ok,
            "edges_ok": self.edges_ok,
            "composition_ok": self.composition_ok,
            "artifacts_detected": self.artifacts_detected,
            "rejection_reason": self.rejection_reason,
        }
