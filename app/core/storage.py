"""Storage utilities for debug artifacts and outputs."""
import json
from pathlib import Path
from typing import Any

from PIL import Image

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class Storage:
    """Manages storage for job artifacts."""

    def __init__(self, job_id: str):
        """
        Initialize storage for a job.

        Args:
            job_id: Unique job identifier
        """
        self.job_id = job_id
        self.debug_dir = settings.DEBUG_ROOT / job_id
        self.output_dir = settings.OUTPUT_ROOT / job_id
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_debug(self, image: Image.Image, filename: str) -> Path:
        """
        Save debug artifact.

        Args:
            image: PIL Image
            filename: Filename

        Returns:
            Path to saved file
        """
        path = self.debug_dir / filename
        image.save(path)
        logger.debug(f"Saved debug: {path}")
        return path

    def save_output(self, image: Image.Image, filename: str = "result.png") -> Path:
        """
        Save output image.

        Args:
            image: PIL Image
            filename: Filename

        Returns:
            Path to saved file
        """
        path = self.output_dir / filename
        image.save(path, quality=95)
        logger.info(f"Saved output: {path}")
        return path

    def save_metadata(self, metadata: dict[str, Any]) -> Path:
        """
        Save job metadata.

        Args:
            metadata: Metadata dictionary

        Returns:
            Path to saved file
        """
        path = self.debug_dir / "metadata.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        logger.debug(f"Saved metadata: {path}")
        return path
