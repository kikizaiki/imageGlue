"""Main rendering pipeline."""
import logging
import time
from pathlib import Path
from typing import Any

from PIL import Image

from app.core.config import settings
from app.core.exceptions import (
    CompositingError,
    DetectionError,
    QualityGateError,
    SegmentationError,
    TemplateError,
    ValidationError,
)
from app.core.storage import Storage
from app.models.schemas import DetectionResult, ValidationResult
from app.services.compositing.compositor import Compositor
from app.services.crop_planning.crop_planner import CropPlanner
from app.services.detection.dog_detector import DogDetector
from app.services.placement.placement_planner import PlacementPlanner
from app.services.quality.quality_gate import QualityGate
from app.services.refinement.kie_refiner import KIERefiner
from app.services.segmentation.background_remover import BackgroundRemover
from app.services.validation.input_validator import InputValidator
from app.utils.ids import generate_job_id

logger = logging.getLogger(__name__)


class RenderPipeline:
    """Main rendering pipeline."""

    def __init__(self):
        """Initialize pipeline."""
        self.validator = InputValidator()
        self.detector = DogDetector()
        self.segmenter = BackgroundRemover()
        self.crop_planner = CropPlanner()
        self.placement_planner = PlacementPlanner()
        self.quality_gate = QualityGate()
        self.refiner = KIERefiner()

    def load_template(self, template_id: str) -> dict[str, Any]:
        """
        Load template configuration.

        Args:
            template_id: Template identifier

        Returns:
            Template configuration

        Raises:
            TemplateError: If template not found
        """
        template_dir = settings.TEMPLATES_ROOT / template_id
        config_path = template_dir / "template_config.json"

        if not config_path.exists():
            raise TemplateError(f"Template not found: {template_id}")

        import json
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        config["_template_dir"] = str(template_dir)
        config["template_id"] = template_id

        logger.info(f"Template loaded: {template_id}")
        return config

    def render(
        self,
        image_path: str | Path,
        template_id: str,
        debug: bool = False,
    ) -> tuple[Image.Image, dict[str, Any]]:
        """
        Render image into template.

        Args:
            image_path: Path to input image
            template_id: Template identifier
            debug: Enable debug output

        Returns:
            Tuple of (final image, metadata)

        Raises:
            Various exceptions for pipeline failures
        """
        job_id = generate_job_id()
        storage = Storage(job_id)
        metadata = {
            "job_id": job_id,
            "template_id": template_id,
            "stages": {},
            "timings": {},
        }

        try:
            # Stage A: Input validation
            stage_start = time.time()
            logger.info(f"Job {job_id}: Stage A - Input Validation")
            validation_result = self.validator.validate(image_path)
            metadata["validation"] = validation_result.to_dict()

            if debug:
                image = Image.open(image_path)
                storage.save_debug(image, "00_original.png")

            metadata["timings"]["validation"] = time.time() - stage_start

            # Load template
            template_config = self.load_template(template_id)

            # Stage B: Subject analysis
            stage_start = time.time()
            logger.info(f"Job {job_id}: Stage B - Subject Analysis")
            image = Image.open(image_path)
            detection = self.detector.detect(image)
            metadata["detection"] = detection.to_dict()

            if debug:
                # Draw detection overlay
                from PIL import ImageDraw

                overlay = image.copy()
                draw = ImageDraw.Draw(overlay)
                bbox = detection.dog_bbox
                draw.rectangle(
                    [(bbox.x1, bbox.y1), (bbox.x2, bbox.y2)],
                    outline="red",
                    width=3,
                )
                if detection.head_bbox:
                    hbbox = detection.head_bbox
                    draw.rectangle(
                        [(hbbox.x1, hbbox.y1), (hbbox.x2, hbbox.y2)],
                        outline="blue",
                        width=2,
                    )
                storage.save_debug(overlay, "01_detection_overlay.png")

            metadata["timings"]["detection"] = time.time() - stage_start

            # Stage C: Segmentation
            stage_start = time.time()
            logger.info(f"Job {job_id}: Stage C - Segmentation")
            cropped = self.crop_planner.extract_crop(
                image, self.crop_planner.plan_crop(image, detection, template_config["placement"])
            )

            if debug:
                storage.save_debug(cropped, "02_crop.png")

            subject_rgba = self.segmenter.remove_background(cropped)
            metadata["segmentation"] = {"size": subject_rgba.size}

            if debug:
                storage.save_debug(subject_rgba, "03_subject_rgba.png")

            metadata["timings"]["segmentation"] = time.time() - stage_start

            # Stage D: Crop planning (already done above)
            # Stage E: Placement planning
            stage_start = time.time()
            logger.info(f"Job {job_id}: Stage E - Placement Planning")
            placement = self.placement_planner.plan_placement(
                subject_rgba.width,
                subject_rgba.height,
                template_config["placement"]["insert_zone"],
                template_config["placement"],
            )
            metadata["placement"] = placement

            metadata["timings"]["placement"] = time.time() - stage_start

            # Stage F: Compositing
            stage_start = time.time()
            logger.info(f"Job {job_id}: Stage F - Compositing")
            compositor = Compositor(template_config)
            final_image = compositor.compose(subject_rgba, placement)
            metadata["compositing"] = {"size": final_image.size}

            if debug:
                storage.save_debug(final_image, "04_composited.png")

            metadata["timings"]["compositing"] = time.time() - stage_start

            # Stage G: Postprocess (handled in compositor)
            # Stage H: Optional AI refinement
            if settings.REFINEMENT_ENABLED:
                stage_start = time.time()
                logger.info(f"Job {job_id}: Stage H - AI Refinement")
                try:
                    final_image = self.refiner.refine(final_image)
                    metadata["refinement"] = {"applied": True}

                    if debug:
                        storage.save_debug(final_image, "05_refined.png")

                except Exception as e:
                    logger.warning(f"Refinement failed: {e}, using original")
                    metadata["refinement"] = {"applied": False, "error": str(e)}

                metadata["timings"]["refinement"] = time.time() - stage_start

            # Stage I: Quality gate
            stage_start = time.time()
            logger.info(f"Job {job_id}: Stage I - Quality Gate")
            quality_score = self.quality_gate.assess(final_image, detection, placement)
            metadata["quality"] = quality_score.to_dict()

            metadata["timings"]["quality_check"] = time.time() - stage_start

            # Save outputs
            output_path = storage.save_output(final_image)
            metadata["output_path"] = str(output_path)

            if debug:
                storage.save_metadata(metadata)

            logger.info(f"Job {job_id}: Completed successfully")
            return final_image, metadata

        except (
            ValidationError,
            DetectionError,
            SegmentationError,
            CompositingError,
            QualityGateError,
            TemplateError,
        ) as e:
            logger.error(f"Job {job_id}: Pipeline failed: {e}")
            metadata["error"] = str(e)
            if debug:
                storage.save_metadata(metadata)
            raise
        except Exception as e:
            logger.error(f"Job {job_id}: Unexpected error: {e}", exc_info=True)
            metadata["error"] = str(e)
            if debug:
                storage.save_metadata(metadata)
            raise
