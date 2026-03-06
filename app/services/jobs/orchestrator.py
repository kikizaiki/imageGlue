"""Job orchestration and pipeline execution."""
import logging
import time
from pathlib import Path

from PIL import Image

from app.core.config import settings
from app.core.exceptions import (
    BackgroundRemovalError,
    CompositingError,
    DetectionError,
    JobProcessingError,
    TemplateNotFoundError,
)
from app.core.storage import JobStorage
from app.domain.enums import EntityType, FitMode, JobStage, JobStatus
from app.domain.models import (
    BBox,
    CropPlan,
    DetectionResult,
    JobMetadata,
    PlacementResult,
)
from app.services.bgremove.base import BaseBackgroundRemovalProvider
from app.services.bgremove.external_api import ExternalApiBackgroundRemovalProvider
from app.services.bgremove.mock_provider import MockBackgroundRemovalProvider
from app.services.bgremove.rembg_provider import RembgBackgroundRemovalProvider
from app.services.composer.geometry import (
    compute_placement,
    expand_bbox,
    expand_bbox_directional,
)
from app.services.composer.image_ops import (
    create_detection_overlay,
    crop_image,
)
from app.services.composer.sandwich import SandwichCompositor
from app.services.detector.base import BaseDetector
from app.services.detector.mock_detector import MockDetector
from app.services.detector.yolo_detector import YoloHeadDetector
from app.services.jobs.status_store import JobStatusStore
from app.services.templates.loader import TemplateLoader

logger = logging.getLogger(__name__)


class JobOrchestrator:
    """Orchestrates the image processing pipeline."""

    def __init__(self):
        """Initialize orchestrator."""
        self.template_loader = TemplateLoader()
        self.status_store = JobStatusStore()
        self._detector: BaseDetector | None = None
        self._bgremoval_provider: BaseBackgroundRemovalProvider | None = None

    def _get_detector(self) -> BaseDetector:
        """Get detector instance (lazy initialization)."""
        if self._detector is None:
            backend = settings.DETECTOR_BACKEND
            if backend == "mock":
                self._detector = MockDetector()
            elif backend == "yolo":
                try:
                    self._detector = YoloHeadDetector()
                except Exception as e:
                    logger.error(f"Failed to initialize YOLO detector: {e}")
                    logger.warning("Falling back to mock detector")
                    self._detector = MockDetector()
            else:
                self._detector = MockDetector()

            logger.info(f"Using detector: {self._detector.get_name()}")
        return self._detector

    def _get_bgremoval_provider(self) -> BaseBackgroundRemovalProvider:
        """Get background removal provider (lazy initialization)."""
        if self._bgremoval_provider is None:
            backend = settings.BGREMOVAL_BACKEND
            if backend == "mock":
                self._bgremoval_provider = MockBackgroundRemovalProvider()
            elif backend == "rembg":
                try:
                    model = settings.BGREMOVAL_REMBG_MODEL
                    self._bgremoval_provider = RembgBackgroundRemovalProvider(model_name=model)
                except Exception as e:
                    logger.error(f"Failed to initialize rembg provider: {e}")
                    logger.warning("Falling back to mock provider")
                    self._bgremoval_provider = MockBackgroundRemovalProvider()
            elif backend == "external":
                try:
                    self._bgremoval_provider = ExternalApiBackgroundRemovalProvider()
                except Exception as e:
                    logger.error(f"Failed to initialize external provider: {e}")
                    logger.warning("Falling back to mock provider")
                    self._bgremoval_provider = MockBackgroundRemovalProvider()
            else:
                self._bgremoval_provider = MockBackgroundRemovalProvider()

            logger.info(f"Using bg removal provider: {self._bgremoval_provider.get_name()}")
        return self._bgremoval_provider

    def process_job(
        self, job_id: str, image: Image.Image, template_id: str, entity_type: EntityType
    ) -> dict:
        """
        Process a job through the full pipeline.

        Args:
            job_id: Job identifier
            image: Input image
            template_id: Template identifier
            entity_type: Expected entity type

        Returns:
            Dictionary with job results including final image path

        Raises:
            JobProcessingError: If processing fails
        """
        timings: dict[str, float] = {}
        storage = JobStorage(job_id)

        try:
            # Initialize metadata
            metadata = JobMetadata(
                job_id=job_id,
                template_id=template_id,
                entity_type=entity_type,
                status=JobStatus.PROCESSING,
                current_stage=JobStage.INGEST,
            )
            self.status_store.save_status(metadata)

            # Stage 0: Ingest
            stage_start = time.time()
            logger.info(f"Job {job_id}: Stage 0 - Ingest")
            storage.save_original(image, "original.png")
            timings["ingest"] = (time.time() - stage_start) * 1000

            # Load template
            template_config = self.template_loader.load(template_id)
            if entity_type.value not in template_config.get("supported_entities", []):
                logger.warning(
                    f"Entity type {entity_type.value} not in template supported entities"
                )

            # Stage 1: Detection
            stage_start = time.time()
            logger.info(f"Job {job_id}: Stage 1 - Detection")
            metadata.current_stage = JobStage.DETECTION
            self.status_store.save_status(metadata)

            detector = self._get_detector()
            detections = detector.detect(image, entity_type)

            if not detections:
                raise DetectionError("No detections found")

            best_detection = detector.select_best_detection(
                detections, image.width, image.height
            )

            if best_detection is None:
                raise DetectionError("Failed to select best detection")

            metadata.detection_result = best_detection
            metadata.detector_name = detector.get_name()
            self.status_store.save_status(metadata)

            # Save detection overlay
            overlay = create_detection_overlay(image, best_detection)
            storage.save_overlay(overlay, "detection_overlay.png")

            timings["detection"] = (time.time() - stage_start) * 1000

            # Stage 2: Crop Planning
            stage_start = time.time()
            logger.info(f"Job {job_id}: Stage 2 - Crop Planning")
            metadata.current_stage = JobStage.CROP_PLANNING
            self.status_store.save_status(metadata)

            placement_config = template_config.get("placement", {})

            # Use directional expansion if configured, otherwise fallback to simple expansion
            crop_expansion_config = placement_config.get("crop_expansion", {})
            if isinstance(crop_expansion_config, dict):
                # Directional expansion
                expanded_bbox = expand_bbox_directional(
                    best_detection.bbox,
                    expand_left=crop_expansion_config.get("expand_left", 0.3),
                    expand_right=crop_expansion_config.get("expand_right", 0.3),
                    expand_top=crop_expansion_config.get("expand_top", 0.2),
                    expand_bottom=crop_expansion_config.get("expand_bottom", 0.5),
                    image_width=image.width,
                    image_height=image.height,
                    vertical_shift=placement_config.get("crop_vertical_shift", 0.0),
                )
            else:
                # Simple expansion (backward compatibility)
                crop_expansion = crop_expansion_config if isinstance(crop_expansion_config, (int, float)) else 1.2
                expanded_bbox = expand_bbox(
                    best_detection.bbox, crop_expansion, image.width, image.height
                )
                # Apply vertical shift if specified
                if placement_config.get("crop_vertical_shift", 0.0) != 0.0:
                    shift = placement_config["crop_vertical_shift"]
                    expanded_bbox = BBox(
                        x1=expanded_bbox.x1,
                        y1=expanded_bbox.y1 + shift,
                        x2=expanded_bbox.x2,
                        y2=expanded_bbox.y2 + shift,
                    )
                    # Clamp
                    expanded_bbox = BBox(
                        x1=max(0, expanded_bbox.x1),
                        y1=max(0, min(expanded_bbox.y1, image.height)),
                        x2=min(image.width, expanded_bbox.x2),
                        y2=min(image.height, expanded_bbox.y2),
                    )

            crop_plan = CropPlan(
                bbox=best_detection.bbox,
                expanded_bbox=expanded_bbox,
                image_width=image.width,
                image_height=image.height,
            )
            metadata.crop_plan = crop_plan
            self.status_store.save_status(metadata)

            timings["crop_planning"] = (time.time() - stage_start) * 1000

            # Stage 3: Background Removal
            stage_start = time.time()
            logger.info(f"Job {job_id}: Stage 3 - Background Removal")
            metadata.current_stage = JobStage.BACKGROUND_REMOVAL
            self.status_store.save_status(metadata)

            cropped_image = crop_image(image, expanded_bbox)
            storage.save_expanded_crop(cropped_image, "expanded_crop.png")
            if settings.DEBUG:
                storage.save_crop(cropped_image, "crop.png")

            bgremoval_provider = self._get_bgremoval_provider()
            subject_image = bgremoval_provider.remove_background(cropped_image)
            storage.save_subject_rgba(subject_image, "subject_rgba.png")
            if settings.DEBUG:
                storage.save_subject(subject_image, "subject.png")

            metadata.bgremoval_provider_name = bgremoval_provider.get_name()
            self.status_store.save_status(metadata)

            timings["background_removal"] = (time.time() - stage_start) * 1000

            # Stage 4: Subject Alignment
            stage_start = time.time()
            logger.info(f"Job {job_id}: Stage 4 - Subject Alignment")
            metadata.current_stage = JobStage.SUBJECT_ALIGNMENT
            self.status_store.save_status(metadata)

            target_area = placement_config["target_area"]
            fit_mode = FitMode(placement_config.get("fit_mode", "contain"))
            anchor_mode = placement_config.get("anchor_mode", "center")
            horizontal_bias = placement_config.get("horizontal_bias", 0.0)
            vertical_bias = placement_config.get("vertical_bias", 0.0)
            min_scale = placement_config.get("min_scale", 0.1)
            max_scale = placement_config.get("max_scale", 10.0)
            padding = placement_config.get("padding", 0)
            scale_multiplier = placement_config.get("scale_multiplier", 1.0)

            placement = compute_placement(
                subject_width=subject_image.width,
                subject_height=subject_image.height,
                target_x=target_area["x"],
                target_y=target_area["y"],
                target_width=target_area["width"],
                target_height=target_area["height"],
                fit_mode=fit_mode,
                anchor_mode=anchor_mode,
                horizontal_bias=horizontal_bias,
                vertical_bias=vertical_bias,
                min_scale=min_scale,
                max_scale=max_scale,
                padding=padding,
                scale_multiplier=scale_multiplier,
            )

            metadata.placement_result = placement
            self.status_store.save_status(metadata)

            timings["subject_alignment"] = (time.time() - stage_start) * 1000

            # Stage 5: Compositing
            stage_start = time.time()
            logger.info(f"Job {job_id}: Stage 5 - Compositing")
            metadata.current_stage = JobStage.COMPOSITING
            self.status_store.save_status(metadata)

            compositor = SandwichCompositor(template_config)
            save_debug = settings.DEBUG or template_config.get("debug", {}).get("save_intermediates", False)
            final_image, debug_images = compositor.compose(
                subject_image, placement.to_dict(), save_debug=save_debug
            )

            # Save debug images
            if save_debug and debug_images:
                if "subject_placed" in debug_images:
                    storage.save_debug_image(debug_images["subject_placed"], "subject_placed.png")
                if "subject_masked_by_visor" in debug_images:
                    storage.save_debug_image(
                        debug_images["subject_masked_by_visor"], "subject_masked_by_visor.png"
                    )

            final_path = storage.save_final(final_image)

            timings["compositing"] = (time.time() - stage_start) * 1000

            # Stage 6: Delivery
            stage_start = time.time()
            logger.info(f"Job {job_id}: Stage 6 - Delivery")
            metadata.current_stage = JobStage.DELIVERY
            metadata.status = JobStatus.COMPLETED
            metadata.timings_ms = timings
            self.status_store.save_status(metadata)

            # Save metadata
            storage.save_metadata(metadata.to_dict())

            timings["delivery"] = (time.time() - stage_start) * 1000
            timings["total"] = sum(timings.values())

            logger.info(f"Job {job_id}: Completed in {timings['total']:.2f}ms")

            return {
                "job_id": job_id,
                "status": "completed",
                "final_image_url": storage.get_url_path("final.png"),
                "metadata": metadata.to_dict(),
            }

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            metadata.status = JobStatus.FAILED
            metadata.error_message = str(e)
            metadata.timings_ms = timings
            self.status_store.save_status(metadata)

            if settings.DEBUG:
                storage.save_metadata(metadata.to_dict())

            raise JobProcessingError(f"Job processing failed: {e}") from e
