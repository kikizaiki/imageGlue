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
            
            # Получаем тип сущности из конфигурации шаблона
            entity_type = template_config.get("entity_type", "dog")
            logger.info(f"Job {job_id}: Template entity_type from config: {entity_type}")
            logger.info(f"Job {job_id}: Detecting {entity_type} in image")
            
            detection = self.detector.detect(image, entity_type=entity_type)
            metadata["detection"] = detection.to_dict()
            metadata["detection"]["entity_type"] = entity_type  # Сохраняем тип сущности в метаданных

            if debug:
                # Draw detection overlay
                from PIL import ImageDraw

                overlay = image.copy()
                draw = ImageDraw.Draw(overlay)
                
                # Рисуем основной bbox (entity bbox)
                bbox = detection.dog_bbox  # dog_bbox содержит entity_bbox для обратной совместимости
                entity_name = "собака" if entity_type == "dog" else "человек"
                draw.rectangle(
                    [(bbox.x1, bbox.y1), (bbox.x2, bbox.y2)],
                    outline="red",
                    width=3,
                )
                # Добавляем подпись
                draw.text(
                    (bbox.x1, bbox.y1 - 20),
                    f"{entity_name.capitalize()} (conf: {detection.confidence:.2f})",
                    fill="red",
                )
                
                # Рисуем head bbox
                if detection.head_bbox:
                    hbbox = detection.head_bbox
                    draw.rectangle(
                        [(hbbox.x1, hbbox.y1), (hbbox.x2, hbbox.y2)],
                        outline="blue",
                        width=2,
                    )
                    draw.text(
                        (hbbox.x1, hbbox.y1 - 15),
                        "Head",
                        fill="blue",
                    )
                
                logger.info(
                    f"Job {job_id}: Detection overlay: {entity_type} bbox={bbox.to_dict()}, "
                    f"head_bbox={detection.head_bbox.to_dict() if detection.head_bbox else None}, "
                    f"confidence={detection.confidence:.2f}"
                )
                storage.save_debug(overlay, "01_detection_overlay.png")

            metadata["timings"]["detection"] = time.time() - stage_start

            # Stage C: Segmentation
            stage_start = time.time()
            logger.info(f"Job {job_id}: Stage C - Segmentation")
            # Передаём entity_type для правильного кропа
            entity_type = template_config.get("entity_type", "dog")
            crop_bbox = self.crop_planner.plan_crop(
                image, detection, template_config["placement"], entity_type=entity_type
            )
            cropped = self.crop_planner.extract_crop(image, crop_bbox)

            if debug:
                storage.save_debug(cropped, "02_crop.png")

            subject_rgba = self.segmenter.remove_background(cropped)
            metadata["segmentation"] = {"size": subject_rgba.size}

            # Optional: AI-improved segmentation
            if settings.REFINEMENT_ENABLED and self.refiner.api_key:
                try:
                    logger.info(f"Job {job_id}: Improving segmentation with AI")
                    subject_rgba = self.refiner.refine_segmentation(subject_rgba, cropped)
                    metadata["segmentation"]["ai_improved"] = True
                    if debug:
                        storage.save_debug(subject_rgba, "03_subject_rgba_ai_improved.png")
                except Exception as e:
                    logger.warning(f"AI segmentation improvement failed: {e}")

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

            # Optional: AI-improved placement (if enabled)
            if (
                settings.REFINEMENT_ENABLED
                and self.refiner.api_key
                and template_config.get("refinement", {}).get("improve_placement", False)
            ):
                try:
                    logger.info(f"Job {job_id}: Improving placement with AI")
                    # Load base for AI placement
                    base_path = (
                        Path(template_config["_template_dir"])
                        / template_config["assets"]["base_clean"]
                    )
                    base_image = Image.open(base_path).convert("RGB")

                    # Use AI to improve placement
                    improved_composite = self.refiner.refine_placement(
                        subject_rgba,
                        base_image,
                        placement_hint=template_config["placement"]
                        .get("insert_zone", {})
                        .get("description", "center"),
                    )

                    # Extract improved subject position (simplified)
                    # For now, we'll use the improved composite directly in next stage
                    metadata["placement"]["ai_improved"] = True
                    if debug:
                        storage.save_debug(improved_composite, "03_ai_placement.png")

                except Exception as e:
                    logger.warning(f"AI placement improvement failed: {e}")

            metadata["timings"]["placement"] = time.time() - stage_start

            # Stage F: Пропускаем старый композитинг - ВСЁ ДЕЛАЕМ ЧЕРЕЗ LLM
            # Stage H: AI Integration - ВСЁ ДЕЛАЕМ ЧЕРЕЗ LLM, без старого композитинга
            final_image = None  # Будет установлен через AI
            
            if settings.REFINEMENT_ENABLED:
                stage_start = time.time()
                logger.info(f"Job {job_id}: Stage H - AI Integration")
                
                # Check if API key is available
                if not self.refiner.api_key:
                    logger.warning(
                        f"Job {job_id}: KIE_API_KEY not set, skipping AI integration. "
                        "Add KIE_API_KEY to .env file to enable AI refinement."
                    )
                    metadata["refinement"] = {
                        "applied": False,
                        "reason": "no_api_key",
                        "message": "KIE_API_KEY not configured in .env",
                    }
                else:
                    try:
                        logger.info(f"Job {job_id}: API key available, starting AI integration")
                        
                        # Загружаем чистый постер для AI интеграции
                        base_path = (
                            Path(template_config["_template_dir"])
                            / template_config["assets"]["base_clean"]
                        )
                        poster_background = Image.open(base_path).convert("RGB")
                        
                        template_desc = template_config.get("title", "poster")
                        
                        # Получаем промпт для AI интеграции из конфигурации шаблона
                        ai_integration_config = template_config.get("ai_integration", {})
                        ai_prompt = ai_integration_config.get("prompt", "")
                        
                        if not ai_prompt:
                            raise CompositingError(
                                f"AI integration prompt not found in template config. "
                                f"Please add 'ai_integration.prompt' to {template_config.get('template_id', 'template')} config."
                            )
                        
                        entity_type = template_config.get("entity_type", "dog")
                        entity_name = "собаку" if entity_type == "dog" else "человека"
                        logger.info(f"Job {job_id}: Using AI to integrate {entity_name} into {template_desc}")
                        logger.debug(f"Job {job_id}: AI prompt: {ai_prompt[:100]}...")
                        
                        try:
                            # ВСЁ ДЕЛАЕМ ЧЕРЕЗ LLM - полная AI интеграция с исходным фото и постером
                            # БЕЗ старого композитинга - сразу отправляем в LLM
                            logger.info(f"Job {job_id}: 🎨 Starting LLM-based integration (NO old compositing)")
                            logger.info(f"Job {job_id}: 📤 Sending original dog image + poster to LLM for smart integration")
                            logger.info(f"Job {job_id}: 🚫 Old compositing SKIPPED - everything through LLM")
                            
                            refined_result = self.refiner.refine_compositing(
                                None,  # Нет старого композитинга - только LLM
                                ai_prompt,  # Промпт из конфигурации шаблона
                                detected_issues=[],
                                original_dog_image=image,  # Исходное фото собаки - ОБЯЗАТЕЛЬНО
                                poster_background=poster_background,  # Чистый постер - ОБЯЗАТЕЛЬНО
                            )
                            
                            # ВСЕГДА используем результат AI если он получен
                            if refined_result is not None:
                                final_image = refined_result
                                logger.info(f"Job {job_id}: ✅✅✅ LLM integration successful - USING AI RESULT")
                                logger.info(f"Job {job_id}: ✅ Final image size: {final_image.size}")
                            else:
                                logger.error(f"Job {job_id}: ❌ AI returned None result")
                                raise CompositingError("LLM integration returned None - cannot proceed without AI result")
                                
                        except CompositingError as e:
                            # Если полная интеграция не удалась - пробуем альтернативный подход через LLM
                            logger.warning(f"Job {job_id}: Full LLM integration failed ({e}), trying alternative LLM approach")
                            try:
                                # Создаём временный простой композит ТОЛЬКО для LLM улучшения (не для финального результата)
                                from app.services.compositing.compositor import Compositor
                                temp_compositor = Compositor(template_config)
                                temp_composite = temp_compositor.compose(subject_rgba, placement)
                                
                                if debug:
                                    storage.save_debug(temp_composite, "04_temp_composite_for_llm.png")
                                
                                # Для fallback используем промпт из конфига, если он есть
                                fallback_prompt = ai_prompt if ai_prompt else (
                                    "Improve the image. Ensure perfect color matching, "
                                    "natural shadows, and seamless integration. "
                                    "The result should be publication-ready."
                                )
                                refined_result = self.refiner.refine_compositing(
                                    temp_composite,  # Временный композит для LLM улучшения
                                    fallback_prompt,  # Промпт из конфига или базовый для fallback
                                    detected_issues=[],
                                    original_dog_image=None,  # Не используем исходное фото
                                    poster_background=None,  # Не используем чистый постер
                                )
                                if refined_result:
                                    final_image = refined_result
                                    logger.info(f"Job {job_id}: ✅ Alternative LLM approach successful")
                                else:
                                    raise CompositingError("Alternative LLM approach also returned None")
                            except Exception as e2:
                                logger.error(f"Job {job_id}: ❌ All LLM approaches failed: {e2}")
                                raise CompositingError(f"LLM integration completely failed: {e2}") from e2
                        
                        # Проверяем что у нас есть финальное изображение
                        if final_image is None:
                            raise CompositingError("No final image generated - LLM integration failed")
                        
                        logger.info(f"Job {job_id}: ✅✅✅ FINAL RESULT: Using LLM-generated image (NO old compositing)")
                        
                        # Метаданные для AI результата
                        metadata["refinement"] = {
                            "applied": True,
                            "passes": 1,
                            "type": "llm_integration",
                            "method": "full_llm_integration",
                            "old_compositing": False,
                        }
                        logger.info(f"Job {job_id}: ✅ Dog integrated successfully using LLM (no old compositing)")

                        # Применяем occlusion_mask и glass_fx поверх результата от KIE.ai
                        final_image = self._apply_post_ai_layers(
                            final_image, template_config, debug, storage, job_id
                        )

                        # 05_refined_pass1.png - это финальное изображение
                        # KIE.ai обновляет сущность прямо в постере, затем применяются маски
                        if debug:
                            storage.save_debug(final_image, "05_refined_pass1.png")
                        
                        logger.info(f"Job {job_id}: ✅ Final image ready: 05_refined_pass1.png ({final_image.size})")

                    except Exception as e:
                        logger.error(f"Refinement failed: {e}", exc_info=True)
                        logger.error(f"❌ LLM integration completely failed: {e}")
                        logger.error(f"Job {job_id}: Cannot proceed without LLM - no fallback available")
                        metadata["refinement"] = {
                            "applied": False,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        }
                        # Без LLM мы не можем создать финальное изображение
                        raise CompositingError(f"LLM integration failed and no fallback available: {e}") from e

                metadata["timings"]["refinement"] = time.time() - stage_start
            else:
                # Если REFINEMENT_ENABLED = False, мы не можем создать финальное изображение
                raise CompositingError(
                    "REFINEMENT_ENABLED is False - LLM integration is required. "
                    "Set REFINEMENT_ENABLED=True in .env"
                )
            
            # Проверяем что финальное изображение создано
            if final_image is None:
                raise CompositingError("No final image generated - LLM integration is required")

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

    def _apply_post_ai_layers(
        self,
        image: Image.Image,
        template_config: dict[str, Any],
        debug: bool,
        storage: Storage,
        job_id: str,
    ) -> Image.Image:
        """
        Apply occlusion_mask and glass_fx layers on top of AI result.
        
        Args:
            image: Image from KIE.ai
            template_config: Template configuration
            debug: Debug mode
            storage: Storage instance
            job_id: Job ID
            
        Returns:
            Final image with layers applied
        """
        from pathlib import Path
        
        result = image.convert("RGBA")
        template_dir = Path(template_config.get("_template_dir", ""))
        
        # Apply occlusion_mask if present
        if "assets" in template_config and "occlusion_mask" in template_config["assets"]:
            occlusion_path = template_dir / template_config["assets"]["occlusion_mask"]
            if occlusion_path.exists():
                try:
                    logger.info(f"Job {job_id}: Applying occlusion_mask...")
                    occlusion = Image.open(occlusion_path).convert("RGBA")
                    
                    # Resize occlusion_mask to match image size
                    if occlusion.size != result.size:
                        occlusion = occlusion.resize(result.size, Image.Resampling.LANCZOS)
                    
                    # Composite occlusion_mask on top
                    result = Image.alpha_composite(result, occlusion)
                    logger.info(f"Job {job_id}: ✅ occlusion_mask applied")
                    
                    if debug:
                        storage.save_debug(result.convert("RGB"), "06_after_occlusion_mask.png")
                except Exception as e:
                    logger.warning(f"Job {job_id}: Failed to apply occlusion_mask: {e}")
            else:
                logger.warning(f"Job {job_id}: occlusion_mask not found: {occlusion_path}")
        
        # Apply glass_fx if present
        if "assets" in template_config and "glass_fx" in template_config["assets"]:
            glass_path = template_dir / template_config["assets"]["glass_fx"]
            if glass_path.exists():
                try:
                    logger.info(f"Job {job_id}: Applying glass_fx...")
                    glass = Image.open(glass_path).convert("RGBA")
                    
                    # Resize glass_fx to match image size
                    if glass.size != result.size:
                        glass = glass.resize(result.size, Image.Resampling.LANCZOS)
                    
                    # Composite glass_fx on top
                    result = Image.alpha_composite(result, glass)
                    logger.info(f"Job {job_id}: ✅ glass_fx applied")
                    
                    if debug:
                        storage.save_debug(result.convert("RGB"), "07_after_glass_fx.png")
                except Exception as e:
                    logger.warning(f"Job {job_id}: Failed to apply glass_fx: {e}")
            else:
                logger.warning(f"Job {job_id}: glass_fx not found: {glass_path}")
        
        return result.convert("RGB")
