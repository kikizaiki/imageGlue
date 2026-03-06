"""Template configuration loader."""
import json
import logging
from pathlib import Path
from typing import Any

from app.core.exceptions import TemplateNotFoundError, TemplateValidationError
from app.domain.enums import EntityType
from app.services.templates.validator import TemplateValidator

logger = logging.getLogger(__name__)


class TemplateLoader:
    """Loads and validates template configurations."""

    def __init__(self, templates_root: Path | None = None):
        """
        Initialize template loader.

        Args:
            templates_root: Root directory for templates. Defaults to ./templates
        """
        if templates_root is None:
            templates_root = Path("templates")
        self.templates_root = Path(templates_root)
        self.validator = TemplateValidator()

    def load(self, template_id: str) -> dict[str, Any]:
        """
        Load a template configuration.

        Args:
            template_id: Template identifier

        Returns:
            Template configuration dictionary

        Raises:
            TemplateNotFoundError: If template directory or config not found
            TemplateValidationError: If template config is invalid
        """
        template_dir = self.templates_root / template_id
        config_path = template_dir / "config.json"

        if not template_dir.exists():
            raise TemplateNotFoundError(f"Template directory not found: {template_dir}")

        if not config_path.exists():
            raise TemplateNotFoundError(
                f"Template config not found: {config_path}"
            )

        try:
            with open(config_path, "r") as f:
                config = json.load(f)

            # Add template_id and resolve paths
            config["template_id"] = template_id
            config["_template_dir"] = str(template_dir)

            # Resolve relative paths
            self._resolve_paths(config, template_dir)

            # Resolve visor mask path if present
            if "visor_mask" in config and "path" in config["visor_mask"]:
                visor_mask_rel = config["visor_mask"]["path"]
                config["visor_mask"]["_resolved_path"] = str(template_dir / visor_mask_rel)

            # Validate
            self.validator.validate(config)

            logger.info(f"Loaded template: {template_id}")
            return config

        except json.JSONDecodeError as e:
            raise TemplateValidationError(f"Invalid JSON in template config: {e}")
        except Exception as e:
            if isinstance(e, (TemplateNotFoundError, TemplateValidationError)):
                raise
            raise TemplateValidationError(f"Failed to load template: {e}")

    def _resolve_paths(self, config: dict[str, Any], template_dir: Path) -> None:
        """
        Resolve relative paths in template config to absolute paths.

        Args:
            config: Template configuration
            template_dir: Template directory
        """
        # Resolve layer paths
        if "layers" in config:
            layers = config["layers"]
            if "background" in layers and "path" in layers["background"]:
                rel_path = layers["background"]["path"]
                layers["background"]["_resolved_path"] = str(
                    template_dir / rel_path
                )
            if "foreground" in layers and "path" in layers["foreground"]:
                rel_path = layers["foreground"]["path"]
                layers["foreground"]["_resolved_path"] = str(
                    template_dir / rel_path
                )

    def list_templates(self) -> list[dict[str, Any]]:
        """
        List all available templates.

        Returns:
            List of template info dictionaries
        """
        templates = []

        if not self.templates_root.exists():
            logger.warning(f"Templates root not found: {self.templates_root}")
            return templates

        for template_dir in self.templates_root.iterdir():
            if not template_dir.is_dir():
                continue

            template_id = template_dir.name
            config_path = template_dir / "config.json"

            if not config_path.exists():
                logger.warning(f"Template {template_id} missing config.json")
                continue

            try:
                config = self.load(template_id)
                templates.append(
                    {
                        "template_id": config["template_id"],
                        "title": config.get("title", template_id),
                        "version": config.get("version", "1.0.0"),
                        "enabled": config.get("enabled", True),
                        "supported_entities": config.get("supported_entities", []),
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to load template {template_id}: {e}")

        return templates
