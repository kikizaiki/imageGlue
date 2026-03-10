"""Application configuration."""
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    APP_ENV: str = "development"
    DEBUG: bool = True

    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    TEMPLATES_ROOT: Path = PROJECT_ROOT / "templates"
    RUNS_ROOT: Path = PROJECT_ROOT / "runs"
    DEBUG_ROOT: Path = RUNS_ROOT / "debug"
    OUTPUT_ROOT: Path = RUNS_ROOT / "output"

    # Input validation
    MIN_IMAGE_WIDTH: int = 200
    MIN_IMAGE_HEIGHT: int = 200
    MAX_IMAGE_SIZE_MB: int = 10
    ALLOWED_FORMATS: set[str] = {".jpg", ".jpeg", ".png", ".webp"}

    # Detection
    DETECTION_CONFIDENCE_THRESHOLD: float = 0.3
    MIN_DOG_AREA_RATIO: float = 0.05  # Минимальная доля площади кадра, занимаемая собакой

    # Segmentation
    SEGMENTATION_BACKEND: Literal["rembg", "kie"] = "rembg"
    REMBG_MODEL: str = "u2net"

    # AI Refinement
    KIE_API_KEY: str = ""
    KIE_API_URL: str = "https://api.kie.ai"  # Base URL для API (createTask, getTaskStatus, etc.)
    KIE_UPLOAD_BASE_URL: str = "https://kieai.redpandaai.co"  # Base URL для File Upload API (отдельная база)
    KIE_PRIMARY_MODEL: str = "gpt-image/1.5-image-to-image"  # Primary model: "gpt-image/1.5-image-to-image" or "google/nano-banana-edit"
    KIE_FALLBACK_MODEL: str = ""  # Fallback model if primary fails (empty = no fallback)
    KIE_MODEL: str = "gpt-image/1.5-image-to-image"  # DEPRECATED: Use KIE_PRIMARY_MODEL instead. Kept for backward compatibility.
    REFINEMENT_ENABLED: bool = True
    REFINEMENT_THRESHOLD: float = 0.6  # Порог качества для запуска refinement
    REFINEMENT_TIMEOUT: int = 120  # Timeout for AI requests in seconds

    # Quality gate
    QUALITY_CHECK_ENABLED: bool = True
    MIN_HEAD_SIZE_RATIO: float = 0.02  # Минимальный размер головы относительно кадра
    MAX_BLUR_THRESHOLD: float = 100.0  # Порог размытия (Laplacian variance)

    # Age-based routing
    REFERENCE_SUBJECT_CLASSIFIER: str = "local"  # "local" or "external"
    TEEN_UNKNOWN_POLICY: str = "route_to_adult"  # "route_to_teen" or "route_to_adult" (default: route_to_adult for safety)

    # Teen flow configuration
    TEEN_FLOW_ENABLED: bool = True
    TEEN_FLOW_DEFAULT_MODE: str = "face_region"  # "face_region" only for teens
    TEEN_FLOW_MODELS: str = (
        "flux-kontext-pro,qwen/image-edit,seedream/4-5-edit,google/nano-banana-edit"
    )  # Comma-separated list
    SAVE_TEEN_DEBUG_ARTIFACTS: bool = True

    def __init__(self, **kwargs):
        """Initialize settings and create directories."""
        super().__init__(**kwargs)
        self.DEBUG_ROOT.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


settings = Settings()
