"""KIE.ai model definitions and exceptions."""
import enum
from typing import Any


class KIEModel(str, enum.Enum):
    """Supported KIE.ai image editing models."""

    GPT_IMAGE_15_I2I = "gpt-image/1.5-image-to-image"
    NANO_BANANA_EDIT = "google/nano-banana-edit"
    FLUX_KONTEXT_PRO = "flux-kontext-pro"
    QWEN_IMAGE_EDIT = "qwen/image-edit"
    SEEDREAM_4_5_EDIT = "seedream/4-5-edit"

    @classmethod
    def from_string(cls, model_name: str) -> "KIEModel":
        """
        Create KIEModel from string.

        Args:
            model_name: Model name string

        Returns:
            KIEModel instance

        Raises:
            UnsupportedModelError: If model is not supported
        """
        try:
            return cls(model_name)
        except ValueError:
            supported = ", ".join([m.value for m in cls])
            raise UnsupportedModelError(
                f"Unsupported KIE model: {model_name}. "
                f"Supported models: {supported}"
            )

    def __str__(self) -> str:
        """Return model name as string."""
        return self.value


class UnsupportedModelError(ValueError):
    """Raised when unsupported KIE model is requested."""

    pass


class KIEValidationError(ValueError):
    """Raised when KIE request validation fails."""

    def __init__(self, message: str, model: str | None = None, **details: Any):
        """
        Initialize validation error.

        Args:
            message: Error message
            model: Model name that caused the error
            **details: Additional error details
        """
        super().__init__(message)
        self.model = model
        self.details = details


class KIETaskError(Exception):
    """Raised when KIE task creation or execution fails."""

    def __init__(
        self,
        message: str,
        model: str | None = None,
        task_id: str | None = None,
        endpoint: str | None = None,
        **details: Any,
    ):
        """
        Initialize task error.

        Args:
            message: Error message
            model: Model name that caused the error
            task_id: Task ID if available
            endpoint: API endpoint that was called
            **details: Additional error details
        """
        super().__init__(message)
        self.model = model
        self.task_id = task_id
        self.endpoint = endpoint
        self.details = details
