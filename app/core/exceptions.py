"""Custom exceptions."""


class ImageGlueException(Exception):
    """Base exception."""

    pass


class ValidationError(ImageGlueException):
    """Input validation error."""

    pass


class DetectionError(ImageGlueException):
    """Detection error."""

    pass


class SegmentationError(ImageGlueException):
    """Segmentation error."""

    pass


class CompositingError(ImageGlueException):
    """Compositing error."""

    pass


class QualityGateError(ImageGlueException):
    """Quality gate rejection."""

    pass


class TemplateError(ImageGlueException):
    """Template error."""

    pass
