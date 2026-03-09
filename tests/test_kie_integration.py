"""Tests for KIE.ai integration."""
import pytest

from app.integrations.kie.builders import (
    build_kie_payload,
    build_kie_payload_for_gpt_image_i2i,
    build_kie_payload_for_nano_banana_edit,
)
from app.integrations.kie.models import (
    KIEModel,
    KIEValidationError,
    UnsupportedModelError,
)


class TestKIEModel:
    """Tests for KIEModel enum."""

    def test_model_enum_values(self):
        """Test that model enum has correct values."""
        assert KIEModel.GPT_IMAGE_15_I2I.value == "gpt-image/1.5-image-to-image"
        assert KIEModel.NANO_BANANA_EDIT.value == "google/nano-banana-edit"

    def test_model_from_string_valid(self):
        """Test creating model from valid string."""
        model = KIEModel.from_string("gpt-image/1.5-image-to-image")
        assert model == KIEModel.GPT_IMAGE_15_I2I

        model = KIEModel.from_string("google/nano-banana-edit")
        assert model == KIEModel.NANO_BANANA_EDIT

    def test_model_from_string_invalid(self):
        """Test creating model from invalid string raises error."""
        with pytest.raises(UnsupportedModelError) as exc_info:
            KIEModel.from_string("invalid-model")
        assert "Unsupported KIE model" in str(exc_info.value)


class TestBuilders:
    """Tests for payload builders."""

    def test_build_gpt_image_i2i_payload(self):
        """Test building payload for GPT Image 1.5."""
        payload = build_kie_payload_for_gpt_image_i2i(
            prompt="Test prompt",
            poster_url="https://example.com/poster.png",
            reference_url="https://example.com/reference.png",
            aspect_ratio="3:2",
            quality="high",
        )

        assert payload["model"] == "gpt-image/1.5-image-to-image"
        assert "input" in payload
        assert payload["input"]["prompt"] == "Test prompt"
        assert "input_urls" in payload["input"]
        assert len(payload["input"]["input_urls"]) == 2
        assert payload["input"]["aspect_ratio"] == "3:2"
        assert payload["input"]["quality"] == "high"

    def test_build_gpt_image_i2i_payload_missing_prompt(self):
        """Test that missing prompt raises validation error."""
        with pytest.raises(KIEValidationError) as exc_info:
            build_kie_payload_for_gpt_image_i2i(prompt="")
        assert "Prompt is required" in str(exc_info.value)

    def test_build_nano_banana_edit_payload(self):
        """Test building payload for Nano Banana Edit."""
        payload = build_kie_payload_for_nano_banana_edit(
            prompt="Test prompt",
            poster_url="https://example.com/poster.png",
            reference_url="https://example.com/reference.png",
            output_format="png",
        )

        assert payload["model"] == "google/nano-banana-edit"
        assert "input" in payload
        assert payload["input"]["prompt"] == "Test prompt"
        assert "image_urls" in payload["input"]
        assert len(payload["input"]["image_urls"]) == 2
        assert payload["input"]["output_format"] == "png"

    def test_build_nano_banana_edit_missing_poster_url(self):
        """Test that missing poster_url raises validation error."""
        with pytest.raises(KIEValidationError) as exc_info:
            build_kie_payload_for_nano_banana_edit(
                prompt="Test prompt",
                poster_url=None,
                reference_url="https://example.com/reference.png",
            )
        assert "poster_url is required" in str(exc_info.value)
        assert exc_info.value.model == "google/nano-banana-edit"

    def test_build_nano_banana_edit_missing_reference_url(self):
        """Test that missing reference_url raises validation error."""
        with pytest.raises(KIEValidationError) as exc_info:
            build_kie_payload_for_nano_banana_edit(
                prompt="Test prompt",
                poster_url="https://example.com/poster.png",
                reference_url=None,
            )
        assert "reference_url is required" in str(exc_info.value)

    def test_build_nano_banana_edit_local_path_error(self):
        """Test that local file path raises validation error."""
        with pytest.raises(KIEValidationError) as exc_info:
            build_kie_payload_for_nano_banana_edit(
                prompt="Test prompt",
                poster_url="/local/path/to/image.png",
                reference_url="https://example.com/reference.png",
            )
        assert "must be a public HTTP(S) URL" in str(exc_info.value)

    def test_build_kie_payload_routing(self):
        """Test that build_kie_payload routes to correct builder."""
        # Test GPT Image 1.5
        payload = build_kie_payload(
            model="gpt-image/1.5-image-to-image",
            prompt="Test",
            poster_url="https://example.com/poster.png",
        )
        assert payload["model"] == "gpt-image/1.5-image-to-image"

        # Test Nano Banana Edit
        payload = build_kie_payload(
            model="google/nano-banana-edit",
            prompt="Test",
            poster_url="https://example.com/poster.png",
            reference_url="https://example.com/reference.png",
        )
        assert payload["model"] == "google/nano-banana-edit"
        assert "image_urls" in payload["input"]

    def test_build_kie_payload_unsupported_model(self):
        """Test that unsupported model raises error."""
        with pytest.raises(UnsupportedModelError):
            build_kie_payload(
                model="invalid-model",
                prompt="Test",
            )


class TestKIEClient:
    """Tests for KIEClient (mocked)."""

    def test_client_initialization(self):
        """Test client initialization."""
        from app.integrations.kie.client import KIEClient

        client = KIEClient(
            api_key="test-key",
            api_url="https://api.test.com",
            primary_model="gpt-image/1.5-image-to-image",
            fallback_model="google/nano-banana-edit",
        )

        assert client.api_key == "test-key"
        assert client.api_url == "https://api.test.com"
        assert client.primary_model == "gpt-image/1.5-image-to-image"
        assert client.fallback_model == "google/nano-banana-edit"

    def test_client_validation_missing_prompt(self):
        """Test that missing prompt raises validation error."""
        from app.integrations.kie.client import KIEClient

        client = KIEClient(api_key="test-key")
        with pytest.raises(KIEValidationError) as exc_info:
            client.create_image_edit_task(prompt="")
        assert "Prompt is required" in str(exc_info.value)
