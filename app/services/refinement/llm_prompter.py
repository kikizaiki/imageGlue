"""LLM-based prompt generation for AI refinement."""
import logging
from typing import Any

import httpx

from app.core.config import settings
from app.core.exceptions import CompositingError

logger = logging.getLogger(__name__)


class LLMPrompter:
    """Generates optimized prompts using LLM (via KIE.ai or similar)."""

    def __init__(self):
        """Initialize LLM prompter."""
        self.api_key = settings.KIE_API_KEY
        self.api_url = settings.KIE_API_URL.rstrip("/")

    def generate_refinement_prompt(
        self,
        template_description: str,
        issues: list[str] | None = None,
    ) -> str:
        """
        Generate optimized refinement prompt using LLM.

        Args:
            template_description: Description of template
            issues: List of detected issues to fix

        Returns:
            Generated prompt
        """
        if not self.api_key:
            # Fallback to template-based prompt
            return self._generate_fallback_prompt(template_description, issues)

        try:
            # Try to use LLM to generate better prompt
            system_prompt = (
                "You are an expert in image compositing and photo editing. "
                "Generate precise, actionable prompts for AI image editing that will improve compositing quality."
            )

            user_prompt = (
                f"Template: {template_description}\n"
                f"Issues to fix: {', '.join(issues) if issues else 'General quality improvement'}\n\n"
                "Generate a detailed, specific prompt for AI image editing that will: "
                "1. Improve edge blending and remove visible seams\n"
                "2. Match lighting and shadows naturally\n"
                "3. Ensure color harmony between elements\n"
                "4. Make the composition look like a single cohesive image\n\n"
                "Return only the prompt, no explanations."
            )

            payload = {
                "model": "gpt-4",  # Adjust based on KIE.ai available models
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 200,
                "temperature": 0.7,
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            # Try chat completion endpoint
            endpoints = [
                "/v1/chat/completions",
                "/api/v1/chat/completions",
                "/chat/completions",
            ]

            for endpoint in endpoints:
                try:
                    with httpx.Client(timeout=30.0) as client:
                        response = client.post(
                            f"{self.api_url}{endpoint}",
                            headers=headers,
                            json=payload,
                        )
                        response.raise_for_status()
                        result = response.json()

                        if "choices" in result and len(result["choices"]) > 0:
                            generated_prompt = result["choices"][0]["message"]["content"].strip()
                            logger.info("Generated prompt using LLM")
                            return generated_prompt

                except Exception as e:
                    logger.debug(f"LLM endpoint {endpoint} failed: {e}")
                    continue

            # Fallback if LLM not available
            return self._generate_fallback_prompt(template_description, issues)

        except Exception as e:
            logger.warning(f"LLM prompt generation failed: {e}, using fallback")
            return self._generate_fallback_prompt(template_description, issues)

    def _generate_fallback_prompt(
        self, template_description: str, issues: list[str] | None
    ) -> str:
        """Generate prompt without LLM."""
        base_prompt = (
            f"Improve the image compositing quality for a {template_description}. "
            "Make all elements blend seamlessly together. "
            "Fix visible edges, improve lighting matching, ensure natural shadows and highlights. "
            "The composition should look like a single cohesive professional image, not like elements were pasted together."
        )

        if issues:
            issue_text = ", ".join(issues)
            return f"{base_prompt} Specifically address: {issue_text}."

        return base_prompt
