"""Refinement strategy router for adult vs teen flows."""
import logging
from enum import Enum
from typing import Any

from app.services.age_routing import SubjectAgeClass

logger = logging.getLogger(__name__)


class RefinementStrategy(str, Enum):
    """Refinement strategy types."""

    ADULT_KIE_REALISTIC = "adult_kie_realistic"
    TEEN_KIE_STYLIZED = "teen_kie_stylized"


class StrategyRouter:
    """Routes to appropriate refinement strategy based on subject age."""

    def __init__(self, unknown_policy: str = "route_to_teen"):
        """
        Initialize strategy router.

        Args:
            unknown_policy: Policy for UNKNOWN age class ("route_to_teen" or "route_to_adult")
        """
        self.unknown_policy = unknown_policy

    def select_refinement_strategy(
        self,
        subject_age_class: SubjectAgeClass,
        config: dict[str, Any] | None = None,
    ) -> RefinementStrategy:
        """
        Select refinement strategy based on subject age class.

        Args:
            subject_age_class: Classified subject age
            config: Optional configuration overrides

        Returns:
            RefinementStrategy
        """
        config = config or {}

        # Check for explicit override in config
        if "refinement_strategy" in config:
            strategy_str = config["refinement_strategy"]
            try:
                return RefinementStrategy(strategy_str)
            except ValueError:
                logger.warning(f"Invalid strategy override: {strategy_str}, using routing logic")

        # Route based on age class
        # CRITICAL: Only TEEN_OR_MINOR should route to teen flow
        # ADULT and UNKNOWN should always route to adult flow for safety
        
        if subject_age_class == SubjectAgeClass.ADULT:
            strategy = RefinementStrategy.ADULT_KIE_REALISTIC
            reason = f"Subject classified as ADULT - using adult flow"
            logger.info(
                f"🎯 Selected strategy: {strategy.value}\n"
                f"   - Subject age class: {subject_age_class.value}\n"
                f"   - Reason: {reason}\n"
                f"   - Provider: KIE.ai (adult realistic)"
            )
            return strategy

        elif subject_age_class == SubjectAgeClass.TEEN_OR_MINOR:
            strategy = RefinementStrategy.TEEN_KIE_STYLIZED
            reason = f"Subject classified as TEEN_OR_MINOR - using teen flow"
            logger.info(
                f"🎯 Selected strategy: {strategy.value}\n"
                f"   - Subject age class: {subject_age_class.value}\n"
                f"   - Reason: {reason}\n"
                f"   - Provider: KIE.ai (teen stylized)"
            )
            return strategy

        else:  # UNKNOWN
            # SAFETY: UNKNOWN should default to adult flow, not teen
            # Only route to teen if explicitly configured AND user understands the risk
            if self.unknown_policy == "route_to_teen":
                strategy = RefinementStrategy.TEEN_KIE_STYLIZED
                reason = f"Subject UNKNOWN, policy={self.unknown_policy} - routing to teen (EXPLICIT POLICY)"
                logger.warning(
                    f"⚠️  Selected strategy: {strategy.value}\n"
                    f"   - Subject age class: {subject_age_class.value}\n"
                    f"   - Reason: {reason}\n"
                    f"   - Provider: KIE.ai (teen stylized)\n"
                    f"   - WARNING: UNKNOWN subject routed to teen flow due to explicit policy"
                )
                return strategy
            else:  # route_to_adult (default and safe)
                strategy = RefinementStrategy.ADULT_KIE_REALISTIC
                reason = f"Subject UNKNOWN, policy={self.unknown_policy} - routing to adult (SAFE DEFAULT)"
                logger.info(
                    f"🎯 Selected strategy: {strategy.value}\n"
                    f"   - Subject age class: {subject_age_class.value}\n"
                    f"   - Reason: {reason}\n"
                    f"   - Provider: KIE.ai (adult realistic)"
                )
                return strategy
