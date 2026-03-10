"""Tests for age-based routing logic."""
import pytest
from PIL import Image

from app.core.config import settings
from app.services.age_routing import AgeRouter, SubjectAgeClass
from app.services.refinement.strategy_router import RefinementStrategy, StrategyRouter


class TestAgeRouting:
    """Test age-based routing logic."""

    def test_adult_routes_to_adult_strategy(self):
        """ADULT should always route to ADULT_KIE_REALISTIC."""
        router = StrategyRouter(unknown_policy="route_to_adult")
        strategy = router.select_refinement_strategy(SubjectAgeClass.ADULT)
        assert strategy == RefinementStrategy.ADULT_KIE_REALISTIC

    def test_teen_routes_to_teen_strategy(self):
        """TEEN_OR_MINOR should always route to TEEN_KIE_STYLIZED."""
        router = StrategyRouter(unknown_policy="route_to_adult")
        strategy = router.select_refinement_strategy(SubjectAgeClass.TEEN_OR_MINOR)
        assert strategy == RefinementStrategy.TEEN_KIE_STYLIZED

    def test_unknown_routes_to_adult_by_default(self):
        """UNKNOWN should route to ADULT_KIE_REALISTIC by default (safe)."""
        router = StrategyRouter(unknown_policy="route_to_adult")
        strategy = router.select_refinement_strategy(SubjectAgeClass.UNKNOWN)
        assert strategy == RefinementStrategy.ADULT_KIE_REALISTIC

    def test_unknown_can_route_to_teen_with_explicit_policy(self):
        """UNKNOWN can route to TEEN_KIE_STYLIZED only with explicit policy."""
        router = StrategyRouter(unknown_policy="route_to_teen")
        strategy = router.select_refinement_strategy(SubjectAgeClass.UNKNOWN)
        assert strategy == RefinementStrategy.TEEN_KIE_STYLIZED

    def test_teen_refiner_never_invoked_for_adult(self):
        """TeenRefiner should never be invoked for ADULT classification."""
        router = StrategyRouter(unknown_policy="route_to_adult")
        strategy = router.select_refinement_strategy(SubjectAgeClass.ADULT)
        # Strategy should be adult, not teen
        assert strategy != RefinementStrategy.TEEN_KIE_STYLIZED
        assert strategy == RefinementStrategy.ADULT_KIE_REALISTIC

    def test_teen_refiner_never_invoked_for_unknown_default(self):
        """TeenRefiner should never be invoked for UNKNOWN with default policy."""
        router = StrategyRouter(unknown_policy="route_to_adult")
        strategy = router.select_refinement_strategy(SubjectAgeClass.UNKNOWN)
        # Strategy should be adult, not teen
        assert strategy != RefinementStrategy.TEEN_KIE_STYLIZED
        assert strategy == RefinementStrategy.ADULT_KIE_REALISTIC

    def test_default_policy_is_route_to_adult(self):
        """Default policy should be route_to_adult for safety."""
        # Check that config default is safe
        assert settings.TEEN_UNKNOWN_POLICY == "route_to_adult"

    def test_routing_matrix(self):
        """Test complete routing matrix."""
        test_cases = [
            # (age_class, unknown_policy, expected_strategy)
            (SubjectAgeClass.ADULT, "route_to_adult", RefinementStrategy.ADULT_KIE_REALISTIC),
            (SubjectAgeClass.ADULT, "route_to_teen", RefinementStrategy.ADULT_KIE_REALISTIC),
            (SubjectAgeClass.TEEN_OR_MINOR, "route_to_adult", RefinementStrategy.TEEN_KIE_STYLIZED),
            (SubjectAgeClass.TEEN_OR_MINOR, "route_to_teen", RefinementStrategy.TEEN_KIE_STYLIZED),
            (SubjectAgeClass.UNKNOWN, "route_to_adult", RefinementStrategy.ADULT_KIE_REALISTIC),
            (SubjectAgeClass.UNKNOWN, "route_to_teen", RefinementStrategy.TEEN_KIE_STYLIZED),
        ]

        for age_class, unknown_policy, expected_strategy in test_cases:
            router = StrategyRouter(unknown_policy=unknown_policy)
            strategy = router.select_refinement_strategy(age_class)
            assert strategy == expected_strategy, (
                f"Failed for age_class={age_class.value}, "
                f"unknown_policy={unknown_policy}, "
                f"expected={expected_strategy.value}, "
                f"got={strategy.value}"
            )

    def test_adult_never_routes_to_kie(self):
        """ADULT should never route to KIE (TEEN_KIE_STYLIZED)."""
        router = StrategyRouter(unknown_policy="route_to_adult")
        strategy = router.select_refinement_strategy(SubjectAgeClass.ADULT)
        assert strategy != RefinementStrategy.TEEN_KIE_STYLIZED
        assert strategy == RefinementStrategy.ADULT_KIE_REALISTIC
        # Note: ADULT_KIE_REALISTIC is the strategy name, but in practice
        # adult flow does NOT use KIE refiner - it uses simple compositing

    def test_unknown_never_routes_to_kie_by_default(self):
        """UNKNOWN should never route to KIE by default."""
        router = StrategyRouter(unknown_policy="route_to_adult")
        strategy = router.select_refinement_strategy(SubjectAgeClass.UNKNOWN)
        assert strategy != RefinementStrategy.TEEN_KIE_STYLIZED
        assert strategy == RefinementStrategy.ADULT_KIE_REALISTIC
        # Note: ADULT_KIE_REALISTIC is the strategy name, but in practice
        # adult flow does NOT use KIE refiner - it uses simple compositing

    def test_only_teen_routes_to_kie(self):
        """Only TEEN_OR_MINOR should route to KIE (TEEN_KIE_STYLIZED)."""
        router = StrategyRouter(unknown_policy="route_to_adult")
        strategy = router.select_refinement_strategy(SubjectAgeClass.TEEN_OR_MINOR)
        assert strategy == RefinementStrategy.TEEN_KIE_STYLIZED
        # This is the only case where KIE refiner should be used
