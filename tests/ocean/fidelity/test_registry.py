"""Unit tests for legoesm.ocean.fidelity.registry."""

from __future__ import annotations

import pytest

from legoesm.ocean.fidelity import registry


@pytest.fixture(autouse=True)
def _clean_registry():
    registry.clear_registry()
    yield
    registry.clear_registry()


def _identity_metric(artifacts):  # pragma: no cover - trivial
    return 1.0


def _identity_reference(artifacts):  # pragma: no cover - trivial
    return 1.0


def test_register_and_lookup():
    case = registry.FidelityCase(
        tier=1, name="igw", metric_fn=_identity_metric,
        reference_fn=_identity_reference,
    )
    registry.register_case(case)
    assert registry.get_case(1, "igw") is case


def test_duplicate_registration_errors():
    case = registry.FidelityCase(
        tier=1, name="igw", metric_fn=_identity_metric,
        reference_fn=_identity_reference,
    )
    registry.register_case(case)
    with pytest.raises(registry.RegistryError, match="already registered"):
        registry.register_case(case)


def test_lookup_missing_errors():
    with pytest.raises(registry.RegistryError, match="no case"):
        registry.get_case(0, "does_not_exist")


def test_get_cases_for_tier_sorted_by_name():
    registry.register_case(registry.FidelityCase(
        tier=1, name="zebra", metric_fn=_identity_metric,
        reference_fn=_identity_reference,
    ))
    registry.register_case(registry.FidelityCase(
        tier=1, name="apple", metric_fn=_identity_metric,
        reference_fn=_identity_reference,
    ))
    cases = registry.get_cases_for_tier(1)
    assert [c.name for c in cases] == ["apple", "zebra"]


def test_get_cases_for_empty_tier_returns_empty_list():
    assert registry.get_cases_for_tier(3) == []


def test_list_tiers_returns_sorted_unique():
    for tier, name in [(2, "a"), (5, "b"), (2, "c"), (0, "d")]:
        registry.register_case(registry.FidelityCase(
            tier=tier, name=name, metric_fn=_identity_metric,
            reference_fn=_identity_reference,
        ))
    assert registry.list_tiers() == [0, 2, 5]


def test_tier_out_of_range_rejected():
    with pytest.raises(registry.RegistryError, match="tier"):
        registry.register_case(registry.FidelityCase(
            tier=9, name="x", metric_fn=_identity_metric,
            reference_fn=_identity_reference,
        ))


def test_bad_ci_marker_rejected():
    with pytest.raises(registry.RegistryError, match="ci_marker"):
        registry.register_case(registry.FidelityCase(
            tier=1, name="x", metric_fn=_identity_metric,
            reference_fn=_identity_reference,
            ci_marker="weekly",  # type: ignore[arg-type]
        ))


def test_non_case_input_rejected():
    with pytest.raises(registry.RegistryError, match="expected FidelityCase"):
        registry.register_case("not a case")  # type: ignore[arg-type]


def test_clear_registry_empties_state():
    registry.register_case(registry.FidelityCase(
        tier=0, name="x", metric_fn=_identity_metric,
        reference_fn=_identity_reference,
    ))
    assert registry.list_tiers() == [0]
    registry.clear_registry()
    assert registry.list_tiers() == []
