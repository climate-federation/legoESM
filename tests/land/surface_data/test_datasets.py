"""Tests for the land-cover dataset registry (reconstruction choice)."""

import pytest

from legoesm.land.surface_data.datasets import (
    KNOWN_LAND_COVER_DATASETS,
    has_gross_transitions,
    validate_land_cover_dataset,
)


def test_known_datasets_validate():
    for name in KNOWN_LAND_COVER_DATASETS:
        assert validate_land_cover_dataset(name) == name


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="unknown land_cover_dataset"):
        validate_land_cover_dataset("sage")


def test_gross_transitions_only_luh():
    # Only LUH2/LUH3 carry native gross transitions (E_LUC fidelity split).
    assert has_gross_transitions("luh2") is True
    assert has_gross_transitions("luh3") is True
    for name in ("clm5", "hyde", "pongratz", "kk10"):
        assert has_gross_transitions(name) is False


def test_gross_transitions_unknown_raises():
    with pytest.raises(ValueError, match="unknown land_cover_dataset"):
        has_gross_transitions("foo")
