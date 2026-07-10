"""Unit tests for the shared PFT growth-form / leaf-habit / pathway classifiers in
``legoesm.land.surface_params`` (``is_woody`` / ``is_evergreen`` / ``is_c4``).

These live next to ``CLM5_PFT_NAMES`` as the SINGLE source of truth shared by
the archetype IC builder (``carbon.global_init``) and its per-pixel validator
(``scripts/validate/land_carbon_equilibrium.py``); this test pins their truth
table so the two consumers can never drift onto different physics.
"""

from __future__ import annotations

from legoesm.land.surface_params import (
    CLM5_PFT_NAMES,
    is_c4,
    is_evergreen,
    is_woody,
)


def test_is_woody_trees_and_shrubs_true():
    for name in (
        "needleleaf_evergreen_boreal",
        "broadleaf_evergreen_tropical",
        "broadleaf_deciduous_temperate",
        "broadleaf_deciduous_boreal_shrub",
        "broadleaf_evergreen_shrub",
    ):
        assert is_woody(name) is True, name


def test_is_woody_grasses_and_crops_false():
    for name in (
        "c3_grass",
        "c4_grass",
        "c3_arctic_grass",
        "crop_c3",
        "crop_c4",
    ):
        assert is_woody(name) is False, name


def test_is_evergreen_truth_table():
    assert is_evergreen("needleleaf_evergreen_temperate") is True
    assert is_evergreen("broadleaf_evergreen_tropical") is True
    assert is_evergreen("broadleaf_evergreen_shrub") is True
    # deciduous woody and all herbaceous PFTs are NOT evergreen
    assert is_evergreen("needleleaf_deciduous_boreal") is False
    assert is_evergreen("broadleaf_deciduous_temperate") is False
    assert is_evergreen("c3_grass") is False
    assert is_evergreen("crop_c4") is False


def test_is_c4_truth_table():
    # EXACTLY the two C4 PFTs are C4; every C3 grass/crop and all woody PFTs are C3.
    assert is_c4("c4_grass") is True
    assert is_c4("crop_c4") is True
    for name in (
        "c3_grass",
        "c3_arctic_grass",
        "crop_c3",
        "broadleaf_evergreen_tropical",
        "needleleaf_evergreen_boreal",
        "bare_soil",
    ):
        assert is_c4(name) is False, name


def test_is_c4_matches_exactly_two_clm5_pfts():
    # The substring classifier must flag EXACTLY {c4_grass, crop_c4} across the CLM5 set --
    # the same two PFTs the C3-only Farquhar approximates, hence the two the delta13C term
    # masks out.
    c4_names = {name for name in CLM5_PFT_NAMES if is_c4(name)}
    assert c4_names == {"c4_grass", "crop_c4"}, c4_names


def test_classifiers_are_deterministic_bools_for_every_clm5_pft():
    # Every CLM5 PFT name must classify to a plain bool (no None / exceptions),
    # and evergreen implies woody (an evergreen grass/crop does not exist), and a C4 PFT is
    # never woody (both C4 PFTs are grasses/crops).
    for name in CLM5_PFT_NAMES:
        w, e, c4 = is_woody(name), is_evergreen(name), is_c4(name)
        assert isinstance(w, bool) and isinstance(e, bool) and isinstance(c4, bool), name
        if e:
            assert w, f"{name}: evergreen must be woody"
        if c4:
            assert not w, f"{name}: C4 PFTs are herbaceous (not woody)"
