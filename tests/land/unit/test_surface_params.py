"""Unit tests for the shared PFT growth-form / leaf-habit / pathway classifiers in
``legoesm.land.surface_params`` (``is_woody`` / ``is_evergreen`` / ``is_c4``).

These live next to ``CLM5_PFT_NAMES`` as the SINGLE source of truth shared by
the archetype IC builder (``carbon.global_init``) and its per-pixel validator
(``scripts/validate/land_carbon_equilibrium.py``); this test pins their truth
table so the two consumers can never drift onto different physics.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
from legoesm.land.surface_params import (
    CLM5_PFT_NAMES,
    is_c4,
    is_c4_pft_id,
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
    # the same two PFTs the leaf-delta13C forward routes through the FAITHFUL C4 discrimination
    # branch (via is_c4_pft_id), and the observed loader keeps C4-banded.
    c4_names = {name for name in CLM5_PFT_NAMES if is_c4(name)}
    assert c4_names == {"c4_grass", "crop_c4"}, c4_names


def test_is_c4_pft_id_vectorized_matches_scalar_and_handles_out_of_range():
    """The vectorized per-id classifier (the SHARED C3/C4 selector for the leaf-delta13C
    forward + the observed mask) agrees with the scalar is_c4 on the name, flags EXACTLY the
    two C4 ids (14 c4_grass, 16 crop_c4), and maps out-of-range ids to False (never crashes)."""
    ids = np.array([1, 7, 13, 14, 15, 16])   # tree, tree, c3_grass, c4_grass, crop_c3, crop_c4
    mask = is_c4_pft_id(ids)
    assert mask.dtype == bool
    npt.assert_array_equal(mask, [is_c4(CLM5_PFT_NAMES[i]) for i in ids])
    npt.assert_array_equal(mask, [False, False, False, True, False, True])
    # full CLM5 sweep flags exactly the two C4 ids
    all_mask = is_c4_pft_id(np.arange(len(CLM5_PFT_NAMES)))
    assert set(np.flatnonzero(all_mask).tolist()) == {14, 16}
    # out-of-range ids -> False, one flag per id
    npt.assert_array_equal(is_c4_pft_id(np.array([14, 999, -1, 4])), [True, False, False, False])


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
