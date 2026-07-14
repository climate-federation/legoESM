"""Tests for the shared anthropogenic-cover -> 17-PFT overlay (HYDE/Pongratz/KK10)."""

from __future__ import annotations

import numpy as np
import pytest

from legoesm.land.surface_params import CLM5_PFT_NAMES, N_PFT_CLM5
from legoesm.land.surface_data.sources.anthropogenic import (
    anthropogenic_to_pft_frac,
    build_anthropogenic_pft_frac,
    c4_crop_fraction_from_base,
    c4_grass_fraction_from_base,
    normalised_pnv_shape,
)

_IDX = {name: i for i, name in enumerate(CLM5_PFT_NAMES)}


def _forest_pnv(ny=2, nx=2):
    pnv = np.zeros((N_PFT_CLM5, ny, nx))
    pnv[_IDX["broadleaf_evergreen_tropical"]] = 1.0
    return pnv


def _f(val, nyear=1, ny=2, nx=2):
    return np.full((nyear, ny, nx), val)


def test_overlay_conserves_to_one_per_cell():
    """Natural + crop + pasture + urban == 1 per cell/year (of land)."""
    crop, pasture, urban = _f(0.2), _f(0.1), _f(0.05)
    out = anthropogenic_to_pft_frac(crop, pasture, urban, _forest_pnv(),
                                    np.full((2, 2), 0.5), np.full((2, 2), 0.5))
    assert out.shape == (1, N_PFT_CLM5, 2, 2)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(out >= 0.0)


def test_overlay_routes_crop_pasture_urban():
    crop, pasture, urban = _f(0.4, ny=1, nx=1), _f(0.2, ny=1, nx=1), _f(0.1, ny=1, nx=1)
    out = anthropogenic_to_pft_frac(
        crop, pasture, urban, _forest_pnv(1, 1),
        c4_grass_frac=np.array([[0.25]]), c4_crop_frac=np.array([[0.5]]))
    np.testing.assert_allclose(out[:, _IDX["crop_c3"]], 0.4 * 0.5)
    np.testing.assert_allclose(out[:, _IDX["crop_c4"]], 0.4 * 0.5)
    np.testing.assert_allclose(out[:, _IDX["c4_grass"]], 0.2 * 0.25)
    np.testing.assert_allclose(out[:, _IDX["c3_grass"]], 0.2 * 0.75)
    np.testing.assert_allclose(out[:, _IDX["bare_soil"]], 0.1)
    # natural residual 1-0.7=0.3 all to the forest PFT
    np.testing.assert_allclose(out[:, _IDX["broadleaf_evergreen_tropical"]], 0.3)


def test_overlay_clips_anthropogenic_over_one():
    """Anthropogenic total > 1 scales down proportionally; total stays 1."""
    crop, pasture, urban = _f(0.8, ny=1, nx=1), _f(0.6, ny=1, nx=1), _f(0.0, ny=1, nx=1)
    out = anthropogenic_to_pft_frac(crop, pasture, urban, _forest_pnv(1, 1),
                                    np.array([[0.0]]), np.array([[0.0]]))
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-12)
    # 0.8 : 0.6 scaled to sum 1 -> crop 4/7, pasture 3/7, no natural residual
    np.testing.assert_allclose(out[:, _IDX["crop_c3"]], 0.8 / 1.4)
    np.testing.assert_allclose(out[:, _IDX["c3_grass"]], 0.6 / 1.4)


def test_normalised_pnv_shape_partial_nan():
    pnv = np.zeros((N_PFT_CLM5, 1, 1))
    pnv[4, 0, 0] = np.nan          # NaN band at a live cell -> 0 weight
    pnv[5, 0, 0] = 1.0
    shape = normalised_pnv_shape(pnv)
    np.testing.assert_allclose(shape[4, 0, 0], 0.0)
    np.testing.assert_allclose(shape[5, 0, 0], 1.0)


def test_c4_fraction_helpers():
    base = np.zeros((N_PFT_CLM5, 1, 1))
    base[_IDX["c3_grass"]] = 3.0; base[_IDX["c4_grass"]] = 1.0
    base[_IDX["crop_c3"]] = 1.0; base[_IDX["crop_c4"]] = 1.0
    np.testing.assert_allclose(c4_grass_fraction_from_base(base)[0, 0], 0.25)
    np.testing.assert_allclose(c4_crop_fraction_from_base(base)[0, 0], 0.5)


def test_build_anthropogenic_end_to_end_conserves():
    base = _forest_pnv(2, 2)
    base[_IDX["c3_grass"]] = 1.0; base[_IDX["c4_grass"]] = 1.0
    anthro = {"crop": _f(0.3), "pasture": _f(0.2), "urban": _f(0.05)}
    out = build_anthropogenic_pft_frac(anthro, base)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-12)


def test_build_anthropogenic_missing_keys_default_zero():
    base = _forest_pnv(1, 1)
    out = build_anthropogenic_pft_frac({"crop": _f(0.5, ny=1, nx=1)}, base)  # no pasture/urban
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(out[:, _IDX["crop_c3"]], 0.5)   # pasture/urban = 0


def test_overlay_land_frac_budget_preserves_gridcell_fractions():
    """With land_frac < 1, cover sums to land_frac and anthro fractions are kept."""
    crop, pasture, urban = _f(0.2, ny=1, nx=1), _f(0.15, ny=1, nx=1), _f(0.02, ny=1, nx=1)
    out = anthropogenic_to_pft_frac(
        crop, pasture, urban, _forest_pnv(1, 1),
        c4_grass_frac=np.array([[0.0]]), c4_crop_frac=np.array([[0.0]]),
        land_frac=0.5)
    np.testing.assert_allclose(out.sum(axis=1), 0.5, atol=1e-12)   # budget, not 1
    np.testing.assert_allclose(out[:, _IDX["crop_c3"]], 0.2)        # grid-cell frac kept
    np.testing.assert_allclose(out[:, _IDX["c3_grass"]], 0.15)      # pasture kept (c4=0)
    np.testing.assert_allclose(out[:, _IDX["bare_soil"]], 0.02)     # urban kept
    # natural budget 0.5-0.37=0.13 all on the forest PFT.
    np.testing.assert_allclose(out[:, _IDX["broadleaf_evergreen_tropical"]], 0.13)


def test_overlay_land_frac_zero_is_all_zero():
    """A non-land cell (land_frac 0) yields an all-zero column, mask preserved."""
    out = anthropogenic_to_pft_frac(
        _f(0.3, ny=1, nx=1), _f(0.2, ny=1, nx=1), _f(0.0, ny=1, nx=1),
        _forest_pnv(1, 1), np.array([[0.0]]), np.array([[0.0]]), land_frac=0.0)
    np.testing.assert_allclose(out, 0.0, atol=1e-12)


def test_overlay_sanitizes_nan_and_negative():
    """NaN/negative source values -> 0 (never poison the sum or make PFT < 0)."""
    crop = np.array([[[np.nan]]])         # missing (coastal regrid) -> 0
    pasture = np.array([[[-0.1]]])        # invalid negative -> 0
    urban = _f(0.1, ny=1, nx=1)
    out = anthropogenic_to_pft_frac(
        crop, pasture, urban, _forest_pnv(1, 1),
        np.array([[0.0]]), np.array([[0.0]]))
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(out[:, _IDX["bare_soil"]], 0.1)     # only urban survived
