"""The transient-year cover selector shared by the surface_data builders."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp

from legoesm.land.boundary_data import cover_fracs, dominant_pft_index


def _gsd(*year_slices, years):
    """Minimal gsd stand-in: cover_fracs only reads pft_frac + years."""
    return SimpleNamespace(
        pft_frac=jnp.asarray(np.stack(year_slices)),   # (nyear, ncol, npft)
        years=jnp.asarray(np.asarray(years, dtype=float)),
    )


def test_cover_fracs_none_is_year_mean():
    y0 = np.array([[1.0, 0.0, 0.0]])
    y1 = np.array([[0.0, 1.0, 0.0]])
    gsd = _gsd(y0, y1, years=[2000, 2010])
    np.testing.assert_allclose(np.asarray(cover_fracs(gsd, None)), 0.5 * (y0 + y1))


def test_cover_fracs_selects_exact_year():
    y0 = np.array([[1.0, 0.0, 0.0]])
    y1 = np.array([[0.0, 1.0, 0.0]])
    gsd = _gsd(y0, y1, years=[2000, 2010])
    np.testing.assert_allclose(np.asarray(cover_fracs(gsd, 2000.0)), y0)
    np.testing.assert_allclose(np.asarray(cover_fracs(gsd, 2010.0)), y1)


def test_cover_fracs_interpolates_between_years():
    y0 = np.array([[1.0, 0.0, 0.0]])
    y1 = np.array([[0.0, 1.0, 0.0]])
    gsd = _gsd(y0, y1, years=[2000, 2010])
    np.testing.assert_allclose(np.asarray(cover_fracs(gsd, 2005.0)), 0.5 * (y0 + y1))


def test_cover_fracs_single_year_ignores_requested_year():
    y0 = np.array([[0.3, 0.7, 0.0]])
    gsd = _gsd(y0, years=[2000])           # nyear == 1
    np.testing.assert_allclose(np.asarray(cover_fracs(gsd, 2050.0)), y0)
    np.testing.assert_allclose(np.asarray(cover_fracs(gsd, None)), y0)


def test_dominant_pft_index_tracks_year():
    # A cell that is PFT 1 in 2000 and PFT 2 in 2010 -> dominant flips with year.
    y0 = np.array([[0.1, 0.8, 0.1]])
    y1 = np.array([[0.1, 0.1, 0.8]])
    gsd = _gsd(y0, y1, years=[2000, 2010])
    assert dominant_pft_index(gsd, 2000.0).tolist() == [1]
    assert dominant_pft_index(gsd, 2010.0).tolist() == [2]
    # Year-mean is symmetric (0.1, 0.45, 0.45); argmax picks the first max -> PFT 1.
    assert dominant_pft_index(gsd, None).tolist() == [1]
