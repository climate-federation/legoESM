"""Tests for the multi-year contiguous CRU-JRA forcing (Phase B).

Covers:
  * shape / concatenation: N-year forcing has N * per-year-steps along axis 0
  * cross-year continuity: no NaN, T is close (not equal) across 12/31 -> 1/1
  * argument validation: year_end < year_start raises
  * synthetic path works multi-year without external data
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.forcing import cru_jra, stage_forcing_years

jax.config.update("jax_enable_x64", True)

_SEC_PER_DAY = 86400.0
_SEC_PER_YEAR = _SEC_PER_DAY * 365.0                    # noleap


def _cols():
    lat = np.radians(np.array([-30.0, 0.0, 45.0]))
    lon = np.radians(np.array([0.0, 120.0, 240.0]))
    return lat, lon


def test_stage_forcing_years_shape_and_concat():
    # Two years of hourly forcing spanning [0, 2*sec_per_year).
    lat, lon = _cols()
    n_per_year = 24                                     # one day/year to keep it fast
    dt = _SEC_PER_HOUR = 3600.0
    tq = np.concatenate([
        np.arange(n_per_year) * dt,                     # year 0 substeps
        _SEC_PER_YEAR + np.arange(n_per_year) * dt,     # year 1 substeps
    ])
    atm = stage_forcing_years(lat, lon, tq,
                              year_start=2000, year_end=2001, data_dir=None)
    assert isinstance(atm, AtmToSurface)
    # Every leaf carries the concatenated leading axis.
    leaves = jax.tree_util.tree_leaves(atm)
    for leaf in leaves:
        assert np.asarray(leaf).shape == (2 * n_per_year, lat.size)
    # Nothing NaN.
    assert np.all(np.isfinite(np.asarray(atm.T_lowest)))


def test_stage_forcing_years_cross_boundary_continuity():
    # Sample steps near the 12/31 -> 1/1 boundary and confirm T is close
    # (not necessarily equal — small sub-6h discontinuity is documented).
    lat, lon = _cols()
    tq_end   = np.array([_SEC_PER_YEAR - 3600.0])       # last hour of year 0
    tq_start = np.array([_SEC_PER_YEAR + 3600.0])       # first hour of year 1
    tq = np.concatenate([tq_end, tq_start])
    atm = stage_forcing_years(lat, lon, tq,
                              year_start=2000, year_end=2001, data_dir=None)
    T = np.asarray(atm.T_lowest)
    assert np.all(np.isfinite(T))
    # For a smooth synthetic climatology, hourly step across the boundary
    # should stay within a physically-plausible jump (< 20 K).
    assert np.all(np.abs(T[0] - T[1]) < 20.0)


def test_year_end_lt_year_start_raises():
    lat, lon = _cols()
    with pytest.raises(ValueError, match="year_end"):
        stage_forcing_years(lat, lon, np.array([0.0, 3600.0]),
                            year_start=2001, year_end=2000, data_dir=None)


def test_single_year_range_matches_stage_forcing():
    # year_start == year_end is a legitimate degenerate case and must produce
    # bit-identical output to the single-year stage_forcing.
    lat, lon = _cols()
    dt = 3600.0
    tq = np.arange(12) * dt
    a_single = cru_jra.stage_forcing(lat, lon, tq, year=2000, data_dir=None)
    a_range  = stage_forcing_years(lat, lon, tq,
                                    year_start=2000, year_end=2000, data_dir=None)
    for k in ("T_lowest", "q_lowest", "sw_down", "precip_total", "cos_zenith"):
        np.testing.assert_allclose(np.asarray(getattr(a_range, k)),
                                   np.asarray(getattr(a_single, k)),
                                   rtol=1e-12, atol=0.0)


def test_no_times_in_range_raises():
    # model_times_s is expressed as seconds since year_start's Jan 1, so times
    # BEYOND the range's end are the only way to fall outside every year window.
    lat, lon = _cols()
    tq = np.array([3.0 * _SEC_PER_YEAR + 3600.0])       # 3 years after year_start
    with pytest.raises(ValueError, match="no model times"):
        stage_forcing_years(lat, lon, tq,
                            year_start=2000, year_end=2001, data_dir=None)
