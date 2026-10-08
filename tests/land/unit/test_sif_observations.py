"""Unit tests for the per-archetype observed-SIF loader
(``legoesm.land.carbon.sif_observations``) + the shared cover-weighted-mean helper it
reuses from ``soc_observations``.

Pure NumPy (no JAX / model step); the loader is a fixed calibration target.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from legoesm.land.carbon.sif_observations import (
    per_archetype_observed_sif,
    synthetic_observed_sif,
)
from legoesm.land.carbon.soc_observations import per_archetype_cover_weighted_mean


def _table(sw, mat_k):
    from legoesm.land.carbon.global_init import ArchetypeTable
    sw = np.asarray(sw, dtype=float)
    n = sw.shape[0]
    return ArchetypeTable(
        pft_id=np.arange(1, n + 1, dtype=int),
        mat_k=np.asarray(mat_k, dtype=float),
        map_yr=np.asarray([1200.0] * n, dtype=float),
        t_seasonal_amp_k=np.asarray([6.0] * n, dtype=float),
        aridity=np.asarray([1.0] * n, dtype=float),
        sw_mean_w=sw,
        soil_class=np.asarray(["loam"] * n, dtype=object),
    )


def test_cover_weighted_mean_hand_value():
    # 2 cells, 2 PFTs; per-cell values [1.5, 3.5].
    vals = np.array([1.5, 3.5])
    cid = np.array([[0, -1], [1, 0]])
    cw = np.array([[0.75, 0.0], [1.0, 0.25]])
    out = per_archetype_cover_weighted_mean(vals, cid, cw, n_arch=2)
    # arch0 = (0.75*1.5 + 0.25*3.5) / (0.75+0.25) = 2.0 ; arch1 = 3.5 (single member).
    npt.assert_allclose(out, [2.0, 3.5], rtol=1e-12)


def test_cover_weighted_mean_zero_cover_is_nan():
    vals = np.array([1.0])
    cid = np.array([[0, -1]])
    cw = np.array([[1.0, 0.0]])
    out = per_archetype_cover_weighted_mean(vals, cid, cw, n_arch=2)
    npt.assert_allclose(out[0], 1.0, rtol=1e-12)
    assert np.isnan(out[1])            # unassigned archetype -> NaN, never 0


def test_cover_weighted_mean_dim_checks():
    with pytest.raises(ValueError):     # cell_values must be 1-D
        per_archetype_cover_weighted_mean(
            np.ones((2, 2)), np.array([[0], [1]]), np.array([[1.0], [1.0]]))
    with pytest.raises(ValueError):     # ncell mismatch
        per_archetype_cover_weighted_mean(
            np.ones(3), np.array([[0], [1]]), np.array([[1.0], [1.0]]))


def test_cover_weighted_mean_excludes_nan_cells():
    """A missing (NaN) cell value is EXCLUDED from the archetype mean (not a fabricated 0),
    from both numerator and denominator."""
    vals = np.array([2.0, np.nan, 4.0])
    cid = np.array([[0], [0], [1]])
    cw = np.array([[1.0], [3.0], [1.0]])
    out = per_archetype_cover_weighted_mean(vals, cid, cw, n_arch=2)
    # arch0 averages ONLY cell0 (cell1 NaN excluded) -> 2.0 (a 0-fill would give 0.5).
    npt.assert_allclose(out, [2.0, 4.0], rtol=1e-12)


def test_cover_weighted_mean_all_missing_is_nan():
    """An archetype whose member cells are ALL missing -> NaN (never 0)."""
    vals = np.array([np.nan, np.nan])
    cid = np.array([[0], [0]])
    cw = np.array([[1.0], [1.0]])
    out = per_archetype_cover_weighted_mean(vals, cid, cw, n_arch=1)
    assert np.isnan(out[0])


def test_observed_sif_delegates_to_shared_mean():
    sif_cell = np.array([0.5, 1.5])
    cid = np.array([[0, -1], [1, 0]])
    cw = np.array([[0.75, 0.0], [1.0, 0.25]])
    got = per_archetype_observed_sif(sif_cell, cid, cw, n_arch=2)
    ref = per_archetype_cover_weighted_mean(sif_cell, cid, cw, n_arch=2)
    npt.assert_allclose(got, ref, rtol=1e-12)


def test_synthetic_observed_sif_positive_and_ordered():
    """Synthetic target: strictly positive, higher for the brighter+warmer archetype."""
    sif = synthetic_observed_sif(_table([150.0, 260.0], [285.0, 300.0]))
    assert np.all(sif > 0.0)
    assert sif[1] > sif[0]              # more shortwave + warmth -> more SIF
    # In the model's native photon-flux O(10) umol/m2/s scale (not radiance).
    assert np.all(sif < 100.0)


def test_load_gridded_obs_sif_reshape_and_mismatch(tmp_path):
    xr = pytest.importorskip("xarray")
    try:
        from legoesm.land.carbon.sif_observations import SIF_VAR_CANDIDATES
        from legoesm.land.carbon.soc_observations import load_gridded_obs
        ds = xr.Dataset({"sif": (("lat", "lon"), np.arange(6.0).reshape(2, 3))})
        path = tmp_path / "sif.nc"
        ds.to_netcdf(path)
    except Exception as exc:                       # no netcdf backend in this env
        pytest.skip(f"netcdf write unavailable: {exc}")
    # 2x3 = 6 cells: correct ncell -> row-major (i_lat, i_lon) flatten.
    out = load_gridded_obs(str(path), SIF_VAR_CANDIDATES, "sif", ncell=6)
    npt.assert_allclose(out, np.arange(6.0), rtol=1e-12)
    # A wrong ncell is a hard error (never a silent misalignment).
    with pytest.raises(SystemExit, match="ncell"):
        load_gridded_obs(str(path), SIF_VAR_CANDIDATES, "sif", ncell=5)


def test_load_gridded_obs_sif_preserves_nan_gaps(tmp_path):
    xr = pytest.importorskip("xarray")
    try:
        from legoesm.land.carbon.sif_observations import SIF_VAR_CANDIDATES
        from legoesm.land.carbon.soc_observations import load_gridded_obs
        arr = np.array([[1.0, np.nan], [3.0, 4.0]])   # a satellite gap at (0,1)
        ds = xr.Dataset({"sif": (("lat", "lon"), arr)})
        path = tmp_path / "sif_gap.nc"
        ds.to_netcdf(path)
    except Exception as exc:
        pytest.skip(f"netcdf write unavailable: {exc}")
    out = load_gridded_obs(str(path), SIF_VAR_CANDIDATES, "sif", ncell=4)
    assert np.isnan(out[1])                            # gap PRESERVED, not fabricated 0
    npt.assert_allclose(out[[0, 2, 3]], [1.0, 3.0, 4.0], rtol=1e-12)


def test_load_gridded_obs_sif_time_averaged(tmp_path):
    xr = pytest.importorskip("xarray")
    try:
        from legoesm.land.carbon.sif_observations import SIF_VAR_CANDIDATES
        from legoesm.land.carbon.soc_observations import load_gridded_obs
        arr = np.stack([np.full((2, 2), 1.0), np.full((2, 2), 3.0)])   # (time=2, 2, 2)
        ds = xr.Dataset({"sif": (("time", "lat", "lon"), arr)})
        path = tmp_path / "sif_t.nc"
        ds.to_netcdf(path)
    except Exception as exc:
        pytest.skip(f"netcdf write unavailable: {exc}")
    out = load_gridded_obs(str(path), SIF_VAR_CANDIDATES, "sif", ncell=4)
    npt.assert_allclose(out, np.full(4, 2.0), rtol=1e-12)              # mean over time
