"""Unit tests for the per-archetype observed-biomass loader
(``legoesm.land.carbon.biomass_observations``) + its per-cell use of the shared
cover-weighted-mean helper (``soc_observations``).

Pure NumPy (no JAX / model step); the loader is a fixed calibration target.  The real
ESA-CCI/GEDI AGB product is a documented fetcher follow-up -- here only the loader (a
pre-regridded per-cell NetCDF) + the labelled synthetic target are exercised.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from legoesm.land.carbon.biomass_observations import (
    per_archetype_observed_biomass,
    synthetic_observed_biomass,
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


def test_observed_biomass_delegates_to_shared_mean():
    """Biomass is per-cell (like SOC/SIF): the archetype target is the shared cover-weighted
    mean of the per-cell biomass (no duplicated numerics)."""
    bio_cell = np.array([1.0, 4.0])
    cid = np.array([[0, -1], [1, 0]])
    cw = np.array([[0.75, 0.0], [1.0, 0.25]])
    got = per_archetype_observed_biomass(bio_cell, cid, cw, n_arch=2)
    ref = per_archetype_cover_weighted_mean(bio_cell, cid, cw, n_arch=2)
    npt.assert_allclose(got, ref, rtol=1e-12)
    # arch0 = (0.75*1 + 0.25*4)/(0.75+0.25) = 1.75 ; arch1 = 4.0 (single member).
    npt.assert_allclose(got, [1.75, 4.0], rtol=1e-12)


def test_observed_biomass_missing_is_nan():
    bio_cell = np.array([np.nan, np.nan])
    cid = np.array([[0], [0]])
    cw = np.array([[1.0], [1.0]])
    out = per_archetype_observed_biomass(bio_cell, cid, cw, n_arch=1)
    assert np.isnan(out[0])                # all-missing archetype -> NaN, never a fake 0


def test_synthetic_observed_biomass_positive_and_ordered():
    bio = synthetic_observed_biomass(_table([120.0, 260.0], [283.0, 300.0]))
    assert np.all(bio > 0.0)
    assert bio[1] > bio[0]                 # more shortwave + warmth -> more standing biomass
    assert np.all(bio < 100.0)             # a sane live-biomass kgC/m2 scale


def test_load_gridded_biomass_reshape_mask_and_mismatch(tmp_path):
    xr = pytest.importorskip("xarray")
    try:
        from legoesm.land.carbon.biomass_observations import load_gridded_biomass
        arr = np.array([[1.0, np.nan], [3.0, 4.0]])      # a product gap at (0,1)
        ds = xr.Dataset({"biomass": (("lat", "lon"), arr)})
        path = tmp_path / "bio.nc"
        ds.to_netcdf(path)
    except Exception as exc:                             # no netcdf backend in this env
        pytest.skip(f"netcdf write unavailable: {exc}")
    out = load_gridded_biomass(str(path), ncell=4)
    assert np.isnan(out[1])                              # gap PRESERVED, not a fabricated 0
    npt.assert_allclose(out[[0, 2, 3]], [1.0, 3.0, 4.0], rtol=1e-12)
    with pytest.raises(SystemExit, match="ncell"):
        load_gridded_biomass(str(path), ncell=5)


def test_load_gridded_biomass_time_averaged(tmp_path):
    xr = pytest.importorskip("xarray")
    try:
        from legoesm.land.carbon.biomass_observations import load_gridded_biomass
        arr = np.stack([np.full((2, 2), 2.0), np.full((2, 2), 6.0)])   # (time=2, 2, 2)
        ds = xr.Dataset({"agb": (("time", "lat", "lon"), arr)})
        path = tmp_path / "bio_t.nc"
        ds.to_netcdf(path)
    except Exception as exc:
        pytest.skip(f"netcdf write unavailable: {exc}")
    out = load_gridded_biomass(str(path), ncell=4)
    npt.assert_allclose(out, np.full(4, 4.0), rtol=1e-12)              # mean over time
