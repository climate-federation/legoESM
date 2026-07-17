"""Unit tests for the per-archetype observed-LAI loader
(``legoesm.land.carbon.lai_observations``) + its per-(cell, PFT) use of the shared
cover-weighted-mean helper (``soc_observations``).

Pure NumPy (no JAX / model step); the loader is a fixed calibration target.  The real
surfdata field ``MONTHLY_LAI`` is exercised via a tiny in-memory NetCDF.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from legoesm.land.carbon.lai_observations import (
    annual_mean_lai,
    per_archetype_observed_lai,
    synthetic_observed_lai,
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


def test_annual_mean_lai_collapses_months_ignoring_nan():
    """12-month climatology -> annual mean over axis 0, NaN gaps ignored (nanmean)."""
    lai = np.stack([np.full((2, 2), float(m)) for m in range(12)])   # (12, 2, 2)
    lai[3, 0, 0] = np.nan                                            # one gap
    out = annual_mean_lai(lai)
    assert out.shape == (2, 2)
    npt.assert_allclose(out[0, 1], np.mean(np.arange(12.0)), rtol=1e-12)
    # cell (0,0): mean of months 0..11 excluding month 3.
    npt.assert_allclose(out[0, 0], np.mean([m for m in range(12) if m != 3]), rtol=1e-12)


def test_annual_mean_lai_rejects_wrong_month_axis():
    with pytest.raises(SystemExit, match="12 months"):
        annual_mean_lai(np.ones((6, 2, 2)))


def test_per_archetype_observed_lai_per_cell_pft_hand_value():
    """LAI is per-(cell, PFT): archetype a averages that PFT's LAI at its member cells
    (a 2-D (ncell, npft) value array through the shared cover-weighted mean)."""
    # 2 cells, 2 PFTs. V[c,p] = LAI of PFT p at cell c.
    lai_cp = np.array([[2.0, 5.0], [3.0, 6.0]])
    cid = np.array([[0, 1], [0, 1]])       # PFT0 -> arch0, PFT1 -> arch1 (both cells)
    cw = np.array([[1.0, 2.0], [3.0, 1.0]])
    out = per_archetype_observed_lai(lai_cp, cid, cw, n_arch=2)
    # arch0 (PFT0): (1*2 + 3*3)/(1+3) = 11/4 = 2.75 ; arch1 (PFT1): (2*5 + 1*6)/3 = 16/3.
    npt.assert_allclose(out, [2.75, 16.0 / 3.0], rtol=1e-12)
    # Equals the shared helper on the SAME (ncell, npft) value array (no duplicated numerics).
    ref = per_archetype_cover_weighted_mean(lai_cp, cid, cw, n_arch=2)
    npt.assert_allclose(out, ref, rtol=1e-12)


def test_per_archetype_observed_lai_masks_missing():
    """A (cell, PFT) with NaN LAI is excluded; an archetype with ALL members missing -> NaN
    (the loss then masks it out)."""
    lai_cp = np.array([[np.nan, 4.0], [np.nan, 6.0]])
    cid = np.array([[0, 1], [0, 1]])
    cw = np.array([[1.0, 1.0], [1.0, 1.0]])
    out = per_archetype_observed_lai(lai_cp, cid, cw, n_arch=2)
    assert np.isnan(out[0])                # PFT0 all-NaN -> NaN, never a fabricated 0
    npt.assert_allclose(out[1], 5.0, rtol=1e-12)


def test_synthetic_observed_lai_positive_and_ordered():
    lai = synthetic_observed_lai(_table([120.0, 260.0], [283.0, 300.0]))
    assert np.all(lai > 0.0)
    assert lai[1] > lai[0]                 # more shortwave + warmth -> more leaf area
    assert np.all(lai < 20.0)              # a sane annual-mean LAI scale


def test_load_surfdata_lai_reshape_mask_and_mismatch(tmp_path):
    """MONTHLY_LAI(12, npft, nlat, nlon) -> annual-mean per (cell, PFT) (ncell, npft);
    bare/degenerate LAI guarded to NaN; grid/PFT mismatch is a hard error."""
    xr = pytest.importorskip("xarray")
    try:
        from legoesm.land.carbon.lai_observations import load_surfdata_lai
        # npft=2, nlat=2, nlon=2 (ncell=4); constant over months so annual-mean = the value.
        base = np.array([[[1.0, 2.0], [3.0, 4.0]],       # PFT0 over (lat, lon)
                         [[5.0, 0.0], [7.0, 8.0]]])       # PFT1; the 0.0 is a bare cell
        lai = np.stack([base] * 12)                       # (12, 2, 2, 2)
        ds = xr.Dataset({"MONTHLY_LAI": (("time", "pft", "lat", "lon"), lai)})
        path = tmp_path / "surf_lai.nc"
        ds.to_netcdf(path)
    except Exception as exc:                              # no netcdf backend in this env
        pytest.skip(f"netcdf write unavailable: {exc}")
    out = load_surfdata_lai(str(path), ncell=4, n_pft=2)
    assert out.shape == (4, 2)                            # (ncell, npft)
    # Row-major (i_lat, i_lon) cells; PFT0 column = [1,2,3,4].
    npt.assert_allclose(out[:, 0], [1.0, 2.0, 3.0, 4.0], rtol=1e-12)
    # PFT1 bare cell (value 0.0 at cell 1) -> NaN (guarded), others preserved.
    assert np.isnan(out[1, 1])
    npt.assert_allclose(out[[0, 2, 3], 1], [5.0, 7.0, 8.0], rtol=1e-12)
    # Wrong PFT count / ncell are hard errors.
    with pytest.raises(SystemExit, match="PFTs"):
        load_surfdata_lai(str(path), ncell=4, n_pft=3)
    with pytest.raises(SystemExit, match="ncell"):
        load_surfdata_lai(str(path), ncell=9, n_pft=2)
