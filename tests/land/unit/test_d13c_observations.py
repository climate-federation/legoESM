"""Unit tests for the per-archetype observed leaf-delta13C loader
(``legoesm.land.carbon.d13c_observations``) + the C4-archetype mask (the C3-only-model
correctness guard) + the shared cover-weighted-mean helper it reuses from ``soc_observations``.

Pure NumPy (no JAX / model step); the loader is a fixed calibration target.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from legoesm.land.carbon.d13c_observations import (
    c4_archetype_mask,
    per_archetype_observed_d13c,
    synthetic_observed_d13c,
)
from legoesm.land.carbon.soc_observations import per_archetype_cover_weighted_mean


def _table(pft_id, mat_k):
    from legoesm.land.carbon.global_init import ArchetypeTable
    pft_id = np.asarray(pft_id, dtype=int)
    n = pft_id.shape[0]
    return ArchetypeTable(
        pft_id=pft_id,
        mat_k=np.asarray(mat_k, dtype=float),
        map_yr=np.asarray([1200.0] * n, dtype=float),
        t_seasonal_amp_k=np.asarray([6.0] * n, dtype=float),
        aridity=np.asarray([1.0] * n, dtype=float),
        sw_mean_w=np.asarray([200.0] * n, dtype=float),
        soil_class=np.asarray(["loam"] * n, dtype=object),
    )


def test_c4_archetype_mask_flags_exactly_the_c4_pfts():
    """The per-archetype C4 mask is True ONLY for the C4 PFTs (14 c4_grass, 16 crop_c4) --
    the two the C3-only Farquhar cannot represent, hence the two the delta13C term excludes."""
    pft_id = np.array([1, 7, 13, 14, 15, 16])   # tree, tree, c3_grass, c4_grass, crop_c3, crop_c4
    mask = c4_archetype_mask(pft_id)
    assert mask.dtype == bool
    npt.assert_array_equal(mask, [False, False, False, True, False, True])


def test_c4_archetype_mask_shape_and_out_of_range():
    """One flag per archetype; an out-of-range PFT id is treated as non-C4 (never crashes)."""
    mask = c4_archetype_mask(np.array([14, 999, -1, 4]))
    npt.assert_array_equal(mask, [True, False, False, False])


def test_synthetic_observed_d13c_in_c3_range_and_finite():
    """Synthetic target: one value per archetype, finite, in the physical C3 leaf band
    (~ -22..-34 permil) so the water-use-efficiency params have a residual to fit."""
    obs = synthetic_observed_d13c(_table([1, 4, 7], [278.0, 295.0, 305.0]))
    assert obs.shape == (3,)
    assert np.all(np.isfinite(obs))
    assert np.all(obs <= -22.0) and np.all(obs >= -34.0), obs


def test_synthetic_observed_d13c_warmer_less_negative():
    """Warmer archetypes discriminate less (lower Ci/Ca) -> a LESS NEGATIVE synthetic
    delta13C, so the sign of the climate gradient is physical."""
    obs = synthetic_observed_d13c(_table([7, 7], [280.0, 305.0]))
    assert obs[1] > obs[0]      # warmer -> less negative


def test_synthetic_observed_d13c_is_pathway_aware_c3_vs_c4():
    """The synthetic target is C3/C4-aware (matching the forward's two faithful pathways): C4
    PFTs (14=c4_grass, 16=crop_c4) get a C4-BAND target (~ -11..-14 permil, distinctly less
    negative), C3 PFTs stay in the C3 band, so the C4 leakiness lever fits a small residual
    (not the ~15 permil a stale C3-band target on a C4 archetype would fabricate)."""
    # broadleaf tree (C3), c3_grass (C3), c4_grass (C4), crop_c4 (C4)
    obs = synthetic_observed_d13c(_table([4, 13, 14, 16], [298.0, 288.0, 300.0, 299.0]))
    mask = c4_archetype_mask(np.array([4, 13, 14, 16]))
    assert np.all(np.isfinite(obs))
    # C4 archetypes in the C4 band, C3 archetypes in the C3 band, cleanly separated
    assert np.all(obs[mask] > -16.0) and np.all(obs[mask] < -10.0), obs
    assert np.all(obs[~mask] <= -22.0) and np.all(obs[~mask] >= -34.0), obs
    assert obs[mask].min() > obs[~mask].max()   # every C4 target less negative than every C3


def test_observed_d13c_delegates_to_shared_mean():
    """The per-archetype observed delta13C is the SHARED cover-weighted mean (no duplicated
    numerics)."""
    d13c_cell = np.array([-25.0, -30.0])
    cid = np.array([[0, -1], [1, 0]])
    cw = np.array([[0.75, 0.0], [1.0, 0.25]])
    got = per_archetype_observed_d13c(d13c_cell, cid, cw, n_arch=2)
    ref = per_archetype_cover_weighted_mean(d13c_cell, cid, cw, n_arch=2)
    npt.assert_allclose(got, ref, rtol=1e-12)
    # arch0 = (0.75*-25 + 0.25*-30)/1.0 = -26.25 ; arch1 = -30 (single member)
    npt.assert_allclose(got, [-26.25, -30.0], rtol=1e-12)


def test_load_gridded_d13c_reshape_and_mismatch(tmp_path):
    xr = pytest.importorskip("xarray")
    try:
        from legoesm.land.carbon.d13c_observations import load_gridded_d13c
        arr = np.array([[-24.0, -26.0, -28.0], [-30.0, -27.0, -25.0]])
        ds = xr.Dataset({"d13c": (("lat", "lon"), arr)})
        path = tmp_path / "d13c.nc"
        ds.to_netcdf(path)
    except Exception as exc:                       # no netcdf backend in this env
        pytest.skip(f"netcdf write unavailable: {exc}")
    out = load_gridded_d13c(str(path), ncell=6)    # 2x3 row-major flatten
    npt.assert_allclose(out, arr.reshape(-1), rtol=1e-12)
    with pytest.raises(SystemExit, match="ncell"):  # wrong ncell -> hard error
        load_gridded_d13c(str(path), ncell=5)


def test_load_gridded_d13c_preserves_nan_gaps(tmp_path):
    xr = pytest.importorskip("xarray")
    try:
        from legoesm.land.carbon.d13c_observations import load_gridded_d13c
        arr = np.array([[-24.0, np.nan], [-30.0, -26.0]])   # a sparse-product gap at (0,1)
        ds = xr.Dataset({"d13c": (("lat", "lon"), arr)})
        path = tmp_path / "d13c_gap.nc"
        ds.to_netcdf(path)
    except Exception as exc:
        pytest.skip(f"netcdf write unavailable: {exc}")
    out = load_gridded_d13c(str(path), ncell=4)
    assert np.isnan(out[1])                            # gap PRESERVED, not fabricated 0
    npt.assert_allclose(out[[0, 2, 3]], [-24.0, -30.0, -26.0], rtol=1e-12)
