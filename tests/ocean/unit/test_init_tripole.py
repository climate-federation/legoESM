"""Direct unit tests for the promoted NEMO tripole geometry + WOA IC loaders
(``legoesm.ocean.init_tripole``).

These functions were moved verbatim out of ``scripts/run/run_omip_core2.py`` so
the coupled-ESM driver (Phase-2 tripole coupler) can build the SAME validated
cold-start without importing from ``scripts/``.  The tests pin the contracts the
coupler relies on:

* ``read_mesh_mask_bathy``  -> land mask in {0,1}; H_bathy = sum_k e3t_0*tmask,
  > 0 on wet columns, 0 on land, shape == grid.
* ``compute_woa_3d``        -> finite T/S of shape (n_lat, n_lon, nlev), zeroed
  on land, deep-filled below the local seafloor.
* ``squeeze_nemo_field_2d`` -> drops leading singleton (t, z) dims.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_squeeze_nemo_field_2d_drops_leading_singletons():
    from legoesm.ocean.init_tripole import squeeze_nemo_field_2d

    a = np.arange(12.0).reshape(1, 1, 3, 4)  # (t, z, y, x)
    out = squeeze_nemo_field_2d(a)
    assert out.shape == (3, 4)
    assert np.array_equal(out, a[0, 0])
    # Already 2-D -> unchanged.
    b = np.ones((5, 6))
    assert squeeze_nemo_field_2d(b).shape == (5, 6)


def _write_synthetic_mesh_mask(path, ny=4, nx=5, nz=3):
    """Tiny synthetic NEMO mesh_mask: a land/ocean checkerboard column and a
    per-level e3t_0 so H_bathy is analytically known."""
    import xarray as xr

    # 2-D surface ocean mask (1 = ocean). Make row 0 all land, rest ocean,
    # plus one interior land cell to exercise the land branch.
    tmaskutil = np.ones((ny, nx), dtype=np.float64)
    tmaskutil[0, :] = 0.0
    tmaskutil[2, 3] = 0.0

    # 3-D tmask: wet down to a per-column depth level. Column (j,i) is wet for
    # k < kbot[j,i]. Land columns (tmaskutil==0) are dry at every level.
    kbot = np.array([[0, 0, 0, 0, 0],
                     [1, 2, 3, 2, 1],
                     [3, 3, 3, 0, 3],
                     [2, 2, 2, 2, 2]], dtype=int)
    tmask = np.zeros((nz, ny, nx), dtype=np.float64)
    for k in range(nz):
        tmask[k] = (k < kbot).astype(np.float64)

    # Uniform layer thickness e3t_0 = 100 m at every level so
    # H_bathy[j,i] = 100 * kbot[j,i].
    e3t0 = np.full((nz, ny, nx), 100.0, dtype=np.float64)

    ds = xr.Dataset(
        {
            "tmaskutil": (("y", "x"), tmaskutil),
            "tmask": (("z", "y", "x"), tmask),
            "e3t_0": (("z", "y", "x"), e3t0),
        }
    )
    ds.to_netcdf(path, engine="scipy")  # NETCDF3 via scipy (no netcdf4 dep)
    return tmaskutil, kbot


def test_read_mesh_mask_bathy(tmp_path):
    from legoesm.ocean.init_tripole import read_mesh_mask_bathy

    mesh = tmp_path / "synthetic_mesh_mask.nc"
    tmaskutil, kbot = _write_synthetic_mesh_mask(mesh)

    land_mask, H_bathy = read_mesh_mask_bathy(str(mesh))

    assert land_mask.shape == tmaskutil.shape == H_bathy.shape
    # Mask is exactly {0, 1}.
    assert set(np.unique(land_mask)).issubset({0.0, 1.0})
    np.testing.assert_array_equal(land_mask, tmaskutil)
    # H_bathy = 100 m * kbot (sum_k e3t_0*tmask).
    np.testing.assert_allclose(H_bathy, 100.0 * kbot)
    # > 0 on wet columns, == 0 on land columns.
    wet = land_mask > 0.5
    assert np.all(H_bathy[wet] > 0.0)
    # Land columns have kbot==0 -> H_bathy==0.
    assert np.all(H_bathy[~wet] == 0.0)


def test_compute_woa_3d_analytic_profiles_masking_and_deepfill():
    """compute_woa_3d on a small regular lat-lon grid with ANALYTIC WOA
    profiles (woa_t=woa_s=None): checks shape, finiteness, land-zeroing, and
    below-seafloor deep-fill.  (The S<1 flood-fill branch is exercised by the
    OMIP tripole runs; analytic profiles never produce S<1.)"""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_tripole import compute_woa_3d
    from legoesm import constants

    n_lat, n_lon, nlev = 8, 16, 6
    grid = create_latlon_grid(n_lat, n_lon)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=2000.0,
                                  dz_surface=10.0, dz_deep=400.0)

    # Ocean everywhere except a land strip; one shallow column to force the
    # below-seafloor deep-fill at depth.
    land_mask = np.ones((n_lat, n_lon), dtype=np.float64)
    land_mask[0, :] = 0.0
    H_bathy = np.full((n_lat, n_lon), 2000.0, dtype=np.float64)
    H_bathy[5, 7] = 50.0  # shallow column: deep levels must be deep-filled

    T, S = compute_woa_3d(grid, z_coord, None, None, H_bathy, land_mask)

    assert T.shape == S.shape == (n_lat, n_lon, nlev)
    assert np.all(np.isfinite(T)) and np.all(np.isfinite(S))
    # Land rows are zeroed (m3 mask).
    assert np.all(T[0] == 0.0) and np.all(S[0] == 0.0)
    # Ocean surface salinity is physical (analytic WOA ~ 34-36 PSU).
    ocean = land_mask > 0.5
    assert np.all(S[..., 0][ocean] > 20.0)
    # Shallow column: the deepest level sits below 50 m -> deep-fill value.
    S_fill = float(getattr(constants, "S_deep_ocean_ref_psu", 34.7))
    assert S[5, 7, -1] == pytest.approx(S_fill, abs=1e-6)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
