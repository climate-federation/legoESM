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

    # T-point coordinates: rows 40.0..41.5N, columns 26.0..30.0E, so that
    # exactly the cells (j=1..2, i=2..4) fall inside CLOSED_SEAS["marmara"].
    gphit = np.repeat(np.array([[40.0], [40.5], [41.0], [41.5]]), nx, axis=1)
    glamt = np.repeat(np.array([[26.0, 26.5, 27.0, 28.0, 30.0]]), ny, axis=0)
    ds = xr.Dataset(
        {
            "tmaskutil": (("y", "x"), tmaskutil),
            "tmask": (("z", "y", "x"), tmask),
            "e3t_0": (("z", "y", "x"), e3t0),
            "gphit": (("y", "x"), gphit),
            "glamt": (("y", "x"), glamt),
        }
    )
    ds.to_netcdf(path, engine="scipy")  # NETCDF3 via scipy (no netcdf4 dep)
    return tmaskutil, kbot


def test_read_mesh_mask_bathy_closed_seas_become_land(tmp_path):
    import pytest
    from legoesm.ocean.init_tripole import CLOSED_SEAS, read_mesh_mask_bathy

    mesh = tmp_path / "synthetic_mesh_mask.nc"
    tmaskutil, kbot = _write_synthetic_mesh_mask(mesh)
    lm0, hb0 = read_mesh_mask_bathy(str(mesh))
    lm, hb = read_mesh_mask_bathy(str(mesh), closed_seas=("marmara",))
    inside = np.zeros_like(tmaskutil, dtype=bool)
    inside[1:3, 2:5] = True                       # 40.5-41.0N x 27-30E
    assert CLOSED_SEAS["marmara"] == (40.3, 41.1, 26.9, 30.0)
    assert np.all(lm[inside] == 0.0) and np.all(hb[inside] == 0.0)
    np.testing.assert_array_equal(lm[~inside], lm0[~inside])
    np.testing.assert_array_equal(hb[~inside], hb0[~inside])
    # the wet count drops by exactly the wet cells inside the box (one of the
    # six is already land: (2,3)); with the knob off nothing changes
    assert int((lm0 > 0.5).sum()) - int((lm > 0.5).sum()) == 5
    np.testing.assert_array_equal(read_mesh_mask_bathy(str(mesh), closed_seas=())[0], lm0)
    with pytest.raises(ValueError, match="unknown closed_seas"):
        read_mesh_mask_bathy(str(mesh), closed_seas=("caspian",))


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


# ---------------------------------------------------------------------------
# mask_to_nemo_domain: the wet domain follows the oracle's domain_cfg
# ---------------------------------------------------------------------------

def _write_mesh_with_halo(path, ny=6, nx=8, nz=2):
    """Mesh frame = inner domain plus one cyclic halo column each side and a
    north-fold row on top (eORCA1's (332, 362) around NEMO's (331, 360)).
    Coordinates are a plain lat/lon ladder so the inner frame is identifiable
    by coordinates alone. One inner cell (a 'lake') is wet here."""
    import xarray as xr

    lat = np.repeat(np.arange(ny, dtype=np.float64)[:, None] * 10.0 - 30.0, nx, 1)
    lon = np.repeat(np.arange(nx, dtype=np.float64)[None, :] * 40.0 - 40.0, ny, 0)
    lon = lon % 360.0
    tmaskutil = np.ones((ny, nx), dtype=np.float64)
    tmaskutil[:, 0] = 0.0
    tmaskutil[:, -1] = 0.0          # raw halo columns are land (the seam wall)
    tmaskutil[-1, :] = 0.0          # dead north-fold row
    tmaskutil[2, 2] = 0.0           # an ordinary land cell
    tmask = np.repeat(tmaskutil[None], nz, 0)
    e3t0 = np.full((nz, ny, nx), 100.0)
    xr.Dataset({
        "tmaskutil": (("y", "x"), tmaskutil),
        "tmask": (("z", "y", "x"), tmask),
        "e3t_0": (("z", "y", "x"), e3t0),
        "gphit": (("y", "x"), lat),
        "glamt": (("y", "x"), lon),
    }).to_netcdf(path, engine="scipy")
    return tmaskutil, lat, lon


def _write_domain_cfg(path, lat, lon, top_level):
    import xarray as xr

    xr.Dataset({
        "top_level": (("y", "x"), top_level.astype(np.int32)),
        "gphit": (("y", "x"), lat),
        "glamt": (("y", "x"), lon),
    }).to_netcdf(path, engine="scipy")


def test_mask_to_nemo_domain_lands_the_lake_and_only_the_lake(tmp_path):
    from legoesm.ocean.init_tripole import mask_to_nemo_domain, read_mesh_mask_bathy

    mesh = tmp_path / "mesh_mask.nc"
    tmaskutil, lat, lon = _write_mesh_with_halo(mesh)
    ny, nx = tmaskutil.shape
    inner = (slice(0, ny - 1), slice(1, nx - 1))   # drop fold row + halo cols
    top = tmaskutil[inner].copy()
    top[3, 4] = 0.0                                # NEMO runs this wet cell DRY
    top[2, 1] = 1.0                                # NEMO wet where the mesh is dry
    dc = tmp_path / "domain_cfg.nc"
    _write_domain_cfg(dc, lat[inner], lon[inner], top)

    out = mask_to_nemo_domain(tmaskutil, lat, lon, str(dc))
    diff = np.argwhere(out != tmaskutil)
    assert diff.tolist() == [[3, 5]]               # inner (3,4) -> mesh (3,5)
    assert out[3, 5] == 0.0
    assert out[2, 2] == 0.0                        # never wetted: no bathymetry
    assert out.shape == tmaskutil.shape

    # Through the loader: the lake column is land AND has zero depth; without
    # the keyword it is wet with the analytic 200 m depth (non-vacuity).
    m0, h0 = read_mesh_mask_bathy(str(mesh))
    assert m0[3, 5] == 1.0 and h0[3, 5] == 200.0
    m1, h1 = read_mesh_mask_bathy(str(mesh), nemo_domain_cfg=str(dc))
    assert m1[3, 5] == 0.0 and h1[3, 5] == 0.0
    assert np.array_equal(np.argwhere(m1 != m0), [[3, 5]])


def test_mask_to_nemo_domain_refuses_a_foreign_domain_cfg(tmp_path):
    from legoesm.ocean.init_tripole import mask_to_nemo_domain

    mesh = tmp_path / "mesh_mask.nc"
    tmaskutil, lat, lon = _write_mesh_with_halo(mesh)
    ny, nx = tmaskutil.shape
    inner = (slice(0, ny - 1), slice(1, nx - 1))
    dc = tmp_path / "domain_cfg.nc"
    # Same shape, coordinates shifted by half a cell: no offset can match.
    _write_domain_cfg(dc, lat[inner] + 5.0, lon[inner], tmaskutil[inner])
    with pytest.raises(ValueError, match="not the domain_cfg of this mesh"):
        mask_to_nemo_domain(tmaskutil, lat, lon, str(dc))


def test_mask_to_nemo_domain_finds_another_offset_across_a_360_wrap(tmp_path):
    """Codex: one tested offset with copied coordinates would pass a hardcoded
    (0, 1) or a plain (unwrapped) longitude difference. Here the domain frame
    sits at offset (1, 2), its longitudes are written 360 deg lower, and the
    lake maps through that offset."""
    from legoesm.ocean.init_tripole import mask_to_nemo_domain

    mesh = tmp_path / "mesh_mask.nc"
    tmaskutil, lat, lon = _write_mesh_with_halo(mesh, ny=7, nx=9)
    ny, nx = tmaskutil.shape
    inner = (slice(1, ny - 1), slice(2, nx - 1))
    top = tmaskutil[inner].copy()
    top[2, 3] = 0.0                                # -> mesh (3, 5)
    dc = tmp_path / "domain_cfg.nc"
    _write_domain_cfg(dc, lat[inner], lon[inner] - 360.0, top)
    out = mask_to_nemo_domain(tmaskutil, lat, lon, str(dc))
    assert np.argwhere(out != tmaskutil).tolist() == [[3, 5]]
