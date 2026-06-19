"""Direct unit test for ``ocean.restart.save_mld_snapshot`` + the shared
``grid_lat2d_lon2d_deg`` coordinate extractor.

The snapshot writer for the offline mixed-layer-depth scorers
(``scripts/validate/compare_mld_dbm.py`` / ``compare_omip_nemo.py``).  Asserts
the written ``.npz`` carries exactly the contract those scorers consume:
``T``/``S``/``land_mask``/``H_bathy`` + ``lat_T``/``lon_T`` (in DEGREES) + a
positive-down strictly-increasing ``z_center_ref`` of length nlev, that the
snapshot drives the canonical ``ocean.diagnostics.mixed_layer_depth`` end-to-end
(the real consumer path), and that the writer fails LOUD on a missing
consumer-required field.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.restart import (
    save_mld_snapshot, grid_lat2d_lon2d_deg,
)
from legoesm.ocean.diagnostics import mixed_layer_depth
from legoesm.ocean.vertical import create_ocean_z_star


def _state_and_geom(n_lat=8, n_lon=16, nlev=10, H_max=4000.0):
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=H_max)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=18.0, T_deep=2.0, S_uniform=35.0,
        H_max=H_max,
    )
    return grid, z_coord, state


def test_grid_lat2d_lon2d_deg_units():
    """The extractor returns 2-D DEGREES (LatLonGrid stores 1-D radian axes)
    + raises on an unknown grid_type (dispatch hardening)."""
    grid = create_latlon_grid(n_lat=8, n_lon=16)
    lat2d, lon2d = grid_lat2d_lon2d_deg(grid, "latlon")
    assert lat2d.shape == (grid.n_lat, grid.n_lon)
    assert lon2d.shape == (grid.n_lat, grid.n_lon)
    assert np.all(np.abs(lat2d) <= 90.0 + 1e-6)      # degrees, not radians
    assert np.all((lon2d >= -1e-6) & (lon2d <= 360.0 + 1e-6))
    # Matches the meshgrid of the rad->deg 1-D axes.
    lon_chk, lat_chk = np.meshgrid(np.rad2deg(np.asarray(grid.lon)),
                                   np.rad2deg(np.asarray(grid.lat)))
    assert np.allclose(lat2d, lat_chk) and np.allclose(lon2d, lon_chk)
    with pytest.raises(ValueError, match="unknown grid_type"):
        grid_lat2d_lon2d_deg(grid, "klein_bottle")


def test_save_mld_snapshot_contract(tmp_path):
    grid, z_coord, state = _state_and_geom()
    nlev = z_coord.n_levels
    shape2d = (grid.n_lat, grid.n_lon)
    out = tmp_path / "snap.npz"
    lat2d, lon2d = grid_lat2d_lon2d_deg(grid, "latlon")
    ret = save_mld_snapshot(
        state, out, z_coord=z_coord,
        lat2d=lat2d, lon2d=lon2d,
        time_s=86400.0 * 30, step=1234,
    )
    assert ret == out and out.exists()
    s = np.load(out)
    keys = set(s.files)
    # Exactly the keys the MLD / OMIP-NEMO scorers read.
    for req in ("T", "S", "land_mask", "H_bathy", "lat_T", "lon_T",
                "z_center_ref"):
        assert req in keys, f"missing {req}"
    # z_center_ref: length nlev, strictly increasing, positive-down.
    zc = np.asarray(s["z_center_ref"])
    assert zc.shape == (nlev,)
    assert np.all(zc > 0.0)
    assert np.all(np.diff(zc) > 0.0)
    # Geometry shapes line up with the field arrays.
    assert s["T"].shape[-1] == nlev
    assert s["H_bathy"].shape == shape2d
    assert s["lat_T"].shape == shape2d
    # lat_T stored in DEGREES (the scorer convention), not radians.
    assert np.all(np.abs(np.asarray(s["lat_T"])) <= 90.0 + 1e-6)
    # Provenance scalars.
    assert int(s["_step"]) == 1234


def test_snapshot_drives_mixed_layer_depth(tmp_path):
    """The written snapshot feeds the canonical MLD diagnostic exactly the way
    compare_mld_dbm._load_snapshot_mld does: wet = z_center_ref < H_bathy."""
    grid, z_coord, state = _state_and_geom()
    shape2d = (grid.n_lat, grid.n_lon)
    out = tmp_path / "snap.npz"
    lat2d, lon2d = grid_lat2d_lon2d_deg(grid, "latlon")
    save_mld_snapshot(state, out, z_coord=z_coord, lat2d=lat2d, lon2d=lon2d)
    s = np.load(out)
    T = np.asarray(s["T"]); S = np.asarray(s["S"])
    z_c = np.asarray(s["z_center_ref"], dtype=np.float64)
    Hb = np.asarray(s["H_bathy"], dtype=np.float64)
    mask2d = np.asarray(s["land_mask"])
    wet = ((z_c[(None,) * Hb.ndim + (slice(None),)] < Hb[..., None])
           & (mask2d[..., None] > 0.5)).astype(np.float64)
    mld = np.asarray(mixed_layer_depth(
        T, S, z_c, delta_sigma=0.03, wet_mask=wet, bottom_depth=Hb))
    assert mld.shape == shape2d
    # Wet columns give a finite, non-negative MLD bounded by the sea floor.
    wet_col = mask2d > 0.5
    assert np.all(np.isfinite(mld[wet_col]))
    assert np.all(mld[wet_col] >= 0.0)
    assert np.all(mld[wet_col] <= Hb[wet_col] + 1e-6)


def test_missing_H_bathy_raises(tmp_path):
    """The writer fails LOUD at write time when the consumer-required H_bathy
    is absent (beats a cryptic SystemExit later in the scorer)."""
    grid, z_coord, state = _state_and_geom()
    lat2d, lon2d = grid_lat2d_lon2d_deg(grid, "latlon")
    state_no_hb = state._replace(H_bathy=None)
    with pytest.raises(ValueError, match="H_bathy"):
        save_mld_snapshot(state_no_hb, tmp_path / "snap.npz", z_coord=z_coord,
                          lat2d=lat2d, lon2d=lon2d)
