"""Unit test for apply_balanced_init (promoted from run_omip_core2 into the
ocean package so the coupled 3D-ocean driver can reuse the OMIP-validated
geostrophic cold-start balance).

The balanced init seeds u/v/eta in geostrophic / level-of-no-motion balance
with the WOA-like baroclinic pressure field so the rest-velocity cold start
does not suffer a violent geostrophic-adjustment shock.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.init_latlon_cgrid import (
    rest_state_latlon_cgrid_ocean,
    apply_balanced_init,
)


def _baroclinic_state(grid, z_coord):
    """All-ocean rest state with a lat-dependent (baroclinic) T field so the
    balanced init has a non-trivial pressure gradient to balance."""
    n_lat, n_lon = grid.n_lat, grid.n_lon
    nlev = z_coord.n_levels
    land = jnp.ones((n_lat, n_lon))
    H = jnp.full((n_lat, n_lon), 5500.0)
    st = rest_state_latlon_cgrid_ocean(
        grid, z_coord, land_mask_override=land, H_bathy_override=H,
    )
    # Warm tropics -> cold poles, decaying with depth: a realistic baroclinic
    # structure (surface-intensified meridional density gradient).
    lat = np.asarray(grid.lat2d)              # (n_lat, n_lon), radians
    zf = np.asarray(z_coord.z_full_ref)       # (nlev,), negative
    T_surf = 2.0 + 26.0 * np.cos(lat) ** 2    # 28 eq -> 2 pole
    decay = np.exp(zf / 500.0)                # surface-intensified
    T = (2.0 + (T_surf[..., None] - 2.0) * decay[None, None, :]).astype(np.float64)
    S = np.full((n_lat, n_lon, nlev), 35.0)
    return st._replace(
        T=st.T.replace(data=jnp.asarray(T)),
        S=st.S.replace(data=jnp.asarray(S)),
    )


def test_balanced_init_runs_and_bounds():
    grid = create_latlon_grid(n_lat=24, n_lon=48)
    z_coord = create_ocean_z_star(10, H_max=5500.0)
    cfg = LatLonCGridOceanConfig.from_flat()
    st = _baroclinic_state(grid, z_coord)

    out = apply_balanced_init(st, grid, z_coord, cfg, max_speed=2.5)

    u = np.asarray(out.u.data)
    v = np.asarray(out.v.data)
    eta = np.asarray(out.eta.data)
    # Finite everywhere.
    assert np.isfinite(u).all() and np.isfinite(v).all() and np.isfinite(eta).all()
    # Velocity respects the max_speed clip; SSH within the physical bound (5 m).
    assert np.nanmax(np.abs(u)) <= 2.5 + 1e-6
    assert np.nanmax(np.abs(v)) <= 2.5 + 1e-6
    assert np.nanmax(np.abs(eta)) <= 5.0 + 1e-6
    # Non-trivial: a baroclinic field must induce SOME geostrophic flow.
    assert np.nanmax(np.abs(u)) > 0.0 or np.nanmax(np.abs(v)) > 0.0
    # T / S unchanged (pure velocity/SSH IC change).
    np.testing.assert_array_equal(np.asarray(out.T.data), np.asarray(st.T.data))
    np.testing.assert_array_equal(np.asarray(out.S.data), np.asarray(st.S.data))


def test_balanced_init_no_ssh_option():
    grid = create_latlon_grid(n_lat=24, n_lon=48)
    z_coord = create_ocean_z_star(10, H_max=5500.0)
    cfg = LatLonCGridOceanConfig.from_flat()
    st = _baroclinic_state(grid, z_coord)
    # eta left at the rest value (0) when with_ssh=False.
    out = apply_balanced_init(st, grid, z_coord, cfg, with_ssh=False)
    np.testing.assert_array_equal(np.asarray(out.eta.data),
                                  np.asarray(st.eta.data))
    assert np.isfinite(np.asarray(out.u.data)).all()
