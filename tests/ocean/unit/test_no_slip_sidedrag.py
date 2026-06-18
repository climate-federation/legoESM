"""Unit tests for the MITgcm-faithful no-slip lateral side-drag operator."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.grids.latlon import create_beta_plane_cgrid_geometry
from legoesm.ocean.dynamics.latlon_cgrid_operators import no_slip_sidedrag_cgrid


def _closed_box(ny, nx, dx, dy):
    """Beta-plane geometry + closed-box masks (1-cell walls, 1=ocean)."""
    g = create_beta_plane_cgrid_geometry(ny, nx, dx_m=dx, dy_m=dy, f0=1e-4, beta=1e-11)
    m = np.ones((ny, nx))
    m[0, :] = 0
    m[-1, :] = 0
    m[:, 0] = 0
    m[:, -1] = 0
    um = np.zeros((ny, nx + 1))
    for j in range(ny):
        for i in range(nx + 1):
            left = m[j, i - 1] if i - 1 >= 0 else 0
            right = m[j, i] if i < nx else 0
            um[j, i] = left * right
    vm = np.zeros((ny + 1, nx))
    for j in range(ny + 1):
        for i in range(nx):
            south = m[j - 1, i] if j - 1 >= 0 else 0
            north = m[j, i] if j < ny else 0
            vm[j, i] = south * north
    return g, jnp.asarray(m), jnp.asarray(um), jnp.asarray(vm)


def test_sidedrag_matches_mitgcm_analytic_formula():
    """-(closedN+closedS)·2·ah·u/dy² at wall-adjacent cells, 0 in the interior."""
    ny = nx = 8
    dx = dy = 20e3
    ah, u0, v0 = 400.0, 0.1, 0.05
    g, m, um, vm = _closed_box(ny, nx, dx, dy)
    u = jnp.full((ny, nx + 1), u0) * um
    v = jnp.full((ny + 1, nx), v0) * vm
    du, dv = no_slip_sidedrag_cgrid(u, v, g, ah, u_mask=um, v_mask=vm, mask=m)
    du, dv = np.asarray(du), np.asarray(dv)

    drag_u_1wall = -2 * ah * u0 / dy**2
    drag_v_1wall = -2 * ah * v0 / dx**2
    # u-cell just inside the south wall (row 1): exactly one closed (S) side.
    np.testing.assert_allclose(du[1, 3], drag_u_1wall, rtol=1e-12)
    # deep-interior u-cell: no wall -> zero drag.
    assert du[4, 4] == 0.0
    # v-cell just inside the west wall (col 1): one closed (W) side.
    np.testing.assert_allclose(dv[3, 1], drag_v_1wall, rtol=1e-12)
    # No interior cell has 2 N/S walls here, so max == 1-wall value.
    np.testing.assert_allclose(np.abs(du).max(), abs(drag_u_1wall), rtol=1e-12)


def test_rectangular_grid_uses_dy_for_u_and_dx_for_v():
    """u-drag scales with 1/dy², v-drag with 1/dx² (no axis swap)."""
    ny = nx = 8
    dx, dy = 20e3, 10e3
    ah, u0, v0 = 400.0, 0.1, 0.1
    g, m, um, vm = _closed_box(ny, nx, dx, dy)
    u = jnp.full((ny, nx + 1), u0) * um
    v = jnp.full((ny + 1, nx), v0) * vm
    du, dv = no_slip_sidedrag_cgrid(u, v, g, ah, u_mask=um, v_mask=vm, mask=m)
    np.testing.assert_allclose(np.asarray(du)[1, 3], -2 * ah * u0 / dy**2, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(dv)[3, 1], -2 * ah * v0 / dx**2, rtol=1e-12)


def test_no_drag_in_open_domain():
    """No walls -> no side-drag anywhere (free-slip-equivalent)."""
    ny = nx = 6
    g = create_beta_plane_cgrid_geometry(ny, nx, dx_m=20e3, dy_m=20e3, f0=1e-4, beta=1e-11)
    m = jnp.ones((ny, nx))
    um = jnp.ones((ny, nx + 1))
    vm = jnp.ones((ny + 1, nx))
    u = jnp.full((ny, nx + 1), 0.1)
    v = jnp.full((ny + 1, nx), 0.1)
    du, dv = no_slip_sidedrag_cgrid(u, v, g, 400.0, u_mask=um, v_mask=vm, mask=m)
    # Interior vertices all wet; only the domain-edge vertices are dry (pole BC),
    # so edge faces see drag, but a fully-wet interior cell sees none.
    assert np.asarray(du)[ny // 2, nx // 2] == 0.0


def test_3d_field_broadcasts_over_levels():
    ny = nx = 8
    g, m, um, vm = _closed_box(ny, nx, 20e3, 20e3)
    u = (jnp.full((ny, nx + 1), 0.1) * um)[..., None] * jnp.ones((1, 1, 3))
    v = (jnp.full((ny + 1, nx), 0.05) * vm)[..., None] * jnp.ones((1, 1, 3))
    du, dv = no_slip_sidedrag_cgrid(u, v, g, 400.0, u_mask=um, v_mask=vm, mask=m)
    assert du.shape == u.shape and dv.shape == v.shape
    # Same drag on every level (uniform velocity).
    np.testing.assert_allclose(np.asarray(du)[..., 0], np.asarray(du)[..., 2])


def test_requires_mask_or_vertex_mask():
    g = create_beta_plane_cgrid_geometry(6, 6, dx_m=1e3, dy_m=1e3, f0=1e-4, beta=1e-11)
    u = jnp.zeros((6, 7))
    v = jnp.zeros((7, 6))
    um = jnp.ones((6, 7))
    vm = jnp.ones((7, 6))
    with pytest.raises(ValueError, match="mask or vertex_mask"):
        no_slip_sidedrag_cgrid(u, v, g, 400.0, u_mask=um, v_mask=vm)


def test_unknown_lateral_side_bc_raises_at_step():
    """Dispatch hardening: an unknown lateral_side_bc fails loudly when stepping."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity import mitgcm_barotropic_gyre_recipe as gyre

    cfg = gyre.build_gyre_config()._replace(lateral_side_bc="bogus")
    z = gyre.create_ocean_z_star(1, H_max=gyre.H_DEPTH_M)
    model = LatLonCGridOceanModel(gyre.build_gyre_geometry(), z, cfg)
    s = gyre.build_gyre_state(z)
    with pytest.raises(ValueError, match="lateral_side_bc must be one of"):
        model.step(s, gyre.DT_S, surface_forcing=gyre.build_gyre_wind())


def test_sidedrag_finite_on_spherical_grid_with_zero_length_polar_faces():
    """Regression: on a SPHERICAL grid the v-face zonal length
    ``dx_v = R·cos(lat_v)·dlon`` is 0 at the polar boundary faces; the unguarded
    ``1/dx_v²`` was inf there and ``inf·0`` (mask) → NaN, blowing up any no-slip run
    on a non-Cartesian grid (the MITgcm baroclinic-gyre oracle).  The ``where(Δ>0)``
    guard makes the drag finite (and zero on those masked faces).  Non-vacuous:
    before the guard this produced NaN."""
    from legoesm.grids.latlon import create_regional_latlon_grid, ensure_geometry
    grid, wall = create_regional_latlon_grid(
        30, 20, lat_south=15.0, lat_north=75.0, lon_west=0.0, lon_east=20.0)
    g = ensure_geometry(grid)
    assert bool((np.asarray(g.dx_v) == 0.0).any())     # polar faces ARE zero-length
    m = np.asarray(wall)
    # Face masks = product of the two adjacent cell masks (closed-basin walls).
    um = np.pad(m, ((0, 0), (1, 0))) * np.pad(m, ((0, 0), (0, 1)))
    vm = np.pad(m, ((1, 0), (0, 0))) * np.pad(m, ((0, 1), (0, 0)))
    u = jnp.ones((g.n_lat, g.n_lon + 1))
    v = jnp.ones((g.n_lat + 1, g.n_lon))
    du, dv = no_slip_sidedrag_cgrid(
        u, v, g, 5000.0, u_mask=jnp.asarray(um), v_mask=jnp.asarray(vm), mask=jnp.asarray(m))
    assert np.all(np.isfinite(np.asarray(du))) and np.all(np.isfinite(np.asarray(dv)))
