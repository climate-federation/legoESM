"""Unit tests for the Hollingsworth KE-gradient correction added to
``ocean_pe_latlon_cgrid.py`` (issue #263).

Verifies:
- The HW form reduces to the centered form for spatially uniform u, v
  (sanity check on the formula).
- The HW form gives a different (smoother) value when there is
  meridional shear in u, matching NEMO 4.2.1 dynkeg.F90 nkeg_HW.
- The default config uses the legacy ``"centered"`` scheme so all
  existing regression tests stay bit-exact.
- An invalid scheme name raises ValueError (per project convention).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest


def _ke_centered(u, v):
    u_cell = 0.5 * (u[:, :-1, :] + u[:, 1:, :])
    v_cell = 0.5 * (v[:-1, :, :] + v[1:, :, :])
    return 0.5 * (u_cell ** 2 + v_cell ** 2)


def _ke_hollingsworth(u, v):
    """Inline mirror of the implementation under test, for verification."""
    u_l = u[:, :-1, :]
    u_r = u[:, 1:, :]
    u_l_jm1 = jnp.concatenate([u_l[:1], u_l[:-1]], axis=0)
    u_l_jp1 = jnp.concatenate([u_l[1:], u_l[-1:]], axis=0)
    u_r_jm1 = jnp.concatenate([u_r[:1], u_r[:-1]], axis=0)
    u_r_jp1 = jnp.concatenate([u_r[1:], u_r[-1:]], axis=0)
    v_s = v[:-1, :, :]
    v_n = v[1:, :, :]
    v_s_im1 = jnp.roll(v_s, shift=+1, axis=1)
    v_s_ip1 = jnp.roll(v_s, shift=-1, axis=1)
    v_n_im1 = jnp.roll(v_n, shift=+1, axis=1)
    v_n_ip1 = jnp.roll(v_n, shift=-1, axis=1)
    zu = 8.0 * (u_l ** 2 + u_r ** 2) \
         + (u_l_jm1 + u_l_jp1) ** 2 \
         + (u_r_jm1 + u_r_jp1) ** 2
    zv = 8.0 * (v_s ** 2 + v_n ** 2) \
         + (v_s_im1 + v_s_ip1) ** 2 \
         + (v_n_im1 + v_n_ip1) ** 2
    return (zu + zv) / 48.0


# ---------- formula consistency ----------

def test_HW_reduces_to_half_u2_for_uniform_u():
    """For uniform u = U, v = 0: K should be 0.5*U² regardless of scheme."""
    n_lat, n_lon, nlev = 8, 10, 1
    U = 2.5
    u = jnp.full((n_lat, n_lon + 1, nlev), U)
    v = jnp.zeros((n_lat + 1, n_lon, nlev))
    K = _ke_hollingsworth(u, v)
    assert bool(jnp.allclose(K, 0.5 * U ** 2))


def test_HW_reduces_to_half_v2_for_uniform_v():
    n_lat, n_lon, nlev = 8, 10, 1
    V = 1.7
    u = jnp.zeros((n_lat, n_lon + 1, nlev))
    v = jnp.full((n_lat + 1, n_lon, nlev), V)
    K = _ke_hollingsworth(u, v)
    assert bool(jnp.allclose(K, 0.5 * V ** 2))


def test_HW_matches_centered_for_uniform_flow():
    """When u and v are both uniform, HW and centered give same K."""
    n_lat, n_lon, nlev = 8, 10, 1
    u = jnp.full((n_lat, n_lon + 1, nlev), 2.0)
    v = jnp.full((n_lat + 1, n_lon, nlev), -1.0)
    K_c = _ke_centered(u, v)
    K_h = _ke_hollingsworth(u, v)
    assert bool(jnp.allclose(K_c, K_h, atol=1e-12))


def test_HW_differs_from_centered_for_meridional_shear():
    """For u with j-direction shear, HW (3-row stencil) gives different
    K than centered (1-row). Both still positive."""
    n_lat, n_lon, nlev = 8, 10, 1
    j = jnp.arange(n_lat)
    u_profile = (j - n_lat / 2) * 0.5  # linear shear in j
    u = jnp.broadcast_to(u_profile[:, None, None], (n_lat, n_lon + 1, nlev))
    v = jnp.zeros((n_lat + 1, n_lon, nlev))
    K_c = _ke_centered(u, v)
    K_h = _ke_hollingsworth(u, v)
    assert not bool(jnp.allclose(K_c, K_h))
    assert bool(jnp.all(K_c >= 0))
    assert bool(jnp.all(K_h >= 0))


def _ke_c2(u, v):
    """Inline mirror of the NEMO nkeg_C2 mean-of-squares implementation."""
    return 0.25 * (
        u[:, :-1, :] ** 2 + u[:, 1:, :] ** 2
        + v[:-1, :, :] ** 2 + v[1:, :, :] ** 2
    )


def test_C2_reduces_to_centered_for_uniform_flow():
    """Mean-of-squares == square-of-mean when the faces are equal."""
    n_lat, n_lon, nlev = 8, 10, 1
    u = jnp.full((n_lat, n_lon + 1, nlev), 2.0)
    v = jnp.full((n_lat + 1, n_lon, nlev), -1.0)
    assert bool(jnp.allclose(_ke_c2(u, v), _ke_centered(u, v), atol=1e-12))


def test_C2_is_mean_of_squares_not_square_of_mean():
    """With zonally-varying u, C2 (mean of squares) must exceed the
    centered (square of the mean) value and match a hand computation."""
    nlev = 1
    # Single T-cell: west face u=1, east face u=3, no v.
    u = jnp.array([1.0, 3.0]).reshape(1, 2, nlev)   # (n_lat=1, n_lon+1=2)
    v = jnp.zeros((2, 1, nlev))                       # (n_lat+1=2, n_lon=1)
    # Mean-of-squares: 0.25*(1 + 9) = 2.5 ; square-of-mean: 0.5*((1+3)/2)² = 2.0
    assert bool(jnp.allclose(_ke_c2(u, v), 2.5, atol=1e-12))
    assert bool(jnp.allclose(_ke_centered(u, v), 2.0, atol=1e-12))
    # They MUST differ — this is the whole point of the C2 option.
    assert not bool(jnp.allclose(_ke_c2(u, v), _ke_centered(u, v)))


# ---------- config integration ----------

def test_config_default_is_centered():
    from legoesm.ocean.state import LatLonCGridOceanConfig
    cfg = LatLonCGridOceanConfig.from_flat()
    assert cfg.ke_gradient_scheme == "centered"


def test_config_can_select_hollingsworth():
    from legoesm.ocean.state import LatLonCGridOceanConfig
    cfg = LatLonCGridOceanConfig.from_flat()._replace(ke_gradient_scheme="hollingsworth")
    assert cfg.ke_gradient_scheme == "hollingsworth"


def test_config_can_select_c2():
    from legoesm.ocean.state import LatLonCGridOceanConfig
    cfg = LatLonCGridOceanConfig.from_flat(ke_gradient_scheme="c2")
    assert cfg.ke_gradient_scheme == "c2"


# ---------- end-to-end via model.step ----------

@pytest.fixture(scope="module")
def small_setup():
    """Build the smallest possible lat-lon C-grid model for a one-step
    sanity check that both KE schemes run without error."""
    from legoesm.grids.latlon import create_regional_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    grid, _ = create_regional_latlon_grid(
        n_lat=20, n_lon=10,
        lat_south=-30.0, lat_north=30.0,
        lon_west=0.0, lon_east=30.0,
    )
    z = create_ocean_z_star(n_levels=10, H_max=4000.0,
                            dz_surface=10.0, dz_deep=500.0)
    state = rest_state_latlon_cgrid_ocean(grid, z, H_max=4000.0)
    return grid, z, state


def test_one_step_centered_KE(small_setup):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.state import LatLonCGridOceanConfig
    grid, z, state = small_setup
    cfg = LatLonCGridOceanConfig.from_flat(ke_gradient_scheme="centered")
    model = LatLonCGridOceanModel(grid, z, cfg)
    new = model.step(state, dt=600.0)
    assert bool(jnp.all(jnp.isfinite(new.eta.data)))


def test_one_step_hollingsworth_KE(small_setup):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.state import LatLonCGridOceanConfig
    grid, z, state = small_setup
    cfg = LatLonCGridOceanConfig.from_flat(ke_gradient_scheme="hollingsworth")
    model = LatLonCGridOceanModel(grid, z, cfg)
    new = model.step(state, dt=600.0)
    assert bool(jnp.all(jnp.isfinite(new.eta.data)))


def test_one_step_c2_KE(small_setup):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.state import LatLonCGridOceanConfig
    grid, z, state = small_setup
    cfg = LatLonCGridOceanConfig.from_flat(ke_gradient_scheme="c2")
    model = LatLonCGridOceanModel(grid, z, cfg)
    new = model.step(state, dt=600.0)
    assert bool(jnp.all(jnp.isfinite(new.eta.data)))


def test_invalid_ke_scheme_raises():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.grids.latlon import create_regional_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    grid, _ = create_regional_latlon_grid(
        n_lat=10, n_lon=10, lat_south=-20.0, lat_north=20.0,
        lon_west=0.0, lon_east=30.0,
    )
    z = create_ocean_z_star(n_levels=5, H_max=2000.0,
                            dz_surface=10.0, dz_deep=500.0)
    state = rest_state_latlon_cgrid_ocean(grid, z, H_max=2000.0)
    cfg = LatLonCGridOceanConfig.from_flat(ke_gradient_scheme="bogus")
    model = LatLonCGridOceanModel(grid, z, cfg)
    with pytest.raises(ValueError, match="ke_gradient_scheme"):
        model.step(state, dt=600.0)


# ---------- production function, regular grid vs tripole fold ----------

def _rand_uv(n_lat, n_lon, nlev=2, seed=0):
    rng = np.random.default_rng(seed)
    u = jnp.asarray(rng.standard_normal((n_lat, n_lon + 1, nlev)))
    u = u.at[:, -1, :].set(u[:, 0, :])  # periodic duplicate column
    v = jnp.asarray(rng.standard_normal((n_lat + 1, n_lon, nlev)))
    return u, v


def test_production_HW_regular_grid_matches_mirror():
    from legoesm.grids.latlon import create_latlon_geometry
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        hollingsworth_kinetic_energy)
    g = create_latlon_geometry(12, 24)
    u, v = _rand_uv(g.n_lat, g.n_lon)
    np.testing.assert_allclose(hollingsworth_kinetic_energy(u, v, g),
                               _ke_hollingsworth(u, v), rtol=1e-12)


def test_production_HW_tripole_top_row_uses_signed_fold_partner():
    """Top row's j+1 u neighbour = vector_sign_u * u[top, perm_u] (NEMO
    lbc_lnk 'U',-1), not a repeat of the top row; all other rows unchanged."""
    from legoesm.grids.tripole import create_synthetic_tripole
    from legoesm.grids.operators_latlon_cgrid import (
        fold_ghost_source_T, fold_perm_u, fold_row)
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        hollingsworth_kinetic_energy)
    g = create_synthetic_tripole(12, 24)
    f = g.fold
    u, v = _rand_uv(g.n_lat, g.n_lon)
    K = hollingsworth_kinetic_energy(u, v, g)
    K_neumann = _ke_hollingsworth(u, v)
    np.testing.assert_allclose(K[:-1], K_neumann[:-1], rtol=1e-12)
    ghost = fold_row(fold_ghost_source_T(u, f), fold_perm_u(f),
                     f.vector_sign_u, f.perm_T.shape[0])[0]
    u_l, u_r = u[-1, :-1], u[-1, 1:]
    zu = (8.0 * (u_l ** 2 + u_r ** 2)
          + (u[-2, :-1] + ghost[:-1]) ** 2 + (u[-2, 1:] + ghost[1:]) ** 2)
    v_s, v_n = v[-2], v[-1]
    zv = (8.0 * (v_s ** 2 + v_n ** 2)
          + (jnp.roll(v_s, 1, 0) + jnp.roll(v_s, -1, 0)) ** 2
          + (jnp.roll(v_n, 1, 0) + jnp.roll(v_n, -1, 0)) ** 2)
    np.testing.assert_allclose(K[-1], (zu + zv) / 48.0, rtol=1e-12)
    assert not np.allclose(K[-1], K_neumann[-1])  # the fold row really changed
