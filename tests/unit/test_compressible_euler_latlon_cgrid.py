"""Unit tests for the lat-lon C-grid non-hydrostatic compressible Euler dycore.

These are v0 smoke / shape / no-NaN tests for the first-cut
implementation in
``src/legoesm/atmosphere/dynamics/compressible_euler_latlon_cgrid.py``.
Full benchmark tests (rising bubble, mountain wave, baroclinic wave)
are wired into the atmosphere test matrix in a follow-up once the
:func:`/codex:adversarial-review` findings on the dycore are addressed.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


# ----------------------------------------------------------------------
# Fixtures: small lat-lon grid + height-coord + state
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_setup():
    """Build a 12x24 lat-lon grid with 8 vertical levels at z up to 30 km."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import (
        create_height_coordinate, compute_terrain_metric,
    )

    grid = create_latlon_grid(n_lat=12, n_lon=24)
    height = create_height_coordinate(n_levels=8, H=30000.0)
    z_s = jnp.zeros((grid.n_lat, grid.n_lon))   # flat terrain
    terrain = compute_terrain_metric(z_s, height)
    return grid, height, terrain


@pytest.fixture(scope="module")
def rest_state(small_setup):
    """Rest state: u = v = w = theta' = rho' = 0."""
    from legoesm.atmosphere.dynamics.compressible_euler_latlon_cgrid import (
        CGridLatLonNonHydrostaticState,
    )
    grid, height, _ = small_setup
    n_lat, n_lon, nlev = grid.n_lat, grid.n_lon, height.n_levels
    zero_uface = jnp.zeros((n_lat, n_lon + 1, nlev))
    zero_vface = jnp.zeros((n_lat + 1, n_lon, nlev))
    zero_cell = jnp.zeros((n_lat, n_lon, nlev))
    zero_w = jnp.zeros((n_lat, n_lon, nlev + 1))
    zero_2d = jnp.zeros((n_lat, n_lon))
    return CGridLatLonNonHydrostaticState(
        u=zero_uface, v=zero_vface, w=zero_w,
        theta_prime=zero_cell, rho_prime=zero_cell,
        phis=zero_2d, tracers={},
    )


# ----------------------------------------------------------------------
# Shape / no-NaN
# ----------------------------------------------------------------------

def test_slow_tendencies_shapes_at_rest(small_setup, rest_state):
    from legoesm.atmosphere.dynamics.compressible_euler_latlon_cgrid import (
        cgrid_latlon_nh_slow_tendencies,
    )
    grid, height, terrain = small_setup
    tend = cgrid_latlon_nh_slow_tendencies(
        rest_state, grid, height, terrain,
    )
    assert tend.du_dt.data.shape == rest_state.u.shape
    assert tend.dv_dt.data.shape == rest_state.v.shape
    assert tend.dw_dt.data.shape == rest_state.w.shape
    assert tend.dtheta_prime_dt.data.shape == rest_state.theta_prime.shape
    assert tend.drho_prime_dt.data.shape == rest_state.rho_prime.shape


def test_slow_tendencies_finite_at_rest(small_setup, rest_state):
    """Slow tendencies must be finite at the rest state."""
    from legoesm.atmosphere.dynamics.compressible_euler_latlon_cgrid import (
        cgrid_latlon_nh_slow_tendencies,
    )
    grid, height, terrain = small_setup
    tend = cgrid_latlon_nh_slow_tendencies(
        rest_state, grid, height, terrain,
    )
    for name in ("du_dt", "dv_dt", "dw_dt", "dtheta_prime_dt", "drho_prime_dt"):
        arr = getattr(tend, name).data
        assert jnp.all(jnp.isfinite(arr)), f"{name} not all finite"


def test_pole_wall_bc_on_dv(small_setup, rest_state):
    """v tendency must vanish at the polar lat-faces."""
    from legoesm.atmosphere.dynamics.compressible_euler_latlon_cgrid import (
        cgrid_latlon_nh_slow_tendencies,
    )
    grid, height, terrain = small_setup
    tend = cgrid_latlon_nh_slow_tendencies(
        rest_state, grid, height, terrain,
    )
    assert jnp.allclose(tend.dv_dt.data[0], 0.0)
    assert jnp.allclose(tend.dv_dt.data[-1], 0.0)


def test_rest_state_remains_at_rest_under_one_step(small_setup, rest_state):
    """One outer step from the rest state must stay (approximately) at rest.

    The reference state ``rho_ref``, ``theta_ref`` is hydrostatically
    balanced by construction, so a rest state with zero perturbations
    is an exact equilibrium of the continuous equations.  The
    discrete dycore should preserve that equilibrium to floating-
    point tolerance.
    """
    from legoesm.atmosphere.dynamics.compressible_euler_latlon_cgrid import (
        cgrid_latlon_nh_step,
    )
    grid, height, terrain = small_setup
    new_state = cgrid_latlon_nh_step(
        rest_state, grid, height, terrain, dt=10.0,
    )
    for name in ("u", "v", "w", "theta_prime", "rho_prime"):
        arr = getattr(new_state, name)
        assert jnp.all(jnp.isfinite(arr)), f"{name} not finite after step"
        # Magnitudes should be tiny (round-off only).
        assert float(jnp.max(jnp.abs(arr))) < 1.0e-6, (
            f"{name} drifted from rest: max|.|={float(jnp.max(jnp.abs(arr)))}"
        )


def test_warm_bubble_drives_upward_motion(small_setup):
    """A constant-pressure warm bubble must give positive dw/dt.

    Classical Skamarock-Klemp warm-bubble sanity check: a +1 K theta'
    blob centered at mid-column, paired with a matching rho'
    perturbation that holds pressure constant (so the Exner
    perturbation vanishes and the pressure-gradient term drops out),
    should excite an upward acceleration via the residual buoyancy
    term ``g * theta'/theta_ref``.

    The constant-pressure constraint follows from
    ``p = R_d * rho * theta * (p/p_0)^kappa`` -> for fixed p we need
    ``(rho/rho_0) * (theta/theta_0) = 1``, i.e.
    ``rho'/rho_0 = -theta'/theta_0`` to first order.
    """
    from legoesm.atmosphere.dynamics.compressible_euler_latlon_cgrid import (
        CGridLatLonNonHydrostaticState, cgrid_latlon_nh_step,
    )
    grid, height, terrain = small_setup
    n_lat, n_lon, nlev = grid.n_lat, grid.n_lon, height.n_levels
    zero_uface = jnp.zeros((n_lat, n_lon + 1, nlev))
    zero_vface = jnp.zeros((n_lat + 1, n_lon, nlev))
    zero_cell = jnp.zeros((n_lat, n_lon, nlev))
    zero_w = jnp.zeros((n_lat, n_lon, nlev + 1))
    zero_2d = jnp.zeros((n_lat, n_lon))

    # Constant-pressure warm bubble at (lat_mid, lon_mid, k_mid).
    i_lat, i_lon, k = n_lat // 2, n_lon // 2, nlev // 2
    dtheta = 1.0
    theta_0_k = float(height.theta_ref[k])
    rho_0_k = float(height.rho_ref[k])
    drho = -rho_0_k * dtheta / theta_0_k

    theta_blob = zero_cell.at[i_lat, i_lon, k].set(dtheta)
    rho_blob = zero_cell.at[i_lat, i_lon, k].set(drho)

    warm_state = CGridLatLonNonHydrostaticState(
        u=zero_uface, v=zero_vface, w=zero_w,
        theta_prime=theta_blob, rho_prime=rho_blob,
        phis=zero_2d, tracers={},
    )

    new_state = cgrid_latlon_nh_step(
        warm_state, grid, height, terrain, dt=2.0,
    )
    # The half-level just above (k_mid -> k_mid+1) should be lifted.
    w_above = float(new_state.w[i_lat, i_lon, k + 1])
    assert math.isfinite(w_above)
    assert w_above > 0.0, f"warm bubble did not lift: w_above={w_above}"
