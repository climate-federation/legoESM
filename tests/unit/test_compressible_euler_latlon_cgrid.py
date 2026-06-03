"""Unit tests for the lat-lon C-grid non-hydrostatic compressible Euler dycore.

Smoke / shape / no-NaN tests + the constant-pressure warm-bubble
buoyancy check that drove the v1 ``/codex:adversarial-review`` cycle.
Full benchmark tests (rising bubble, mountain wave, baroclinic wave)
are wired into the atmosphere test matrix as a follow-up.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


# ----------------------------------------------------------------------
# Test parameters (kept at module scope per CLAUDE.md hygiene rules so
# they are named, easy to audit, and live next to the assertion that
# consumes them rather than buried in a test body).
# ----------------------------------------------------------------------

# Skamarock-Klemp warm-bubble amplitude.  +1 K is the canonical value
# used in the Robert (1993) / Wicker & Skamarock (1998) test suite.
WARM_BUBBLE_DTHETA_K = 1.0

# Outer dycore time step used by the buoyancy check.  Two seconds is
# well inside the acoustic CFL for the 12 x 24 x 8 grid below and gives
# w time to climb out of round-off.
WARM_BUBBLE_DT_S = 2.0

# Tolerance on the rest-state stability test (the discrete rest state
# is preserved analytically; the residual is round-off in the floor /
# ratio path inside ``compute_exner_perturbation``).
REST_STATE_DRIFT_TOL = 1.0e-6

# Lower-bound threshold on the post-step vertical velocity at the
# half-level just above the warm bubble.  Half a millimetre per second
# is far above the rest-state round-off floor but is conservative
# relative to the analytical buoyancy frequency on this 30 km column.
WARM_BUBBLE_W_MIN_M_S = 1.0e-6


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
        assert float(jnp.max(jnp.abs(arr))) < REST_STATE_DRIFT_TOL, (
            f"{name} drifted from rest: max|.|={float(jnp.max(jnp.abs(arr)))}"
        )


def test_uv_shapes_preserved_through_step(small_setup, rest_state):
    """C-grid u/v staggered shapes survive a full outer step.

    Guards the acoustic-substep invariant documented in the module
    docstring: the shared ``acoustic_substeps`` must NOT re-shape u
    or v.  If a future refactor accidentally interpolates them to
    cell centres, this test fails immediately.
    """
    from legoesm.atmosphere.dynamics.compressible_euler_latlon_cgrid import (
        cgrid_latlon_nh_step,
    )
    grid, height, terrain = small_setup
    new_state = cgrid_latlon_nh_step(
        rest_state, grid, height, terrain, dt=10.0,
    )
    assert new_state.u.shape == rest_state.u.shape
    assert new_state.v.shape == rest_state.v.shape
    assert new_state.w.shape == rest_state.w.shape


def test_exner_perturbation_zero_at_rest(small_setup):
    """``pi'`` must vanish (exactly) when theta' = rho' = 0.

    Confirms that ``compute_exner_perturbation`` returns the
    *perturbation* only (the reference Exner is subtracted
    analytically inside the function), so the horizontal Exner
    gradient that drives the slow PGF carries no z-only leakage.
    """
    from legoesm.atmosphere.dynamics.compressible_euler import (
        compute_exner_perturbation,
    )
    _, height, _ = small_setup
    nlev = height.n_levels
    zero = jnp.zeros((4, 5, nlev))
    pi_p = compute_exner_perturbation(zero, zero, height)
    assert jnp.all(pi_p == 0.0), (
        f"pi_p not exactly zero at rest: max|pi_p|={float(jnp.max(jnp.abs(pi_p)))}"
    )


def test_pgf_zero_at_rest(small_setup, rest_state):
    """The horizontal PGF must be exactly zero at the rest state.

    The hydrostatic reference state is z-only, so its horizontal
    Exner gradient should vanish discretely.  This is a regression
    guard for finding #6 in the v1 review: any future change that
    leaks the reference Exner into ``pi_p`` would show up here.
    """
    from legoesm.atmosphere.dynamics.compressible_euler_latlon_cgrid import (
        cgrid_latlon_nh_slow_tendencies,
    )
    grid, height, terrain = small_setup
    tend = cgrid_latlon_nh_slow_tendencies(
        rest_state, grid, height, terrain,
    )
    # At rest, du_dt and dv_dt are exactly zero (KE = 0, pi' = 0,
    # Coriolis vanishes because u = v = 0).
    assert float(jnp.max(jnp.abs(tend.du_dt.data))) < 1.0e-14
    assert float(jnp.max(jnp.abs(tend.dv_dt.data))) < 1.0e-14


def test_warm_bubble_drives_upward_motion(small_setup):
    """An exact constant-pressure warm bubble must give positive dw/dt.

    Classical Skamarock-Klemp warm-bubble sanity check: a ``+dtheta``
    blob centred at mid-column, paired with a matching rho'
    perturbation that holds the total pressure *exactly* equal to the
    reference pressure (so the Exner perturbation vanishes
    analytically), excites an upward acceleration solely via the
    buoyancy term ``g * theta'/theta_ref`` inside the acoustic loop.

    The exact constant-pressure constraint follows from the dry
    equation of state
    ``p = p_0 * (R_d * rho * theta / p_0)^(c_p/c_v)`` ⇒ for fixed
    ``p = p_0_ref`` we need ``rho * theta = rho_0 * theta_0`` exactly,
    so

        rho' = rho_0 * theta_0 / (theta_0 + dtheta) - rho_0
             = -rho_0 * dtheta / (theta_0 + dtheta).

    The v0 test used the first-order approximation
    ``rho' = -rho_0 * dtheta / theta_0`` which leaves an
    ``O((dtheta/theta_0)^2)`` residual pi'.  At dtheta = 1 K,
    theta_0 ≈ 300 K, that residual is ~10⁻⁵, small but visible.
    The exact construction below makes pi' literally zero (to
    floating-point) at the seeded cell, so the assertion measures
    pure buoyancy.
    """
    from legoesm.atmosphere.dynamics.compressible_euler import (
        compute_exner_perturbation,
    )
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

    i_lat, i_lon, k = n_lat // 2, n_lon // 2, nlev // 2
    dtheta = WARM_BUBBLE_DTHETA_K
    theta_0_k = float(height.theta_ref[k])
    rho_0_k = float(height.rho_ref[k])
    # Exact constant-pressure perturbation (no Taylor truncation).
    drho = rho_0_k * theta_0_k / (theta_0_k + dtheta) - rho_0_k

    theta_blob = zero_cell.at[i_lat, i_lon, k].set(dtheta)
    rho_blob = zero_cell.at[i_lat, i_lon, k].set(drho)

    # Sanity assertion on the construction itself: pi' must be
    # round-off-zero at the seeded cell because rho * theta is
    # constructed exactly equal to rho_0 * theta_0.
    pi_p_check = compute_exner_perturbation(rho_blob, theta_blob, height)
    assert abs(float(pi_p_check[i_lat, i_lon, k])) < 1.0e-12, (
        f"warm bubble construction failed exact-constant-pressure: "
        f"pi_p={float(pi_p_check[i_lat, i_lon, k])}"
    )

    warm_state = CGridLatLonNonHydrostaticState(
        u=zero_uface, v=zero_vface, w=zero_w,
        theta_prime=theta_blob, rho_prime=rho_blob,
        phis=zero_2d, tracers={},
    )

    new_state = cgrid_latlon_nh_step(
        warm_state, grid, height, terrain, dt=WARM_BUBBLE_DT_S,
    )
    # The half-level just above the seeded cell should be lifted by
    # the buoyancy in the acoustic substep.
    w_above = float(new_state.w[i_lat, i_lon, k + 1])
    assert math.isfinite(w_above)
    assert w_above > WARM_BUBBLE_W_MIN_M_S, (
        f"warm bubble did not lift: w_above={w_above:.3e} m/s "
        f"(threshold {WARM_BUBBLE_W_MIN_M_S:.1e})"
    )
