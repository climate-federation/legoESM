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

# --- Slow-tendency application + mass-conservation regressions -------
# Uniform zonal wind for the slow-advection application check [m/s].
SLOW_ADV_U_M_S = 10.0
# Outer step used by the bug-fix regression tests [s]; well inside both
# the acoustic (dt/6 substeps on 3.75 km layers) and advective CFL.
REGRESSION_DT_S = 2.0
# Interior half-level w seeded inside the default 10 km top sponge [m/s].
SPONGE_TEST_W_M_S = 0.1
# Divergent-wind mass-conservation test parameters.
MASS_TEST_U_M_S = 10.0
MASS_TEST_RHO_FRAC = 0.02      # rho' patch amplitude as fraction of rho_ref
MASS_TEST_N_STEPS = 10
MASS_TEST_REL_TOL = 1.0e-12
# Threshold for "the compression term actually acted" at a column far
# from the seeded rho' patch (analytic scale: rho_ref*div(v)*t ~ 3e-5).
MASS_TEST_FAR_RHO_MIN = 1.0e-8
# Terrain-metric mass test: half-amplitude of the zonal surface ridge
# [m] (z_s in [0, 3000] on the 30 km column -> J in [0.9, 1.0]).
TERRAIN_RIDGE_HALF_M = 1500.0
TERRAIN_TEST_N_STEPS = 5


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
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_latlon_cgrid import (
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
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_latlon_cgrid import (
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
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_latlon_cgrid import (
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
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_latlon_cgrid import (
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
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_latlon_cgrid import (
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
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_latlon_cgrid import (
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
    from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
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
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_latlon_cgrid import (
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
    from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
        compute_exner_perturbation,
    )
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_latlon_cgrid import (
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


# ----------------------------------------------------------------------
# Bug-fix regressions: slow tendencies must reach the prognostic state,
# and horizontal continuity must conserve dry mass (flux form).
# ----------------------------------------------------------------------


def _zero_fields(grid, height):
    """Zero C-grid field set (u, v, w, cell, 2d) for building test states."""
    n_lat, n_lon, nlev = grid.n_lat, grid.n_lon, height.n_levels
    return (
        jnp.zeros((n_lat, n_lon + 1, nlev)),
        jnp.zeros((n_lat + 1, n_lon, nlev)),
        jnp.zeros((n_lat, n_lon, nlev + 1)),
        jnp.zeros((n_lat, n_lon, nlev)),
        jnp.zeros((n_lat, n_lon)),
    )


def _divergent_zonal_wind(grid, nlev):
    """Divergent lon-periodic zonal wind on u faces: u = U0 sin(lon_face).

    ``sin(2*pi)`` is not bitwise equal to ``sin(0)``, so the terminal
    face is copied from face 0 explicitly -- the telescoped lon flux
    then cancels EXACTLY rather than to ~1e-15, and the test measures
    the scheme, not the seam.
    """
    lon_face = jnp.arange(grid.n_lon + 1) * grid.dlon
    u = (
        MASS_TEST_U_M_S * jnp.sin(lon_face)[None, :, None]
        * jnp.ones((grid.n_lat, grid.n_lon + 1, nlev))
    )
    return u.at[:, -1].set(u[:, 0])


def _dry_mass(s, grid, height, terrain):
    """Global dry mass sum((rho_ref + rho') * J * dz * area) [kg]."""
    rho_total = height.rho_ref[None, None, :] + s.rho_prime
    return float(jnp.sum(
        rho_total
        * terrain.jacobian[..., None]
        * height.dz[None, None, :]
        * grid.area[:, :, None]
    ))


def test_step_applies_slow_theta_advection(small_setup):
    """Slow theta' advection must reach the prognostic state.

    Regression for the v1 defect where ``cgrid_latlon_nh_step`` advanced
    only u and v by the slow tendency before the acoustic loop (which by
    contract ignores ``slow_tend``), silently discarding the slow w /
    theta' / rho' tendencies: stepping a theta' blob with a uniform
    zonal wind produced BIT-IDENTICAL theta' to stepping it with u = 0.
    With the fix, the advective displacement ``dt * dtheta'/dt|_slow``
    shows up in the stepped state (plus an O(few %) acoustic-response
    difference, hence the 0.25 safety factor on the expected scale).
    """
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_latlon_cgrid import (
        CGridLatLonNonHydrostaticState,
        cgrid_latlon_nh_slow_tendencies,
        cgrid_latlon_nh_step,
    )
    grid, height, terrain = small_setup
    n_lat, n_lon, nlev = grid.n_lat, grid.n_lon, height.n_levels
    zero_u, zero_v, zero_w, zero_cell, zero_2d = _zero_fields(grid, height)

    i_lat, i_lon, k = n_lat // 2, n_lon // 2, nlev // 2
    theta_blob = zero_cell.at[i_lat, i_lon, k].set(WARM_BUBBLE_DTHETA_K)

    state_u0 = CGridLatLonNonHydrostaticState(
        u=zero_u, v=zero_v, w=zero_w,
        theta_prime=theta_blob, rho_prime=zero_cell,
        phis=zero_2d, tracers={},
    )
    state_adv = state_u0._replace(u=jnp.full_like(zero_u, SLOW_ADV_U_M_S))

    stepped_adv = cgrid_latlon_nh_step(
        state_adv, grid, height, terrain, dt=REGRESSION_DT_S,
    )
    stepped_u0 = cgrid_latlon_nh_step(
        state_u0, grid, height, terrain, dt=REGRESSION_DT_S,
    )

    diff = float(jnp.max(jnp.abs(
        stepped_adv.theta_prime - stepped_u0.theta_prime
    )))
    slow_adv = cgrid_latlon_nh_slow_tendencies(
        state_adv, grid, height, terrain,
    )
    expected = REGRESSION_DT_S * float(
        jnp.max(jnp.abs(slow_adv.dtheta_prime_dt.data))
    )
    assert expected > 0.0, "advective slow tendency vanished unexpectedly"
    assert diff > 0.25 * expected, (
        f"slow theta' advection not applied by the step: "
        f"max|theta'(u={SLOW_ADV_U_M_S}) - theta'(u=0)|={diff:.3e} K, "
        f"expected O(dt*tend)={expected:.3e} K"
    )


def test_step_applies_w_sponge(small_setup):
    """The Rayleigh sponge on w must damp w inside the top sponge layer.

    Horizontally-uniform w seeded at half-level k=1 (z_half[1] = 26.25 km,
    inside the default 10 km top sponge on the 30 km column) with
    u = v = theta' = rho' = 0: the slow dw/dt reduces to the sponge term
    ``-sponge(z) * w`` alone (advection and hyperdiff vanish).
    Controlled comparison -- the ONLY variable is ``sponge_coeff`` (on
    vs 0); pre-fix both runs were bit-identical because the slow dw/dt
    was discarded by the outer step.
    """
    from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
        CompressibleEulerConfig,
    )
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_latlon_cgrid import (
        CGridLatLonCompressibleEulerConfig,
        CGridLatLonNonHydrostaticState,
        cgrid_latlon_nh_step,
    )
    grid, height, terrain = small_setup
    zero_u, zero_v, zero_w, zero_cell, zero_2d = _zero_fields(grid, height)

    # Rigid lid: only interior half-levels may carry w.
    w_seed = zero_w.at[:, :, 1].set(SPONGE_TEST_W_M_S)
    state = CGridLatLonNonHydrostaticState(
        u=zero_u, v=zero_v, w=w_seed,
        theta_prime=zero_cell, rho_prime=zero_cell,
        phis=zero_2d, tracers={},
    )
    cfg_on = CGridLatLonCompressibleEulerConfig()          # default sponge
    cfg_off = CGridLatLonCompressibleEulerConfig(
        euler=CompressibleEulerConfig(sponge_coeff=0.0),
    )

    stepped_on = cgrid_latlon_nh_step(
        state, grid, height, terrain, dt=REGRESSION_DT_S, config=cfg_on,
    )
    stepped_off = cgrid_latlon_nh_step(
        state, grid, height, terrain, dt=REGRESSION_DT_S, config=cfg_off,
    )

    w_on = stepped_on.w[:, :, 1]
    w_off = stepped_off.w[:, :, 1]
    assert not bool(jnp.allclose(w_on, w_off)), (
        "sponge had no effect on w: slow dw/dt is being discarded"
    )
    # Rayleigh damping (-sponge*w) must REDUCE |w| relative to no-sponge.
    assert float(jnp.max(jnp.abs(w_on))) < float(jnp.max(jnp.abs(w_off))), (
        f"sponge did not damp w: max|w_on|={float(jnp.max(jnp.abs(w_on))):.4e}"
        f" >= max|w_off|={float(jnp.max(jnp.abs(w_off))):.4e}"
    )


def test_dry_mass_conserved_under_divergent_wind(small_setup):
    """Global dry mass must be conserved under divergent horizontal flow.

    Budget: M = sum((rho_ref + rho') * J * dz * area).  The horizontal
    continuity leg is conservative flux form on TOTAL density
    (-div_h(rho_total v_h)), which telescopes exactly (periodic lon,
    v = 0 pole walls), and the acoustic vertical leg telescopes per
    column (rho_total*w flux, w = 0 at the rigid lid/bottom), so
    in - out - dStorage = 0 and M drifts only at round-off.

    The v1 ADVECTIVE form (-v . grad rho') violated this: its global
    integral contains sum(rho' * div v) != 0 under divergent flow, a
    measured leak of ~3e-8 relative over these 10 steps (and it dropped
    the -rho_total*div(v) compression physics entirely -- checked below
    via a column far from the seeded rho' patch, which only the
    compression term can reach on this timescale).
    """
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_latlon_cgrid import (
        CGridLatLonNonHydrostaticState,
        cgrid_latlon_nh_step,
    )
    grid, height, terrain = small_setup
    n_lat, n_lon, nlev = grid.n_lat, grid.n_lon, height.n_levels
    _, zero_v, zero_w, zero_cell, zero_2d = _zero_fields(grid, height)

    u_div = _divergent_zonal_wind(grid, nlev)
    # rho' patch (fraction of rho_ref so the perturbation is mild at
    # every level) centred on the grid -- gives the advective form a
    # nonzero sum(rho' * div v) leak to regress against.
    i_lat, i_lon = n_lat // 2, n_lon // 2
    rho_patch = zero_cell.at[
        i_lat - 1:i_lat + 2, i_lon - 1:i_lon + 2, :
    ].set(MASS_TEST_RHO_FRAC * height.rho_ref[None, None, :])

    state = CGridLatLonNonHydrostaticState(
        u=u_div, v=zero_v, w=zero_w,
        theta_prime=zero_cell, rho_prime=rho_patch,
        phis=zero_2d, tracers={},
    )

    # Dry-mass functional: rho_ref is static, so all drift lives in rho'.
    m0 = _dry_mass(state, grid, height, terrain)
    s = state
    for _ in range(MASS_TEST_N_STEPS):
        s = cgrid_latlon_nh_step(
            s, grid, height, terrain, dt=REGRESSION_DT_S,
        )
        assert bool(jnp.all(jnp.isfinite(s.rho_prime)))
    m1 = _dry_mass(s, grid, height, terrain)

    rel_drift = abs(m1 - m0) / abs(m0)
    assert rel_drift < MASS_TEST_REL_TOL, (
        f"dry mass drifted under divergent wind: "
        f"|dM|/M = {rel_drift:.3e} (tol {MASS_TEST_REL_TOL:.1e})"
    )

    # Compression term must actually act: a column far from the rho'
    # patch (blob advection ~ U0 * t ~ 200 m << cell) develops rho'
    # from -rho_ref * div(v) alone.  Pre-fix this column stayed
    # EXACTLY zero (slow drho'/dt discarded; acoustic loop is columnar).
    far_lat, far_lon = n_lat // 4, 0
    far_rho = float(jnp.max(jnp.abs(s.rho_prime[far_lat, far_lon, :])))
    assert far_rho > MASS_TEST_FAR_RHO_MIN, (
        f"horizontal compression term missing: |rho'| at far column = "
        f"{far_rho:.3e} (threshold {MASS_TEST_FAR_RHO_MIN:.1e})"
    )


def test_dry_mass_conserved_over_terrain(small_setup):
    """Dry mass must be conserved with a NON-flat terrain metric (J != 1).

    The horizontal continuity leg carries the metric Jacobian inside the
    flux, ``-(1/J) div_h(J rho_total v_h)``: the J-weighted budget
    ``M = sum(rho_total * J * dz * area)`` then telescopes for ANY J.
    Without the J inside the flux (the round-1 form
    ``-div_h(rho_total v_h)``), the J-weighted integral does not
    telescope and mass leaks immediately over sloped terrain (measured
    8.4e-8 relative over these 5 steps, vs 9.6e-16 with the fix).

    Scope: this asserts the MASS budget only.  The rest of the v1 slow
    dynamics (PGF metric term, surface kinematic w BC) is still
    flat-terrain physics -- documented at the flux call site.
    """
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_latlon_cgrid import (
        CGridLatLonNonHydrostaticState,
        cgrid_latlon_nh_step,
    )
    from legoesm.grids.vertical import compute_terrain_metric
    grid, height, _flat = small_setup
    nlev = height.n_levels
    _, zero_v, zero_w, zero_cell, zero_2d = _zero_fields(grid, height)

    # Zonal surface ridge: z_s in [0, 2*TERRAIN_RIDGE_HALF_M] -> J in
    # [0.9, 1.0] on the 30 km column.
    z_s = TERRAIN_RIDGE_HALF_M * (1.0 + jnp.sin(grid.lon2d))
    terrain = compute_terrain_metric(z_s, height)

    state = CGridLatLonNonHydrostaticState(
        u=_divergent_zonal_wind(grid, nlev), v=zero_v, w=zero_w,
        theta_prime=zero_cell, rho_prime=zero_cell,
        phis=zero_2d, tracers={},
    )

    m0 = _dry_mass(state, grid, height, terrain)
    s = state
    for _ in range(TERRAIN_TEST_N_STEPS):
        s = cgrid_latlon_nh_step(
            s, grid, height, terrain, dt=REGRESSION_DT_S,
        )
        assert bool(jnp.all(jnp.isfinite(s.rho_prime)))
    m1 = _dry_mass(s, grid, height, terrain)

    rel_drift = abs(m1 - m0) / abs(m0)
    assert rel_drift < MASS_TEST_REL_TOL, (
        f"dry mass drifted over sloped terrain metric: "
        f"|dM|/M = {rel_drift:.3e} (tol {MASS_TEST_REL_TOL:.1e})"
    )


def test_sponge_profile_shape_forwarded(small_setup):
    """``sponge_profile_shape`` must reach the sponge, and typos must raise.

    The shared config exposes ``sponge_profile_shape`` ("sin2" |
    "sam_rational"); the v1 slow-tendency call omitted the ``shape``
    argument, silently pinning this dycore to "sin2".  With w seeded in
    the sponge layer the two tapers give measurably different slow
    dw/dt (sam_rational ramps to ~0.98*coeff at z_half[1] vs ~0.69 for
    sin2).  Unknown shapes must raise (dispatch hardening).
    """
    from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
        CompressibleEulerConfig,
    )
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_latlon_cgrid import (
        CGridLatLonCompressibleEulerConfig,
        CGridLatLonNonHydrostaticState,
        cgrid_latlon_nh_slow_tendencies,
    )
    grid, height, terrain = small_setup
    zero_u, zero_v, zero_w, zero_cell, zero_2d = _zero_fields(grid, height)
    state = CGridLatLonNonHydrostaticState(
        u=zero_u, v=zero_v, w=zero_w.at[:, :, 1].set(SPONGE_TEST_W_M_S),
        theta_prime=zero_cell, rho_prime=zero_cell,
        phis=zero_2d, tracers={},
    )

    def dw(shape):
        cfg = CGridLatLonCompressibleEulerConfig(
            euler=CompressibleEulerConfig(sponge_profile_shape=shape),
        )
        return cgrid_latlon_nh_slow_tendencies(
            state, grid, height, terrain, cfg,
        ).dw_dt.data

    dw_sin2 = dw("sin2")
    dw_sam = dw("sam_rational")
    assert not bool(jnp.allclose(dw_sin2[:, :, 1], dw_sam[:, :, 1])), (
        "sponge_profile_shape is not forwarded: sin2 and sam_rational "
        "produced identical slow dw/dt in the sponge layer"
    )
    with pytest.raises(ValueError):
        dw("not_a_sponge_shape")
