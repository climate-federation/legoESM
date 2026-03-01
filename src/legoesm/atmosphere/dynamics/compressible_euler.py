"""Fully compressible non-hydrostatic Euler equations on the cubed-sphere.

Solves the compressible Euler equations in height-based terrain-following
(z*) coordinates using reference-state subtraction:

    du/dt   =  (zeta+f)·v - dB/dx - c_p·theta·d(pi')/dx + D_u + sponge
    dv/dt   = -(zeta+f)·u - dB/dy - c_p·theta·d(pi')/dy + D_v + sponge
    dw/dt   = -c_p·theta·(1/J)·d(pi')/dz* - g·(theta'/theta_0) + sponge
    d(theta')/dt = -v·grad(theta) - (w/J)·d(theta)/dz* + D_theta + Q/(rho·c_p)
    d(rho')/dt   = -(1/J)·[div_h(J·rho·v_h) + d(rho·w)/dz*]

where:
    pi = (p/p_0)^kappa          Exner function (dimensionless)
    B = 0.5·(u^2 + v^2)        Bernoulli function (kinetic energy only)
    J = (H - z_s) / H          Jacobian of z* transform
    primes = perturbation from 1D reference state

Time integration uses split-explicit: SSP-RK3 for slow modes with
forward-backward acoustic substeps for fast (sound/gravity) waves.

References
----------
- Skamarock & Klemp (2008): A Time-Split Nonhydrostatic Atmospheric Model.
- Klemp et al. (2007): Terrain-Following Coordinate.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState, NonHydrostaticTendencies
from legoesm.core.operators_3d import (
    vorticity_3d,
    gradient_x_3d,
    gradient_y_3d,
    divergence_3d,
    hyperdiffusion_3d,
    vertical_gradient_half_to_full,
    vertical_advection_height,
    vertical_divergence_height,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.vertical import HeightCoordinate, TerrainMetric
from legoesm.timestepping.split_explicit import (
    split_explicit_step,
    SplitExplicitConfig,
)
from legoesm import constants


class CompressibleEulerConfig(NamedTuple):
    """Configuration for the non-hydrostatic compressible Euler model."""
    g: float = constants.g
    hyperdiff_coeff: float = 0.0
    sponge_width: float = 10000.0   # Sponge layer width from model top [m]
    sponge_coeff: float = 0.05      # Maximum Rayleigh damping rate [1/s]
    n_acoustic_substeps: int = 6
    small_earth_factor: float = 1.0
    use_conservation_fixer: bool = True
    use_coriolis: bool = True       # Set False for f=0 tests (e.g. DCMIP TC3)
    semi_implicit_acoustic: bool = False  # Use tridiagonal solve for acoustic substeps


# ==============================================================================
# Equation of state
# ==============================================================================

def compute_exner_perturbation(
    rho_prime: jax.Array,
    theta_prime: jax.Array,
    height_coord: HeightCoordinate,
) -> jax.Array:
    """Compute Exner function perturbation from density and theta perturbations.

    The full (dimensionless) Exner function is:
        pi = (R_d · rho · theta / p_0)^(R_d/c_v)

    The perturbation is pi' = pi_total - pi_0.

    Parameters
    ----------
    rho_prime : jax.Array
        Density perturbation [kg/m^3], shape (6, n, n, nlev).
    theta_prime : jax.Array
        Potential temperature perturbation [K], shape (6, n, n, nlev).
    height_coord : HeightCoordinate
        Vertical coordinate with reference state.

    Returns
    -------
    jax.Array
        Exner perturbation [-], shape (6, n, n, nlev).
    """
    R_d = constants.R_d
    c_v = constants.c_vd

    rho_0 = height_coord.rho_ref  # (nlev,)
    theta_0 = height_coord.theta_ref  # (nlev,)
    pi_0 = height_coord.exner_ref  # (nlev,)

    # Ratio form to avoid catastrophic cancellation in pi_total - pi_0.
    # Since pi = (R_d*rho*theta/p_0)^(R_d/c_v), we have:
    #   pi_total/pi_0 = ((rho_0+rho')*(theta_0+theta') / (rho_0*theta_0))^(R_d/c_v)
    #                 = ((1 + rho'/rho_0)*(1 + theta'/theta_0))^(R_d/c_v)
    #   pi' = pi_0 * (ratio^exponent - 1)
    # This is exact: zero when rho'=theta'=0, no large-value subtraction.
    exponent = R_d / c_v
    ratio = (1.0 + rho_prime / rho_0) * (1.0 + theta_prime / theta_0)
    return pi_0 * (ratio ** exponent - 1.0)


# ==============================================================================
# Sponge layer
# ==============================================================================

def _sponge_profile(
    z_full: jax.Array,
    H: float,
    sponge_width: float,
    sponge_coeff: float,
) -> jax.Array:
    """Compute Rayleigh damping coefficient profile.

    Increases smoothly from 0 to sponge_coeff over the top sponge_width
    meters using a sin^2 taper.

    Returns shape (nlev,).
    """
    z_sponge_bottom = H - sponge_width
    # Fraction into sponge layer: 0 below, 1 at top
    frac = jnp.clip((z_full - z_sponge_bottom) / sponge_width, 0.0, 1.0)
    return sponge_coeff * jnp.sin(0.5 * jnp.pi * frac) ** 2


# ==============================================================================
# Slow tendency computation
# ==============================================================================

def compressible_euler_slow_tendencies(
    state: NonHydrostaticState,
    grid: CubedSphereGrid,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
    config: CompressibleEulerConfig,
    physics_tendency: NonHydrostaticTendencies | None = None,
) -> NonHydrostaticTendencies:
    """Compute slow (advective) tendencies for the compressible Euler equations.

    These are evaluated once per RK3 stage and held constant during
    acoustic substeps. They include:
    1. Coriolis force
    2. Horizontal pressure gradient (Exner function)
    3. Horizontal and vertical advection of u, v, theta
    4. Kinetic energy gradient
    5. Hyperdiffusion
    6. Sponge layer damping

    The fast tendencies (acoustic: vertical PGF for w, continuity for rho)
    are handled in the acoustic substeps.
    """
    u = state.u.data          # (6, n, n, nlev)
    v = state.v.data
    w = state.w.data          # (6, n, n, nlev+1)
    theta_p = state.theta_prime.data
    rho_p = state.rho_prime.data
    tracers = state.tracers.data  # (6, n, n, nlev, n_tracers)

    g = config.g
    c_p = constants.c_pd
    R_d = constants.R_d
    rho_0 = height_coord.rho_ref        # (nlev,)
    theta_0 = height_coord.theta_ref    # (nlev,)
    dz = height_coord.dz                # (nlev,)
    dz_half = height_coord.dz_half      # (nlev-1,)
    J = terrain_metric.jacobian         # (6, n, n)

    theta_total = theta_0 + theta_p     # (6, n, n, nlev)
    rho_total = rho_0 + rho_p

    # --- 1. Exner perturbation and horizontal pressure gradient ---
    pi_prime = compute_exner_perturbation(rho_p, theta_p, height_coord)

    dpi_dx = gradient_x_3d(pi_prime, grid)
    dpi_dy = gradient_y_3d(pi_prime, grid)

    # --- 2. Vorticity and Coriolis ---
    zeta = vorticity_3d(u, v, grid)
    abs_vor = (zeta + grid.f[..., None]) if config.use_coriolis else zeta

    # --- 3. Kinetic energy gradient ---
    K = 0.5 * (u**2 + v**2)
    dK_dx = gradient_x_3d(K, grid)
    dK_dy = gradient_y_3d(K, grid)

    # --- 4. Horizontal momentum (vector-invariant form) ---
    du_dt = abs_vor * v - dK_dx - c_p * theta_total * dpi_dx
    dv_dt = -abs_vor * u - dK_dy - c_p * theta_total * dpi_dy

    # --- 5. Vertical advection of u, v ---
    du_dt = du_dt + vertical_advection_height(u, w, dz, dz_half, J)
    dv_dt = dv_dt + vertical_advection_height(v, w, dz, dz_half, J)

    # --- 6. Theta equation: horizontal + vertical advection ---
    dtheta_dx = gradient_x_3d(theta_total, grid)
    dtheta_dy = gradient_y_3d(theta_total, grid)
    horiz_adv_theta = -(u * dtheta_dx + v * dtheta_dy)

    vert_adv_theta = vertical_advection_height(theta_total, w, dz, dz_half, J)

    dtheta_p_dt = horiz_adv_theta + vert_adv_theta

    # --- 7. Continuity: horizontal divergence contribution ---
    # d(rho')/dt includes -(1/J) * div_h(J * rho * v_h) - vertical divergence
    # Horizontal part: -div_h(rho * u, rho * v)
    # For simplicity, use advective form: -u·grad(rho) - rho·div(v)
    rho_u = rho_total * u
    rho_v = rho_total * v
    div_rho_v = divergence_3d(rho_u, rho_v, grid)

    # Vertical mass flux divergence
    rho_half = 0.5 * (rho_total[..., :-1] + rho_total[..., 1:])
    rho_w = jnp.zeros_like(w)
    rho_w = rho_w.at[..., 1:-1].set(rho_half * w[..., 1:-1])
    vert_div = vertical_divergence_height(rho_w, dz, J)

    drho_p_dt = -div_rho_v - vert_div

    # --- 8. Tracer advection ---
    n_tracers = tracers.shape[-1] if tracers.ndim > 3 else 0
    if n_tracers > 0:
        tracers_t = jnp.moveaxis(tracers, -1, 0)  # (n_tracers, 6, n, n, nlev)

        def _single_tracer_tendency(q):
            dq_dx = gradient_x_3d(q, grid)
            dq_dy = gradient_y_3d(q, grid)
            horiz_adv_q = -(u * dq_dx + v * dq_dy)
            vert_adv_q = vertical_advection_height(q, w, dz, dz_half, J)
            return horiz_adv_q + vert_adv_q

        dtracers_dt_t = jax.vmap(_single_tracer_tendency)(tracers_t)
        dtracers_dt = jnp.moveaxis(dtracers_dt_t, 0, -1)
    else:
        dtracers_dt = jnp.zeros_like(tracers)

    # --- 9. Hyperdiffusion ---
    if config.hyperdiff_coeff > 0:
        du_dt = du_dt + hyperdiffusion_3d(u, grid, config.hyperdiff_coeff)
        dv_dt = dv_dt + hyperdiffusion_3d(v, grid, config.hyperdiff_coeff)
        dtheta_p_dt = dtheta_p_dt + hyperdiffusion_3d(
            theta_p, grid, config.hyperdiff_coeff
        )

    # --- 10. Sponge layer (Rayleigh damping toward reference state) ---
    sponge = _sponge_profile(
        height_coord.z_full, height_coord.H,
        config.sponge_width, config.sponge_coeff,
    )
    du_dt = du_dt - sponge * u
    dv_dt = dv_dt - sponge * v
    dtheta_p_dt = dtheta_p_dt - sponge * theta_p

    # Sponge for w at half levels
    sponge_half = _sponge_profile(
        height_coord.z_half, height_coord.H,
        config.sponge_width, config.sponge_coeff,
    )

    # --- 11. w tendency (slow part only: buoyancy is in acoustic step) ---
    # The w equation has slow contributions from horizontal advection
    w_full = 0.5 * (w[..., :-1] + w[..., 1:])  # interpolate to full levels
    dw_dx = gradient_x_3d(w_full, grid)
    dw_dy = gradient_y_3d(w_full, grid)
    horiz_adv_w = -(u * dw_dx + v * dw_dy)

    # Map back to half levels by averaging
    horiz_adv_w_half = jnp.zeros_like(w)
    horiz_adv_w_half = horiz_adv_w_half.at[..., 1:-1].set(
        0.5 * (horiz_adv_w[..., :-1] + horiz_adv_w[..., 1:])
    )

    dw_dt = horiz_adv_w_half - sponge_half * w

    # --- 12. Add physics tendencies if provided ---
    if physics_tendency is not None:
        du_dt = du_dt + physics_tendency.du_dt.data
        dv_dt = dv_dt + physics_tendency.dv_dt.data
        dw_dt = dw_dt + physics_tendency.dw_dt.data
        dtheta_p_dt = dtheta_p_dt + physics_tendency.dtheta_prime_dt.data
        drho_p_dt = drho_p_dt + physics_tendency.drho_prime_dt.data
        dtracers_dt = dtracers_dt + physics_tendency.dtracers_dt.data

    # --- Build tendency pytree ---
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    dims_tr = ("face", "x", "y", "level", "tracer")

    return NonHydrostaticTendencies(
        du_dt=Field(data=du_dt, name="du_dt", dims=dims_3d, units="m/s^2"),
        dv_dt=Field(data=dv_dt, name="dv_dt", dims=dims_3d, units="m/s^2"),
        dw_dt=Field(data=dw_dt, name="dw_dt", dims=dims_w, units="m/s^2"),
        dtheta_prime_dt=Field(
            data=dtheta_p_dt, name="dtheta_prime_dt", dims=dims_3d, units="K/s",
        ),
        drho_prime_dt=Field(
            data=drho_p_dt, name="drho_prime_dt", dims=dims_3d, units="kg/m^3/s",
        ),
        dphis_dt=Field(
            data=jnp.zeros_like(state.phis.data),
            name="dphis_dt", dims=dims_2d, units="m^2/s^3",
        ),
        dtracers_dt=Field(
            data=dtracers_dt, name="dtracers_dt", dims=dims_tr, units="1/s",
        ),
    )


# ==============================================================================
# Acoustic substeps (forward-backward)
# ==============================================================================

def acoustic_substeps(
    state: NonHydrostaticState,
    slow_tend: NonHydrostaticTendencies,
    dt_s: float,
    n_substeps: int,
    config: SplitExplicitConfig,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
    euler_config: CompressibleEulerConfig,
) -> NonHydrostaticState:
    """Run N acoustic substeps using forward-backward scheme.

    Each substep:
    1. Forward: update w using vertical Exner gradient + buoyancy
    2. Backward: update rho' using 3D divergence
    3. Backward: update theta' using vertical advection by w

    Parameters
    ----------
    state : NonHydrostaticState
        State after slow tendency update.
    slow_tend : NonHydrostaticTendencies
        Slow tendencies (held constant during substeps).
    dt_s : float
        Acoustic substep size [seconds].
    n_substeps : int
        Number of substeps.
    config : SplitExplicitConfig
    height_coord : HeightCoordinate
    terrain_metric : TerrainMetric
    euler_config : CompressibleEulerConfig

    Returns
    -------
    NonHydrostaticState
        State after all acoustic substeps.
    """
    g = euler_config.g
    c_p = constants.c_pd
    R_d = constants.R_d
    dz = height_coord.dz
    theta_0 = height_coord.theta_ref
    rho_0 = height_coord.rho_ref
    J = terrain_metric.jacobian

    # Extract mutable arrays
    w = state.w.data
    theta_p = state.theta_prime.data
    rho_p = state.rho_prime.data

    def substep_body(i, carry):
        w_c, theta_p_c, rho_p_c = carry

        theta_total = theta_0 + theta_p_c
        rho_total = rho_0 + rho_p_c

        # --- Forward: update w ---
        # Exner perturbation at full levels
        pi_p = compute_exner_perturbation(rho_p_c, theta_p_c, height_coord)

        # Exner gradient at half levels: d(pi')/dz* from full to half
        # Interior half levels only (1..nlev-1)
        dpi_dz_inner = (pi_p[..., :-1] - pi_p[..., 1:]) / (
            0.5 * (dz[:-1] + dz[1:])
        )

        # Theta at half levels (interpolated)
        theta_half_inner = 0.5 * (theta_total[..., :-1] + theta_total[..., 1:])

        # Buoyancy: -g * theta'/theta_0 at half levels
        theta_p_half = 0.5 * (theta_p_c[..., :-1] + theta_p_c[..., 1:])
        theta_0_half = 0.5 * (theta_0[:-1] + theta_0[1:])
        buoyancy = -g * theta_p_half / theta_0_half

        # w tendency at interior half levels
        dw_dt_inner = (
            -c_p * theta_half_inner * dpi_dz_inner / J[..., None]
            + buoyancy
        )

        # Update w (only interior levels, BCs stay at 0)
        w_new = w_c.at[..., 1:-1].set(
            w_c[..., 1:-1] + dt_s * dw_dt_inner
        )

        # --- Backward: update rho' using continuity ---
        # Vertical mass flux divergence with updated w
        rho_half = 0.5 * (rho_total[..., :-1] + rho_total[..., 1:])
        rho_w = jnp.zeros_like(w_new)
        rho_w = rho_w.at[..., 1:-1].set(rho_half * w_new[..., 1:-1])

        vert_div = (rho_w[..., :-1] - rho_w[..., 1:]) / dz
        vert_div = vert_div / J[..., None]

        rho_p_new = rho_p_c - dt_s * vert_div

        # --- Backward: update theta' using vertical w advection ---
        # d(theta')/dt from acoustic vertical advection only
        w_full = 0.5 * (w_new[..., :-1] + w_new[..., 1:])
        # Centered vertical gradient of theta_total at interior full levels
        dtheta_dz = jnp.zeros_like(theta_total)
        nlev = theta_total.shape[-1]
        if nlev > 2:
            dz_half_val = height_coord.dz_half  # (nlev-1,)
            dz_centered = dz_half_val[:-1] + dz_half_val[1:]  # (nlev-2,)
            inner_grad = (theta_total[..., :-2] - theta_total[..., 2:]) / dz_centered
            dtheta_dz = dtheta_dz.at[..., 1:-1].set(inner_grad)

        theta_p_new = theta_p_c - dt_s * w_full / J[..., None] * dtheta_dz

        return (w_new, theta_p_new, rho_p_new)

    # Run substeps via fori_loop
    w_final, theta_p_final, rho_p_final = jax.lax.fori_loop(
        0, n_substeps, substep_body, (w, theta_p, rho_p)
    )

    # Return updated state
    return NonHydrostaticState(
        u=state.u,
        v=state.v,
        w=state.w.replace(data=w_final),
        theta_prime=state.theta_prime.replace(data=theta_p_final),
        rho_prime=state.rho_prime.replace(data=rho_p_final),
        phis=state.phis,
        tracers=state.tracers,
    )


# ==============================================================================
# Semi-implicit acoustic substeps (tridiagonal)
# ==============================================================================

def acoustic_substeps_semi_implicit(
    state: NonHydrostaticState,
    slow_tend: NonHydrostaticTendencies,
    dt_s: float,
    n_substeps: int,
    config: SplitExplicitConfig,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
    euler_config: CompressibleEulerConfig,
) -> NonHydrostaticState:
    """Semi-implicit acoustic substeps using tridiagonal solve for w.

    Instead of a forward Euler update for w (explicit), the vertical
    pressure gradient term is treated implicitly by solving a tridiagonal
    system for w at each substep. This removes the acoustic CFL
    constraint in the vertical direction, enabling larger time steps
    and longer stable integrations.

    The implicit equation for w at interior half-levels is:

        (1 + dt_s^2 * c_s^2 / dz^2 / J^2) * w_new = w_old + dt_s * RHS_explicit

    where c_s^2 = c_p * R_d * T_ref is the linearized sound speed squared.
    The resulting tridiagonal system is solved per column via the Thomas
    algorithm, with jax.vmap over all columns.

    Parameters
    ----------
    state : NonHydrostaticState
        State after slow tendency update.
    slow_tend : NonHydrostaticTendencies
        Slow tendencies (held constant during substeps).
    dt_s : float
        Acoustic substep size [seconds].
    n_substeps : int
        Number of substeps.
    config : SplitExplicitConfig
    height_coord : HeightCoordinate
    terrain_metric : TerrainMetric
    euler_config : CompressibleEulerConfig

    Returns
    -------
    NonHydrostaticState
        State after all acoustic substeps.
    """
    from legoesm.timestepping.tridiagonal import thomas_solve_batched

    g = euler_config.g
    c_p = constants.c_pd
    R_d = constants.R_d
    c_v = constants.c_vd
    dz = height_coord.dz
    dz_half = height_coord.dz_half
    theta_0 = height_coord.theta_ref
    rho_0 = height_coord.rho_ref
    J = terrain_metric.jacobian  # (6, n, n)

    # Linearized sound speed squared: c_s^2 = gamma * R_d * T_ref
    # where T_ref = theta_0 * pi_0 and gamma = c_p / c_v
    gamma = c_p / c_v
    T_ref = theta_0 * height_coord.exner_ref  # (nlev,)
    cs2 = gamma * R_d * T_ref  # (nlev,)

    # Sound speed at half levels (interior): average of adjacent full levels
    cs2_half = 0.5 * (cs2[:-1] + cs2[1:])  # (nlev-1,)

    # Extract mutable arrays
    w = state.w.data       # (..., nlev+1)
    theta_p = state.theta_prime.data  # (..., nlev)
    rho_p = state.rho_prime.data      # (..., nlev)

    nlev = theta_p.shape[-1]
    n_inner = nlev - 1  # Number of interior half-levels

    # Precompute tridiagonal matrix coefficients for the implicit w solve.
    # The implicit equation at interior half-level k (k=1..nlev-1) is:
    #   -alpha * w[k-1] + (1 + 2*alpha) * w[k] - alpha * w[k+1] = RHS
    # where alpha = dt_s^2 * cs2_half[k] / dz_k^2 / J^2
    # But interior half-levels use dz between adjacent full levels.
    # dz at half-level k is 0.5*(dz[k-1]+dz[k]) for interior levels.

    dz_inner = 0.5 * (dz[:-1] + dz[1:])  # (nlev-1,)

    def substep_body(i, carry):
        w_c, theta_p_c, rho_p_c = carry

        theta_total = theta_0 + theta_p_c
        rho_total = rho_0 + rho_p_c

        # --- Explicit RHS for w (same as forward step) ---
        pi_p = compute_exner_perturbation(rho_p_c, theta_p_c, height_coord)
        dpi_dz_inner = (pi_p[..., :-1] - pi_p[..., 1:]) / dz_inner

        theta_half_inner = 0.5 * (theta_total[..., :-1] + theta_total[..., 1:])
        theta_p_half = 0.5 * (theta_p_c[..., :-1] + theta_p_c[..., 1:])
        theta_0_half = 0.5 * (theta_0[:-1] + theta_0[1:])
        buoyancy = -g * theta_p_half / theta_0_half

        dw_dt_inner = (
            -c_p * theta_half_inner * dpi_dz_inner / J[..., None]
            + buoyancy
        )

        # RHS of tridiagonal system: w_old + dt_s * explicit_tendency
        rhs = w_c[..., 1:-1] + dt_s * dw_dt_inner  # (6,n,n,nlev-1)

        # --- Build tridiagonal coefficients for implicit solve ---
        # alpha_k = dt_s^2 * cs2_half[k] / (dz_inner[k] * J)^2
        alpha = dt_s**2 * cs2_half / (dz_inner * J[..., None])**2  # (6,n,n,nlev-1)

        # Sub-diagonal: -alpha (for k > 0 in interior)
        a_tri = jnp.zeros_like(alpha)
        a_tri = a_tri.at[..., 1:].set(-alpha[..., 1:])

        # Main diagonal: 1 + 2*alpha (interior), adjusted at boundaries
        # At k=0 (top interior): w[k-1] = w[0] = 0 (BC), so only +alpha from below
        # At k=n_inner-1 (bottom interior): w[k+1] = w[nlev] = 0 (BC), so only +alpha from above
        b_tri = 1.0 + 2.0 * alpha
        # Boundary corrections: first row has no upper neighbor in implicit part, last row no lower
        b_tri = b_tri.at[..., 0].set(1.0 + alpha[..., 0])
        b_tri = b_tri.at[..., -1].set(1.0 + alpha[..., -1])

        # Super-diagonal: -alpha (for k < n_inner-1)
        c_tri = jnp.zeros_like(alpha)
        c_tri = c_tri.at[..., :-1].set(-alpha[..., :-1])

        # Solve tridiagonal system
        w_inner_new = thomas_solve_batched(a_tri, b_tri, c_tri, rhs)

        # Update w (boundaries stay at 0)
        w_new = w_c.at[..., 1:-1].set(w_inner_new)

        # --- Backward: update rho' using updated w ---
        rho_half = 0.5 * (rho_total[..., :-1] + rho_total[..., 1:])
        rho_w = jnp.zeros_like(w_new)
        rho_w = rho_w.at[..., 1:-1].set(rho_half * w_new[..., 1:-1])
        vert_div = (rho_w[..., :-1] - rho_w[..., 1:]) / dz
        vert_div = vert_div / J[..., None]
        rho_p_new = rho_p_c - dt_s * vert_div

        # --- Backward: update theta' using vertical w advection ---
        w_full = 0.5 * (w_new[..., :-1] + w_new[..., 1:])
        dtheta_dz = jnp.zeros_like(theta_total)
        if nlev > 2:
            dz_centered = dz_half[:-1] + dz_half[1:]
            inner_grad = (theta_total[..., :-2] - theta_total[..., 2:]) / dz_centered
            dtheta_dz = dtheta_dz.at[..., 1:-1].set(inner_grad)
        theta_p_new = theta_p_c - dt_s * w_full / J[..., None] * dtheta_dz

        return (w_new, theta_p_new, rho_p_new)

    # Run substeps via fori_loop
    w_final, theta_p_final, rho_p_final = jax.lax.fori_loop(
        0, n_substeps, substep_body, (w, theta_p, rho_p)
    )

    return NonHydrostaticState(
        u=state.u,
        v=state.v,
        w=state.w.replace(data=w_final),
        theta_prime=state.theta_prime.replace(data=theta_p_final),
        rho_prime=state.rho_prime.replace(data=rho_p_final),
        phis=state.phis,
        tracers=state.tracers,
    )


# ==============================================================================
# Model class
# ==============================================================================

class CompressibleEulerModel:
    """Non-hydrostatic compressible Euler model on the cubed-sphere.

    Parameters
    ----------
    grid : CubedSphereGrid
        Horizontal grid.
    height_coord : HeightCoordinate
        Vertical coordinate with reference state.
    terrain_metric : TerrainMetric
        Terrain-following metric terms.
    config : CompressibleEulerConfig, optional
        Model configuration.

    Example
    -------
    >>> grid = create_cubed_sphere(48)
    >>> hc = create_height_coordinate(88, 40000.0, theta_fn)
    >>> tm = compute_terrain_metric(z_s, hc)
    >>> model = CompressibleEulerModel(grid, hc, tm)
    >>> state_new = model.step(state, dt=10.0)
    """

    def __init__(
        self,
        grid: CubedSphereGrid,
        height_coord: HeightCoordinate,
        terrain_metric: TerrainMetric,
        config: CompressibleEulerConfig | None = None,
    ):
        self.grid = grid
        self.height_coord = height_coord
        self.terrain_metric = terrain_metric
        self.config = config or CompressibleEulerConfig()

    def tendencies(
        self,
        state: NonHydrostaticState,
        physics_tendency: NonHydrostaticTendencies | None = None,
    ) -> NonHydrostaticTendencies:
        """Compute slow tendencies (pure function wrapper)."""
        return compressible_euler_slow_tendencies(
            state, self.grid, self.height_coord, self.terrain_metric,
            self.config, physics_tendency,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: NonHydrostaticState,
        dt: float,
    ) -> NonHydrostaticState:
        """Advance one time step using split-explicit RK3.

        Parameters
        ----------
        state : NonHydrostaticState
            Current state.
        dt : float
            Time step [seconds].

        Returns
        -------
        NonHydrostaticState : State after one time step.
        """
        se_config = SplitExplicitConfig(
            n_substeps=self.config.n_acoustic_substeps,
        )

        def slow_tendency_fn(s):
            tend = compressible_euler_slow_tendencies(
                s, self.grid, self.height_coord, self.terrain_metric,
                self.config,
            )
            return NonHydrostaticState(
                u=s.u.replace(data=tend.du_dt.data),
                v=s.v.replace(data=tend.dv_dt.data),
                w=s.w.replace(data=tend.dw_dt.data),
                theta_prime=s.theta_prime.replace(data=tend.dtheta_prime_dt.data),
                rho_prime=s.rho_prime.replace(data=tend.drho_prime_dt.data),
                phis=s.phis.replace(data=jnp.zeros_like(s.phis.data)),
                tracers=s.tracers.replace(data=tend.dtracers_dt.data),
            )

        def acoustic_update_fn(s, slow_tend, dt_s, n_sub, cfg):
            if self.config.semi_implicit_acoustic:
                return acoustic_substeps_semi_implicit(
                    s, slow_tend, dt_s, n_sub, cfg,
                    self.height_coord, self.terrain_metric, self.config,
                )
            return acoustic_substeps(
                s, slow_tend, dt_s, n_sub, cfg,
                self.height_coord, self.terrain_metric, self.config,
            )

        return split_explicit_step(
            state, slow_tendency_fn, acoustic_update_fn, dt, se_config,
        )

    @partial(jax.jit, static_argnums=(0, 3))
    def step_with_physics(
        self,
        state: NonHydrostaticState,
        dt: float,
        physics_fn=None,
    ) -> NonHydrostaticState:
        """Advance one time step with physics forcing.

        Parameters
        ----------
        state : NonHydrostaticState
        dt : float
        physics_fn : callable, optional
            Function (state, grid, height_coord, terrain_metric) -> tendencies.
        """
        se_config = SplitExplicitConfig(
            n_substeps=self.config.n_acoustic_substeps,
        )

        def slow_tendency_fn(s):
            phys = None
            if physics_fn is not None:
                phys = physics_fn(
                    s, self.grid, self.height_coord, self.terrain_metric,
                )
            tend = compressible_euler_slow_tendencies(
                s, self.grid, self.height_coord, self.terrain_metric,
                self.config, phys,
            )
            return NonHydrostaticState(
                u=s.u.replace(data=tend.du_dt.data),
                v=s.v.replace(data=tend.dv_dt.data),
                w=s.w.replace(data=tend.dw_dt.data),
                theta_prime=s.theta_prime.replace(data=tend.dtheta_prime_dt.data),
                rho_prime=s.rho_prime.replace(data=tend.drho_prime_dt.data),
                phis=s.phis.replace(data=jnp.zeros_like(s.phis.data)),
                tracers=s.tracers.replace(data=tend.dtracers_dt.data),
            )

        def acoustic_update_fn(s, slow_tend, dt_s, n_sub, cfg):
            if self.config.semi_implicit_acoustic:
                return acoustic_substeps_semi_implicit(
                    s, slow_tend, dt_s, n_sub, cfg,
                    self.height_coord, self.terrain_metric, self.config,
                )
            return acoustic_substeps(
                s, slow_tend, dt_s, n_sub, cfg,
                self.height_coord, self.terrain_metric, self.config,
            )

        return split_explicit_step(
            state, slow_tendency_fn, acoustic_update_fn, dt, se_config,
        )

    def integrate(
        self,
        state: NonHydrostaticState,
        duration: float,
        dt: float,
        save_every: int = 1,
        physics_fn=None,
    ) -> tuple[NonHydrostaticState, list[NonHydrostaticState]]:
        """Integrate forward for a given duration.

        Parameters
        ----------
        state : NonHydrostaticState
        duration : float
            Total integration time [seconds].
        dt : float
        save_every : int
        physics_fn : callable, optional

        Returns
        -------
        final_state, trajectory
        """
        n_steps = int(duration / dt)
        trajectory = [state]

        for i in range(n_steps):
            if physics_fn is not None:
                state = self.step_with_physics(state, dt, physics_fn)
            else:
                state = self.step(state, dt)
            if (i + 1) % save_every == 0:
                trajectory.append(state)

        return state, trajectory
