"""Shared utilities for the non-hydrostatic compressible Euler equations.

This module provides shared infrastructure used by all non-hydrostatic
compressible Euler solvers (C-D grid cubed-sphere, lat-lon FV, MPAS,
spectral):

- ``CompressibleEulerConfig`` — base configuration NamedTuple
- ``compute_exner_perturbation`` — Exner function perturbation from EOS
- ``_sponge_profile`` — Rayleigh damping profile
- ``acoustic_substeps`` — forward-backward acoustic substeps
- ``acoustic_substeps_semi_implicit`` — tridiagonal implicit acoustic substeps

The A-grid cubed-sphere slow-tendency solver that previously lived here
has been removed.  Use ``cdgrid_compressible_euler_slow_tendencies`` from
``compressible_euler_cdgrid.py`` instead.

Equations solved
----------------
Compressible-Euler dry-air system in perturbation form. The prognostic
state ``NonHydrostaticState`` carries

    u, v          horizontal velocity components
    w             vertical velocity at half-levels
    theta_prime   potential-temperature perturbation about ref(z)
    rho_prime     density perturbation about ref(z)
    phis          surface geopotential (diagnostic)
    tracers       passive + reactive tracer mixing ratios

with the full state recovered as

    theta(x, y, z, t) = theta_ref(z) + theta_prime(x, y, z, t)
    rho(x, y, z, t)   = rho_ref(z)   + rho_prime(x, y, z, t).

The reference profile ``(theta_ref, rho_ref)`` is supplied by
``HeightCoordinate`` and is hydrostatically balanced. Subtracting it
keeps the acoustic-substep pressure-gradient terms well-conditioned
because the dominant background ``g * rho_ref`` cancels analytically.

Acoustic substepping uses the Skamarock-Klemp split-explicit scheme:
slow horizontal advection + tracer flux divergence + diffusion are
frozen for ``n_acoustic_substeps`` short substeps that resolve the
fast acoustic modes. Vertical acoustic terms can be advanced either
forward-backward (default) or with a tridiagonal implicit solve when
``semi_implicit_acoustic=True`` — see ``acoustic_substeps`` and
``acoustic_substeps_semi_implicit`` for the exact update formulas.

Grid callback contract (existing consumers)
-------------------------------------------
The acoustic substep routines in this module are grid-agnostic with
respect to the horizontal stencil: each existing consumer
(``compressible_euler_cdgrid.py``, ``compressible_euler_mpas.py``,
``spectral_nh.py``) assembles the horizontal pressure-gradient and
flux-divergence contributions in its own slow-tendency routine and
then calls the shared substep kernel for the vertically coupled
acoustic update.

For that pattern to work, every dycore caller currently must provide:

- ``HeightCoordinate`` exposing ``rho_ref``, ``theta_ref``, ``dz`` and
  the half-level / full-level arrangement used by the chosen Lorenz
  staggering.
- ``TerrainMetric`` exposing the column-local Jacobian ``J`` and
  half-level scale-factor used inside the vertical implicit solve.
- A ``physics_fn`` callable that returns physics tendencies on the
  same state pytree as the dycore, applied between split-explicit
  outer stages.

This contract is **descriptive, not prescriptive**: it documents the
shape of what cubed-sphere C-D and MPAS Voronoi do today.
Plane-specific or lat-lon-specific extensions land in their own
modules in follow-up PRs of the CRM rollout and may add new optional
callbacks (vertical-tridiagonal coefficient assembly, periodic-halo
operator) without changing the existing signatures.

Conservation invariant
----------------------
Discrete dry-air mass on each existing grid is ``sum_{cells} rho * J
* area_cell * dz``. When ``CompressibleEulerConfig.fix_mass=True`` the
dycore applies a uniform additive correction to ``rho_prime`` so that
this sum equals a stored target (``anchor_mass_to_initial=True``
anchors the target to ``t=0`` and prevents drift). The correction is
constant per outer step and so preserves all spatial gradients used by
the slow-tendency routine.

References
----------
- Skamarock & Klemp (2008): A Time-Split Nonhydrostatic Atmospheric Model.
- Klemp et al. (2007): Terrain-Following Coordinate.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.state import NonHydrostaticState, NonHydrostaticTendencies
from legoesm.grids.vertical import HeightCoordinate, TerrainMetric
from legoesm.timestepping.split_explicit import SplitExplicitConfig
from legoesm.timestepping.tridiagonal import thomas_solve_batched
from legoesm.atmosphere.physics.thermodynamics import sanitize_theta_rho
from legoesm import constants


class CompressibleEulerConfig(NamedTuple):
    """Configuration for the non-hydrostatic compressible Euler model."""
    g: float = constants.g
    hyperdiff_coeff: float = 0.0
    hyperdiff_rho_coeff: float = 0.0
    hyperdiff_w_coeff: float = 0.0
    sponge_width: float = 10000.0   # Sponge layer width from model top [m]
    sponge_coeff: float = 0.05      # Maximum Rayleigh damping rate [1/s]
    n_acoustic_substeps: int = 6
    small_earth_factor: float = 1.0
    use_coriolis: bool = True       # Set False for f=0 tests (e.g. DCMIP TC3)
    semi_implicit_acoustic: bool = False  # Use tridiagonal solve for acoustic substeps
    outer_integrator: str = "ssp_rk3"  # "ssp_rk3" | "ssp_rk34"/"ssp34" | "ssp_rk54"/"ssp45"
    fix_mass: bool = False            # Apply NH mass conservation fixer
    anchor_mass_to_initial: bool = False  # Anchor to initial mass (prevents drift)
    acoustic_off_centering: float = 0.0   # Off-centering parameter beta for acoustic steps
                                          # 0.0 = centered (neutral), 0.1 = slightly damped
                                          # Damps vertically-propagating acoustic modes
                                          # without horizontal CFL constraint (Skamarock 2008)


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
    rho_rel = 1.0 + rho_prime / jnp.clip(rho_0, 1.0e-9, None)
    theta_rel = 1.0 + theta_prime / jnp.clip(theta_0, 50.0, None)
    ratio = jnp.clip(rho_rel * theta_rel, 1.0e-12, 1.0e12)
    return pi_0 * jnp.expm1(exponent * jnp.log(ratio))


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
    dz = height_coord.dz
    theta_0 = height_coord.theta_ref
    rho_0 = height_coord.rho_ref
    J = terrain_metric.jacobian

    # Extract mutable arrays
    w = state.w.data
    theta_p = state.theta_prime.data
    rho_p = state.rho_prime.data

    # Off-centering parameter for acoustic damping (Skamarock & Klemp 2008).
    # beta > 0 introduces a small amount of temporal diffusion that damps
    # vertically-propagating acoustic/gravity wave noise without affecting
    # the horizontal CFL constraint.  Typical value: 0.1 for long runs.
    beta = euler_config.acoustic_off_centering

    def substep_body(i, carry):
        w_c, theta_p_c, rho_p_c = carry

        theta_total, rho_total = sanitize_theta_rho(
            theta_0 + theta_p_c,
            rho_0 + rho_p_c,
        )

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

        # Buoyancy: +g * theta'/theta_0 at half levels
        # Derived from: buoyancy = -c_p * theta' * d(pi_0)/dz
        # Using hydrostatic balance d(pi_0)/dz = -g/(c_p*theta_0)
        theta_p_half = 0.5 * (theta_p_c[..., :-1] + theta_p_c[..., 1:])
        theta_0_half = 0.5 * (theta_0[:-1] + theta_0[1:])
        buoyancy = g * theta_p_half / theta_0_half

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
        # Vertical mass flux divergence with updated w; ``rho_w`` has
        # zero at top/bottom (rigid BC).  Single Pad HLO op replaces
        # alloc-zeros + scatter.
        rho_half = 0.5 * (rho_total[..., :-1] + rho_total[..., 1:])
        pad_axes_w = ((0, 0),) * (w_new.ndim - 1)
        rho_w = jnp.pad(rho_half * w_new[..., 1:-1], (*pad_axes_w, (1, 1)))

        vert_div = (rho_w[..., :-1] - rho_w[..., 1:]) / dz
        vert_div = vert_div / J[..., None]

        rho_p_new = rho_p_c - dt_s * vert_div

        # --- Off-centering: damp acoustic mode via time-averaging ---
        # rho_p_damped = (1+beta)*rho_p_new - beta*rho_p_old
        # For beta=0: no damping (centered). For beta>0: dissipative.
        rho_p_new = (1.0 + beta) * rho_p_new - beta * rho_p_c

        # --- Backward: update theta' using vertical w advection ---
        # d(theta')/dt from acoustic vertical advection only.
        # ``dtheta_dz`` is zero at top/bottom (no ghost), centred in
        # the interior; single Pad HLO op replaces alloc-zeros +
        # scatter.
        w_full = 0.5 * (w_new[..., :-1] + w_new[..., 1:])
        nlev = theta_total.shape[-1]
        if nlev > 2:
            dz_half_val = height_coord.dz_half  # (nlev-1,)
            dz_centered = dz_half_val[:-1] + dz_half_val[1:]  # (nlev-2,)
            inner_grad = (theta_total[..., :-2] - theta_total[..., 2:]) / dz_centered
            pad_axes_t = ((0, 0),) * (theta_total.ndim - 1)
            dtheta_dz = jnp.pad(inner_grad, (*pad_axes_t, (1, 1)))
        else:
            dtheta_dz = jnp.zeros_like(theta_total)

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
    g = euler_config.g
    c_p = constants.c_pd
    R_d = constants.R_d
    c_v = constants.c_vd
    dz = height_coord.dz
    dz_half = height_coord.dz_half
    theta_0 = height_coord.theta_ref
    rho_0 = height_coord.rho_ref
    J = terrain_metric.jacobian  # (6, n, n)
    beta = euler_config.acoustic_off_centering

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

    # Precompute tridiagonal matrix coefficients for the implicit w solve.
    # The implicit equation at interior half-level k (k=1..nlev-1) is:
    #   -alpha * w[k-1] + (1 + 2*alpha) * w[k] - alpha * w[k+1] = RHS
    # where alpha = dt_s^2 * cs2_half[k] / dz_k^2 / J^2
    # But interior half-levels use dz between adjacent full levels.
    # dz at half-level k is 0.5*(dz[k-1]+dz[k]) for interior levels.

    dz_inner = 0.5 * (dz[:-1] + dz[1:])  # (nlev-1,)

    def substep_body(i, carry):
        w_c, theta_p_c, rho_p_c = carry

        theta_total, rho_total = sanitize_theta_rho(
            theta_0 + theta_p_c,
            rho_0 + rho_p_c,
        )

        # --- Explicit RHS for w (same as forward step) ---
        pi_p = compute_exner_perturbation(rho_p_c, theta_p_c, height_coord)
        dpi_dz_inner = (pi_p[..., :-1] - pi_p[..., 1:]) / dz_inner

        theta_half_inner = 0.5 * (theta_total[..., :-1] + theta_total[..., 1:])
        theta_p_half = 0.5 * (theta_p_c[..., :-1] + theta_p_c[..., 1:])
        theta_0_half = 0.5 * (theta_0[:-1] + theta_0[1:])
        buoyancy = g * theta_p_half / theta_0_half

        dw_dt_inner = (
            -c_p * theta_half_inner * dpi_dz_inner / J[..., None]
            + buoyancy
        )

        # RHS of tridiagonal system: w_old + dt_s * explicit_tendency
        rhs = w_c[..., 1:-1] + dt_s * dw_dt_inner  # (6,n,n,nlev-1)

        # --- Build tridiagonal coefficients for implicit solve ---
        # alpha_k = dt_s^2 * cs2_half[k] / (dz_inner[k] * J)^2
        alpha = dt_s**2 * cs2_half / (dz_inner * J[..., None])**2  # (6,n,n,nlev-1)

        # Sub-diagonal: 0 at k=0, -alpha for k > 0.  Single Pad HLO op
        # replaces alloc-zeros + scatter.
        pad_axes_a = ((0, 0),) * (alpha.ndim - 1)
        a_tri = jnp.pad(-alpha[..., 1:], (*pad_axes_a, (1, 0)))

        # Main diagonal: 1 + alpha + alpha_interior, where alpha_interior
        # is alpha with the boundary entries zeroed.  This collapses
        # ``b_tri = 1 + 2*alpha`` + 2 boundary scatters into 1 Pad HLO op
        # (the slice + Pad share intermediates).
        # At boundaries (k=0, k=-1) the implicit BC sets w_outside=0 so
        # the diagonal is 1 + alpha; in the interior it is 1 + 2*alpha.
        alpha_interior = jnp.pad(alpha[..., 1:-1], (*pad_axes_a, (1, 1)))
        b_tri = 1.0 + alpha + alpha_interior

        # Super-diagonal: -alpha for k < n_inner-1, 0 at k=-1.  Single
        # Pad HLO op replaces alloc-zeros + scatter.
        c_tri = jnp.pad(-alpha[..., :-1], (*pad_axes_a, (0, 1)))

        # Solve tridiagonal system
        w_inner_new = thomas_solve_batched(a_tri, b_tri, c_tri, rhs)

        # Update w (boundaries stay at 0)
        w_new = w_c.at[..., 1:-1].set(w_inner_new)

        # --- Backward: update rho' using updated w ---
        # ``rho_w`` has zero at top/bottom interfaces (rigid lid / rigid
        # bottom).  Single Pad HLO op replaces alloc-zeros + scatter.
        rho_half = 0.5 * (rho_total[..., :-1] + rho_total[..., 1:])
        pad_axes_w = ((0, 0),) * (w_new.ndim - 1)
        rho_w = jnp.pad(rho_half * w_new[..., 1:-1], (*pad_axes_w, (1, 1)))
        vert_div = (rho_w[..., :-1] - rho_w[..., 1:]) / dz
        vert_div = vert_div / J[..., None]
        rho_p_new = rho_p_c - dt_s * vert_div

        # Off-centering: damp acoustic mode (Skamarock & Klemp 2008)
        rho_p_new = (1.0 + beta) * rho_p_new - beta * rho_p_c

        # --- Backward: update theta' using vertical w advection ---
        # ``dtheta_dz`` is zero at top/bottom (one-sided would require
        # ghost cells); centred difference fills the interior.  Single
        # Pad HLO op replaces alloc-zeros + scatter.
        w_full = 0.5 * (w_new[..., :-1] + w_new[..., 1:])
        if nlev > 2:
            dz_centered = dz_half[:-1] + dz_half[1:]
            inner_grad = (theta_total[..., :-2] - theta_total[..., 2:]) / dz_centered
            pad_axes_t = ((0, 0),) * (theta_total.ndim - 1)
            dtheta_dz = jnp.pad(inner_grad, (*pad_axes_t, (1, 1)))
        else:
            dtheta_dz = jnp.zeros_like(theta_total)
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
