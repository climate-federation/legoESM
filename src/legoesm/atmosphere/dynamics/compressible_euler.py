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
    si_w_vertical_filter_nu: float = 0.0  # Vertical Laplacian filter for w inside
                                          # each SI acoustic substep:
                                          #   w[k] += nu * (w[k-1] - 2*w[k] + w[k+1])
                                          # Default 0.0 = off (preserves the iter-78
                                          # contract of the SI scheme).
                                          # Empirical: the SI dycore has a structural
                                          # exponential mode (growth rate ~1.25/step)
                                          # triggered by ANY perturbed theta' IC
                                          # (warm-bubble tests xfailed at iter-65 had
                                          # the same root). The mode saturates at
                                          # max|w|~15 m/s with active moist physics
                                          # (gray+Kessler at N=128 dx=2km nlev=30 ran
                                          # 100 days stably) but blows up to NaN with
                                          # cleaner physics (RRTMGP+Morrison) or a
                                          # dry dycore. Setting nu>=0.4 fully damps
                                          # the unstable mode at every dt tested
                                          # (0.5-20 s). 0.5 = explicit-diffusion CFL
                                          # bound; above that the filter itself NaNs.
                                          # Recommended: 0.3-0.4 for runs with
                                          # perturbed theta' IC; 0.0 for clean Wing
                                          # IC + active moist physics.
    implicit_buoyancy: bool = False       # Klemp-Wilhelmson 1978 implicit-buoyancy
                                          # in the SI acoustic substep. Substitutes
                                          # theta_p_new = theta_p_c
                                          #   - dt_s*w_full/J * dtheta_ref/dz
                                          # into the buoyancy term g*theta_p/theta_0
                                          # of the w-equation. Adds three nearest-
                                          # neighbour bands to the existing implicit
                                          # PG tridiagonal: kappa = 0.25*dt_s^2*g/
                                          # (theta_0_half*J). Stabilizes the
                                          # w-theta gravity-wave feedback that
                                          # destabilizes the plane NH dycore at
                                          # coarse vertical resolution
                                          # (dz~1000 m) with stratified ICs.
                                          # Only active when
                                          # semi_implicit_acoustic=True.
    # ---- Plane-only fields (PR3c) ----
    # The following two knobs are consumed ONLY by the doubly-periodic
    # plane non-hydrostatic dycore
    # (:mod:`legoesm.atmosphere.dynamics.compressible_euler_plane`).
    # Cubed-sphere, MPAS, and spectral NH dycores ignore them entirely
    # — each of those has its own Smagorinsky knob in its own
    # ``*CompressibleEulerConfig`` (e.g. ``CDGridCompressibleEulerConfig.
    # smagorinsky_cs``). Defaults of ``0.0`` and ``1.0`` make this a
    # no-op on every dycore.
    # PR3c is a HORIZONTAL-ONLY pilot Smag closure: the strain magnitude
    # includes ∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y only; vertical shear
    # (∂u/∂z, ∂v/∂z, ∂w/∂x, ∂w/∂y, ∂w/∂z) is deferred to a 3D extension PR.
    # The Prandtl number controls the *thermal* (K_h) leg of that same
    # closure; the horizontal-only caveat applies to both fields below.
    smagorinsky_cs: float = 0.0           # Smagorinsky-Lilly LES coefficient.
                                          # K_m = (C_s * Δ)^2 * |S|. Typical 0.1-0.25.
                                          # 0.0 disables (Python on/off gate).
    smagorinsky_prandtl: float = 1.0      # Turbulent Prandtl number K_h = K_m / Pr.
                                          # Plane LES default 1.0; classical atmosphere
                                          # value is ~1/3 for stable stratification.
                                          # Must be > 0 when smagorinsky_cs > 0
                                          # (validate_plane_config enforces).
    horizontal_advection_scheme: str = "upwind1"
                                          # Horizontal advection of theta_prime, u, v, w
                                          # (and tracers) on the plane dycore. Three
                                          # choices, defined by
                                          # HORIZONTAL_ADVECTION_HALO_REQUIREMENT in
                                          # compressible_euler_plane.py:
                                          # "upwind1" — first-order upwind (cheap, very
                                          # dispersive at coarse dx).
                                          # "van_leer" — 2nd-order TVD, stencil 4
                                          # (iter-183 production: 3x wall-time speedup
                                          # vs upwind1 at dt=20 thanks to lower
                                          # numerical diffusion + monotonicity).
                                          # "weno5" — 5th-order WENO-Z, stencil 6
                                          # (least grid-scale dispersion but ~3x
                                          # per-step cost; opt-in for sharp-front
                                          # problems).
                                          # Class default stays "upwind1" for back-
                                          # compat with iter-7 fixtures; the
                                          # production driver overrides to van_leer
                                          # at parse_args time (iter-183).
                                          # Consumed by
                                          # ``compressible_euler_plane.py`` only;
                                          # cubed-sphere / MPAS ignore it.
    vertical_theta_diffusion: float = 0.0
                                          # Explicit vertical Laplacian diffusivity
                                          # on theta_prime [m^2/s], applied per outer
                                          # RK stage. Damps the buoyancy-driven
                                          # gravity-wave amplification that
                                          # destabilises the plane NH dycore at
                                          # dx ~ 2 km / dt > 0.5 s with a coarse
                                          # vertical grid (dz ~ 1000 m). Rigid (zero)
                                          # boundary condition at top + bottom.
                                          # Typical effective value: nu_v ~ 1e3-5e3
                                          # so dt * nu_v / dz^2 stays below ~0.1
                                          # (explicit-Euler CFL bound). 0.0 disables.
                                          # Consumed by plane dycore only.


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


def _acoustic_column_kernel(
    w_c: jax.Array,
    theta_p_c: jax.Array,
    rho_p_c: jax.Array,
    height_coord: HeightCoordinate,
    J: jax.Array,
    dt_s: float,
    beta: float,
    g: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Single forward-backward acoustic substep (column-local algebra).

    This kernel encapsulates the vertical-only update of ``(w, theta',
    rho')`` for one acoustic substep. It is grid-agnostic in the
    horizontal: any number of leading axes broadcast through the
    ``[..., k]`` slicing.

    Contract (vertical = last axis)
    -------------------------------
    - ``w_c`` : ``(..., nlev+1)`` — vertical velocity at Lorenz half
      (interface) levels. Rigid boundaries ``w_c[..., 0] = w_c[..., -1]
      = 0`` must already hold on input; the kernel updates only the
      interior ``[..., 1:-1]`` slice and leaves the boundary values
      untouched.
    - ``theta_p_c`` : ``(..., nlev)`` — potential-temperature
      perturbation at full (cell-centre) levels.
    - ``rho_p_c`` : ``(..., nlev)`` — density perturbation at full
      levels.
    - ``height_coord`` : ``HeightCoordinate`` providing
      ``rho_ref(nlev,)``, ``theta_ref(nlev,)``, ``exner_ref(nlev,)``,
      ``dz(nlev,)`` (full-level spacing), ``dz_half(nlev-1,)``
      (half-level spacing between adjacent cell centres).
    - ``J`` : Jacobian, shape broadcastable to the horizontal leading
      axes of ``w_c``, ``theta_p_c``, ``rho_p_c``. The kernel uses
      ``J[..., None]`` to broadcast over the vertical axis.
    - ``dt_s`` : substep size in seconds (Python scalar).
    - ``beta`` : Skamarock-Klemp off-centering parameter in ``[0, 1)``.
      ``0`` is centred, larger values damp vertically-propagating
      acoustic modes.
    - ``g`` : gravitational acceleration [m/s^2].

    Returns
    -------
    (w_new, theta_p_new, rho_p_new)
        Updated arrays with the same shapes as the inputs. Rigid w
        boundaries preserved.
    """
    c_p = constants.c_pd
    dz = height_coord.dz
    theta_0 = height_coord.theta_ref
    rho_0 = height_coord.rho_ref

    theta_total, rho_total = sanitize_theta_rho(
        theta_0 + theta_p_c,
        rho_0 + rho_p_c,
    )

    # --- Forward: update w ---
    pi_p = compute_exner_perturbation(rho_p_c, theta_p_c, height_coord)

    dpi_dz_inner = (pi_p[..., :-1] - pi_p[..., 1:]) / (
        0.5 * (dz[:-1] + dz[1:])
    )

    theta_half_inner = 0.5 * (theta_total[..., :-1] + theta_total[..., 1:])

    theta_p_half = 0.5 * (theta_p_c[..., :-1] + theta_p_c[..., 1:])
    theta_0_half = 0.5 * (theta_0[:-1] + theta_0[1:])
    buoyancy = g * theta_p_half / theta_0_half

    dw_dt_inner = (
        -c_p * theta_half_inner * dpi_dz_inner / J[..., None]
        + buoyancy
    )

    w_new = w_c.at[..., 1:-1].set(
        w_c[..., 1:-1] + dt_s * dw_dt_inner
    )

    # --- Backward: update rho' using continuity ---
    rho_half = 0.5 * (rho_total[..., :-1] + rho_total[..., 1:])
    pad_axes_w = ((0, 0),) * (w_new.ndim - 1)
    rho_w = jnp.pad(rho_half * w_new[..., 1:-1], (*pad_axes_w, (1, 1)))

    vert_div = (rho_w[..., :-1] - rho_w[..., 1:]) / dz
    vert_div = vert_div / J[..., None]

    rho_p_new = rho_p_c - dt_s * vert_div
    if beta != 0.0:
        rho_p_new = (1.0 + beta) * rho_p_new - beta * rho_p_c

    # --- Backward: update theta' using vertical w advection ---
    w_full = 0.5 * (w_new[..., :-1] + w_new[..., 1:])
    nlev = theta_total.shape[-1]
    if nlev > 2:
        dz_half_val = height_coord.dz_half  # (nlev-1,)
        dz_centered = dz_half_val[:-1] + dz_half_val[1:]  # (nlev-2,)
        inner_grad = (theta_total[..., :-2] - theta_total[..., 2:]) / dz_centered
        top_grad = (theta_total[..., 0:1] - theta_total[..., 1:2]) / dz_half_val[0]
        bottom_grad = (
            theta_total[..., -2:-1] - theta_total[..., -1:]
        ) / dz_half_val[-1]
        dtheta_dz = jnp.concatenate([top_grad, inner_grad, bottom_grad], axis=-1)
    else:
        dtheta_dz = jnp.zeros_like(theta_total)

    theta_p_new = theta_p_c - dt_s * w_full / J[..., None] * dtheta_dz

    return (w_new, theta_p_new, rho_p_new)


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

    The per-substep vertical algebra is factored into
    :func:`_acoustic_column_kernel` so it can be reused by other dycores
    (currently the future plane dycore in the CRM rollout, PR2b).

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
    J = terrain_metric.jacobian

    w = state.w.data
    theta_p = state.theta_prime.data
    rho_p = state.rho_prime.data

    beta = euler_config.acoustic_off_centering

    def substep_body(i, carry):
        w_c, theta_p_c, rho_p_c = carry
        return _acoustic_column_kernel(
            w_c, theta_p_c, rho_p_c,
            height_coord, J, dt_s, beta, g,
        )

    # Python-loop unroll (n_substeps is compile-time static via
    # SplitExplicitConfig). See semi-implicit variant for full rationale.
    w_final, theta_p_final, rho_p_final = (w, theta_p, rho_p)
    for _i in range(int(n_substeps)):
        w_final, theta_p_final, rho_p_final = substep_body(
            _i, (w_final, theta_p_final, rho_p_final),
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
# Semi-implicit acoustic substeps (tridiagonal)
# ==============================================================================

def precompute_si_tridiag_bands(
    height_coord: HeightCoordinate,
    J: jax.Array,
    dt_s: float,
    g: float,
    implicit_buoyancy: bool = False,
    nlev: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Loop-invariant pieces of the semi-implicit acoustic tridiag system.

    ``alpha``, ``a_tri``, ``b_tri``, ``c_tri`` and the implicit-buoyancy
    band additions only depend on ``dt_s``, ``height_coord``, ``J``, and
    ``g`` — they do NOT depend on the substep carry ``(w, theta_p,
    rho_p)``. Callers running a multi-substep ``jax.lax.fori_loop``
    can call this once outside the loop and pass the result into
    :func:`_semi_implicit_acoustic_column_kernel` via
    ``precomputed_tridiag`` to skip recomputing them every iteration.

    Contract
    --------
    The returned ``(a_tri, b_tri, c_tri)`` are tied to the EXACT
    ``(dt_s, height_coord, J, g, implicit_buoyancy)`` passed here.
    When forwarded to :func:`_semi_implicit_acoustic_column_kernel`,
    the same ``dt_s``, ``height_coord``, ``J``, ``g`` MUST be passed
    to the kernel (used for the RHS / backward updates). When
    ``precomputed_tridiag is not None`` the kernel IGNORES its own
    ``implicit_buoyancy`` flag because the bands already encode it —
    so callers should not mix bands built with ``implicit_buoyancy=A``
    and a kernel call with ``implicit_buoyancy=B≠A`` expecting B to
    take effect. The mismatch is silent at runtime.

    Returns
    -------
    (a_tri, b_tri, c_tri) — each broadcastable to ``(*spatial, nlev-1)``.
    """
    c_p = constants.c_pd
    R_d = constants.R_d
    c_v = constants.c_vd
    dz = height_coord.dz
    dz_half = height_coord.dz_half
    theta_0 = height_coord.theta_ref
    gamma = c_p / c_v
    T_ref = theta_0 * height_coord.exner_ref
    cs2 = gamma * R_d * T_ref
    cs2_half = 0.5 * (cs2[:-1] + cs2[1:])
    dz_inner = 0.5 * (dz[:-1] + dz[1:])
    theta_0_half = 0.5 * (theta_0[:-1] + theta_0[1:])
    if nlev is None:
        nlev = int(theta_0.shape[-1])

    alpha = dt_s ** 2 * cs2_half / (dz_inner * J[..., None]) ** 2
    pad_axes_a = ((0, 0),) * (alpha.ndim - 1)
    a_tri = jnp.pad(-alpha[..., 1:], (*pad_axes_a, (1, 0)))
    alpha_interior = jnp.pad(alpha[..., 1:-1], (*pad_axes_a, (1, 1)))
    b_tri = 1.0 + alpha + alpha_interior
    c_tri = jnp.pad(-alpha[..., :-1], (*pad_axes_a, (0, 1)))

    if implicit_buoyancy and nlev > 2:
        dz_centered = dz_half[:-1] + dz_half[1:]
        inner_grad_ref = (theta_0[:-2] - theta_0[2:]) / dz_centered
        top_grad_ref = (theta_0[0:1] - theta_0[1:2]) / dz_half[0]
        bottom_grad_ref = (theta_0[-2:-1] - theta_0[-1:]) / dz_half[-1]
        dtheta_ref_dz = jnp.concatenate(
            [top_grad_ref, inner_grad_ref, bottom_grad_ref], axis=-1,
        )
        kappa = 0.25 * dt_s ** 2 * g / (theta_0_half * J[..., None])
        d_above = dtheta_ref_dz[:-1]
        d_below = dtheta_ref_dz[1:]
        a_buoy_full = kappa * d_above
        b_buoy_full = kappa * (d_above + d_below)
        c_buoy_full = kappa * d_below
        a_buoy = jnp.pad(a_buoy_full[..., 1:], (*pad_axes_a, (1, 0)))
        c_buoy = jnp.pad(c_buoy_full[..., :-1], (*pad_axes_a, (0, 1)))
        a_tri = a_tri + a_buoy
        b_tri = b_tri + b_buoy_full
        c_tri = c_tri + c_buoy

    return (a_tri, b_tri, c_tri)


def _semi_implicit_acoustic_column_kernel(
    w_c: jax.Array,
    theta_p_c: jax.Array,
    rho_p_c: jax.Array,
    height_coord: HeightCoordinate,
    J: jax.Array,
    dt_s: float,
    beta: float,
    g: float,
    implicit_buoyancy: bool = False,
    precomputed_tridiag: tuple[jax.Array, jax.Array, jax.Array] | None = None,
    si_w_vertical_filter_nu: float = 0.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Single semi-implicit acoustic substep (column-local algebra).

    Layout-agnostic counterpart of :func:`_acoustic_column_kernel`.
    Treats the vertical pressure-gradient term in the w equation
    implicitly via a per-column tridiagonal Thomas solve so the
    vertical acoustic CFL constraint is lifted.

    Inputs follow the same contract as the explicit kernel:
    ``[..., nlev+1]`` w (rigid lid/bottom), ``[..., nlev]`` theta'+rho',
    ``J`` broadcastable to the horizontal leading axes.

    Boundary contract (post iter-69)
    --------------------------------
    Unlike :func:`_acoustic_column_kernel` (explicit) which preserves
    the input boundary values ``w_c[..., 0]`` and ``w_c[..., -1]``,
    this kernel **overwrites** them with 0 (rigid lid/bottom BC) on
    output. Callers MUST already obey the rigid BC on entry; this
    enforcement is a fusion optimization, not new physics. If a future
    layout exposes a nonzero-w boundary (e.g., moving bottom), revert
    to the explicit kernel's ``at[..., 1:-1].set(...)`` pattern.

    When ``implicit_buoyancy=True`` the buoyancy contribution
    ``g * theta_p_half / theta_0_half`` in the w-equation is treated
    implicitly by substituting the backward theta'-update into the
    buoyancy term (Klemp-Wilhelmson 1978). This augments the
    tridiagonal system with three nearest-neighbour bands proportional
    to the mean-state stratification ``dtheta_ref/dz`` and closes the
    w<->theta gravity-wave feedback that otherwise grows at coarse
    vertical resolution.
    """
    c_p = constants.c_pd
    R_d = constants.R_d
    c_v = constants.c_vd
    dz = height_coord.dz
    dz_half = height_coord.dz_half
    theta_0 = height_coord.theta_ref
    rho_0 = height_coord.rho_ref
    gamma = c_p / c_v
    T_ref = theta_0 * height_coord.exner_ref           # (nlev,)
    cs2 = gamma * R_d * T_ref                          # (nlev,)
    cs2_half = 0.5 * (cs2[:-1] + cs2[1:])              # (nlev-1,)
    dz_inner = 0.5 * (dz[:-1] + dz[1:])                # (nlev-1,)
    nlev = theta_p_c.shape[-1]

    theta_total, rho_total = sanitize_theta_rho(
        theta_0 + theta_p_c, rho_0 + rho_p_c,
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
    rhs = w_c[..., 1:-1] + dt_s * dw_dt_inner

    # --- Tridiagonal coefficients for implicit w solve ---
    # Hoisted fast path: callers running a multi-substep loop can pass
    # the precomputed (a_tri, b_tri, c_tri) — they are loop-invariant
    # (depend only on dt_s, height_coord, J, g) — to skip the rebuild
    # every iteration. See :func:`precompute_si_tridiag_bands`.
    if precomputed_tridiag is not None:
        a_tri, b_tri, c_tri = precomputed_tridiag
    else:
        alpha = dt_s**2 * cs2_half / (dz_inner * J[..., None])**2
        pad_axes_a = ((0, 0),) * (alpha.ndim - 1)
        a_tri = jnp.pad(-alpha[..., 1:], (*pad_axes_a, (1, 0)))
        alpha_interior = jnp.pad(alpha[..., 1:-1], (*pad_axes_a, (1, 1)))
        b_tri = 1.0 + alpha + alpha_interior
        c_tri = jnp.pad(-alpha[..., :-1], (*pad_axes_a, (0, 1)))

        if implicit_buoyancy and nlev > 2:
            # Mean-state d(theta_ref)/dz at full levels (sign convention
            # matches the backward theta'-update used downstream).
            dz_centered = dz_half[:-1] + dz_half[1:]
            inner_grad_ref = (theta_0[:-2] - theta_0[2:]) / dz_centered
            top_grad_ref = (theta_0[0:1] - theta_0[1:2]) / dz_half[0]
            bottom_grad_ref = (theta_0[-2:-1] - theta_0[-1:]) / dz_half[-1]
            dtheta_ref_dz = jnp.concatenate(
                [top_grad_ref, inner_grad_ref, bottom_grad_ref], axis=-1,
            )  # shape (nlev,)
            # kappa at interior half-levels k_int=0..nlev-2, (..., nlev-1).
            kappa = 0.25 * dt_s ** 2 * g / (theta_0_half * J[..., None])
            d_above = dtheta_ref_dz[:-1]
            d_below = dtheta_ref_dz[1:]
            a_buoy_full = kappa * d_above
            b_buoy_full = kappa * (d_above + d_below)
            c_buoy_full = kappa * d_below
            # Boundary handling: drop sub-diag at k_int=0, super-diag at -1.
            a_buoy = jnp.pad(a_buoy_full[..., 1:], (*pad_axes_a, (1, 0)))
            c_buoy = jnp.pad(c_buoy_full[..., :-1], (*pad_axes_a, (0, 1)))
            a_tri = a_tri + a_buoy
            b_tri = b_tri + b_buoy_full
            c_tri = c_tri + c_buoy

    w_inner_new = thomas_solve_batched(a_tri, b_tri, c_tri, rhs)
    # Rigid lid/bottom: w=0 at top and bottom interfaces. Padding with
    # 0 fuses better than `w_c.at[..., 1:-1].set(...)` because pad is a
    # simple HLO op without the dynamic-update-slice fusion barrier on
    # the cuSPARSE custom-call output.
    pad_axes_w = ((0, 0),) * (w_inner_new.ndim - 1)
    w_new = jnp.pad(w_inner_new, (*pad_axes_w, (1, 1)))

    # SI-stability filter: vertical Laplacian damping on w. Defaults
    # off (nu=0.0). Empirical fix for the structural exponential mode
    # that grows in the SI scheme with perturbed theta' IC — see
    # CompressibleEulerConfig.si_w_vertical_filter_nu docstring.
    if si_w_vertical_filter_nu > 0.0:
        w_above = w_new[..., :-2]
        w_below = w_new[..., 2:]
        w_interior = w_new[..., 1:-1]
        w_filt = w_interior + si_w_vertical_filter_nu * (
            w_above - 2.0 * w_interior + w_below
        )
        w_new = w_new.at[..., 1:-1].set(w_filt)

    # --- Backward: update rho' using continuity ---
    rho_half = 0.5 * (rho_total[..., :-1] + rho_total[..., 1:])
    rho_w = jnp.pad(rho_half * w_inner_new, (*pad_axes_w, (1, 1)))
    vert_div = (rho_w[..., :-1] - rho_w[..., 1:]) / dz
    vert_div = vert_div / J[..., None]
    rho_p_new = rho_p_c - dt_s * vert_div
    if beta != 0.0:
        rho_p_new = (1.0 + beta) * rho_p_new - beta * rho_p_c

    # --- Backward: update theta' using w-advection of theta_total ---
    w_full = 0.5 * (w_new[..., :-1] + w_new[..., 1:])
    if nlev > 2:
        dz_centered = dz_half[:-1] + dz_half[1:]
        inner_grad = (theta_total[..., :-2] - theta_total[..., 2:]) / dz_centered
        top_grad = (theta_total[..., 0:1] - theta_total[..., 1:2]) / dz_half[0]
        bottom_grad = (theta_total[..., -2:-1] - theta_total[..., -1:]) / dz_half[-1]
        dtheta_dz = jnp.concatenate([top_grad, inner_grad, bottom_grad], axis=-1)
    else:
        dtheta_dz = jnp.zeros_like(theta_total)
    theta_p_new = theta_p_c - dt_s * w_full / J[..., None] * dtheta_dz
    return (w_new, theta_p_new, rho_p_new)


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

    Boundary contract (post iter-69): the substep body overwrites
    ``w[..., 0]`` and ``w[..., -1]`` with 0 (rigid lid/bottom BC) on
    every iteration via ``jnp.pad(w_inner_new, ..., (1, 1))``. Callers
    MUST obey the rigid BC on input. This is a fusion optimization, not
    new physics; revert to ``at[..., 1:-1].set`` if a moving boundary
    is ever introduced.

    Substep loop unroll (post iter-70): ``n_substeps`` MUST be a
    Python ``int`` (it always is when passed through ``SplitExplicitConfig``).
    The function uses a plain Python ``for _ in range(n_substeps)`` to
    fully unroll the substep sequence so XLA can fuse across iterations.
    Passing a traced ``n_substeps`` (e.g., from ``lax.cond``) will fail
    at trace time with ``TracerIntegerConversionError`` — that error is
    the correct guard, do not silence it with ``int(...)``.

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
    J = terrain_metric.jacobian  # (*spatial,) — cubed-sphere (6,n,n), latlon (ny,nx), etc.
    beta = euler_config.acoustic_off_centering
    implicit_buoyancy = euler_config.implicit_buoyancy

    # Extract mutable arrays
    w = state.w.data       # (..., nlev+1)
    theta_p = state.theta_prime.data  # (..., nlev)
    rho_p = state.rho_prime.data      # (..., nlev)

    nlev = theta_p.shape[-1]

    # Loop-invariant pieces: dz_inner used in the explicit RHS dpi/dz,
    # and the tridiag bands (alpha + buoyancy) shared across substeps.
    # Hoisted out of the fori_loop via precompute_si_tridiag_bands.
    # NOTE: dz_inner is also recomputed inside precompute_si_tridiag_bands;
    # the recomputation is a 5-character expression and XLA folds it. The
    # helper consumes it for alpha; we use it here for dpi/dz in the RHS.
    dz_inner = 0.5 * (dz[:-1] + dz[1:])  # (nlev-1,)
    a_tri_pre, b_tri_pre, c_tri_pre = precompute_si_tridiag_bands(
        height_coord, J, dt_s, g, implicit_buoyancy, nlev=nlev,
    )

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
        rhs = w_c[..., 1:-1] + dt_s * dw_dt_inner

        # Tridiag system reused across substeps (loop-invariant).
        a_tri, b_tri, c_tri = a_tri_pre, b_tri_pre, c_tri_pre

        # Solve tridiagonal system
        w_inner_new = thomas_solve_batched(a_tri, b_tri, c_tri, rhs)

        # Construct full w_new via pad-with-0 (rigid lid/bottom BC),
        # avoiding the dynamic-update-slice fusion barrier that the
        # `w_c.at[..., 1:-1].set(...)` pattern emits on top of the
        # cuSPARSE custom-call output.
        pad_axes_w = ((0, 0),) * (w_inner_new.ndim - 1)
        w_new = jnp.pad(w_inner_new, (*pad_axes_w, (1, 1)))

        # --- Backward: update rho' using updated w ---
        # ``rho_w`` has zero at top/bottom interfaces (rigid lid / rigid
        # bottom).  Single Pad HLO op replaces alloc-zeros + scatter.
        rho_half = 0.5 * (rho_total[..., :-1] + rho_total[..., 1:])
        rho_w = jnp.pad(rho_half * w_inner_new, (*pad_axes_w, (1, 1)))
        vert_div = (rho_w[..., :-1] - rho_w[..., 1:]) / dz
        vert_div = vert_div / J[..., None]
        rho_p_new = rho_p_c - dt_s * vert_div

        # Off-centering: damp acoustic mode (Skamarock & Klemp 2008).
        # Python-guard when beta=0 (default config) — skips a kernel
        # in the substep tail that XLA may not fully fold.
        if beta != 0.0:
            rho_p_new = (1.0 + beta) * rho_p_new - beta * rho_p_c

        # --- Backward: update theta' using vertical w advection ---
        # ``dtheta_dz`` is zero at top/bottom (one-sided would require
        # ghost cells); centred difference fills the interior.  Single
        # Pad HLO op replaces alloc-zeros + scatter.
        w_full = 0.5 * (w_new[..., :-1] + w_new[..., 1:])
        if nlev > 2:
            dz_centered = dz_half[:-1] + dz_half[1:]
            inner_grad = (theta_total[..., :-2] - theta_total[..., 2:]) / dz_centered
            top_grad = (theta_total[..., 0:1] - theta_total[..., 1:2]) / dz_half[0]
            bottom_grad = (
                theta_total[..., -2:-1] - theta_total[..., -1:]
            ) / dz_half[-1]
            dtheta_dz = jnp.concatenate(
                [top_grad, inner_grad, bottom_grad], axis=-1,
            )
        else:
            dtheta_dz = jnp.zeros_like(theta_total)
        theta_p_new = theta_p_c - dt_s * w_full / J[..., None] * dtheta_dz

        return (w_new, theta_p_new, rho_p_new)

    # Python-loop unroll: n_substeps is compile-time static so XLA can
    # fuse the post-cuSPARSE tail of one substep with the pre-cuSPARSE
    # head of the next. lax.fori_loop kept the substeps as a while-loop
    # and prevented inter-iteration fusion.
    w_final, theta_p_final, rho_p_final = (w, theta_p, rho_p)
    for _i in range(int(n_substeps)):
        w_final, theta_p_final, rho_p_final = substep_body(
            _i, (w_final, theta_p_final, rho_p_final),
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
