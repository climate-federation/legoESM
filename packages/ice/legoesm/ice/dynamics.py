"""Sea ice dynamics: EVP momentum solver and stress divergence.

Solves the ice momentum equation using Elastic-Viscous-Plastic (EVP)
subcycling (Hunke & Dukowicz 1997):

    m du/dt = tau_air + tau_ocean - mfk x u + div(sigma)

where m = rho_ice * h is ice mass per unit area, tau_air and tau_ocean
are wind and ocean drag stresses, f is the Coriolis parameter,
and sigma is the internal stress tensor from the VP/EVP rheology.

The sea-surface-tilt / ocean-pressure-gradient force ``- m g grad(eta_ocean)``
(H&D97 eq. 2; CICE ``strtltx``) is implemented as an OPTIONAL ``-g grad(eta)``
acceleration on both the EVP and mEVP velocity updates (``ssh_grad_x/y``,
default None -> no tilt).  Without it, drift is biased under strong SSH
gradients (e.g. the Beaufort Gyre) because only the geostrophic-current part
enters, via ``tau_ocean``.
# ponytail: the ice-side term + its unit test are in place; the remaining wire
# is the coupler passing the prognostic ocean SSH gradient at the ice C-grid
# (a 3D-ocean-coupling follow-up).  Until then ssh_grad defaults None (no-op).

The EVP solver subcycles N_evp times per dynamical timestep. Each
subcycle updates stress via elastic relaxation toward the VP solution,
then advances velocity semi-implicitly.

Uses ``jax.lax.fori_loop`` (production) or ``jax.lax.scan``
(differentiable) following the barotropic substep pattern.

All functions are JAX-compatible (differentiable, JIT-friendly).

References
----------
- Hunke, E. C. & Dukowicz, J. K. (1997): An elastic-viscous-plastic model
  for sea ice dynamics. J. Phys. Oceanogr., 27, 1849-1867.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.core.field import Field
from legoesm.core.operators import gradient_x, gradient_y
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.halo_latlon import pad_halo_latlon
from legoesm.grids.latlon import LatLonGrid
from legoesm.grids.voronoi import VoronoiMesh
from legoesm.ice.rheology import (
    ice_strength,
    strain_rates,
    evp_stress_update,
    mevp_stress_update,
    cell_gradient_voronoi,
)
from legoesm.ice.config import SeaIceConfig

# Canonical defaults for the dynamics solver kwargs live on ``SeaIceConfig`` (the
# single source of truth callers pass in); the function signatures below default
# to these so no empirical literal is buried in a signature. C_ai/C_oi are the
# air-ice / ocean-ice drag coefficients (= drag_atm / drag_ocean).
_DYN_DEFAULTS = SeaIceConfig()
# Loop-iteration COUNTS are fixed module constants, never config/trainable
# parameters: an integer subcycle count is a discrete control-flow / static
# value, cannot be differentiated, and must not be exposed to a tuning or
# backprop loop (it would also force a recompile if traced). The EVP and mEVP
# solvers share the same default subcycle count.
_N_EVP_DEFAULT = 120
# mEVP pseudo-time relaxation parameters (Bouillon 2013 / Kimmritz 2015) — solver
# numerics not carried on SeaIceConfig; declared here so they are not inline
# signature literals. Stability requires alpha, beta >= ~2*N_mevp.
_MEVP_ALPHA_DEFAULT = 500.0
_MEVP_BETA_DEFAULT = 500.0
# Ice-area presence threshold [-]: cells with concentration below this carry no
# dynamics (distinct from the h_ice_min thickness floor).
_ICE_PRESENCE_THRESHOLD = 0.01


def _is_latlon_grid(grid) -> bool:
    return isinstance(grid, LatLonGrid)


def _is_voronoi_mesh(grid) -> bool:
    return isinstance(grid, VoronoiMesh)


def _grid_coriolis(grid) -> jnp.ndarray:
    """Cell-centered Coriolis parameter array for any supported grid."""
    return grid.grid_coriolis


# ==============================================================================
# Raw-array operator wrappers (avoid Field allocation in fori_loop)
# ==============================================================================

def _gradient_x_raw(data: jnp.ndarray, grid: CubedSphereGrid) -> jnp.ndarray:
    f = Field(data=data, name="f", dims=("face", "x", "y"), units="")
    return gradient_x(f, grid).data


def _gradient_y_raw(data: jnp.ndarray, grid: CubedSphereGrid) -> jnp.ndarray:
    f = Field(data=data, name="f", dims=("face", "x", "y"), units="")
    return gradient_y(f, grid).data


# ==============================================================================
# Stress divergence
# ==============================================================================

def _stress_divergence_cubed_sphere(
    sigma_11: jnp.ndarray,
    sigma_22: jnp.ndarray,
    sigma_12: jnp.ndarray,
    grid: CubedSphereGrid,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Stress divergence on the cubed sphere."""
    Fx = _gradient_x_raw(sigma_11, grid) + _gradient_y_raw(sigma_12, grid)
    Fy = _gradient_x_raw(sigma_12, grid) + _gradient_y_raw(sigma_22, grid)
    return Fx, Fy


def _stress_divergence_latlon(
    sigma_11: jnp.ndarray,
    sigma_22: jnp.ndarray,
    sigma_12: jnp.ndarray,
    grid: LatLonGrid,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Stress divergence on a lat-lon A-grid with spherical metric (F10).

    Centered finite differences in local-Cartesian coordinates, with
    periodic-lon + pole-fold halo, giving the ``(1/h)∂/∂`` gradient terms.
    The divergence of a symmetric tensor on a sphere (h1 = r cosθ, h2 = r)
    additionally carries spherical-METRIC terms from the Christoffel symbols:

        F_x = ds11_dx + ds12_dy - (2 σ_12 tanθ) / r
        F_y = ds12_dx + ds22_dy + ((σ_11 - σ_22) tanθ) / r

    where ds11_dx = (1/(r cosθ)) ∂σ_11/∂λ etc.  The metric terms were
    previously omitted (consistent with the old metric-free strain rate);
    they are now included to match the metric-aware ``_strain_rates_latlon``
    so the momentum balance is correct away from the equator.

    All three stress components are folded with the scalar pole halo: at the
    pole the local (east, north) basis rotates 180°, under which
    σ_11 → σ_11, σ_22 → σ_22, σ_12 → σ_12 (all even), so the scalar halo has
    the correct parity.  (Velocity ``u``, ``v`` are odd — see
    ``_strain_rates_latlon``.)

    The metric uses the EXACT ``tanθ`` (matching ``_strain_rates_latlon``) on
    every cell-center row — not clipped — so the strain-rate and stress-
    divergence metric coefficients are identical (work-conjugate).  Grids avoid
    the exact pole; ``step_sea_ice`` rejects pole-reaching lat-lon grids under
    EVP/mEVP.  The residual missing tripolar fold means very-high-latitude
    lat-lon EVP should be cross-checked against the cubed-sphere / MPAS backends.
    """
    s11_pad = pad_halo_latlon(sigma_11, halo=1)
    s22_pad = pad_halo_latlon(sigma_22, halo=1)
    s12_pad = pad_halo_latlon(sigma_12, halo=1)
    dx = grid.dx
    dy = grid.dy[:, None]

    ds11_dx = (s11_pad[1:-1, 2:] - s11_pad[1:-1, :-2]) / dx
    ds12_dy = (s12_pad[2:, 1:-1] - s12_pad[:-2, 1:-1]) / dy
    ds12_dx = (s12_pad[1:-1, 2:] - s12_pad[1:-1, :-2]) / dx
    ds22_dy = (s22_pad[2:, 1:-1] - s22_pad[:-2, 1:-1]) / dy

    # Spherical-metric coefficient tanθ / r, computed as sinθ/(r cosθ) with
    # |cosθ| floored at 1e-12 ONLY at the exact pole (never binds on real
    # cell-centered grids -> exact tanθ; NaN-safety, identical to the strain
    # rate so the two stay work-conjugate).
    cos_lat = jnp.cos(grid.lat)
    cos_safe = jnp.where(jnp.abs(cos_lat) < 1e-12, 1e-12, cos_lat)
    metric = (jnp.sin(grid.lat) / cos_safe / grid.radius)[:, None]   # (n_lat, 1)

    Fx = ds11_dx + ds12_dy - 2.0 * sigma_12 * metric
    Fy = ds12_dx + ds22_dy + (sigma_11 - sigma_22) * metric
    return Fx, Fy


def _stress_divergence_voronoi(
    sigma_11: jnp.ndarray,
    sigma_22: jnp.ndarray,
    sigma_12: jnp.ndarray,
    mesh: VoronoiMesh,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Stress divergence on a Voronoi mesh via Green-Gauss cell gradients.

    Treats σ_ij as scalars at cell centers in the local east-north
    basis (same convention as the lat-lon path; valid because a
    180° basis rotation leaves all three components even).

        F_x = ∂σ_11/∂x + ∂σ_12/∂y
        F_y = ∂σ_12/∂x + ∂σ_22/∂y
    """
    ds11_dx, _ = cell_gradient_voronoi(sigma_11, mesh)
    ds12_dx, ds12_dy = cell_gradient_voronoi(sigma_12, mesh)
    _, ds22_dy = cell_gradient_voronoi(sigma_22, mesh)
    Fx = ds11_dx + ds12_dy
    Fy = ds12_dx + ds22_dy
    return Fx, Fy


def stress_divergence(
    sigma_11: jnp.ndarray,
    sigma_22: jnp.ndarray,
    sigma_12: jnp.ndarray,
    grid,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Compute the divergence of the internal stress tensor.

    Grid-agnostic dispatcher:
        * ``CubedSphereGrid`` → cubed-sphere FD gradients (treats σ
          as local scalar; cross-face rotation incurs O(dx) error
          at coarse resolution, damped by EVP elastic relaxation).
        * ``LatLonGrid`` → A-grid centered FD with periodic-lon +
          pole-fold halo.
        * ``VoronoiMesh`` → Green-Gauss cell-centered gradient.

    F_x = ∂σ_11/∂x + ∂σ_12/∂y, F_y = ∂σ_12/∂x + ∂σ_22/∂y.
    """
    if _is_voronoi_mesh(grid):
        return _stress_divergence_voronoi(sigma_11, sigma_22, sigma_12, grid)
    if _is_latlon_grid(grid):
        return _stress_divergence_latlon(sigma_11, sigma_22, sigma_12, grid)
    if isinstance(grid, CubedSphereGrid):
        return _stress_divergence_cubed_sphere(sigma_11, sigma_22, sigma_12, grid)
    raise TypeError(
        f"unsupported grid {type(grid).__name__} for stress_divergence"
    )


# ==============================================================================
# Free-drift velocity
# ==============================================================================

def free_drift_velocity(
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    wind_u: jnp.ndarray,
    wind_v: jnp.ndarray,
    drag_ocean: float = _DYN_DEFAULTS.drag_ocean,
    drag_atm: float = _DYN_DEFAULTS.drag_atm,
    rho_air: float = constants.rho_air,
    rho_ocean: float = constants.rho_ocean,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Diagnostic ice velocity from the linearised free-drift balance.

    Steady-state, no-Coriolis, no-internal-stress momentum balance:

        ρ_air · C_ai · |U_a - u_i| (U_a - u_i)
            = ρ_oc  · C_oi · |u_i - U_w| (u_i - U_w)

    Approximating ``|U_a - u_i| ≈ |U_a - U_w|`` in the slow-ice limit
    gives the Zubov-style estimate

        u_i = U_w + α · (U_a - U_w),
        α   = sqrt(ρ_air · C_ai / (ρ_oc · C_oi)).

    With the CICE-default drag coefficients (``C_ai = 1.3e-3``,
    ``C_oi = 5.5e-3``) and densities (``ρ_air ≈ 1.225``, ``ρ_oc ≈ 1025``)
    this yields ``α ≈ 0.017`` — the canonical "2 % of wind" Nansen /
    Zubov drift rule (Leppäranta 2011, eq. 6.42).

    The previous implementation linearly combined ``drag_ocean * U_w +
    (drag_atm * ρ_air / ρ_ice) * U_a``, which treats the dimensionless
    drag coefficients as if they were velocity-mapping ratios and uses
    ρ_ice in the denominator instead of ρ_oc.  That formula produced
    ice drift roughly two orders of magnitude smaller than the physical
    Zubov estimate and is **not** a defensible placeholder for
    diagnostic use.

    Parameters
    ----------
    ocean_u, ocean_v : arrays
        Ocean surface currents [m/s].
    wind_u, wind_v : arrays
        Atmospheric wind [m/s].
    drag_ocean : float
        Ocean-ice drag coefficient ``C_oi`` [-].
    drag_atm : float
        Air-ice drag coefficient ``C_ai`` [-].
    rho_air : float
        Reference dry-air density [kg/m^3].
    rho_ocean : float
        Reference seawater density [kg/m^3].

    Returns
    -------
    u_ice, v_ice : arrays
        Diagnostic ice velocity [m/s].
    """
    alpha = jnp.sqrt(rho_air * drag_atm / (rho_ocean * drag_ocean))
    u_ice = ocean_u + alpha * (wind_u - ocean_u)
    v_ice = ocean_v + alpha * (wind_v - ocean_v)
    return u_ice, v_ice


# ==============================================================================
# Air and ocean stress on ice
# ==============================================================================

def air_ice_stress(
    u_ice: jnp.ndarray,
    v_ice: jnp.ndarray,
    wind_u: jnp.ndarray,
    wind_v: jnp.ndarray,
    rho_air: float = constants.rho_air,
    C_ai: float = _DYN_DEFAULTS.drag_atm,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute air-ice drag stress.

    tau_ai = rho_air * C_ai * |u_wind - u_ice| * (u_wind - u_ice)

    Parameters
    ----------
    u_ice, v_ice : arrays
        Ice velocity [m/s].
    wind_u, wind_v : arrays
        Atmospheric wind [m/s].
    rho_air : float
    C_ai : float
        Air-ice drag coefficient.

    Returns
    -------
    tau_x, tau_y : arrays
        Air stress on ice [N/m^2].
    """
    du = wind_u - u_ice
    dv = wind_v - v_ice
    speed = jnp.sqrt(du ** 2 + dv ** 2 + 1e-10)
    tau_x = rho_air * C_ai * speed * du
    tau_y = rho_air * C_ai * speed * dv
    return tau_x, tau_y


def ocean_ice_stress(
    u_ice: jnp.ndarray,
    v_ice: jnp.ndarray,
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    rho_ocean: float = constants.rho_ocean,
    C_oi: float = _DYN_DEFAULTS.drag_ocean,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute ocean-ice drag stress.

    tau_oi = rho_ocean * C_oi * |u_ocean - u_ice| * (u_ocean - u_ice)

    Parameters
    ----------
    u_ice, v_ice : arrays
        Ice velocity [m/s].
    ocean_u, ocean_v : arrays
        Ocean surface currents [m/s].
    rho_ocean : float
    C_oi : float
        Ocean-ice drag coefficient.

    Returns
    -------
    tau_x, tau_y : arrays
        Ocean stress on ice [N/m^2].
    """
    du = ocean_u - u_ice
    dv = ocean_v - v_ice
    speed = jnp.sqrt(du ** 2 + dv ** 2 + 1e-10)
    tau_x = rho_ocean * C_oi * speed * du
    tau_y = rho_ocean * C_oi * speed * dv
    return tau_x, tau_y


# ==============================================================================
# EVP solver
# ==============================================================================

def evp_solver(
    u_ice: jnp.ndarray,
    v_ice: jnp.ndarray,
    sigma_11: jnp.ndarray,
    sigma_22: jnp.ndarray,
    sigma_12: jnp.ndarray,
    h_ice: jnp.ndarray,
    concentration: jnp.ndarray,
    wind_u: jnp.ndarray,
    wind_v: jnp.ndarray,
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    grid,
    dt: float,
    N_evp: int = _N_EVP_DEFAULT,
    e_yield: float = _DYN_DEFAULTS.e_yield,
    P_star: float = _DYN_DEFAULTS.P_star,
    C_strength: float = _DYN_DEFAULTS.C_strength,
    T_evp: float = _DYN_DEFAULTS.T_evp,
    Delta_min: float = _DYN_DEFAULTS.Delta_min,
    rho_ice: float = constants.rho_ice,
    rho_air: float = constants.rho_air,
    rho_ocean: float = constants.rho_ocean,
    C_ai: float = _DYN_DEFAULTS.drag_atm,
    C_oi: float = _DYN_DEFAULTS.drag_ocean,
    differentiable: bool = False,
    h_ice_min: float = _DYN_DEFAULTS.h_ice_min,
    ssh_grad_x: jnp.ndarray | None = None,
    ssh_grad_y: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run the EVP subcycled momentum solver.

    Subcycles N_evp times per dynamical timestep dt. Each subcycle:
    1. Compute strain rates from current velocity
    2. Update stress via EVP relaxation
    3. Compute stress divergence
    4. Advance velocity semi-implicitly with Coriolis

    Parameters
    ----------
    u_ice, v_ice : arrays (6, n, n)
        Initial ice velocity [m/s].
    sigma_11, sigma_22, sigma_12 : arrays (6, n, n)
        Initial stress tensor [N/m].
    h_ice : array (6, n, n)
        Ice thickness [m].
    concentration : array (6, n, n)
        Ice concentration [0-1].
    wind_u, wind_v : arrays (6, n, n)
        Atmospheric wind [m/s].
    ocean_u, ocean_v : arrays (6, n, n)
        Ocean surface currents [m/s].
    grid : CubedSphereGrid
    dt : float
        Full dynamical timestep [s].
    N_evp : int
        Number of EVP subcycles.
    e_yield : float
        Yield curve eccentricity.
    P_star, C_strength : float
        Ice strength parameters.
    T_evp : float
        EVP damping timescale ratio.
    Delta_min : float
        Deformation-rate regulariser [1/s] threaded into the EVP
        stress-update — keeps the plastic yield curve smooth at zero
        deformation.  Default 2e-9 (CICE convention).
    rho_ice, rho_air, rho_ocean : float
        Densities [kg/m^3].
    C_ai, C_oi : float
        Drag coefficients.
    differentiable : bool
        Use scan (True) or fori_loop (False).

    Returns
    -------
    u_new, v_new : arrays (6, n, n)
        Updated ice velocity [m/s].
    sigma_11_new, sigma_22_new, sigma_12_new : arrays (6, n, n)
        Updated stress tensor [N/m].
    """
    # Fail-early validation of the ITERATION parameters only (mirrors
    # mevp_solver, which guards N_mevp/alpha/beta — not the physics scalars).
    # N_evp is a loop count and T_evp is iteration-coupled, so both are always
    # STATIC Python values → boolean control flow is doctrine-permitted here.
    # ``e_yield`` / ``Delta_min`` are TUNABLE physics params that can arrive as
    # JAX tracers via the trainable-override path under jit/grad, so they MUST
    # NOT enter a Python ``np.isfinite`` check (it would fail at trace time).
    if N_evp < 1:
        raise ValueError(f"evp_solver: N_evp must be >= 1, got {N_evp}.")
    if not np.isfinite(T_evp) or T_evp <= 0.0:
        raise ValueError(
            f"evp_solver: T_evp must be a finite scalar > 0; got {T_evp}.")

    dt_s = dt / N_evp  # subcycle timestep

    # Ice mass per unit ice-covered area: rho_ice · h.  In the CICE
    # equation of motion, both the wind/ocean stress AND the mass scale
    # by concentration A: m_grid · du/dt = A · tau_a + A · tau_o + ...,
    # which simplifies to rho_ice · h · du/dt = tau_a + tau_o + ...
    # because air_ice_stress / ocean_ice_stress already return stress
    # per unit ice-covered area (no A factor).  Including concentration
    # in m_ice without also weighting the stresses introduces a 1/A
    # over-acceleration in the marginal-ice zone.  An earlier audit
    # iteration claimed the original ``rho_ice · max(h, 0.01)`` was
    # missing concentration weighting; Codex GPT-5 review caught the
    # bookkeeping mistake, and the original form is correct.
    m_ice = rho_ice * jnp.maximum(h_ice, h_ice_min)

    # Ice strength (constant during subcycling)
    P = ice_strength(h_ice, concentration, P_star, C_strength)

    # Coriolis parameter at cell centers
    f = _grid_coriolis(grid).astype(u_ice.dtype)
    alpha = 0.5 * f * dt_s

    # Ice mask: only compute dynamics where ice exists
    ice_mask = concentration > _ICE_PRESENCE_THRESHOLD

    def substep_body(i, carry):
        u_c, v_c, s11_c, s22_c, s12_c = carry

        # 1. Strain rates from current velocity
        eps_11, eps_22, eps_12 = strain_rates(u_c, v_c, grid)

        # 2. EVP stress update
        s11_new, s22_new, s12_new = evp_stress_update(
            s11_c, s22_c, s12_c,
            eps_11, eps_22, eps_12,
            P, e_yield, T_evp, dt_s, N_evp, Delta_min,
        )

        # 3. Stress divergence
        Fx, Fy = stress_divergence(s11_new, s22_new, s12_new, grid)

        # 4. External forces (recomputed with current velocity).  Air
        #    stress stays explicit.  Ocean drag is split (Hunke &
        #    Dukowicz 1997 / CICE ``stepu``): the magnitude ``vrel`` is
        #    evaluated at the OLD velocity u^n (explicit), but the linear
        #    u^{n+1} factor is treated IMPLICITLY by folding ``vrel`` into
        #    the 2x2 diagonal below.  This removes the explicit stability
        #    bound dt_s < 2 m / vrel that blows up in thin-ice MIZ cells.
        tau_air_x, tau_air_y = air_ice_stress(
            u_c, v_c, wind_u, wind_v, rho_air, C_ai,
        )
        du_ocn = ocean_u - u_c
        dv_ocn = ocean_v - v_c
        # vrel = rho_ocean · C_oi · |u_ocn − u^n|  [kg m^-2 s^-1].  The
        # 1e-10 floor keeps the sqrt gradient finite at zero shear
        # (AD-safe), matching ocean_ice_stress.
        vrel = rho_ocean * C_oi * jnp.sqrt(
            du_ocn ** 2 + dv_ocn ** 2 + 1e-10)

        # 5. Explicit force per unit mass = air stress + internal-stress
        #    divergence ONLY (ocean drag enters implicitly in step 6).
        ax = (tau_air_x + Fx) / m_ice
        ay = (tau_air_y + Fy) / m_ice
        # Sea-surface-tilt / ocean-pressure-gradient force (H&D97 eq. 2; CICE
        # ``strtltx``): ``-m g grad(eta)`` per unit area -> ``-g grad(eta)`` per
        # unit mass (mass cancels).  ``eta`` is the ocean SSH; the ice drifts
        # DOWN the sea-surface slope.  ``ssh_grad_*`` default None (no tilt)
        # until the coupler plumbs ocean SSH to the ice C-grid; supplied as a
        # prescribed gradient it is exercised directly (unit test).  The
        # ``is not None`` check is a static feature gate (not a traced branch).
        if ssh_grad_x is not None:
            ax = ax - constants.g * ssh_grad_x
        if ssh_grad_y is not None:
            ay = ay - constants.g * ssh_grad_y

        # 6. Semi-implicit velocity update.  Sign convention (per-area
        #    momentum, k = up):
        #      m (u^{n+1}-u^n)/dt_s = tau_air + vrel (u_ocn - u^{n+1})
        #                             + F - m f k×u  (+f v in u-eq,
        #                             -f u in v-eq; Coriolis unchanged).
        #    The implicit drag adds r = dt_s vrel / m_ice to the diagonal:
        #      D u^{n+1} - alpha v^{n+1} = rhs_u
        #      alpha u^{n+1} + D v^{n+1} = rhs_v
        #    with D = 1 + r, alpha = 0.5 f dt_s, det = D^2 + alpha^2.
        #    r = 0 recovers the original Coriolis-only solve exactly.
        drag = dt_s * vrel / m_ice
        D = 1.0 + drag
        rhs_u = u_c + dt_s * ax + drag * ocean_u + alpha * v_c
        rhs_v = v_c + dt_s * ay + drag * ocean_v - alpha * u_c
        det = D ** 2 + alpha ** 2
        u_new = (D * rhs_u + alpha * rhs_v) / det
        v_new = (D * rhs_v - alpha * rhs_u) / det

        # Zero velocity / stress where no ice.  Cast the boolean mask
        # to float once and multiply — fuses naturally with the
        # following stage and gives gradients a smooth zero (vs. the
        # branchless ``select`` that ``jnp.where`` lowers to).
        ice_mask_f = ice_mask.astype(u_new.dtype)
        u_new = u_new * ice_mask_f
        v_new = v_new * ice_mask_f
        s11_new = s11_new * ice_mask_f
        s22_new = s22_new * ice_mask_f
        s12_new = s12_new * ice_mask_f

        return (u_new, v_new, s11_new, s22_new, s12_new)

    init_carry = (u_ice, v_ice, sigma_11, sigma_22, sigma_12)

    if differentiable:
        def scan_body(carry, _):
            new_carry = substep_body(0, carry)
            return new_carry, None

        (u_f, v_f, s11_f, s22_f, s12_f), _ = jax.lax.scan(
            scan_body, init_carry, xs=None, length=N_evp,
        )
    else:
        u_f, v_f, s11_f, s22_f, s12_f = jax.lax.fori_loop(
            0, N_evp, substep_body, init_carry,
        )

    return u_f, v_f, s11_f, s22_f, s12_f


# ==============================================================================
# mEVP solver
# ==============================================================================

def mevp_solver(
    u_ice: jnp.ndarray,
    v_ice: jnp.ndarray,
    sigma_11: jnp.ndarray,
    sigma_22: jnp.ndarray,
    sigma_12: jnp.ndarray,
    h_ice: jnp.ndarray,
    concentration: jnp.ndarray,
    wind_u: jnp.ndarray,
    wind_v: jnp.ndarray,
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    grid,
    dt: float,
    N_mevp: int = _N_EVP_DEFAULT,
    e_yield: float = _DYN_DEFAULTS.e_yield,
    P_star: float = _DYN_DEFAULTS.P_star,
    C_strength: float = _DYN_DEFAULTS.C_strength,
    alpha_mevp: float = _MEVP_ALPHA_DEFAULT,
    beta_mevp: float = _MEVP_BETA_DEFAULT,
    Delta_min: float = _DYN_DEFAULTS.Delta_min,
    rho_ice: float = constants.rho_ice,
    rho_air: float = constants.rho_air,
    rho_ocean: float = constants.rho_ocean,
    C_ai: float = _DYN_DEFAULTS.drag_atm,
    C_oi: float = _DYN_DEFAULTS.drag_ocean,
    differentiable: bool = False,
    h_ice_min: float = _DYN_DEFAULTS.h_ice_min,
    ssh_grad_x: jnp.ndarray | None = None,
    ssh_grad_y: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""Run the modified-EVP pseudo-time momentum solver.

    Iterates ``N_mevp`` pseudo-time steps toward the implicit VP solution
    using the Bouillon (2013) / Kimmritz (2015) update with two
    relaxation parameters ``alpha`` (stress) and ``beta`` (velocity):

        σ^(p+1) = (1 − 1/α) σ^p + (1/α) σ_VP(ε(u^p))

        (β + 1) m u^(p+1) + Δt · m f k × u^(p+1)
            = β m u^p + m u^n + Δt · (∇·σ^(p+1) + τ_a + τ_o)

    where the air- and ocean-stress are evaluated at the current
    pseudo-time velocity ``u^p`` (Picard iteration) and the Coriolis
    term is treated implicitly on ``u^(p+1)``.  At convergence
    (``u^(p+1) ≈ u^p``) the discrete momentum balance with implicit VP
    rheology is recovered, irrespective of ``alpha`` and ``beta``.

    Compared with the Hunke & Dukowicz (1997) EVP path (``evp_solver``):
        * No subcycle timestep ``dt_s = dt / N``: the relaxation is in
          pseudo-time, not physical time, so the elastic-wave CFL
          constraint (``c_E · dt_s / dx ≤ 1``) does not apply.
        * No ``T_evp`` damping ratio: ``alpha`` and ``beta`` directly
          set the relaxation rate.  Larger values → slower per-iteration
          relaxation but better stability margin.
        * No N-dependence in the per-iteration relaxation: doubling
          ``N_mevp`` halves the residual roughly, rather than changing
          the per-iteration scaling.

    Parameters
    ----------
    u_ice, v_ice : arrays (6, n, n)
        Initial ice velocity ``u^n`` at the start of the dynamic step
        [m/s].  Held fixed during the pseudo-time iteration.
    sigma_11, sigma_22, sigma_12 : arrays (6, n, n)
        Initial stress tensor [N/m].
    h_ice : array (6, n, n)
        Ice thickness [m].
    concentration : array (6, n, n)
        Ice concentration [0-1].
    wind_u, wind_v : arrays (6, n, n)
        Atmospheric wind [m/s].
    ocean_u, ocean_v : arrays (6, n, n)
        Ocean surface currents [m/s].
    grid : CubedSphereGrid
    dt : float
        Full dynamical timestep [s].
    N_mevp : int
        Number of mEVP pseudo-time iterations.
    e_yield : float
        Yield curve eccentricity.
    P_star, C_strength : float
        Ice strength parameters.
    alpha_mevp : float
        mEVP stress relaxation parameter.  Stability requires
        ``alpha · beta ≥ (e²/4) · γ²``; default 500 matches CICE / FESOM
        recommendations for hourly Δt on coarse-to-medium grids.
    beta_mevp : float
        mEVP velocity relaxation parameter.  Typically set equal to
        ``alpha`` (Kimmritz 2015).
    Delta_min : float
        Deformation-rate regulariser [1/s].
    rho_ice, rho_air, rho_ocean : float
        Densities [kg/m^3].
    C_ai, C_oi : float
        Drag coefficients.
    differentiable : bool
        Use scan (True) or fori_loop (False).

    Returns
    -------
    u_new, v_new : arrays (6, n, n)
        Updated ice velocity [m/s].
    sigma_11_new, sigma_22_new, sigma_12_new : arrays (6, n, n)
        Updated stress tensor [N/m].

    Raises
    ------
    ValueError
        If ``N_mevp`` < 1, ``alpha_mevp`` < 1 (extrapolation past the VP
        target), or ``beta_mevp`` <= 0 (degenerate velocity update —
        ``alpha * beta = 0`` cannot satisfy the Kimmritz stability bound
        for any nonzero ice viscosity).

    Notes
    -----
    Configuration parameters ``N_mevp``, ``alpha_mevp``, ``beta_mevp``,
    ``e_yield``, ``P_star``, ``C_strength``, ``Delta_min``, ``rho_*``,
    ``C_*``, and ``differentiable`` are **static** Python scalars
    (they appear in the iteration count, the relaxation factors and
    the validation branches).  Pass them as Python ``int``/``float``
    constants; they must NOT be JAX-traced arrays.  When wrapping the
    solver in ``jax.jit`` declare them with ``static_argnames`` so the
    Python branches do not run on tracers.

    The Kimmritz stability bound ``alpha * beta >= (e_yield**2 / 4) *
    gamma**2`` with ``gamma`` proportional to ``zeta_max * dt / (m * L**2)``
    is **not** enforced automatically.  ``gamma`` depends on the realised
    bulk viscosity ``zeta = P / (2 Delta)`` which varies in space and time,
    so a static check would be either overly conservative or unreliable.
    Users running fine grids, large ``dt``, or thin marginal ice
    (small ``m``) should raise ``alpha_mevp`` and ``beta_mevp`` and verify
    convergence by comparing two runs with doubled ``N_mevp``.

    The ``differentiable=False`` ``fori_loop`` path matches the EVP
    convention: production runs prefer ``fori_loop`` for compile-time
    speed; training / VJP workflows must set ``differentiable=True`` so
    the iteration becomes a ``lax.scan`` with native reverse-mode AD.
    """
    if N_mevp < 1:
        raise ValueError(
            f"mevp_solver: N_mevp must be >= 1, got {N_mevp}."
        )
    if not np.isfinite(alpha_mevp) or alpha_mevp < 1.0:
        raise ValueError(
            f"mevp_solver: alpha_mevp must be a finite scalar >= 1; "
            f"got {alpha_mevp}.  NaN/inf silently poisons the stress "
            f"update; alpha<1 extrapolates past the VP target."
        )
    if not np.isfinite(beta_mevp) or beta_mevp <= 0.0:
        raise ValueError(
            f"mevp_solver: beta_mevp must be a finite scalar > 0; got "
            f"{beta_mevp}.  beta_mevp=0 makes alpha*beta=0 (Kimmritz "
            f"bound violated) and degenerates the velocity update to "
            f"explicit Euler; NaN/inf silently poisons the iteration."
        )

    # Per-area ice mass.  See evp_solver for the bookkeeping note —
    # the bulk-stress functions already return stress per unit
    # ice-covered area, so m = rho_ice · h is the correct scaling.
    m_ice = rho_ice * jnp.maximum(h_ice, h_ice_min)

    # Ice strength is held fixed during the pseudo-time relaxation
    # (depends only on the start-of-step h, A).
    P = ice_strength(h_ice, concentration, P_star, C_strength)

    # Implicit Coriolis AND implicit ocean drag: solve the 2x2 system
    #   A · u^(p+1) − B · v^(p+1) = rhs_u
    #   B · u^(p+1) + A · v^(p+1) = rhs_v
    # with B = Δt · f (Coriolis) and A = β + 1 + Δt·vrel/m (base mEVP
    # damping β+1 plus the IMPLICIT ocean-drag rate; Hunke & Dukowicz
    # 1997 / CICE ``stepu``).  ``vrel`` (the drag magnitude) is evaluated
    # at the current pseudo-velocity u^p and so ``A`` and the determinant
    # are recomputed inside the subcycle body below.
    f = _grid_coriolis(grid).astype(u_ice.dtype)
    A_cor = beta_mevp + 1.0
    B_cor = dt * f

    # u^n (held fixed during pseudo-time iteration)
    u_n = u_ice
    v_n = v_ice

    # Ice mask — only update where ice exists
    ice_mask = concentration > _ICE_PRESENCE_THRESHOLD

    def substep_body(i, carry):
        u_p, v_p, s11_p, s22_p, s12_p = carry

        # 1. Strain rates from pseudo-time velocity
        eps_11, eps_22, eps_12 = strain_rates(u_p, v_p, grid)

        # 2. mEVP stress update toward VP target
        s11_new, s22_new, s12_new = mevp_stress_update(
            s11_p, s22_p, s12_p,
            eps_11, eps_22, eps_12,
            P, e_yield, alpha_mevp, Delta_min,
        )

        # 3. Stress divergence
        Fx, Fy = stress_divergence(s11_new, s22_new, s12_new, grid)

        # 4. External stresses at u^p (Picard linearisation).  Air stress
        #    stays explicit; ocean drag is split — magnitude ``vrel`` at
        #    the current pseudo-velocity u^p (explicit), linear u^(p+1)
        #    factor IMPLICIT via the 2x2 diagonal below.
        tau_air_x, tau_air_y = air_ice_stress(
            u_p, v_p, wind_u, wind_v, rho_air, C_ai,
        )
        du_ocn = ocean_u - u_p
        dv_ocn = ocean_v - v_p
        # vrel = rho_ocean · C_oi · |u_ocn − u^p|  [kg m^-2 s^-1]; 1e-10
        # floor keeps the sqrt gradient finite (AD-safe).
        vrel = rho_ocean * C_oi * jnp.sqrt(
            du_ocn ** 2 + dv_ocn ** 2 + 1e-10)

        # 5. Explicit RHS forcing / unit mass = air stress + internal
        #    stress divergence ONLY (ocean drag enters implicitly below).
        ax = (tau_air_x + Fx) / m_ice
        ay = (tau_air_y + Fy) / m_ice
        # Sea-surface-tilt force ``-g grad(eta)`` per unit mass (H&D97 eq. 2;
        # CICE ``strtltx``); default None -> no tilt until the coupler plumbs
        # ocean SSH.  Static feature gate (not a traced branch).
        if ssh_grad_x is not None:
            ax = ax - constants.g * ssh_grad_x
        if ssh_grad_y is not None:
            ay = ay - constants.g * ssh_grad_y

        # 6. mEVP velocity update (Kimmritz 2015 eq. 8) with implicit
        #    Coriolis AND implicit ocean drag.  Sign convention (k = up):
        #    beta(u^(p+1)-u^p) + (u^(p+1)-u^n) = dt/m [tau_air + F
        #      + vrel(u_ocn - u^(p+1))] - dt f k×u^(p+1); Coriolis +f v
        #    in u-eq, -f u in v-eq (unchanged).  drag = dt·vrel/m adds to
        #    the diagonal A; drag = 0 recovers the Coriolis-only solve.
        drag = dt * vrel / m_ice
        A_drag = A_cor + drag
        inv_det = 1.0 / (A_drag ** 2 + B_cor ** 2)
        rhs_u = beta_mevp * u_p + u_n + dt * ax + drag * ocean_u
        rhs_v = beta_mevp * v_p + v_n + dt * ay + drag * ocean_v
        u_new = (A_drag * rhs_u + B_cor * rhs_v) * inv_det
        v_new = (-B_cor * rhs_u + A_drag * rhs_v) * inv_det

        # Zero in ice-free cells (smooth-grad multiply, as in evp_solver)
        ice_mask_f = ice_mask.astype(u_new.dtype)
        u_new = u_new * ice_mask_f
        v_new = v_new * ice_mask_f
        s11_new = s11_new * ice_mask_f
        s22_new = s22_new * ice_mask_f
        s12_new = s12_new * ice_mask_f

        return (u_new, v_new, s11_new, s22_new, s12_new)

    init_carry = (u_ice, v_ice, sigma_11, sigma_22, sigma_12)

    if differentiable:
        def scan_body(carry, _):
            new_carry = substep_body(0, carry)
            return new_carry, None

        (u_f, v_f, s11_f, s22_f, s12_f), _ = jax.lax.scan(
            scan_body, init_carry, xs=None, length=N_mevp,
        )
    else:
        u_f, v_f, s11_f, s22_f, s12_f = jax.lax.fori_loop(
            0, N_mevp, substep_body, init_carry,
        )

    return u_f, v_f, s11_f, s22_f, s12_f
