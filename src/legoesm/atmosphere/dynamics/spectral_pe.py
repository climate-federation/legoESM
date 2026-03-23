"""Spectral Primitive Equation Model using spherical harmonic transforms.

Solves the hydrostatic primitive equations on the sphere using the
vorticity-divergence formulation with pseudospectral (transform) method
in sigma-pressure coordinates.

Prognostic variables (in spectral space):
    vor_hat  : Relative vorticity SH coefficients, (n_sh, nlev)
    div_hat  : Divergence SH coefficients, (n_sh, nlev)
    T_hat    : Temperature SH coefficients, (n_sh, nlev)
    lnps_hat : Log(surface pressure) SH coefficients, (n_sh,)

Equations (vorticity-divergence form, Bourke 1972):
    d(vor)/dt  = -div((vor+f)*v) + curl(vert_adv)
    d(div)/dt  = curl((vor+f)*v) - lap(K + Phi + R_d*T*lnps) + div(vert_adv)
    d(T)/dt    = -div(T*v) + T*div(v) - sigma_dot*dT/dsigma + kappa*T*omega/p
    d(lnps)/dt = -integral(div*dsigma)

References
----------
- Bourke, W. (1972). An Efficient, One-Level, Primitive-Equation Spectral
  Model. Monthly Weather Review, 100, 683-689.
- Hack, J. J. & Jakob, R. (1992). Description of a Global Shallow Water
  Model Based on the Spectral Transform Method. NCAR TN-343+STR.
- Hoskins, B. J. & Simmons, A. J. (1975). A multi-layer spectral model
  and the semi-implicit method. Quart. J. R. Met. Soc., 101, 637-655.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.grids.gaussian import (
    GaussianGrid,
    sh_analysis,
    sh_synthesis,
    sh_analysis_3d,
    sh_synthesis_3d,
    sh_analysis_oc2_3d,
    sh_analysis_dmu_3d,
    uv_from_vordiv_3d,
    spectral_hyperdiffusion_3d,
    _sh_synthesis_H,
)
from legoesm.grids.vertical import (
    SigmaCoordinate,
    HybridSigmaPressureCoordinate,
    pressure_from_hybrid,
    dp_from_hybrid,
    compute_geopotential_hybrid,
    compute_mass_flux_hybrid,
    vertical_advection_hybrid,
    compute_omega_hybrid,
)
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm import constants

_LNPS_MIN = float(jnp.log(100.0))
_LNPS_MAX = float(jnp.log(2.0e6))
_COS_LAT_MIN = 1.0e-6


# =============================================================================
# State and config
# =============================================================================

class SpectralHydrostaticState(NamedTuple):
    """State for the spectral hydrostatic primitive equations.

    3D spectral fields: shape (n_sh, nlev) complex128
    2D spectral fields: shape (n_sh,) complex128
    """
    vor_hat: Field    # Spectral relative vorticity [1/s]
    div_hat: Field    # Spectral divergence [1/s]
    T_hat: Field      # Spectral temperature [K]
    lnps_hat: Field   # Spectral log(surface pressure) [-]
    phis_hat: Field   # Spectral surface geopotential [m^2/s^2] (static)


class SpectralPEConfig(NamedTuple):
    """Configuration for spectral primitive equation model."""
    g: float = constants.g
    hyperdiff_coeff: float = 0.0
    hyperdiff_order: int = 2
    time_integrator: str = "ssp_rk3"  # "ssp_rk3", "ssp_rk34", or "ssp_rk54"
    semi_implicit: bool = False      # Use Hoskins-Simmons semi-implicit
    si_T_ref: float = 300.0         # Reference temperature for linearization [K]
    si_alpha: float = 0.5           # Implicitness (0.5 = Crank-Nicolson)
    si_substeps: int = 1            # Internal SI substeps per external model step
    si_hyperdiff_boost: float = 1.0  # Multiply hyperdiffusion in SI mode
    # Sponge layer (implicit multiplicative filter at model top)
    sponge_sigma: float = 0.1       # Sigma below which sponge is active (damps above)
    sponge_tau: float = 0.0         # E-folding time at model top [s] (0 = off)
    # Level-dependent hyperdiffusion scaling (stronger at low pressures)
    hyperdiff_pscale: float = 0.0   # Power-law exponent: nu_k = nu * (p_ref/p_k)^exp (0 = off)
    # Temperature floor (positivity protection)
    T_min: float = 50.0             # Minimum temperature [K]
    # Post-step spectral filter (damps highest wavenumbers)
    spectral_filter_order: int = 8   # Sharpness of spectral filter
    spectral_filter_strength: float = 0.0  # Retention at n=n_max (0=off, 0.01=aggressive)
    # Tendency truncation to prevent aliasing from cubic nonlinearities.
    dealiasing_fraction: float = 0.667  # 2/3 rule for cubic nonlinearities
    # Implicit (multiplicative) hyperdiffusion.  Applied as a post-step
    # filter: coeff_new = coeff_old * exp(-nu * [n(n+1)/a^2]^p * dt).
    # This is UNCONDITIONALLY STABLE, unlike explicit (tendency-based)
    # hyperdiffusion which is unstable with leapfrog time integration.
    # Set implicit_hyperdiff=True to use this instead of explicit.
    implicit_hyperdiff: bool = False
    # Pressure floor for adiabatic heating (limits 1/p at model top)
    p_floor: float = 10.0           # Pa; adiabatic uses max(p, p_floor) to prevent omega/p overflow
    # Robert-Asselin filter for leapfrog (controls computational mode)
    robert_asselin_coeff: float = 0.05  # Filter coefficient (0 = off, 0.05-0.1 typical)


# =============================================================================
# Internal vertical helpers (generic shapes, no cubed-sphere assumptions)
# =============================================================================

def _compute_geopotential_gaussian(T, p_s, sigma_coord, phis):
    """Simmons-Burridge geopotential on Gaussian grid.

    Same math as vertical.compute_geopotential but with generic
    broadcasting: T is (..., nlev), p_s is (...), phis is (...).
    """
    R_d = constants.R_d
    ln_ratio = sigma_coord.ln_ratio   # (nlev,)
    alpha = sigma_coord.alpha         # (nlev,)

    dPhi = R_d * T * ln_ratio         # broadcast: (..., nlev)

    dPhi_reversed = dPhi[..., ::-1]
    cumsum_reversed = jnp.cumsum(dPhi_reversed, axis=-1)
    cumsum = cumsum_reversed[..., ::-1]

    Phi_above = phis[..., None] + cumsum

    Phi_below = jnp.concatenate(
        [Phi_above[..., 1:], phis[..., None]], axis=-1,
    )

    Phi_full = Phi_below + alpha * R_d * T
    return Phi_full


def _compute_sigma_dot_gaussian(div_3d, sigma_coord):
    """Sigma-dot on arbitrary grid shape. div_3d is (..., nlev)."""
    dsigma = sigma_coord.dsigma
    fractional_sigma = sigma_coord.fractional_sigma

    div_dsigma = div_3d * dsigma
    D_total = jnp.sum(div_dsigma, axis=-1, keepdims=True)
    cumsum_div = jnp.cumsum(div_dsigma, axis=-1)

    sigma_dot_inner = fractional_sigma * D_total - cumsum_div

    shape_2d = div_3d.shape[:-1]
    zero_top = jnp.zeros((*shape_2d, 1))
    sigma_dot = jnp.concatenate([zero_top, sigma_dot_inner], axis=-1)
    sigma_dot = sigma_dot.at[..., -1].set(0.0)
    return sigma_dot


def _vertical_advection_sigma_gaussian(field, sigma_dot, sigma_coord):
    """Vertical advection -sigma_dot * dfield/dsigma (upwind). Generic shapes."""
    sigma_dot_full = 0.5 * (sigma_dot[..., :-1] + sigma_dot[..., 1:])
    dsigma_bwd = sigma_coord.dsigma_full
    df_bwd = jnp.diff(field, axis=-1)

    grad_bwd = jnp.concatenate(
        [jnp.zeros((*field.shape[:-1], 1)),
         df_bwd / dsigma_bwd],
        axis=-1,
    )
    grad_fwd = jnp.concatenate(
        [df_bwd / dsigma_bwd,
         jnp.zeros((*field.shape[:-1], 1))],
        axis=-1,
    )

    grad = jnp.where(sigma_dot_full > 0, grad_bwd, grad_fwd)
    return -sigma_dot_full * grad


def _compute_omega_gaussian(sigma_dot, p_s, dp_s_dt, sigma_coord):
    """Pressure velocity omega = sigma * dp_s/dt + p_s * sigma_dot_full."""
    sigma_full = sigma_coord.sigma_full
    sigma_dot_full = 0.5 * (sigma_dot[..., :-1] + sigma_dot[..., 1:])
    omega = sigma_full * dp_s_dt[..., None] + p_s[..., None] * sigma_dot_full
    return omega


# =============================================================================
# Tendency computation
# =============================================================================

def spectral_pe_tendencies(
    state: SpectralHydrostaticState,
    grid: GaussianGrid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    config: SpectralPEConfig,
    physics_tendency: SpectralHydrostaticState | None = None,
) -> SpectralHydrostaticState:
    """Compute spectral tendencies for the hydrostatic PE.

    Uses the pseudospectral transform method:
    1. Transform prognostic fields to grid space
    2. Compute nonlinear products on grid
    3. Transform products to spectral space
    4. Assemble tendencies using spectral operators

    Returns tendencies in the same pytree structure as state (for SSP-RK3).
    """
    _hybrid = isinstance(sigma_coord, HybridSigmaPressureCoordinate)

    a = grid.radius
    R_d = constants.R_d
    kappa = constants.kappa

    # Dealiasing mask: zero wavenumbers above dealiasing_fraction * n_max
    # to prevent spectral aliasing from cubic nonlinearities.
    if config.dealiasing_fraction > 0:
        n_cut = int(config.dealiasing_fraction * grid.n_max)
        _dealias = jnp.where(grid.ls <= n_cut, 1.0, 0.0)
        _dealias_3d = _dealias[:, None]  # (n_sh, 1) for 3D fields
    else:
        _dealias = None
        _dealias_3d = None

    # --- 1. Transform to grid space ---
    vor = sh_synthesis_3d(grid, state.vor_hat.data)   # (n_lat, n_lon, nlev)
    div = sh_synthesis_3d(grid, state.div_hat.data)
    T = sh_synthesis_3d(grid, state.T_hat.data)
    # Smooth positivity protection (C∞ differentiable, avoids kink in jnp.maximum)
    T = config.T_min + jax.nn.softplus(T - config.T_min)
    lnps_raw = sh_synthesis(grid, state.lnps_hat.data)
    # Smooth two-sided clip via nested softplus
    lnps = _LNPS_MIN + jax.nn.softplus(lnps_raw - _LNPS_MIN)
    lnps = _LNPS_MAX - jax.nn.softplus(_LNPS_MAX - lnps)
    # (n_lat, n_lon)
    phis = sh_synthesis(grid, state.phis_hat.data)

    # --- 2. Velocities ---
    u_cos, v_cos = uv_from_vordiv_3d(
        grid, state.vor_hat.data, state.div_hat.data,
    )  # (n_lat, n_lon, nlev)
    cos_lat_3d = jnp.clip(grid.cos_lat[:, None, None], _COS_LAT_MIN, None)
    u = u_cos / cos_lat_3d
    v = v_cos / cos_lat_3d

    # --- 3. Pressure ---
    p_s = jnp.exp(lnps)
    if _hybrid:
        p_full = pressure_from_hybrid(sigma_coord, p_s)
        dp = dp_from_hybrid(sigma_coord, p_s)
    else:
        p_full = p_s[..., None] * sigma_coord.sigma_full  # (n_lat, n_lon, nlev)

    # --- 4. Geopotential ---
    if _hybrid:
        Phi = compute_geopotential_hybrid(T, p_s, sigma_coord, phis)
    else:
        Phi = _compute_geopotential_gaussian(T, p_s, sigma_coord, phis)

    # --- 5. Kinetic energy ---
    K = 0.5 * (u * u + v * v)

    # --- 6. Absolute vorticity ---
    abs_vor = vor + grid.f[..., None]

    # --- 7. Vertical velocity ---
    if _hybrid:
        mass_flux = compute_mass_flux_hybrid(div, p_s, sigma_coord)
    else:
        sigma_dot = _compute_sigma_dot_gaussian(div, sigma_coord)

    # --- 8. Surface pressure tendency ---
    if _hybrid:
        D_total_p = jnp.sum(div * dp, axis=-1)
        dlnps_dt_grid = -D_total_p / (p_s * sigma_coord.B_range)
        dp_s_dt_grid = p_s * dlnps_dt_grid
    else:
        dsigma = sigma_coord.dsigma
        sigma_top = sigma_coord.sigma_half[0]
        sigma_range = 1.0 - sigma_top
        D_total = jnp.sum(div * dsigma, axis=-1)
        dlnps_dt_grid = -D_total / sigma_range
        dp_s_dt_grid = p_s * dlnps_dt_grid

    # --- 9. Spectral operators ---
    im_over_a = 1j * grid.ms.astype(jnp.float64) / a  # (n_sh,)
    one_over_a = 1.0 / a

    # --- 10. Vorticity fluxes: (zeta+f)*u*cos, (zeta+f)*v*cos ---
    A_vor = abs_vor * u_cos   # (n_lat, n_lon, nlev)
    B_vor = abs_vor * v_cos

    # Spectral divergence of vorticity flux -> dvor/dt
    flux_vor_div = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, A_vor)
        - one_over_a * sh_analysis_dmu_3d(grid, B_vor)
    )  # (n_sh, nlev)

    # Spectral curl of vorticity flux -> ddiv/dt contribution
    flux_vor_curl = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, B_vor)
        + one_over_a * sh_analysis_dmu_3d(grid, A_vor)
    )

    # --- 11. Pressure gradient force (correct form, NOT Bourke E-variable) ---
    # The PGF divergence is: -∇²(K + Φ) - ∇·(R_d·T·∇lnps)
    # Split using T = T_ref + T':
    #   = -∇²(K + Φ) - R_d·T_ref·∇²(lnps) - ∇·(R_d·T'·∇(lnps))
    # The first two terms use spectral Laplacian (exact).
    # The third term is computed as a grid-point product + spectral divergence.
    #
    # NOTE: The Bourke (1972) E-variable form E = K + Φ + R_d·T·lnps
    # is NOT used because -∇²(R_d·T·lnps) ≠ -∇·(R_d·T·∇lnps).
    # The E-variable adds a spurious same-level T→D coupling
    # (-R_d·lnps_0·∇²T') that is unstable when combined with
    # adiabatic heating.
    T_ref = config.si_T_ref
    KPhi = K + Phi
    KPhi_hat = sh_analysis_3d(grid, KPhi)

    # Compute ∇(lnps) on grid (needed for PGF correction and adiabatic)
    dfdlon = sh_synthesis(grid, 1j * grid.ms * state.lnps_hat.data)
    cos_lat_2d = jnp.clip(grid.cos_lat[:, None], _COS_LAT_MIN, None)
    dlnps_dx = dfdlon / (a * cos_lat_2d)
    dfdtheta_cos = _sh_synthesis_H(grid, state.lnps_hat.data)
    dlnps_dy = -dfdtheta_cos / (a * cos_lat_2d)

    # PGF correction: -∇·(R_d·T'·∇lnps) computed as spectral div of grid product
    T_prime_pgf = T - T_ref
    pgf_Fx_cos = R_d * T_prime_pgf * dlnps_dx[..., None] * cos_lat_3d
    pgf_Fy_cos = R_d * T_prime_pgf * dlnps_dy[..., None] * cos_lat_3d
    pgf_correction_hat = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, pgf_Fx_cos)
        - one_over_a * sh_analysis_dmu_3d(grid, pgf_Fy_cos)
    )

    # --- 12. Horizontal tendencies ---
    dvor_hat = -flux_vor_div
    ddiv_hat = (
        flux_vor_curl
        - grid.lap[:, None] * KPhi_hat
        - R_d * T_ref * grid.lap[:, None] * state.lnps_hat.data[:, None]
        - pgf_correction_hat
    )

    # --- 13. Temperature equation ---
    # Horizontal: dT/dt = -v·∇T = -div(T*v) + T*div(v)
    # Reference-temperature subtraction (Simmons & Burridge 1981):
    # For uniform T_ref, -div(T*v) + T*div = -div(T'*v) + T'*div
    # where T' = T - T_ref.  This eliminates the O(T_ref * ε) cancellation
    # error that otherwise destabilises the isothermal rest state.
    T_ref = config.si_T_ref
    T_prime = T - T_ref

    T_prime_u_cos = T_prime * u_cos
    T_prime_v_cos = T_prime * v_cos

    flux_T_div = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, T_prime_u_cos)
        - one_over_a * sh_analysis_dmu_3d(grid, T_prime_v_cos)
    )

    T_prime_div = T_prime * div
    T_prime_div_hat = sh_analysis_3d(grid, T_prime_div)

    dT_hat = -flux_T_div + T_prime_div_hat

    # Vertical advection of T
    if _hybrid:
        vert_adv_T = vertical_advection_hybrid(T, mass_flux, p_s, sigma_coord)
    else:
        vert_adv_T = _vertical_advection_sigma_gaussian(T, sigma_dot, sigma_coord)
    dT_hat = dT_hat + sh_analysis_3d(grid, vert_adv_T)

    # Adiabatic heating: kappa * T * omega / p
    if _hybrid:
        omega = compute_omega_hybrid(mass_flux, p_s, dp_s_dt_grid, sigma_coord)
    else:
        omega = _compute_omega_gaussian(sigma_dot, p_s, dp_s_dt_grid, sigma_coord)
    p_adiab = jnp.maximum(p_full, config.p_floor) if config.p_floor > 0 else p_full
    adiabatic = kappa * T * omega / p_adiab

    # Material derivative correction: kappa * T * v . grad(lnps)
    v_dot_grad_lnps = u * dlnps_dx[..., None] + v * dlnps_dy[..., None]
    adiabatic = adiabatic + kappa * T * v_dot_grad_lnps

    dT_hat = dT_hat + sh_analysis_3d(grid, adiabatic)

    # --- 14. Vertical advection of momentum ---
    if _hybrid:
        vert_adv_u = vertical_advection_hybrid(u, mass_flux, p_s, sigma_coord)
        vert_adv_v = vertical_advection_hybrid(v, mass_flux, p_s, sigma_coord)
    else:
        vert_adv_u = _vertical_advection_sigma_gaussian(u, sigma_dot, sigma_coord)
        vert_adv_v = _vertical_advection_sigma_gaussian(v, sigma_dot, sigma_coord)

    # Convert to spectral vor/div contributions
    vert_u_cos = vert_adv_u * grid.cos_lat[:, None, None]
    vert_v_cos = vert_adv_v * grid.cos_lat[:, None, None]

    # curl(vert_adv) -> dvor_hat
    vert_vor_tend = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, vert_v_cos)
        + one_over_a * sh_analysis_dmu_3d(grid, vert_u_cos)
    )
    # div(vert_adv) -> ddiv_hat
    vert_div_tend = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, vert_u_cos)
        - one_over_a * sh_analysis_dmu_3d(grid, vert_v_cos)
    )

    dvor_hat = dvor_hat + vert_vor_tend
    ddiv_hat = ddiv_hat + vert_div_tend

    # --- 15. Surface pressure tendency (spectral) ---
    dlnps_hat = sh_analysis(grid, dlnps_dt_grid)

    # --- 16. Spectral hyperdiffusion ---
    hyperdiff_coeff = config.hyperdiff_coeff
    if config.semi_implicit and config.si_hyperdiff_boost != 1.0:
        hyperdiff_coeff = hyperdiff_coeff * config.si_hyperdiff_boost

    if hyperdiff_coeff > 0 and not config.implicit_hyperdiff:
        base_diff_vor = spectral_hyperdiffusion_3d(
            grid, state.vor_hat.data, hyperdiff_coeff, config.hyperdiff_order,
        )
        base_diff_div = spectral_hyperdiffusion_3d(
            grid, state.div_hat.data, hyperdiff_coeff, config.hyperdiff_order,
        )
        base_diff_T = spectral_hyperdiffusion_3d(
            grid, state.T_hat.data, hyperdiff_coeff, config.hyperdiff_order,
        )

        if config.hyperdiff_pscale > 0:
            # Level-dependent scaling: (p_ref/p_k)^exponent
            # Stronger diffusion at low pressures (upper atmosphere)
            if _hybrid:
                sigma_full = sigma_coord.A_full + sigma_coord.B_full
            else:
                sigma_full = sigma_coord.sigma_full
            p_ref_sigma = sigma_full[-1]  # near-surface reference
            scale = (p_ref_sigma / jnp.clip(sigma_full, 1e-6, None)) ** config.hyperdiff_pscale
            scale = scale[None, :]  # (1, nlev)
            base_diff_vor = base_diff_vor * scale
            base_diff_div = base_diff_div * scale
            base_diff_T = base_diff_T * scale

        dvor_hat = dvor_hat + base_diff_vor
        ddiv_hat = ddiv_hat + base_diff_div
        dT_hat = dT_hat + base_diff_T

    # --- 17. Add physics tendencies if provided ---
    if physics_tendency is not None:
        dvor_hat = dvor_hat + physics_tendency.vor_hat.data
        ddiv_hat = ddiv_hat + physics_tendency.div_hat.data
        dT_hat = dT_hat + physics_tendency.T_hat.data
        dlnps_hat = dlnps_hat + physics_tendency.lnps_hat.data

    # Apply dealiasing truncation to prevent aliasing instability
    if _dealias_3d is not None:
        dvor_hat = dvor_hat * _dealias_3d
        ddiv_hat = ddiv_hat * _dealias_3d
        dT_hat = dT_hat * _dealias_3d
        dlnps_hat = dlnps_hat * _dealias

    # Return as same pytree structure (for SSP-RK3)
    return SpectralHydrostaticState(
        vor_hat=state.vor_hat.replace(data=dvor_hat),
        div_hat=state.div_hat.replace(data=ddiv_hat),
        T_hat=state.T_hat.replace(data=dT_hat),
        lnps_hat=state.lnps_hat.replace(data=dlnps_hat),
        phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
    )


def _compute_spectral_filter(ls, n_max, order=8, cutoff_fraction=0.65):
    """Compute an exponential spectral filter.

    Applies exp(-alpha * (n/n_max)^order) where alpha is chosen so that
    the filter value at n_max equals cutoff_fraction.

    Parameters
    ----------
    ls : jax.Array, shape (n_sh,)
        Total wavenumber for each spectral coefficient.
    n_max : int
        Maximum wavenumber.
    order : int
        Filter order (higher = sharper cutoff).
    cutoff_fraction : float
        Filter value at n=n_max.

    Returns
    -------
    filter : jax.Array, shape (n_sh,)
        Multiplicative filter in [cutoff_fraction, 1].
    """
    alpha = -jnp.log(cutoff_fraction)
    ratio = ls / n_max
    return jnp.exp(-alpha * ratio**order)


def _compute_sponge_factor(sigma_full, sponge_sigma, sponge_tau, dt):
    """Compute multiplicative sponge damping factor per level.

    Returns exp(-damping_rate * dt) where damping_rate uses a sin² profile
    ramping from zero at sponge_sigma to 1/sponge_tau at sigma=0.
    This is an implicit (unconditionally stable) sponge filter applied
    after each time step.

    Returns shape (nlev,) array of damping factors in [0, 1].
    """
    sponge_arg = jnp.clip(
        (sponge_sigma - sigma_full) / sponge_sigma, 0.0, 1.0,
    )
    damping_rate = jnp.sin(0.5 * jnp.pi * sponge_arg) ** 2 / sponge_tau
    return jnp.exp(-damping_rate * dt)


def _apply_sponge_filter(state, sponge_factor, ms):
    """Apply multiplicative sponge damping to vor, div, and T' at top levels.

    Damps vor and div toward zero.  Damps T perturbations (m != 0 modes)
    toward the zonal mean so that the mean thermal structure is preserved
    but eddy T anomalies are suppressed.

    Parameters
    ----------
    state : SpectralHydrostaticState
    sponge_factor : jax.Array, shape (nlev,)
        Per-level damping factors in [0, 1].
    ms : jax.Array, shape (n_sh,)
        Zonal wavenumber for each spectral coefficient.
    """
    sf = sponge_factor[None, :]  # (1, nlev)

    vor_hat_damped = state.vor_hat.data * sf
    div_hat_damped = state.div_hat.data * sf

    # For temperature, only damp non-zonal modes (m != 0) to preserve
    # the mean thermal stratification
    is_zonal = (ms == 0)[:, None]  # (n_sh, 1) bool
    T_sf = jnp.where(is_zonal, 1.0, sf)  # no damping for m=0
    T_hat_damped = state.T_hat.data * T_sf

    return state._replace(
        vor_hat=state.vor_hat.replace(data=vor_hat_damped),
        div_hat=state.div_hat.replace(data=div_hat_damped),
        T_hat=state.T_hat.replace(data=T_hat_damped),
    )


def _apply_spectral_filter_to_state(state, spectral_filter):
    """Apply exponential spectral filter to all prognostic fields.

    Parameters
    ----------
    state : SpectralHydrostaticState
    spectral_filter : jax.Array, shape (n_sh,)
        Multiplicative filter per spectral coefficient.
    """
    sf_3d = spectral_filter[:, None]  # (n_sh, 1) for 3D fields
    sf_2d = spectral_filter           # (n_sh,) for 2D fields

    return state._replace(
        vor_hat=state.vor_hat.replace(data=state.vor_hat.data * sf_3d),
        div_hat=state.div_hat.replace(data=state.div_hat.data * sf_3d),
        T_hat=state.T_hat.replace(data=state.T_hat.data * sf_3d),
        lnps_hat=state.lnps_hat.replace(data=state.lnps_hat.data * sf_2d),
    )


# =============================================================================
# Model class
# =============================================================================

class SpectralPrimitiveEquationModel:
    """Spectral primitive equation model on the sphere.

    Parameters
    ----------
    grid : GaussianGrid
        Gaussian grid with precomputed SH transform matrices.
    sigma_coord : SigmaCoordinate
        Vertical sigma coordinate.
    config : SpectralPEConfig, optional
        Model configuration.
    """

    def __init__(
        self,
        grid: GaussianGrid,
        sigma_coord: SigmaCoordinate,
        config: SpectralPEConfig | None = None,
        *,
        allow_unsupported_backend: bool = False,
        legoesm_config=None,
    ):
        self.sigma_coord = sigma_coord
        cfg = config or SpectralPEConfig()

        # Force implicit hyperdiffusion for leapfrog integrators
        if cfg.time_integrator in ("leapfrog", "leapfrog_si") and not cfg.implicit_hyperdiff:
            import warnings
            warnings.warn(
                f"Explicit hyperdiffusion is unstable with {cfg.time_integrator} "
                f"time integration. Forcing implicit_hyperdiff=True.",
                stacklevel=2,
            )
            cfg = cfg._replace(implicit_hyperdiff=True)

        # Warn if dealiasing is off
        if cfg.dealiasing_fraction == 0.0:
            import warnings
            warnings.warn(
                "dealiasing_fraction=0.0: spectral aliasing from cubic "
                "nonlinearities is not suppressed. Set dealiasing_fraction=0.667 "
                "for production runs.",
                stacklevel=2,
            )

        self.config = cfg
        self._use_cpu_for_spectral = False
        self._cpu_device = None
        self._default_device = None
        self._si_data = None
        self._si_dt = None
        self._si_data_lf = None  # SI data for leapfrog (dt_eff = 2*dt)
        self._si_dt_lf = None
        self._sponge_factor = None
        self._sponge_dt = None
        # Leapfrog state management
        self._state_prev = None  # Previous time level for leapfrog
        # Precompute spectral filter (time-independent)
        self._spectral_filter = None
        if self.config.spectral_filter_strength > 0:
            self._spectral_filter = _compute_spectral_filter(
                grid.ls if not self._use_cpu_for_spectral else self.grid.ls,
                grid.n_max if not self._use_cpu_for_spectral else self.grid.n_max,
                order=self.config.spectral_filter_order,
                cutoff_fraction=self.config.spectral_filter_strength,
            )

        if self.config.si_substeps < 1:
            raise ValueError(
                f"si_substeps must be >= 1, got {self.config.si_substeps!r}",
            )

        # Precompute implicit hyperdiffusion filter (unconditionally stable)
        self._hyperdiff_filter = None
        self._hyperdiff_filter_dt = None

        if legoesm_config is not None:
            allow_unsupported_backend = bool(
                legoesm_config.get(
                    "atmosphere.spectral.allow_unsupported", False
                )
            )

        from legoesm.core.hardware import get_backend
        backend = get_backend()
        if backend == "METAL":
            self._use_cpu_for_spectral = True
            self._cpu_device = jax.devices("cpu")[0]
            self._default_device = jax.devices()[0]
            self.grid = jax.device_put(grid, self._cpu_device)
        else:
            self.grid = grid
            from legoesm.core.hardware import check_spectral_backend
            check_spectral_backend(
                allow_unsupported=allow_unsupported_backend,
            )

    def _ensure_si_data(self, dt: float):
        """Lazily precompute semi-implicit matrices and refresh when dt changes."""
        if not self.config.semi_implicit:
            return

        from legoesm.timestepping.semi_implicit import precompute_si_matrices

        dt_si = float(dt) / float(self.config.si_substeps)
        if self._si_data is None or self._si_dt != dt_si:
            self._si_data = precompute_si_matrices(
                self.grid, self.sigma_coord,
                T_ref=self.config.si_T_ref,
                alpha=self.config.si_alpha,
                dt=dt_si,
            )
            self._si_dt = dt_si

    def _ensure_sponge_factor(self, dt: float):
        """Lazily precompute sponge damping factors and refresh when dt changes."""
        if self.config.sponge_tau <= 0:
            return
        if self._sponge_factor is not None and self._sponge_dt == dt:
            return

        _hybrid = isinstance(self.sigma_coord, HybridSigmaPressureCoordinate)
        if _hybrid:
            sigma_full = self.sigma_coord.A_full + self.sigma_coord.B_full
        else:
            sigma_full = self.sigma_coord.sigma_full

        self._sponge_factor = _compute_sponge_factor(
            sigma_full, self.config.sponge_sigma, self.config.sponge_tau, dt,
        )
        self._sponge_dt = dt

    def _ensure_hyperdiff_filter(self, dt: float):
        """Lazily precompute implicit hyperdiffusion filter."""
        if not self.config.implicit_hyperdiff or self.config.hyperdiff_coeff <= 0:
            return
        if self._hyperdiff_filter is not None and self._hyperdiff_filter_dt == dt:
            return
        nu = self.config.hyperdiff_coeff
        order = self.config.hyperdiff_order
        # Eigenvalue: -[n(n+1)/a^2]^order
        eig = (self.grid.ls * (self.grid.ls + 1) / self.grid.radius ** 2) ** order
        # For leapfrog, effective dt is 2*dt
        integrator = self.config.time_integrator.lower()
        dt_eff = 2.0 * dt if 'leapfrog' in integrator else dt
        # Multiplicative filter: exp(-nu * eig * dt_eff)
        self._hyperdiff_filter = jnp.exp(-nu * eig * dt_eff)
        self._hyperdiff_filter_dt = dt

    def _apply_implicit_hyperdiff(self, state):
        """Apply implicit (multiplicative) hyperdiffusion filter.

        Divergence gets 2x stronger damping than vorticity and temperature
        to preferentially suppress gravity wave noise from nonlinear
        baroclinic eddy breakdown (standard practice in operational GCMs).
        """
        if self._hyperdiff_filter is None:
            return state
        hf = self._hyperdiff_filter
        hf_3d = hf[:, None]  # (n_sh, 1) for 3D fields
        hf_div = hf ** 2  # stronger damping for divergence
        hf_div_3d = hf_div[:, None]
        return state._replace(
            vor_hat=state.vor_hat.replace(data=state.vor_hat.data * hf_3d),
            div_hat=state.div_hat.replace(data=state.div_hat.data * hf_div_3d),
            T_hat=state.T_hat.replace(data=state.T_hat.data * hf_3d),
            # lnps and phis are NOT diffused (mass conservation)
        )

    def _ensure_si_data_leapfrog(self, dt: float):
        """Precompute SI matrices for leapfrog (dt_eff = 2*dt)."""
        from legoesm.timestepping.semi_implicit import precompute_si_matrices

        dt_eff = 2.0 * float(dt)
        if self._si_data_lf is None or self._si_dt_lf != dt_eff:
            self._si_data_lf = precompute_si_matrices(
                self.grid, self.sigma_coord,
                T_ref=self.config.si_T_ref,
                alpha=self.config.si_alpha,
                dt=dt_eff,
            )
            self._si_dt_lf = dt_eff

    def reset_leapfrog(self):
        """Reset leapfrog state (next step will use Euler startup)."""
        self._state_prev = None

    def _do_step(self, state, dt, tendency_fn):
        """Core step: explicit RK3/RK54 or semi-implicit RK3, then sponge."""
        if self.config.semi_implicit:
            from legoesm.timestepping.semi_implicit import ssp_rk3_step_si
            n_substeps = int(self.config.si_substeps)
            dt_si = dt / float(n_substeps)

            if n_substeps == 1:
                result = ssp_rk3_step_si(
                    state, tendency_fn, dt_si, self._si_data, self.grid,
                )
            else:
                def si_substep(_, s):
                    return ssp_rk3_step_si(
                        s, tendency_fn, dt_si, self._si_data, self.grid,
                    )
                result = jax.lax.fori_loop(0, n_substeps, si_substep, state)
        else:
            result = dispatch_integrator(
                state, tendency_fn, dt, self.config.time_integrator,
            )

        # Apply implicit sponge filter (unconditionally stable)
        if self._sponge_factor is not None:
            result = _apply_sponge_filter(result, self._sponge_factor, self.grid.ms)

        # Apply spectral filter (damps highest wavenumbers)
        if self._spectral_filter is not None:
            result = _apply_spectral_filter_to_state(result, self._spectral_filter)

        return result

    def step(self, state: SpectralHydrostaticState, dt: float, physics_fn=None) -> SpectralHydrostaticState:
        """Advance one time step, optionally with physics forcing.

        Dispatches to leapfrog+SI or SSP-RK3 based on config.time_integrator.
        """
        integrator = self.config.time_integrator.lower()
        if integrator in ("leapfrog", "leapfrog_si"):
            return self._leapfrog_step(state, dt, physics_fn)

        self._ensure_si_data(dt)
        self._ensure_sponge_factor(dt)
        return self._step_jit(state, dt, physics_fn)

    def _leapfrog_step(self, state, dt, physics_fn=None):
        """Leapfrog + SI step with Robert-Asselin filter + implicit diffusion.

        First call: forward Euler + SI (startup).
        Subsequent calls: leapfrog + SI + RA filter + implicit hyperdiffusion.
        """
        self._ensure_sponge_factor(dt)
        self._ensure_hyperdiff_filter(dt)

        if self._state_prev is None:
            # --- First step: forward Euler + SI ---
            self._ensure_si_data(dt)  # SI matrices for dt
            # Also precompute leapfrog SI for next step (avoids stale jit)
            self._ensure_si_data_leapfrog(dt)
            result = self._euler_si_jit(state, dt, physics_fn)
            # Apply sponge and spectral filter
            if self._sponge_factor is not None:
                result = _apply_sponge_filter(result, self._sponge_factor, self.grid.ms)
            if self._spectral_filter is not None:
                result = _apply_spectral_filter_to_state(result, self._spectral_filter)
            # Implicit hyperdiffusion (unconditionally stable)
            result = self._apply_implicit_hyperdiff(result)
            self._state_prev = state
            return result
        else:
            # --- Leapfrog + SI ---
            self._ensure_si_data_leapfrog(dt)
            state_np1 = self._leapfrog_si_jit(
                state, self._state_prev, dt, physics_fn,
            )
            # Apply sponge and spectral filter
            if self._sponge_factor is not None:
                state_np1 = _apply_sponge_filter(
                    state_np1, self._sponge_factor, self.grid.ms,
                )
            if self._spectral_filter is not None:
                state_np1 = _apply_spectral_filter_to_state(
                    state_np1, self._spectral_filter,
                )
            # Implicit hyperdiffusion (unconditionally stable with leapfrog)
            state_np1 = self._apply_implicit_hyperdiff(state_np1)
            # Robert-Asselin filter on time-n state
            gamma = self.config.robert_asselin_coeff
            if gamma > 0:
                from legoesm.timestepping.semi_implicit import robert_asselin_filter
                state_n_filtered, state_np1_filtered = robert_asselin_filter(
                    self._state_prev, state, state_np1, gamma,
                )
            else:
                state_n_filtered = state
                state_np1_filtered = state_np1
            self._state_prev = state_n_filtered
            return state_np1_filtered

    def step_with_physics(
        self,
        state: SpectralHydrostaticState,
        dt: float,
        physics_fn=None,
    ) -> SpectralHydrostaticState:
        """Backward-compatible wrapper for step() with physics."""
        return self.step(state, dt, physics_fn=physics_fn)

    def _leapfrog_step_with_physics(self, state, dt, physics_fn):
        """Backward-compatible wrapper for _leapfrog_step() with physics."""
        return self._leapfrog_step(state, dt, physics_fn=physics_fn)

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def _euler_si_jit(self, state, dt, physics_fn=None):
        """JIT-compiled Euler + SI step (leapfrog startup), optionally with physics."""
        def tendency_fn(s):
            phys = None
            if physics_fn is not None:
                _phys_result = physics_fn(s, self.grid, self.sigma_coord)
                phys = _phys_result[0] if type(_phys_result) is tuple else _phys_result
            return spectral_pe_tendencies(
                s, self.grid, self.sigma_coord, self.config, phys,
            )
        from legoesm.timestepping.semi_implicit import euler_si_step
        return euler_si_step(state, tendency_fn, dt, self._si_data, self.grid)

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def _leapfrog_si_jit(self, state_n, state_nm1, dt, physics_fn=None):
        """JIT-compiled leapfrog + SI step, optionally with physics."""
        def tendency_fn(s):
            phys = None
            if physics_fn is not None:
                _phys_result = physics_fn(s, self.grid, self.sigma_coord)
                phys = _phys_result[0] if type(_phys_result) is tuple else _phys_result
            return spectral_pe_tendencies(
                s, self.grid, self.sigma_coord, self.config, phys,
            )
        from legoesm.timestepping.semi_implicit import leapfrog_si_step
        return leapfrog_si_step(
            state_n, state_nm1, tendency_fn, dt, self._si_data_lf, self.grid,
        )

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def _step_jit(
        self,
        state: SpectralHydrostaticState,
        dt: float,
        physics_fn=None,
    ) -> SpectralHydrostaticState:
        """JIT-compiled inner step (SI matrices already precomputed), optionally with physics."""
        def tendency_fn(s):
            phys = None
            if physics_fn is not None:
                _phys_result = physics_fn(s, self.grid, self.sigma_coord)
                phys = _phys_result[0] if type(_phys_result) is tuple else _phys_result
            return spectral_pe_tendencies(
                s, self.grid, self.sigma_coord, self.config, phys,
            )

        if self._use_cpu_for_spectral:
            state_cpu = jax.device_put(state, self._cpu_device)
            result_cpu = self._do_step(state_cpu, dt, tendency_fn)
            return jax.device_put(result_cpu, self._default_device)

        return self._do_step(state, dt, tendency_fn)

    # Keep old names as aliases for backward compatibility
    _euler_si_physics_jit = _euler_si_jit
    _leapfrog_si_physics_jit = _leapfrog_si_jit
    _step_with_physics_jit = _step_jit

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def _step_on_cpu(
        self,
        state: SpectralHydrostaticState,
        dt: float,
        physics_fn=None,
    ) -> SpectralHydrostaticState:
        """Step on CPU without device transfers, optionally with physics."""
        def tendency_fn(s):
            phys = None
            if physics_fn is not None:
                _phys_result = physics_fn(s, self.grid, self.sigma_coord)
                phys = _phys_result[0] if type(_phys_result) is tuple else _phys_result
            return spectral_pe_tendencies(
                s, self.grid, self.sigma_coord, self.config, phys,
            )
        return self._do_step(state, dt, tendency_fn)

    # Keep old name as alias for backward compatibility
    _step_on_cpu_with_physics = _step_on_cpu

    def integrate(
        self,
        state: SpectralHydrostaticState,
        duration: float,
        dt: float,
        save_every: int = 1,
        physics_fn=None,
    ) -> tuple[SpectralHydrostaticState, list]:
        """Integrate forward for a given duration (Python loop).

        On Metal, batches CPU transfers: transfer state to CPU once,
        run all steps on CPU, then transfer results back to Metal.
        This avoids per-step CPU↔Metal round-trips.
        """
        n_steps = int(duration / dt)
        self._ensure_si_data(dt)
        self._ensure_sponge_factor(dt)

        if self._use_cpu_for_spectral:
            return self._integrate_on_cpu(state, n_steps, dt, save_every, physics_fn)

        trajectory = [state]
        for i in range(n_steps):
            state = self.step(state, dt, physics_fn=physics_fn)
            if (i + 1) % save_every == 0:
                trajectory.append(state)
        return state, trajectory

    def _integrate_on_cpu(self, state, n_steps, dt, save_every, physics_fn=None):
        """Batch integration on CPU: transfer once, not per step."""
        state_cpu = jax.device_put(state, self._cpu_device)
        trajectory_cpu = [state_cpu]

        for i in range(n_steps):
            state_cpu = self._step_on_cpu(state_cpu, dt, physics_fn)
            if (i + 1) % save_every == 0:
                trajectory_cpu.append(state_cpu)

        # Transfer back to Metal
        state_out = jax.device_put(state_cpu, self._default_device)
        trajectory_out = [
            jax.device_put(s, self._default_device) for s in trajectory_cpu
        ]
        return state_out, trajectory_out


# =============================================================================
# Initialization helpers
# =============================================================================

def isothermal_rest_state_spectral(
    grid: GaussianGrid,
    sigma_coord: SigmaCoordinate,
    T_init: float = 300.0,
    p_s_init: float = 1e5,
    phis: jnp.ndarray | None = None,
) -> SpectralHydrostaticState:
    """Create an isothermal rest-state initial condition in spectral space.

    All fields are at rest (zero winds) with uniform temperature and
    uniform surface pressure.

    Parameters
    ----------
    grid : GaussianGrid
        Spectral/Gaussian grid.
    sigma_coord : SigmaCoordinate
        Vertical coordinate.
    T_init : float
        Initial temperature [K].
    p_s_init : float
        Initial surface pressure [Pa].
    phis : jnp.ndarray or None
        Surface geopotential [m^2/s^2], shape (n_lat, n_lon). If None,
        flat terrain is used. When provided, surface pressure is reduced
        hydrostatically: p_s = p_s_init * exp(-phis / (R_d * T_init)).
    """
    from legoesm import constants

    nlev = sigma_coord.n_levels
    n_sh = grid.n_sh

    dims_3d = ("spectral", "level")
    dims_2d = ("spectral",)

    # Zero winds -> zero vorticity and divergence
    vor_hat = jnp.zeros((n_sh, nlev), dtype=jnp.complex128)
    div_hat = jnp.zeros((n_sh, nlev), dtype=jnp.complex128)

    # Uniform temperature: only the n=0,m=0 mode is nonzero
    T_grid = jnp.full((grid.n_lat, grid.n_lon), T_init, dtype=jnp.float64)
    T_hat_2d = sh_analysis(grid, T_grid)  # (n_sh,)
    T_hat = jnp.broadcast_to(T_hat_2d[:, None], (n_sh, nlev)).copy()

    # Surface pressure (hydrostatic adjustment for topography)
    if phis is not None:
        phis_grid = jnp.asarray(phis, dtype=jnp.float64)
        p_s_grid = p_s_init * jnp.exp(-phis_grid / (constants.R_d * T_init))
        lnps_grid = jnp.log(p_s_grid)
    else:
        lnps_grid = jnp.full(
            (grid.n_lat, grid.n_lon), jnp.log(p_s_init), dtype=jnp.float64,
        )
        phis_grid = jnp.zeros((grid.n_lat, grid.n_lon), dtype=jnp.float64)

    lnps_hat = sh_analysis(grid, lnps_grid)

    # Topography in spectral space
    phis_hat_data = sh_analysis(grid, phis_grid)

    return SpectralHydrostaticState(
        vor_hat=Field(data=vor_hat, name="vor_hat", dims=dims_3d, units="1/s"),
        div_hat=Field(data=div_hat, name="div_hat", dims=dims_3d, units="1/s"),
        T_hat=Field(data=T_hat, name="T_hat", dims=dims_3d, units="K"),
        lnps_hat=Field(data=lnps_hat, name="lnps_hat", dims=dims_2d, units=""),
        phis_hat=Field(data=phis_hat_data, name="phis_hat", dims=dims_2d, units="m^2/s^2"),
    )


# =============================================================================
# Baroclinic wave initialization (spectral)
# =============================================================================

def baroclinic_wave_init_spectral(
    grid: GaussianGrid,
    sigma_coord: SigmaCoordinate,
    perturbed: bool = True,
) -> SpectralHydrostaticState:
    """Initialize Jablonowski-Williamson baroclinic wave in spectral space.

    Evaluates the analytic JW06 balanced state on Gaussian grid points,
    then transforms u,v → vorticity/divergence via spectral analysis.

    Parameters
    ----------
    grid : GaussianGrid
        Gaussian grid with SH transform matrices.
    sigma_coord : SigmaCoordinate
        Vertical sigma coordinate.
    perturbed : bool
        If True, add the exponential perturbation to trigger instability.

    Returns
    -------
    SpectralHydrostaticState
        Initial state for the baroclinic wave test.
    """
    import numpy as np
    from legoesm.atmosphere.physics.baroclinic_wave import (
        evaluate_pressure_temperature,
        find_z_for_pressure,
        compute_zonal_wind,
        exponential_perturbation,
        P0,
    )

    nlev = sigma_coord.n_levels
    n_sh = grid.n_sh

    # Grid coordinates as numpy for the analytic solution
    lat_np = np.array(grid.lat)        # (n_lat,)
    lon_np = np.array(grid.lon2d[0])   # (n_lon,) — all rows same longitude
    sigma_full = np.array(sigma_coord.sigma_full)  # (nlev,)

    n_lat = grid.n_lat
    n_lon = grid.n_lon

    # Create 2D lat/lon for each level
    lat_2d = np.broadcast_to(lat_np[:, None], (n_lat, n_lon))
    lon_2d = np.array(grid.lon2d)

    # Allocate 3D fields (n_lat, n_lon, nlev)
    u_3d = np.zeros((n_lat, n_lon, nlev))
    v_3d = np.zeros((n_lat, n_lon, nlev))
    T_3d = np.zeros((n_lat, n_lon, nlev))

    # Compute initial conditions level by level
    for k in range(nlev):
        # Target pressure at this sigma level
        p_target = np.full((n_lat, n_lon), sigma_full[k] * P0)

        # Find height where p(z, lat) = p_target
        z_k = find_z_for_pressure(p_target, lat_2d)

        # Compute temperature at this height
        _, T_k = evaluate_pressure_temperature(z_k, lat_2d)

        # Compute zonal wind from gradient-wind balance
        u_k = compute_zonal_wind(z_k, lat_2d, T_k)

        # Add perturbation if requested
        if perturbed:
            u_k = u_k + exponential_perturbation(lat_2d, lon_2d, z_k)

        u_3d[:, :, k] = u_k
        T_3d[:, :, k] = T_k

    # Convert to JAX float64
    u_jax = jnp.array(u_3d, dtype=jnp.float64)
    v_jax = jnp.array(v_3d, dtype=jnp.float64)
    T_jax = jnp.array(T_3d, dtype=jnp.float64)

    # --- Transform u,v to spectral vorticity/divergence ---
    # The spectral PE uses vorticity = curl(v) and divergence = div(v).
    # From (u, v) on the Gaussian grid, we compute:
    #   vor_hat = curl operator applied to (u*cos, v*cos)
    #   div_hat = div operator applied to (u*cos, v*cos)
    a = grid.radius
    im_over_a = 1j * grid.ms.astype(jnp.float64) / a
    one_over_a = 1.0 / a

    cos_lat_3d = grid.cos_lat[:, None, None]
    u_cos = u_jax * cos_lat_3d
    v_cos = v_jax * cos_lat_3d

    # vor_hat = (im/a) * SH{v*cos/cos^2} + (1/a) * SH_dmu{u*cos}
    vor_hat = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, v_cos)
        + one_over_a * sh_analysis_dmu_3d(grid, u_cos)
    )
    # div_hat = (im/a) * SH{u*cos/cos^2} - (1/a) * SH_dmu{v*cos}
    div_hat = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, u_cos)
        - one_over_a * sh_analysis_dmu_3d(grid, v_cos)
    )

    # Temperature to spectral
    T_hat = sh_analysis_3d(grid, T_jax)

    # Uniform surface pressure (no topography for JW06)
    lnps_grid = jnp.full(
        (n_lat, n_lon), jnp.log(P0), dtype=jnp.float64,
    )
    lnps_hat = sh_analysis(grid, lnps_grid)

    # No topography
    phis_hat = jnp.zeros(n_sh, dtype=jnp.complex128)

    dims_3d = ("spectral", "level")
    dims_2d = ("spectral",)

    return SpectralHydrostaticState(
        vor_hat=Field(data=vor_hat, name="vor_hat", dims=dims_3d, units="1/s"),
        div_hat=Field(data=div_hat, name="div_hat", dims=dims_3d, units="1/s"),
        T_hat=Field(data=T_hat, name="T_hat", dims=dims_3d, units="K"),
        lnps_hat=Field(data=lnps_hat, name="lnps_hat", dims=dims_2d, units=""),
        phis_hat=Field(data=phis_hat, name="phis_hat", dims=dims_2d, units="m^2/s^2"),
    )


# =============================================================================
# Diagnostic utilities
# =============================================================================

def spectral_pe_to_grid(
    state: SpectralHydrostaticState,
    grid: GaussianGrid,
    sigma_coord: SigmaCoordinate,
) -> dict[str, jax.Array]:
    """Convert spectral PE state to grid-point fields for diagnostics.

    Returns
    -------
    dict with keys: 'u', 'v', 'T', 'vor', 'div', 'lnps', 'p_s', 'phis'
    """
    vor = sh_synthesis_3d(grid, state.vor_hat.data)
    div = sh_synthesis_3d(grid, state.div_hat.data)
    T = sh_synthesis_3d(grid, state.T_hat.data)
    lnps = jnp.clip(
        sh_synthesis(grid, state.lnps_hat.data),
        _LNPS_MIN,
        _LNPS_MAX,
    )
    phis = sh_synthesis(grid, state.phis_hat.data)

    u_cos, v_cos = uv_from_vordiv_3d(
        grid, state.vor_hat.data, state.div_hat.data,
    )
    cos_lat_3d = jnp.clip(grid.cos_lat[:, None, None], _COS_LAT_MIN, None)
    u = u_cos / cos_lat_3d
    v = v_cos / cos_lat_3d

    return {
        'u': u, 'v': v, 'T': T,
        'vor': vor, 'div': div,
        'lnps': lnps, 'p_s': jnp.exp(lnps),
        'phis': phis,
    }
