"""Hydrostatic Primitive Equations on the cubed-sphere (A-grid dynamics).

Uses A-grid operators throughout (vorticity, gradients, kinetic energy,
divergence) to avoid the Hollingsworth-type energy inconsistency that
arises when A-grid prognostic winds are converted to D-grid and back.

Cubed-sphere panel edges are treated with variable-specific blending,
analogous to the polar filter used on latitude-longitude grids.

State is stored on the A-grid (cell centres).

References
----------
- Simmons & Burridge (1981): Energy and Angular-Momentum Conserving Scheme
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState, HydrostaticTendencies
from legoesm.core.operators_3d import (
    vorticity_3d as _vorticity_3d,
    gradient_x_3d as _gradient_x_3d,
    gradient_y_3d as _gradient_y_3d,
    divergence_3d as _divergence_3d,
    hyperdiffusion_3d as _hyperdiffusion_3d,
    laplacian_compact_3d as _laplacian_compact_3d,
)
from legoesm.core.operators import gradient_x, gradient_y
from legoesm.core.conservation import zero_mean_tendency
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.cubed_sphere_cdgrid import (
    CubedSphereCDGrid,        # kept for backward-compatible __init__ signature
    create_cubed_sphere_cdgrid,
)
from legoesm.grids.vertical import (
    SigmaCoordinate,
    HybridSigmaPressureCoordinate,
    pressure_from_sigma,
    pressure_from_hybrid,
    dp_from_hybrid,
    compute_geopotential,
    compute_geopotential_hybrid,
    compute_sigma_dot,
    compute_mass_flux_hybrid,
    vertical_advection,
    vertical_advection_hybrid,
    compute_pressure_velocity,
    compute_omega_hybrid,
)
from legoesm.grids.edge_blending import (
    blend_scalar_cube_edges,
    blend_vector_cube_edges,
)
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import IntegrationMixin
from legoesm import constants


class CDGridPrimitiveEquationConfig(NamedTuple):
    """Configuration for the C-D grid hydrostatic PE model.

    For explicit time integration without semi-implicit gravity wave
    treatment, set ``n_barotropic_substeps > 1`` to subcycle the
    barotropic (external gravity wave) mode.  The barotropic substep
    advances surface pressure and the column-mean divergent flow with
    dt_baro = dt / n_barotropic_substeps, while the baroclinic modes
    (temperature, internal wind structure) evolve on the full dt.

    Alternatively, set ``implicit_grav_wave_damping > 0`` to apply a
    linearized implicit correction to p_s after each step, which damps
    the fastest gravity wave mode without substeps.
    """
    g: float = constants.g
    A_h: float = 0.0              # Laplacian viscosity [m^2/s]
    hyperdiff_coeff: float = 0.0
    hyperdiff_ps_coeff: float = 0.0
    div_damp_coeff: float = 0.0   # Divergence damping coefficient [m^2/s]
    implicit_grav_wave_damping: float = 0.0
        # Implicit damping factor for the external gravity wave mode.
        # Applied as an exponential filter: ps_new *= exp(-alpha * dt * lap(ps))
        # where alpha = implicit_grav_wave_damping.
        # Typical value: 0.5 * c_grav^2 * dt / dx^2 where c_grav ~ 300 m/s.
        # This is a simplified semi-implicit treatment that selectively
        # damps divergent modes without a full barotropic solve.
    T_min: float = 50.0            # Temperature floor [K] (positivity protection)
    p_floor: float = 100.0         # Pressure floor [Pa] for adiabatic heating (limits 1/p)
    sponge_sigma: float = 0.15     # Rayleigh sponge activates above this sigma
    sponge_tau_sec: float = 3600.0 # e-folding time at model top [s]
    edge_blend_uv: float = 0.05       # Edge blend strength for winds
    edge_blend_T: float = 0.05        # Edge blend strength for temperature
    edge_blend_p_s: float = 0.05      # Edge blend strength for surface pressure
    edge_blend_width: int = 1         # Edge blend stencil width
    use_conservation_fixer: bool = True
    fix_mass: bool = True
    anchor_mass_to_initial: bool = False
    time_integrator: str = "ssp_rk3"
    T_diss_coeff: float = 0.0
        # Velocity-dependent Laplacian dissipation for the temperature
        # equation.  Mimics upwind advection's built-in diffusivity:
        #   ν_T = T_diss_coeff · |v| · Δx
        # Typically not needed when A_h (Laplacian viscosity) is used,
        # since A_h already damps intermediate-scale T noise.
        # Typical range when used: 0.1–0.5.  0 disables (default).


def cdgrid_hydrostatic_tendencies(
    state: HydrostaticState,
    grid: CubedSphereGrid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    cdgrid: CubedSphereCDGrid,
    config: CDGridPrimitiveEquationConfig = CDGridPrimitiveEquationConfig(),
    physics_tendency: HydrostaticTendencies | None = None,
) -> HydrostaticTendencies:
    """Compute tendencies for the hydrostatic primitive equations.

    Momentum uses A-grid operators (vorticity, gradient, KE) to avoid
    the energy inconsistency that arises from the A→D→A round-trip when
    prognostic winds are stored at cell centres but the vector-invariant
    formulation is evaluated at D-grid corners.

    Continuity (dp_s/dt, sigma-dot) uses C-grid divergence from
    D-grid-interpolated winds, preserving exact discrete continuity
    closure.

    Parameters
    ----------
    state : HydrostaticState
    grid : CubedSphereGrid
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
    cdgrid : CubedSphereCDGrid
    config : CDGridPrimitiveEquationConfig
    physics_tendency : HydrostaticTendencies, optional

    Returns
    -------
    HydrostaticTendencies
    """
    _hybrid = isinstance(sigma_coord, HybridSigmaPressureCoordinate)

    u = state.u.data       # (6, n, n, nlev) — A-grid
    v = state.v.data
    T = state.T.data
    p_s = state.p_s.data   # (6, n, n)
    phis = state.phis.data

    R_d = constants.R_d
    kappa = constants.kappa

    # Positivity protections
    T = jnp.maximum(T, config.T_min)
    p_s = jnp.clip(p_s, 100.0, 2.0e6)

    # --- 1. Pressure at full levels ---
    if _hybrid:
        p_full = pressure_from_hybrid(sigma_coord, p_s)
        dp = dp_from_hybrid(sigma_coord, p_s)
    else:
        p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)

    # --- 2. Geopotential via hydrostatic integration ---
    if _hybrid:
        Phi = compute_geopotential_hybrid(T, p_s, sigma_coord, phis)
    else:
        Phi = compute_geopotential(T, p_s, sigma_coord, phis)

    # --- 3. Kinetic energy and Bernoulli function (A-grid) ---
    K = 0.5 * (u**2 + v**2)
    B = K + Phi

    # --- 4. Horizontal dynamics (A-grid operators) ---
    # Vorticity at A-grid centres
    zeta = _vorticity_3d(u, v, grid)
    abs_vor = zeta + grid.f[..., None]

    # Bernoulli function gradient at A-grid centres
    dB_dx = _gradient_x_3d(B, grid)
    dB_dy = _gradient_y_3d(B, grid)

    # Pressure gradient correction: -R_d * T * ∇(ln p_s) at A-grid
    ln_ps = jnp.log(p_s)
    ln_ps_field = Field(data=ln_ps, name="ln_ps", dims=("face", "x", "y"),
                        units="", staggering="cell")
    dln_ps_dx = gradient_x(ln_ps_field, grid).data   # (6, n, n)
    dln_ps_dy = gradient_y(ln_ps_field, grid).data

    pg_corr_x = R_d * T * dln_ps_dx[..., None]
    pg_corr_y = R_d * T * dln_ps_dy[..., None]

    du_dt_data = abs_vor * v - dB_dx - pg_corr_x
    dv_dt_data = -abs_vor * u - dB_dy - pg_corr_y

    # Divergence damping (A-grid)
    if config.div_damp_coeff > 0:
        div_damp = _divergence_3d(u, v, grid)
        ddiv_dx = _gradient_x_3d(div_damp, grid)
        ddiv_dy = _gradient_y_3d(div_damp, grid)
        du_dt_data = du_dt_data - config.div_damp_coeff * ddiv_dx
        dv_dt_data = dv_dt_data - config.div_damp_coeff * ddiv_dy

    # --- 5. Surface pressure tendency and vertical motion ---
    # CRITICAL: dp_s/dt and σ̇ MUST use the SAME divergence operator.
    # Using A-grid divergence for full consistency with A-grid momentum.
    div_v = _divergence_3d(u, v, grid)  # (6, n, n, nlev)

    if _hybrid:
        D_total_p = jnp.sum(div_v * dp, axis=-1)
        dp_s_dt_data = -D_total_p / sigma_coord.B_range
        dp_s_dt_data = zero_mean_tendency(dp_s_dt_data, grid)

        mass_flux = compute_mass_flux_hybrid(div_v, p_s, sigma_coord)
        vert_adv_T = vertical_advection_hybrid(T, mass_flux, p_s, sigma_coord)
        vert_adv_u = vertical_advection_hybrid(u, mass_flux, p_s, sigma_coord)
        vert_adv_v = vertical_advection_hybrid(v, mass_flux, p_s, sigma_coord)

        omega = compute_omega_hybrid(mass_flux, p_s, dp_s_dt_data, sigma_coord)
        p_adiab = jnp.maximum(p_full, config.p_floor)
    else:
        dsigma = sigma_coord.dsigma
        sigma_top = sigma_coord.sigma_half[0]
        sigma_range = 1.0 - sigma_top

        D_total = jnp.sum(div_v * dsigma, axis=-1)
        dp_s_dt_data = -p_s * D_total / sigma_range
        dp_s_dt_data = zero_mean_tendency(dp_s_dt_data, grid)

        sigma_dot = compute_sigma_dot(div_v, sigma_coord)
        vert_adv_T = vertical_advection(T, sigma_dot, sigma_coord)
        vert_adv_u = vertical_advection(u, sigma_dot, sigma_coord)
        vert_adv_v = vertical_advection(v, sigma_dot, sigma_coord)

        omega = compute_pressure_velocity(sigma_dot, p_s, dp_s_dt_data, sigma_coord)
        p_adiab = jnp.maximum(p_full, config.p_floor)

    du_dt_data = du_dt_data + vert_adv_u
    dv_dt_data = dv_dt_data + vert_adv_v

    # --- 6. Thermodynamic equation ---
    # Centered A-grid advection: -v·∇T
    dT_dx = _gradient_x_3d(T, grid)
    dT_dy = _gradient_y_3d(T, grid)
    horiz_adv_T = -(u * dT_dx + v * dT_dy)

    # Adiabatic heating: κ·T·ω/p
    # ω = σ·∂p_s/∂t + p_s·σ̇  (computed by compute_pressure_velocity)
    # The v·∇(ln p_s) completes the full material derivative of p_s.
    adiabatic = kappa * T * omega / p_adiab
    v_dot_grad_lnps = u * dln_ps_dx[..., None] + v * dln_ps_dy[..., None]
    adiabatic = adiabatic + kappa * T * v_dot_grad_lnps

    dT_dt_data = horiz_adv_T + vert_adv_T + adiabatic

    # --- 6b. Velocity-dependent temperature dissipation ---
    # Adds |v|·Δx-proportional Laplacian diffusion to prevent the
    # feedback loop: grid-scale T noise → pressure gradient →
    # stronger winds → more T noise.  Equivalent to blending the
    # centered scheme with a first-order upwind scheme.
    if config.T_diss_coeff > 0:
        wind_speed = jnp.sqrt(u**2 + v**2)
        dx_local = grid.dx[..., None]  # (6, n, n, 1)
        nu_T = config.T_diss_coeff * wind_speed * dx_local
        lap_T = _laplacian_compact_3d(T, grid)
        dT_dt_data = dT_dt_data + nu_T * lap_T

    # --- 7. Diffusion ---
    # 7a. Laplacian viscosity (∇²) — damps INTERMEDIATE-scale modes that
    # ∇⁴ hyperdiffusion misses.  Essential for stability of centered
    # advection during violent adjustments (e.g., spinup from rest).
    # The ∇⁴ hyperdiffusion damps grid-scale in ~1 hr but intermediate
    # scales (wavenumber n/4) in ~28 hrs — too slow vs the ~4 hr
    # computational instability from nonlinear kinetic-energy aliasing.
    if config.A_h > 0:
        lap_u = _laplacian_compact_3d(u, grid)
        lap_v = _laplacian_compact_3d(v, grid)
        lap_T = _laplacian_compact_3d(T, grid)
        du_dt_data = du_dt_data + config.A_h * lap_u
        dv_dt_data = dv_dt_data + config.A_h * lap_v
        dT_dt_data = dT_dt_data + config.A_h * lap_T

    # 7b. Hyperdiffusion (∇⁴) — scale-selective damping of grid-scale noise
    if config.hyperdiff_coeff > 0:
        du_dt_data = du_dt_data + _hyperdiffusion_3d(u, grid, config.hyperdiff_coeff)
        dv_dt_data = dv_dt_data + _hyperdiffusion_3d(v, grid, config.hyperdiff_coeff)
        dT_dt_data = dT_dt_data + _hyperdiffusion_3d(T, grid, config.hyperdiff_coeff)

    if config.hyperdiff_ps_coeff > 0:
        from legoesm.core.operators import hyperdiffusion
        ps_field = Field(data=p_s, name="p_s", dims=("face", "x", "y"),
                         units="Pa", staggering="cell")
        diff_ps = hyperdiffusion(ps_field, grid, config.hyperdiff_ps_coeff)
        dp_s_dt_data = dp_s_dt_data + diff_ps.data

    # --- 8. Upper-atmosphere Rayleigh sponge ---
    if config.sponge_tau_sec > 0 and config.sponge_sigma > 0:
        sigma_full = sigma_coord.sigma_full
        sponge_frac = jnp.clip(
            (config.sponge_sigma - sigma_full) / config.sponge_sigma, 0.0, 1.0
        )
        sponge_rate = sponge_frac**2 / config.sponge_tau_sec
        du_dt_data = du_dt_data - sponge_rate * u
        dv_dt_data = dv_dt_data - sponge_rate * v

    # --- 9. Physics ---
    if physics_tendency is not None:
        du_dt_data = du_dt_data + physics_tendency.du_dt.data
        dv_dt_data = dv_dt_data + physics_tendency.dv_dt.data
        dT_dt_data = dT_dt_data + physics_tendency.dT_dt.data
        dp_s_dt_data = dp_s_dt_data + physics_tendency.dp_s_dt.data

    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")

    return HydrostaticTendencies(
        du_dt=Field(data=du_dt_data, name="du_dt", dims=dims_3d, units="m/s^2"),
        dv_dt=Field(data=dv_dt_data, name="dv_dt", dims=dims_3d, units="m/s^2"),
        dT_dt=Field(data=dT_dt_data, name="dT_dt", dims=dims_3d, units="K/s"),
        dp_s_dt=Field(data=dp_s_dt_data, name="dp_s_dt", dims=dims_2d, units="Pa/s"),
        dphis_dt=Field(
            data=jnp.zeros_like(phis), name="dphis_dt", dims=dims_2d, units="m^2/s^3"
        ),
    )


def fv3_hydrostatic_tendencies(
    state: HydrostaticState,
    grid: CubedSphereGrid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    cdgrid: CubedSphereCDGrid,
    config: CDGridPrimitiveEquationConfig = CDGridPrimitiveEquationConfig(),
    physics_tendency: HydrostaticTendencies | None = None,
) -> HydrostaticTendencies:
    """FV3-style tendencies using D-grid vorticity and Arakawa-Lamb gradient.

    Momentum uses the vector-invariant form on the C-D grid:
    - Vorticity from D-grid winds via exact circulation (avoids H-K instability)
    - KE at cell centers from C-grid velocities
    - Bernoulli gradient at D-grid corners via 4-point Arakawa-Lamb formula
    - Momentum tendencies computed at D-grid corners, then averaged to A-grid

    Continuity uses C-grid divergence for exact discrete mass conservation.

    Temperature uses PPM flux-form transport per level.

    Parameters
    ----------
    state : HydrostaticState
        Winds stored at A-grid cell centres; internally converted to D-grid.
    grid : CubedSphereGrid
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
    cdgrid : CubedSphereCDGrid
    config : CDGridPrimitiveEquationConfig
    physics_tendency : HydrostaticTendencies, optional

    Returns
    -------
    HydrostaticTendencies
    """
    from legoesm.core.operators_cdgrid import (
        dgrid_vorticity,
        dgrid_to_cgrid,
        cgrid_divergence,
        _arakawa_lamb_gradient,
        _interp_center_to_corner,
        _interp_corner_to_center,
        _laplacian_dgrid,
        _pad_halo_auto,
    )
    from legoesm.grids.halo import pad_halo, pad_halo_vector

    _hybrid = isinstance(sigma_coord, HybridSigmaPressureCoordinate)

    u = state.u.data       # (6, n, n, nlev) — A-grid
    v = state.v.data
    T = state.T.data
    p_s = state.p_s.data   # (6, n, n)
    phis = state.phis.data
    n = grid.n
    nlev = u.shape[-1]

    R_d = constants.R_d
    kappa = constants.kappa

    # Positivity protections
    T = jnp.maximum(T, config.T_min)
    p_s = jnp.clip(p_s, 100.0, 2.0e6)

    # --- 1. Convert A-grid winds to D-grid ---
    # Interpolate cell-centre winds to cell corners via halo-aware averaging
    def _a_to_d_level(u_k, v_k):
        """Convert one level of A-grid winds to D-grid corners."""
        u_pad, v_pad = pad_halo_vector(
            u_k, v_k,
            grid.cos_angle, grid.sin_angle,
            grid.cos_angle_padded, grid.sin_angle_padded,
            interp_offsets=grid.halo_interp_offsets,
        )
        u_d = 0.25 * (u_pad[:, :-1, :-1] + u_pad[:, 1:, :-1]
                       + u_pad[:, :-1, 1:] + u_pad[:, 1:, 1:])
        v_d = 0.25 * (v_pad[:, :-1, :-1] + v_pad[:, 1:, :-1]
                       + v_pad[:, :-1, 1:] + v_pad[:, 1:, 1:])
        return u_d, v_d

    # vmap over levels
    u_t = jnp.moveaxis(u, -1, 0)  # (nlev, 6, n, n)
    v_t = jnp.moveaxis(v, -1, 0)
    u_d_t, v_d_t = jax.vmap(_a_to_d_level)(u_t, v_t)  # (nlev, 6, n+1, n+1) each
    u_d = jnp.moveaxis(u_d_t, 0, -1)  # (6, n+1, n+1, nlev)
    v_d = jnp.moveaxis(v_d_t, 0, -1)

    # --- 2. D-grid → C-grid ---
    u_c, v_c = dgrid_to_cgrid(u_d, v_d, cdgrid)  # (6, n+1, n, nlev), (6, n, n+1, nlev)

    # --- 3. Pressure and geopotential ---
    if _hybrid:
        p_full = pressure_from_hybrid(sigma_coord, p_s)
        dp = dp_from_hybrid(sigma_coord, p_s)
    else:
        p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)

    if _hybrid:
        Phi = compute_geopotential_hybrid(T, p_s, sigma_coord, phis)
    else:
        Phi = compute_geopotential(T, p_s, sigma_coord, phis)

    # --- 4. KE at cell centers from C-grid velocities ---
    u_center = 0.5 * (u_c[:, :-1, :, :] + u_c[:, 1:, :, :])  # (6, n, n, nlev)
    v_center = 0.5 * (v_c[:, :, :-1, :] + v_c[:, :, 1:, :])
    KE = 0.5 * (u_center ** 2 + v_center ** 2)

    # --- 5. Bernoulli function B = KE + Φ (cell centers) ---
    B = KE + Phi

    # --- 6. D-grid vorticity via circulation ---
    zeta = dgrid_vorticity(u_d, v_d, cdgrid)      # (6, n, n, nlev)
    zeta_abs = zeta + grid.f[..., None]

    # Vorticity at D-grid corners (interpolated)
    zeta_corner = _interp_center_to_corner(zeta_abs, cdgrid)  # (6, n+1, n+1, nlev)

    # --- 7. Gradients at D-grid corners via Arakawa-Lamb ---
    dB_dx, dB_dy = _arakawa_lamb_gradient(B, cdgrid)  # (6, n+1, n+1, nlev)

    # Pressure gradient correction at D-grid corners
    ln_ps = jnp.log(p_s)
    dln_dx, dln_dy = _arakawa_lamb_gradient(ln_ps, cdgrid)  # (6, n+1, n+1)
    T_corner = _interp_center_to_corner(T, cdgrid)  # (6, n+1, n+1, nlev)
    pg_corr_x = R_d * T_corner * dln_dx[..., None]  # (6, n+1, n+1, nlev)
    pg_corr_y = R_d * T_corner * dln_dy[..., None]

    # --- 8. D-grid momentum tendencies ---
    du_d_dt = zeta_corner * v_d - dB_dx - pg_corr_x
    dv_d_dt = -zeta_corner * u_d - dB_dy - pg_corr_y

    # Divergence damping at D-grid
    if config.div_damp_coeff > 0:
        div_v = cgrid_divergence(u_c, v_c, cdgrid)  # (6, n, n, nlev)
        ddiv_dx, ddiv_dy = _arakawa_lamb_gradient(div_v, cdgrid)
        du_d_dt = du_d_dt - config.div_damp_coeff * ddiv_dx
        dv_d_dt = dv_d_dt - config.div_damp_coeff * ddiv_dy

    # Diffusion at D-grid corners
    if config.A_h > 0:
        du_d_dt = du_d_dt + config.A_h * _laplacian_dgrid(u_d, cdgrid)
        dv_d_dt = dv_d_dt + config.A_h * _laplacian_dgrid(v_d, cdgrid)

    if config.hyperdiff_coeff > 0:
        du_d_dt = du_d_dt - config.hyperdiff_coeff * _laplacian_dgrid(
            _laplacian_dgrid(u_d, cdgrid), cdgrid)
        dv_d_dt = dv_d_dt - config.hyperdiff_coeff * _laplacian_dgrid(
            _laplacian_dgrid(v_d, cdgrid), cdgrid)

    # --- 9. Convert D-grid tendencies back to A-grid ---
    du_dt_data = _interp_corner_to_center(du_d_dt)  # (6, n, n, nlev)
    dv_dt_data = _interp_corner_to_center(dv_d_dt)

    # --- 10. Surface pressure tendency using C-grid divergence ---
    div_v_c = cgrid_divergence(u_c, v_c, cdgrid)  # (6, n, n, nlev)

    if _hybrid:
        D_total_p = jnp.sum(div_v_c * dp, axis=-1)
        dp_s_dt_data = -D_total_p / sigma_coord.B_range
        dp_s_dt_data = zero_mean_tendency(dp_s_dt_data, grid)

        mass_flux = compute_mass_flux_hybrid(div_v_c, p_s, sigma_coord)
        vert_adv_T = vertical_advection_hybrid(T, mass_flux, p_s, sigma_coord)
        vert_adv_u = vertical_advection_hybrid(u, mass_flux, p_s, sigma_coord)
        vert_adv_v = vertical_advection_hybrid(v, mass_flux, p_s, sigma_coord)

        omega = compute_omega_hybrid(mass_flux, p_s, dp_s_dt_data, sigma_coord)
        p_adiab = jnp.maximum(p_full, config.p_floor)
    else:
        dsigma = sigma_coord.dsigma
        sigma_top = sigma_coord.sigma_half[0]
        sigma_range = 1.0 - sigma_top

        D_total = jnp.sum(div_v_c * dsigma, axis=-1)
        dp_s_dt_data = -p_s * D_total / sigma_range
        dp_s_dt_data = zero_mean_tendency(dp_s_dt_data, grid)

        sigma_dot = compute_sigma_dot(div_v_c, sigma_coord)
        vert_adv_T = vertical_advection(T, sigma_dot, sigma_coord)
        vert_adv_u = vertical_advection(u, sigma_dot, sigma_coord)
        vert_adv_v = vertical_advection(v, sigma_dot, sigma_coord)

        omega = compute_pressure_velocity(sigma_dot, p_s, dp_s_dt_data, sigma_coord)
        p_adiab = jnp.maximum(p_full, config.p_floor)

    du_dt_data = du_dt_data + vert_adv_u
    dv_dt_data = dv_dt_data + vert_adv_v

    # --- 11. Thermodynamic equation ---
    # Horizontal advection: centered A-grid advection -v·∇T
    dT_dx = _gradient_x_3d(T, grid)
    dT_dy = _gradient_y_3d(T, grid)
    horiz_adv_T = -(u * dT_dx + v * dT_dy)

    # Adiabatic heating
    ln_ps_field = Field(data=ln_ps, name="ln_ps", dims=("face", "x", "y"),
                        units="", staggering="cell")
    dln_ps_dx = gradient_x(ln_ps_field, grid).data
    dln_ps_dy = gradient_y(ln_ps_field, grid).data
    adiabatic = kappa * T * omega / p_adiab
    v_dot_grad_lnps = u * dln_ps_dx[..., None] + v * dln_ps_dy[..., None]
    adiabatic = adiabatic + kappa * T * v_dot_grad_lnps

    dT_dt_data = horiz_adv_T + vert_adv_T + adiabatic

    # Temperature dissipation
    if config.T_diss_coeff > 0:
        wind_speed = jnp.sqrt(u**2 + v**2)
        dx_local = grid.dx[..., None]
        nu_T = config.T_diss_coeff * wind_speed * dx_local
        lap_T = _laplacian_compact_3d(T, grid)
        dT_dt_data = dT_dt_data + nu_T * lap_T

    # Temperature diffusion
    if config.A_h > 0:
        lap_T = _laplacian_compact_3d(T, grid)
        dT_dt_data = dT_dt_data + config.A_h * lap_T

    if config.hyperdiff_coeff > 0:
        dT_dt_data = dT_dt_data + _hyperdiffusion_3d(T, grid, config.hyperdiff_coeff)

    if config.hyperdiff_ps_coeff > 0:
        from legoesm.core.operators import hyperdiffusion
        ps_field = Field(data=p_s, name="p_s", dims=("face", "x", "y"),
                         units="Pa", staggering="cell")
        diff_ps = hyperdiffusion(ps_field, grid, config.hyperdiff_ps_coeff)
        dp_s_dt_data = dp_s_dt_data + diff_ps.data

    # --- 12. Upper-atmosphere Rayleigh sponge ---
    if config.sponge_tau_sec > 0 and config.sponge_sigma > 0:
        sigma_full = sigma_coord.sigma_full
        sponge_frac = jnp.clip(
            (config.sponge_sigma - sigma_full) / config.sponge_sigma, 0.0, 1.0
        )
        sponge_rate = sponge_frac**2 / config.sponge_tau_sec
        du_dt_data = du_dt_data - sponge_rate * u
        dv_dt_data = dv_dt_data - sponge_rate * v

    # --- 13. Physics ---
    if physics_tendency is not None:
        du_dt_data = du_dt_data + physics_tendency.du_dt.data
        dv_dt_data = dv_dt_data + physics_tendency.dv_dt.data
        dT_dt_data = dT_dt_data + physics_tendency.dT_dt.data
        dp_s_dt_data = dp_s_dt_data + physics_tendency.dp_s_dt.data

    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")

    return HydrostaticTendencies(
        du_dt=Field(data=du_dt_data, name="du_dt", dims=dims_3d, units="m/s^2"),
        dv_dt=Field(data=dv_dt_data, name="dv_dt", dims=dims_3d, units="m/s^2"),
        dT_dt=Field(data=dT_dt_data, name="dT_dt", dims=dims_3d, units="K/s"),
        dp_s_dt=Field(data=dp_s_dt_data, name="dp_s_dt", dims=dims_2d, units="Pa/s"),
        dphis_dt=Field(
            data=jnp.zeros_like(phis), name="dphis_dt", dims=dims_2d, units="m^2/s^3"
        ),
    )


def _apply_edge_blend(
    state: HydrostaticState,
    grid: CubedSphereGrid,
    config: CDGridPrimitiveEquationConfig,
) -> HydrostaticState:
    """Apply cubed-sphere edge blending to prognostic variables."""
    u_data = state.u.data
    v_data = state.v.data
    T_data = state.T.data
    p_s_data = state.p_s.data

    if config.edge_blend_uv > 0.0:
        u_data, v_data = blend_vector_cube_edges(
            u_data, v_data, grid.cos_angle, grid.sin_angle,
            config.edge_blend_uv, width=config.edge_blend_width,
        )
    if config.edge_blend_T > 0.0:
        T_data = blend_scalar_cube_edges(
            T_data, config.edge_blend_T, width=config.edge_blend_width,
        )
    if config.edge_blend_p_s > 0.0:
        p_s_data = blend_scalar_cube_edges(
            p_s_data, config.edge_blend_p_s, width=config.edge_blend_width,
        )

    return state._replace(
        u=state.u.replace(data=u_data),
        v=state.v.replace(data=v_data),
        T=state.T.replace(data=T_data),
        p_s=state.p_s.replace(data=p_s_data),
    )


class CDGridPrimitiveEquationModel(IntegrationMixin):
    """Hydrostatic PE model on the cubed-sphere with FV3 C-D grid dynamics.

    Uses FV3-style vector-invariant form:
    - D-grid vorticity via circulation (avoids Hollingsworth-Kallberg)
    - Arakawa-Lamb gradient at D-grid corners
    - C-grid divergence for exact mass conservation

    Parameters
    ----------
    grid : CubedSphereGrid
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
    config : CDGridPrimitiveEquationConfig, optional
    """

    def __init__(
        self,
        grid: CubedSphereGrid,
        sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
        config: CDGridPrimitiveEquationConfig | None = None,
    ):
        self.grid = grid
        self.sigma_coord = sigma_coord
        self.config = config or CDGridPrimitiveEquationConfig()
        self.cdgrid = create_cubed_sphere_cdgrid(grid)
        self._target_mass = None

    def tendencies(
        self,
        state: HydrostaticState,
        physics_tendency: HydrostaticTendencies | None = None,
    ) -> HydrostaticTendencies:
        return fv3_hydrostatic_tendencies(
            state, self.grid, self.sigma_coord, self.cdgrid,
            self.config, physics_tendency,
        )

    @partial(jax.jit, static_argnums=(0, 3))
    def step(self, state: HydrostaticState, dt: float, physics_fn=None) -> HydrostaticState:
        """Advance one time step."""
        def tendency_fn(s):
            phys = None
            if physics_fn is not None:
                phys = physics_fn(s, self.grid, self.sigma_coord)
            tend = fv3_hydrostatic_tendencies(
                s, self.grid, self.sigma_coord, self.cdgrid,
                self.config, phys,
            )
            return HydrostaticState(
                u=s.u.replace(data=tend.du_dt.data),
                v=s.v.replace(data=tend.dv_dt.data),
                T=s.T.replace(data=tend.dT_dt.data),
                p_s=s.p_s.replace(data=tend.dp_s_dt.data),
                phis=s.phis.replace(data=jnp.zeros_like(s.phis.data)),
            )

        state_new = dispatch_integrator(
            state, tendency_fn, dt, self.config.time_integrator,
        )

        # Implicit gravity wave damping — post-step Laplacian diffusion on p_s.
        if self.config.implicit_grav_wave_damping > 0:
            from legoesm.core.operators import laplacian_compact
            alpha = self.config.implicit_grav_wave_damping
            lap_ps = laplacian_compact(state_new.p_s.data, self.grid)
            p_s_damped = state_new.p_s.data + alpha * dt * lap_ps
            p_s_damped = jnp.maximum(p_s_damped, 100.0)
            state_new = state_new._replace(
                p_s=state_new.p_s.replace(data=p_s_damped),
            )

        # Conservation fixer
        if self.config.use_conservation_fixer and self.config.fix_mass:
            if self.config.anchor_mass_to_initial:
                from legoesm.core.conservation import fix_mass_hydrostatic_target
                from legoesm.core.operators import global_integral
                if self._target_mass is None:
                    self._target_mass = global_integral(state.p_s, self.grid)
                state_new = fix_mass_hydrostatic_target(
                    state_new, self._target_mass, self.grid,
                )
            else:
                from legoesm.core.conservation import fix_mass_hydrostatic
                state_new = fix_mass_hydrostatic(state_new, state, self.grid)

        return state_new

    def step_with_physics(self, state, dt, physics_fn=None):
        return self.step(state, dt, physics_fn=physics_fn)
