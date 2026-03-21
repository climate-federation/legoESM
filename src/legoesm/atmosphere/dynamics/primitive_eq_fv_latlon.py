"""FV3-Style Hydrostatic Primitive Equations on the lat-lon grid.

Uses unsplit PPM for:
- Surface pressure tendency (replaces centered divergence)
- Temperature horizontal advection (replaces centered gradient)

Momentum remains in vector-invariant form (same as centered).
Vertical advection and thermodynamics are unchanged.
Includes a Fourier polar filter for CFL stability near the poles.

References
----------
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core (FV3)
- Colella & Woodward (1984): The Piecewise Parabolic Method (PPM)
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState, HydrostaticTendencies
from legoesm.core.operators_latlon import (
    gradient_x as _gradient_x_2d,
    gradient_y as _gradient_y_2d,
    hyperdiffusion as _hyperdiffusion_2d,
)
from legoesm.core.operators_latlon_3d import (
    vorticity_3d as _vorticity_3d,
    gradient_x_3d as _gradient_x_3d,
    gradient_y_3d as _gradient_y_3d,
    divergence_3d as _divergence_3d,
    laplacian_3d as _laplacian_3d,
    hyperdiffusion_3d as _hyperdiffusion_3d,
)
from legoesm.core.operators_fv_latlon import fv_flux_divergence_latlon
from legoesm.core.operators_fv_latlon_3d import fv_scalar_advection_latlon_3d
from legoesm.core.conservation import (
    zero_mean_tendency_latlon,
    fix_mass_hydrostatic_latlon,
)
from legoesm.grids.latlon import LatLonGrid
from legoesm.grids.polar_filter import (
    compute_polar_filter_mask,
    fourier_filter,
    fourier_filter_3d,
)
from legoesm.grids.vertical import (
    SigmaCoordinate,
    pressure_from_sigma,
    compute_geopotential,
    compute_sigma_dot,
    vertical_advection,
    compute_pressure_velocity,
)
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import IntegrationMixin
from legoesm import constants


class FVLatLonPrimitiveEquationConfig(NamedTuple):
    """Configuration for the FV hydrostatic PE model on a lat-lon grid."""
    g: float = constants.g
    A_h: float = 0.0               # Laplacian viscosity [m^2/s]
    hyperdiff_coeff: float = 0.0
    hyperdiff_ps_coeff: float = 0.0
    use_conservation_fixer: bool = True
    fix_mass: bool = True
    time_integrator: str = "ssp_rk3"
    use_limiter: bool = True
    use_polar_filter: bool = True
    polar_filter_cutoff_deg: float = 60.0
    polar_filter_max_wave_speed: float = 300.0


def fv_latlon_hydrostatic_tendencies(
    state: HydrostaticState,
    grid: LatLonGrid,
    sigma_coord: SigmaCoordinate,
    config: FVLatLonPrimitiveEquationConfig = FVLatLonPrimitiveEquationConfig(),
    physics_tendency: HydrostaticTendencies | None = None,
    polar_mask: jnp.ndarray | None = None,
) -> HydrostaticTendencies:
    """Compute tendencies for the FV hydrostatic PE on a lat-lon grid.

    Differences from centered lat-lon PE:
    - Surface pressure: FV flux divergence instead of centered divergence
    - Temperature: FV scalar advection instead of centered gradient
    - Momentum: unchanged (vector-invariant)

    Parameters
    ----------
    state : HydrostaticState
    grid : LatLonGrid
    sigma_coord : SigmaCoordinate
    config : FVLatLonPrimitiveEquationConfig
    physics_tendency : HydrostaticTendencies, optional
    polar_mask : jax.Array, optional

    Returns
    -------
    HydrostaticTendencies
    """
    u = state.u.data       # (n_lat, n_lon, nlev)
    v = state.v.data
    T = state.T.data
    p_s = state.p_s.data   # (n_lat, n_lon)
    phis = state.phis.data

    R_d = constants.R_d
    kappa = constants.kappa
    dsigma = sigma_coord.dsigma

    p_s = jnp.clip(p_s, 100.0, 2.0e6)

    # --- 1. Pressure at full levels ---
    p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)

    # --- 2. Geopotential via hydrostatic integration ---
    Phi = compute_geopotential(T, p_s, sigma_coord, phis)

    # --- 3. Kinetic energy and Bernoulli function ---
    K = 0.5 * (u**2 + v**2)
    B = K + Phi

    # --- 4. Horizontal dynamics ---
    zeta = _vorticity_3d(u, v, grid)
    abs_vor = zeta + grid.f[:, :, None]

    dB_dx = _gradient_x_3d(B, grid)
    dB_dy = _gradient_y_3d(B, grid)

    # Pressure gradient correction: -R_d * T * grad(ln p_s)
    ln_ps = jnp.log(p_s)
    ln_ps_field = Field(data=ln_ps, name="ln_ps", dims=("lat", "lon"),
                        units="", staggering="cell")
    dln_ps_dx = _gradient_x_2d(ln_ps_field, grid).data
    dln_ps_dy = _gradient_y_2d(ln_ps_field, grid).data

    pg_corr_x = R_d * T * dln_ps_dx[:, :, None]
    pg_corr_y = R_d * T * dln_ps_dy[:, :, None]

    du_dt_data = abs_vor * v - dB_dx - pg_corr_x
    dv_dt_data = -abs_vor * u - dB_dy - pg_corr_y

    # --- 5. Surface pressure tendency and sigma-dot ---
    # CRITICAL: dp_s/dt and σ̇ MUST use the SAME divergence operator.
    # Using FV flux divergence for dp_s/dt but centered divergence for
    # σ̇ breaks the discrete continuity closure.
    div_v = _divergence_3d(u, v, grid)

    sigma_top = sigma_coord.sigma_half[0]
    sigma_range = 1.0 - sigma_top

    # dp_s/dt from centered divergence (same operator as σ̇)
    D_total = jnp.sum(div_v * dsigma[None, None, :], axis=-1)
    dp_s_dt_data = -p_s * D_total / sigma_range
    dp_s_dt_data = zero_mean_tendency_latlon(dp_s_dt_data, grid)

    sigma_dot = compute_sigma_dot(div_v, sigma_coord)

    # --- 7. Vertical advection of T, u, v ---
    vert_adv_T = vertical_advection(T, sigma_dot, sigma_coord)
    vert_adv_u = vertical_advection(u, sigma_dot, sigma_coord)
    vert_adv_v = vertical_advection(v, sigma_dot, sigma_coord)

    du_dt_data = du_dt_data + vert_adv_u
    dv_dt_data = dv_dt_data + vert_adv_v

    # --- 8. Thermodynamic equation with FV horizontal advection ---
    horiz_adv_T = fv_scalar_advection_latlon_3d(
        T, u, v, grid,
        limiter=config.use_limiter,
    )

    # Adiabatic heating: kappa * T * omega / p
    omega = compute_pressure_velocity(sigma_dot, p_s, dp_s_dt_data, sigma_coord)
    adiabatic = kappa * T * omega / p_full

    # Missing term from full material derivative of p_s
    v_dot_grad_lnps = u * dln_ps_dx[:, :, None] + v * dln_ps_dy[:, :, None]
    adiabatic = adiabatic + kappa * T * v_dot_grad_lnps

    dT_dt_data = horiz_adv_T + vert_adv_T + adiabatic

    # --- 8b. Laplacian viscosity (∇²) ---
    # Damps intermediate-scale modes that ∇⁴ hyperdiffusion misses.
    if config.A_h > 0:
        lap_u = _laplacian_3d(u, grid)
        lap_v = _laplacian_3d(v, grid)
        lap_T = _laplacian_3d(T, grid)
        du_dt_data = du_dt_data + config.A_h * lap_u
        dv_dt_data = dv_dt_data + config.A_h * lap_v
        dT_dt_data = dT_dt_data + config.A_h * lap_T

    # --- 9. Hyperdiffusion ---
    if config.hyperdiff_coeff > 0:
        diff_u = _hyperdiffusion_3d(u, grid, config.hyperdiff_coeff)
        diff_v = _hyperdiffusion_3d(v, grid, config.hyperdiff_coeff)
        diff_T = _hyperdiffusion_3d(T, grid, config.hyperdiff_coeff)
        du_dt_data = du_dt_data + diff_u
        dv_dt_data = dv_dt_data + diff_v
        dT_dt_data = dT_dt_data + diff_T

    if config.hyperdiff_ps_coeff > 0:
        ps_field = Field(data=p_s, name="p_s", dims=("lat", "lon"),
                         units="Pa", staggering="cell")
        diff_ps = _hyperdiffusion_2d(ps_field, grid, config.hyperdiff_ps_coeff)
        dp_s_dt_data = dp_s_dt_data + diff_ps.data

    # --- 9b. Polar filter on tendencies ---
    if config.use_polar_filter and polar_mask is not None:
        du_dt_data = fourier_filter_3d(du_dt_data, grid, polar_mask)
        dv_dt_data = fourier_filter_3d(dv_dt_data, grid, polar_mask)
        dT_dt_data = fourier_filter_3d(dT_dt_data, grid, polar_mask)
        dp_s_dt_data = fourier_filter(dp_s_dt_data, grid, polar_mask)

    # --- 10. Physics ---
    if physics_tendency is not None:
        du_dt_data = du_dt_data + physics_tendency.du_dt.data
        dv_dt_data = dv_dt_data + physics_tendency.dv_dt.data
        dT_dt_data = dT_dt_data + physics_tendency.dT_dt.data
        dp_s_dt_data = dp_s_dt_data + physics_tendency.dp_s_dt.data

    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")

    return HydrostaticTendencies(
        du_dt=Field(data=du_dt_data, name="du_dt", dims=dims_3d, units="m/s^2"),
        dv_dt=Field(data=dv_dt_data, name="dv_dt", dims=dims_3d, units="m/s^2"),
        dT_dt=Field(data=dT_dt_data, name="dT_dt", dims=dims_3d, units="K/s"),
        dp_s_dt=Field(data=dp_s_dt_data, name="dp_s_dt", dims=dims_2d, units="Pa/s"),
        dphis_dt=Field(
            data=jnp.zeros_like(phis), name="dphis_dt", dims=dims_2d, units="m^2/s^3"
        ),
    )


from legoesm.core.filters import filter_state as _filter_state


class FVLatLonPrimitiveEquationModel(IntegrationMixin):
    """FV3-style hydrostatic PE model on the lat-lon grid.

    Same interface as LatLonPrimitiveEquationModel but uses PPM + Lin-Rood
    for surface pressure and temperature horizontal transport.

    Parameters
    ----------
    grid : LatLonGrid
    sigma_coord : SigmaCoordinate
    config : FVLatLonPrimitiveEquationConfig, optional
    dt : float
        Time step for polar filter mask precomputation.
    """

    def __init__(
        self,
        grid: LatLonGrid,
        sigma_coord: SigmaCoordinate,
        config: FVLatLonPrimitiveEquationConfig | None = None,
        dt: float = 600.0,
    ):
        self.grid = grid
        self.sigma_coord = sigma_coord
        self.config = config or FVLatLonPrimitiveEquationConfig()

        if self.config.use_polar_filter:
            self.polar_mask = compute_polar_filter_mask(
                grid, dt,
                self.config.polar_filter_max_wave_speed,
                self.config.polar_filter_cutoff_deg,
            )
        else:
            self.polar_mask = None

    def tendencies(
        self,
        state: HydrostaticState,
        physics_tendency: HydrostaticTendencies | None = None,
    ) -> HydrostaticTendencies:
        """Compute tendencies (pure function wrapper)."""
        return fv_latlon_hydrostatic_tendencies(
            state, self.grid, self.sigma_coord, self.config,
            physics_tendency, self.polar_mask,
        )

    @partial(jax.jit, static_argnums=(0, 3))
    def step(self, state: HydrostaticState, dt: float, physics_fn=None) -> HydrostaticState:
        """Advance one time step, optionally with physics forcing."""
        if self.polar_mask is not None:
            state = _filter_state(state, self.grid, self.polar_mask)

        def tendency_fn(s):
            phys = None
            if physics_fn is not None:
                phys, _ = physics_fn(s, self.grid, self.sigma_coord)
            tend = fv_latlon_hydrostatic_tendencies(
                s, self.grid, self.sigma_coord, self.config, phys,
                self.polar_mask,
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

        if self.config.use_conservation_fixer and self.config.fix_mass:
            state_new = fix_mass_hydrostatic_latlon(state_new, state, self.grid)

        return state_new

    def step_with_physics(self, state, dt, physics_fn=None):
        """Backward-compatible wrapper for step() with physics."""
        return self.step(state, dt, physics_fn=physics_fn)

    # integrate() and integrate_scan() inherited from IntegrationMixin
