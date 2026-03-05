"""Hydrostatic Primitive Equations on the latitude-longitude grid.

Direct translation of primitive_eq.py (cubed-sphere) to the lat-lon grid.
Same vector-invariant PE equations, same Simmons-Burridge vertical
discretization, but using lat-lon operators and the polar filter for
CFL stability near the poles.

The hydrostatic primitive equations in σ-coordinates:

    dp_s/dt = -∫₀¹ div(p_s · v) dσ                      [surface pressure]
    du/dt   =  (ζ+f)·v - ∂B/∂x - R_d·T·∂ln(p_s)/∂x     [x-momentum]
    dv/dt   = -(ζ+f)·u - ∂B/∂y - R_d·T·∂ln(p_s)/∂y     [y-momentum]
    dT/dt   = -v·∇T - σ̇·∂T/∂σ + κ·T·ω/p               [thermodynamic]

References
----------
- Simmons & Burridge (1981)
- Held & Suarez (1994)
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
    hyperdiffusion_3d as _hyperdiffusion_3d,
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
from legoesm.timestepping.ssp_rk3 import ssp_rk3_step
from legoesm import constants


class LatLonPrimitiveEquationConfig(NamedTuple):
    """Configuration for the lat-lon hydrostatic primitive equation model."""
    g: float = constants.g
    hyperdiff_coeff: float = 0.0
    hyperdiff_ps_coeff: float = 0.0
    use_conservation_fixer: bool = True
    fix_mass: bool = True
    use_polar_filter: bool = True
    polar_filter_cutoff_deg: float = 60.0
    polar_filter_max_wave_speed: float = 300.0  # external gravity wave [m/s]


# ==============================================================================
# Tendency computation
# ==============================================================================

def latlon_hydrostatic_tendencies(
    state: HydrostaticState,
    grid: LatLonGrid,
    sigma_coord: SigmaCoordinate,
    config: LatLonPrimitiveEquationConfig = LatLonPrimitiveEquationConfig(),
    physics_tendency: HydrostaticTendencies | None = None,
    polar_mask: jnp.ndarray | None = None,
) -> HydrostaticTendencies:
    """Compute tendencies for the hydrostatic PE on a lat-lon grid.

    Parameters
    ----------
    state : HydrostaticState
        Current model state.
    grid : LatLonGrid
        Horizontal grid.
    sigma_coord : SigmaCoordinate
        Vertical coordinate.
    config : LatLonPrimitiveEquationConfig
        Model configuration.
    physics_tendency : HydrostaticTendencies, optional
        Physics tendencies to add.
    polar_mask : jax.Array, optional
        Precomputed polar filter mask, shape (n_lat, n_freq).

    Returns
    -------
    HydrostaticTendencies : Time derivatives.
    """
    u = state.u.data       # (n_lat, n_lon, nlev)
    v = state.v.data
    T = state.T.data
    p_s = state.p_s.data   # (n_lat, n_lon)
    phis = state.phis.data

    R_d = constants.R_d
    kappa = constants.kappa
    dsigma = sigma_coord.dsigma  # (nlev,)
    p_s = jnp.clip(p_s, 100.0, 2.0e6)

    # --- 1. Pressure at full levels ---
    p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)

    # --- 2. Geopotential via hydrostatic integration ---
    Phi = compute_geopotential(T, p_s, sigma_coord, phis)

    # --- 3. Kinetic energy and Bernoulli function ---
    K = 0.5 * (u**2 + v**2)
    B = K + Phi

    # --- 4. Horizontal dynamics ---

    # 4a. Vorticity
    zeta = _vorticity_3d(u, v, grid)
    abs_vor = zeta + grid.f[:, :, None]  # broadcast f (n_lat, n_lon) -> 3D

    # 4b. Bernoulli gradient
    dB_dx = _gradient_x_3d(B, grid)
    dB_dy = _gradient_y_3d(B, grid)

    # 4c. Pressure gradient force (σ-coordinate correction)
    ln_ps = jnp.log(p_s)
    ln_ps_field = Field(data=ln_ps, name="ln_ps", dims=("lat", "lon"),
                        units="", staggering="cell")
    dln_ps_dx = _gradient_x_2d(ln_ps_field, grid).data
    dln_ps_dy = _gradient_y_2d(ln_ps_field, grid).data

    pg_corr_x = R_d * T * dln_ps_dx[:, :, None]
    pg_corr_y = R_d * T * dln_ps_dy[:, :, None]

    # 4d. Vector-invariant momentum equations
    du_dt_data = abs_vor * v - dB_dx - pg_corr_x
    dv_dt_data = -abs_vor * u - dB_dy - pg_corr_y

    # --- 5. Surface pressure tendency and sigma-dot ---
    div_v = _divergence_3d(u, v, grid)

    sigma_top = sigma_coord.sigma_half[0]
    sigma_range = 1.0 - sigma_top

    D_vel_total = jnp.sum(
        div_v * dsigma[None, None, :], axis=-1
    )
    dp_s_dt_data = -p_s * D_vel_total / sigma_range

    sigma_dot = compute_sigma_dot(div_v, sigma_coord)

    # --- 7. Vertical advection ---
    vert_adv_T = vertical_advection(T, sigma_dot, sigma_coord)
    vert_adv_u = vertical_advection(u, sigma_dot, sigma_coord)
    vert_adv_v = vertical_advection(v, sigma_dot, sigma_coord)

    du_dt_data = du_dt_data + vert_adv_u
    dv_dt_data = dv_dt_data + vert_adv_v

    # --- 8. Thermodynamic equation ---
    dT_dx = _gradient_x_3d(T, grid)
    dT_dy = _gradient_y_3d(T, grid)
    horiz_adv_T = -(u * dT_dx + v * dT_dy)

    omega = compute_pressure_velocity(sigma_dot, p_s, dp_s_dt_data, sigma_coord)
    adiabatic = kappa * T * omega / p_full

    v_dot_grad_lnps = u * dln_ps_dx[:, :, None] + v * dln_ps_dy[:, :, None]
    adiabatic = adiabatic + kappa * T * v_dot_grad_lnps

    dT_dt_data = horiz_adv_T + vert_adv_T + adiabatic

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

    # --- 10. Add physics tendencies ---
    if physics_tendency is not None:
        du_dt_data = du_dt_data + physics_tendency.du_dt.data
        dv_dt_data = dv_dt_data + physics_tendency.dv_dt.data
        dT_dt_data = dT_dt_data + physics_tendency.dT_dt.data
        dp_s_dt_data = dp_s_dt_data + physics_tendency.dp_s_dt.data

    # --- Build tendency pytree ---
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


# ==============================================================================
# Model class
# ==============================================================================

def _filter_state(
    state: HydrostaticState,
    grid: LatLonGrid,
    mask: jnp.ndarray,
) -> HydrostaticState:
    """Apply polar filter to prognostic state variables."""
    return HydrostaticState(
        u=state.u.replace(data=fourier_filter_3d(state.u.data, grid, mask)),
        v=state.v.replace(data=fourier_filter_3d(state.v.data, grid, mask)),
        T=state.T.replace(data=fourier_filter_3d(state.T.data, grid, mask)),
        p_s=state.p_s.replace(data=fourier_filter(state.p_s.data, grid, mask)),
        phis=state.phis,
    )


class LatLonPrimitiveEquationModel:
    """Hydrostatic primitive equation model on a lat-lon grid.

    Parameters
    ----------
    grid : LatLonGrid
        Horizontal grid.
    sigma_coord : SigmaCoordinate
        Vertical coordinate.
    config : LatLonPrimitiveEquationConfig, optional
        Model configuration.
    dt : float
        Time step [seconds], needed to precompute the polar filter mask.
    """

    def __init__(
        self,
        grid: LatLonGrid,
        sigma_coord: SigmaCoordinate,
        config: LatLonPrimitiveEquationConfig | None = None,
        dt: float = 600.0,
    ):
        self.grid = grid
        self.sigma_coord = sigma_coord
        self.config = config or LatLonPrimitiveEquationConfig()

        # Precompute polar filter mask (constant array, not retraced)
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
        return latlon_hydrostatic_tendencies(
            state, self.grid, self.sigma_coord, self.config, physics_tendency,
            self.polar_mask,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: HydrostaticState,
        dt: float,
    ) -> HydrostaticState:
        """Advance one time step using SSP-RK3."""
        # Filter state BEFORE RK3 so the original state used in the
        # RK3 linear combinations is clean.
        if self.polar_mask is not None:
            state = _filter_state(state, self.grid, self.polar_mask)

        def tendency_fn(s):
            tend = latlon_hydrostatic_tendencies(
                s, self.grid, self.sigma_coord, self.config,
                polar_mask=self.polar_mask,
            )
            return HydrostaticState(
                u=s.u.replace(data=tend.du_dt.data),
                v=s.v.replace(data=tend.dv_dt.data),
                T=s.T.replace(data=tend.dT_dt.data),
                p_s=s.p_s.replace(data=tend.dp_s_dt.data),
                phis=s.phis.replace(data=jnp.zeros_like(s.phis.data)),
            )

        state_new = ssp_rk3_step(state, tendency_fn, dt)

        if self.config.use_conservation_fixer and self.config.fix_mass:
            from legoesm.core.conservation import fix_mass_hydrostatic_latlon
            state_new = fix_mass_hydrostatic_latlon(state_new, state, self.grid)

        return state_new

    @partial(jax.jit, static_argnums=(0, 3))
    def step_with_physics(
        self,
        state: HydrostaticState,
        dt: float,
        physics_fn=None,
    ) -> HydrostaticState:
        """Advance one time step with physics forcing.

        Parameters
        ----------
        state : HydrostaticState
            Current state.
        dt : float
            Time step [seconds].
        physics_fn : callable, optional
            Function (state, grid, sigma_coord) -> HydrostaticTendencies.
        """
        # Filter state BEFORE RK3
        if self.polar_mask is not None:
            state = _filter_state(state, self.grid, self.polar_mask)

        def tendency_fn(s):
            phys = None
            if physics_fn is not None:
                phys = physics_fn(s, self.grid, self.sigma_coord)
            tend = latlon_hydrostatic_tendencies(
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

        state_new = ssp_rk3_step(state, tendency_fn, dt)

        if self.config.use_conservation_fixer and self.config.fix_mass:
            from legoesm.core.conservation import fix_mass_hydrostatic_latlon
            state_new = fix_mass_hydrostatic_latlon(state_new, state, self.grid)

        return state_new

    def integrate(
        self,
        state: HydrostaticState,
        duration: float,
        dt: float,
        save_every: int = 1,
        physics_fn=None,
    ) -> tuple[HydrostaticState, list[HydrostaticState]]:
        """Integrate forward for a given duration."""
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
