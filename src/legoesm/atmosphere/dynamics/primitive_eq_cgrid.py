"""Hydrostatic PE with divergence damping on the cubed-sphere.

Extends the centered PE solver with C-grid-style divergence damping and
PPM transport for mass/scalars. This is the FD analog of
primitive_eq_fc_cgrid.py, using the same A-grid storage with:

1. PPM flux divergence for surface pressure (resolves 2Δx checkerboard)
2. PPM scalar advection for temperature (resolves 2Δx checkerboard)
3. 2nd + 4th order divergence damping on momentum
4. Fixed hyperdiffusion (compact inner ∇²) for velocity

References
----------
- Lin (2004): FV3 divergence damping strategy
- Simmons & Burridge (1981): Hydrostatic PE formulation
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState, HydrostaticTendencies
from legoesm.core.operators import (
    gradient_x,
    gradient_y,
    hyperdiffusion,
    global_integral,
)
from legoesm.core.operators_3d import (
    vorticity_3d as _vorticity_3d,
    gradient_x_3d as _gradient_x_3d,
    gradient_y_3d as _gradient_y_3d,
    divergence_3d as _divergence_3d,
    hyperdiffusion_3d as _hyperdiffusion_3d,
    laplacian_compact_3d as _laplacian_compact_3d,
    fv_flux_divergence_3d as _fv_flux_divergence_3d,
    fv_scalar_advection_3d as _fv_scalar_advection_3d,
)
from legoesm.core.conservation import zero_mean_tendency
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.vertical import (
    SigmaCoordinate,
    pressure_from_sigma,
    compute_geopotential,
    compute_sigma_dot,
    vertical_advection,
    compute_pressure_velocity,
)
from legoesm.timestepping.ssp_rk3 import ssp_rk3_step
from legoesm.timestepping.ssp_rk34 import ssp_rk34_step
from legoesm.timestepping.ssp_rk54 import ssp_rk54_step
from legoesm.atmosphere.dynamics.edge_blending import (
    blend_scalar_cube_edges,
    blend_vector_cube_edges,
)
from legoesm import constants


class CGPrimitiveEquationConfig(NamedTuple):
    """Configuration for the C-grid PE model on cubed-sphere."""
    g: float = constants.g
    hyperdiff_coeff: float = 0.0
    hyperdiff_ps_coeff: float = 0.0
    div_damp_2: float = 0.0       # 2nd-order divergence damping [m²/s]
    div_damp_4: float = 0.0       # 4th-order divergence damping [m⁴/s]
    use_conservation_fixer: bool = True
    fix_mass: bool = True
    anchor_mass_to_initial: bool = False
    time_integrator: str = "ssp_rk3"
    edge_blend_uv: float = 0.0
    edge_blend_T: float = 0.0
    edge_blend_p_s: float = 0.0
    edge_blend_width: int = 1


def cgrid_hydrostatic_tendencies(
    state: HydrostaticState,
    grid: CubedSphereGrid,
    sigma_coord: SigmaCoordinate,
    config: CGPrimitiveEquationConfig = CGPrimitiveEquationConfig(),
    physics_tendency: HydrostaticTendencies | None = None,
) -> HydrostaticTendencies:
    """Compute C-grid hydrostatic PE tendencies."""
    u = state.u.data
    v = state.v.data
    T = state.T.data
    p_s = state.p_s.data
    phis = state.phis.data

    R_d = constants.R_d
    kappa = constants.kappa
    dsigma = sigma_coord.dsigma

    p_s = jnp.clip(p_s, 100.0, 2.0e6)

    p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)
    Phi = compute_geopotential(T, p_s, sigma_coord, phis)

    K = 0.5 * (u**2 + v**2)
    B = K + Phi

    zeta = _vorticity_3d(u, v, grid)
    abs_vor = zeta + grid.f[..., None]

    dB_dx = _gradient_x_3d(B, grid)
    dB_dy = _gradient_y_3d(B, grid)

    ln_ps = jnp.log(p_s)
    ln_ps_field = Field(data=ln_ps, name="ln_ps", dims=("face", "x", "y"),
                        units="", staggering="cell")
    dln_ps_dx = gradient_x(ln_ps_field, grid).data
    dln_ps_dy = gradient_y(ln_ps_field, grid).data

    pg_corr_x = R_d * T * dln_ps_dx[..., None]
    pg_corr_y = R_d * T * dln_ps_dy[..., None]

    du_dt_data = abs_vor * v - dB_dx - pg_corr_x
    dv_dt_data = -abs_vor * u - dB_dy - pg_corr_y

    # --- Divergence damping (2nd + 4th order) ---
    div_v = _divergence_3d(u, v, grid)

    if config.div_damp_2 > 0:
        ddiv_dx = _gradient_x_3d(div_v, grid)
        ddiv_dy = _gradient_y_3d(div_v, grid)
        du_dt_data = du_dt_data + config.div_damp_2 * ddiv_dx
        dv_dt_data = dv_dt_data + config.div_damp_2 * ddiv_dy

    if config.div_damp_4 > 0:
        lap_div = _laplacian_compact_3d(div_v, grid)
        grad_lap_x = _gradient_x_3d(lap_div, grid)
        grad_lap_y = _gradient_y_3d(lap_div, grid)
        du_dt_data = du_dt_data - config.div_damp_4 * grad_lap_x
        dv_dt_data = dv_dt_data - config.div_damp_4 * grad_lap_y

    # --- Surface pressure tendency via PPM ---
    sigma_top = sigma_coord.sigma_half[0]
    sigma_range = 1.0 - sigma_top

    div_ps_v = _fv_flux_divergence_3d(
        jnp.broadcast_to(p_s[..., None], T.shape),
        u, v, grid,
    )
    dp_s_dt_data = jnp.sum(
        div_ps_v * dsigma[None, None, None, :], axis=-1,
    ) / sigma_range
    dp_s_dt_data = zero_mean_tendency(dp_s_dt_data, grid)

    # Centered divergence for sigma-dot diagnostic
    sigma_dot = compute_sigma_dot(div_v, sigma_coord)

    vert_adv_T = vertical_advection(T, sigma_dot, sigma_coord)
    vert_adv_u = vertical_advection(u, sigma_dot, sigma_coord)
    vert_adv_v = vertical_advection(v, sigma_dot, sigma_coord)

    du_dt_data = du_dt_data + vert_adv_u
    dv_dt_data = dv_dt_data + vert_adv_v

    # --- Temperature: centered advection (energy-consistent with momentum) ---
    # PPM T advection is energy-inconsistent with centered vector-invariant
    # momentum on the cubed sphere, causing exponential T growth at face
    # boundaries. Centered T advection preserves energy consistency;
    # the 2Δx checkerboard mode is handled by hyperdiffusion with the
    # compact inner Laplacian.
    dT_dx = _gradient_x_3d(T, grid)
    dT_dy = _gradient_y_3d(T, grid)
    horiz_adv_T = -(u * dT_dx + v * dT_dy)

    omega = compute_pressure_velocity(sigma_dot, p_s, dp_s_dt_data, sigma_coord)
    adiabatic = kappa * T * omega / p_full

    v_dot_grad_lnps = u * dln_ps_dx[..., None] + v * dln_ps_dy[..., None]
    adiabatic = adiabatic + kappa * T * v_dot_grad_lnps

    dT_dt_data = horiz_adv_T + vert_adv_T + adiabatic

    # --- Hyperdiffusion (velocity + temperature) ---
    # The compact inner ∇² in hyperdiffusion resolves the 2Δx checkerboard
    # that centered T advection cannot see.
    if config.hyperdiff_coeff > 0:
        du_dt_data = du_dt_data + _hyperdiffusion_3d(u, grid, config.hyperdiff_coeff)
        dv_dt_data = dv_dt_data + _hyperdiffusion_3d(v, grid, config.hyperdiff_coeff)
        dT_dt_data = dT_dt_data + _hyperdiffusion_3d(T, grid, config.hyperdiff_coeff)

    if config.hyperdiff_ps_coeff > 0:
        ps_field = Field(data=p_s, name="p_s", dims=("face", "x", "y"),
                         units="Pa", staggering="cell")
        dp_s_dt_data = dp_s_dt_data + hyperdiffusion(
            ps_field, grid, config.hyperdiff_ps_coeff
        ).data

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


def _apply_cgrid_hydro_edge_blend(
    state: HydrostaticState,
    grid: CubedSphereGrid,
    config: CGPrimitiveEquationConfig,
) -> HydrostaticState:
    """Apply variable-specific cubed-sphere edge blending."""
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
            T_data, config.edge_blend_T, width=config.edge_blend_width
        )
    if config.edge_blend_p_s > 0.0:
        p_s_data = blend_scalar_cube_edges(
            p_s_data, config.edge_blend_p_s, width=config.edge_blend_width
        )

    return state._replace(
        u=state.u.replace(data=u_data),
        v=state.v.replace(data=v_data),
        T=state.T.replace(data=T_data),
        p_s=state.p_s.replace(data=p_s_data),
    )


class CGPrimitiveEquationModel:
    """C-grid PE model on the cubed-sphere.

    Uses PPM transport for mass/scalars, centered vector-invariant
    momentum, and divergence damping.

    Parameters
    ----------
    grid : CubedSphereGrid
    sigma_coord : SigmaCoordinate
    config : CGPrimitiveEquationConfig, optional
    """

    def __init__(
        self,
        grid: CubedSphereGrid,
        sigma_coord: SigmaCoordinate,
        config: CGPrimitiveEquationConfig | None = None,
    ):
        self.grid = grid
        self.sigma_coord = sigma_coord
        self.config = config or CGPrimitiveEquationConfig()
        self._target_mass = None

    def tendencies(
        self,
        state: HydrostaticState,
        physics_tendency: HydrostaticTendencies | None = None,
    ) -> HydrostaticTendencies:
        return cgrid_hydrostatic_tendencies(
            state, self.grid, self.sigma_coord,
            self.config, physics_tendency,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: HydrostaticState, dt: float) -> HydrostaticState:
        def tendency_fn(s):
            tend = cgrid_hydrostatic_tendencies(
                s, self.grid, self.sigma_coord, self.config,
            )
            return HydrostaticState(
                u=s.u.replace(data=tend.du_dt.data),
                v=s.v.replace(data=tend.dv_dt.data),
                T=s.T.replace(data=tend.dT_dt.data),
                p_s=s.p_s.replace(data=tend.dp_s_dt.data),
                phis=s.phis.replace(data=jnp.zeros_like(s.phis.data)),
            )

        integrator = self.config.time_integrator.lower()
        if integrator in ("ssp_rk54", "ssp54", "ssp45", "rk54"):
            state_new = ssp_rk54_step(state, tendency_fn, dt)
        elif integrator in ("ssp_rk34", "ssp34", "rk34"):
            state_new = ssp_rk34_step(state, tendency_fn, dt)
        elif integrator in ("ssp_rk3", "ssp3", "rk3"):
            state_new = ssp_rk3_step(state, tendency_fn, dt)
        else:
            raise ValueError(
                f"Unsupported time_integrator={self.config.time_integrator!r}"
            )

        if (
            self.config.edge_blend_uv > 0.0
            or self.config.edge_blend_T > 0.0
            or self.config.edge_blend_p_s > 0.0
        ):
            state_new = _apply_cgrid_hydro_edge_blend(
                state_new, self.grid, self.config
            )

        if self.config.use_conservation_fixer and self.config.fix_mass:
            if self.config.anchor_mass_to_initial:
                from legoesm.core.conservation import fix_mass_hydrostatic_target
                if self._target_mass is None:
                    self._target_mass = global_integral(state.p_s, self.grid)
                state_new = fix_mass_hydrostatic_target(
                    state_new, self._target_mass, self.grid,
                )
            else:
                from legoesm.core.conservation import fix_mass_hydrostatic
                state_new = fix_mass_hydrostatic(state_new, state, self.grid)

        return state_new

    @partial(jax.jit, static_argnums=(0, 3))
    def step_with_physics(
        self,
        state: HydrostaticState,
        dt: float,
        physics_fn=None,
    ) -> HydrostaticState:
        def tendency_fn(s):
            phys = None
            if physics_fn is not None:
                phys = physics_fn(s, self.grid, self.sigma_coord)
            tend = cgrid_hydrostatic_tendencies(
                s, self.grid, self.sigma_coord, self.config, phys,
            )
            return HydrostaticState(
                u=s.u.replace(data=tend.du_dt.data),
                v=s.v.replace(data=tend.dv_dt.data),
                T=s.T.replace(data=tend.dT_dt.data),
                p_s=s.p_s.replace(data=tend.dp_s_dt.data),
                phis=s.phis.replace(data=jnp.zeros_like(s.phis.data)),
            )

        integrator = self.config.time_integrator.lower()
        if integrator in ("ssp_rk54", "ssp54", "ssp45", "rk54"):
            state_new = ssp_rk54_step(state, tendency_fn, dt)
        elif integrator in ("ssp_rk34", "ssp34", "rk34"):
            state_new = ssp_rk34_step(state, tendency_fn, dt)
        elif integrator in ("ssp_rk3", "ssp3", "rk3"):
            state_new = ssp_rk3_step(state, tendency_fn, dt)
        else:
            raise ValueError(
                f"Unsupported time_integrator={self.config.time_integrator!r}"
            )

        if (
            self.config.edge_blend_uv > 0.0
            or self.config.edge_blend_T > 0.0
            or self.config.edge_blend_p_s > 0.0
        ):
            state_new = _apply_cgrid_hydro_edge_blend(
                state_new, self.grid, self.config
            )

        if self.config.use_conservation_fixer and self.config.fix_mass:
            if self.config.anchor_mass_to_initial:
                from legoesm.core.conservation import fix_mass_hydrostatic_target
                if self._target_mass is None:
                    self._target_mass = global_integral(state.p_s, self.grid)
                state_new = fix_mass_hydrostatic_target(
                    state_new, self._target_mass, self.grid,
                )
            else:
                from legoesm.core.conservation import fix_mass_hydrostatic
                state_new = fix_mass_hydrostatic(state_new, state, self.grid)

        return state_new

    def integrate(
        self,
        state: HydrostaticState,
        duration: float,
        dt: float,
        save_every: int = 1,
        physics_fn=None,
    ) -> tuple[HydrostaticState, list[HydrostaticState]]:
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

    def integrate_scan(
        self,
        state: HydrostaticState,
        n_steps: int,
        dt: float,
    ) -> tuple[HydrostaticState, HydrostaticState]:
        def scan_fn(state, _):
            new_state = self.step(state, dt)
            return new_state, new_state
        final_state, trajectory = jax.lax.scan(
            scan_fn, state, jnp.arange(n_steps)
        )
        return final_state, trajectory
