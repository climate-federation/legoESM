"""FC-Gram Hydrostatic PE with divergence damping on the cubed-sphere.

Same as primitive_eq_fc.py but adds divergence damping to momentum.
This is the PE analog of shallow_water_cgrid.py.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState, HydrostaticTendencies
from legoesm.core.operators import global_integral
from legoesm.core.operators_fc import (
    FCOperatorConfig,
    build_fc_config,
    fc_gradient_x,
    fc_gradient_y,
    fc_hyperdiffusion,
    fc_flux_divergence,
)
from legoesm.core.operators_fc_3d import (
    fc_curl_z_3d,
    fc_gradient_x_3d,
    fc_gradient_y_3d,
    fc_divergence_3d,
    fc_hyperdiffusion_3d,
    fc_scalar_advection_3d,
    fc_divergence_damping_3d,
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


class FCCGPrimitiveEquationConfig(NamedTuple):
    """Configuration for the FC + div-damping PE model."""
    g: float = constants.g
    hyperdiff_coeff: float = 0.0
    hyperdiff_ps_coeff: float = 0.0
    use_conservation_fixer: bool = True
    fix_mass: bool = True
    anchor_mass_to_initial: bool = False
    time_integrator: str = "ssp_rk3"
    edge_blend_uv: float = 0.0
    edge_blend_T: float = 0.0
    edge_blend_p_s: float = 0.0
    edge_blend_width: int = 1
    div_damp_2: float = 0.0
    div_damp_4: float = 0.0
    fc_d: int = 2
    fc_C: int = 4
    fc_degree: int = 5


def fc_cgrid_hydrostatic_tendencies(
    state: HydrostaticState,
    grid: CubedSphereGrid,
    sigma_coord: SigmaCoordinate,
    fc_config: FCOperatorConfig,
    config: FCCGPrimitiveEquationConfig = FCCGPrimitiveEquationConfig(),
    physics_tendency: HydrostaticTendencies | None = None,
) -> HydrostaticTendencies:
    """Compute FC + div-damping hydrostatic PE tendencies."""
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

    zeta = fc_curl_z_3d(u, v, grid, fc_config)
    abs_vor = zeta + grid.f[..., None]

    dB_dx = fc_gradient_x_3d(B, grid, fc_config)
    dB_dy = fc_gradient_y_3d(B, grid, fc_config)

    ln_ps = jnp.log(p_s)
    dln_ps_dx = fc_gradient_x(ln_ps, grid, fc_config)
    dln_ps_dy = fc_gradient_y(ln_ps, grid, fc_config)

    pg_corr_x = R_d * T * dln_ps_dx[..., None]
    pg_corr_y = R_d * T * dln_ps_dy[..., None]

    du_dt_data = abs_vor * v - dB_dx - pg_corr_x
    dv_dt_data = -abs_vor * u - dB_dy - pg_corr_y

    # --- Divergence damping (applied per-level) ---
    du_damp, dv_damp = fc_divergence_damping_3d(u, v, grid, fc_config)
    du_dt_data = du_dt_data + du_damp
    dv_dt_data = dv_dt_data + dv_damp

    # Surface pressure tendency
    sigma_top = sigma_coord.sigma_half[0]
    sigma_range = 1.0 - sigma_top

    u_int = jnp.sum(u * dsigma[None, None, None, :], axis=-1)
    v_int = jnp.sum(v * dsigma[None, None, None, :], axis=-1)

    dp_s_dt_data = fc_flux_divergence(
        p_s, u_int / sigma_range, v_int / sigma_range, grid, fc_config,
    )
    dp_s_dt_data = zero_mean_tendency(dp_s_dt_data, grid)

    div_v = fc_divergence_3d(u, v, grid, fc_config)
    sigma_dot = compute_sigma_dot(div_v, sigma_coord)

    vert_adv_T = vertical_advection(T, sigma_dot, sigma_coord)
    vert_adv_u = vertical_advection(u, sigma_dot, sigma_coord)
    vert_adv_v = vertical_advection(v, sigma_dot, sigma_coord)

    du_dt_data = du_dt_data + vert_adv_u
    dv_dt_data = dv_dt_data + vert_adv_v

    horiz_adv_T = fc_scalar_advection_3d(T, u, v, grid, fc_config)

    omega = compute_pressure_velocity(sigma_dot, p_s, dp_s_dt_data, sigma_coord)
    adiabatic = kappa * T * omega / p_full

    v_dot_grad_lnps = u * dln_ps_dx[..., None] + v * dln_ps_dy[..., None]
    adiabatic = adiabatic + kappa * T * v_dot_grad_lnps

    dT_dt_data = horiz_adv_T + vert_adv_T + adiabatic

    if config.hyperdiff_coeff > 0:
        du_dt_data = du_dt_data + fc_hyperdiffusion_3d(u, grid, fc_config, config.hyperdiff_coeff)
        dv_dt_data = dv_dt_data + fc_hyperdiffusion_3d(v, grid, fc_config, config.hyperdiff_coeff)
        dT_dt_data = dT_dt_data + fc_hyperdiffusion_3d(T, grid, fc_config, config.hyperdiff_coeff)

    if config.hyperdiff_ps_coeff > 0:
        dp_s_dt_data = dp_s_dt_data + fc_hyperdiffusion(
            p_s, grid, fc_config, config.hyperdiff_ps_coeff)

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


class FCCGPrimitiveEquationModel:
    """FC + divergence-damping PE model on the cubed-sphere.

    Parameters
    ----------
    grid : CubedSphereGrid
    sigma_coord : SigmaCoordinate
    config : FCCGPrimitiveEquationConfig, optional
    """

    def __init__(
        self,
        grid: CubedSphereGrid,
        sigma_coord: SigmaCoordinate,
        config: FCCGPrimitiveEquationConfig | None = None,
    ):
        self.grid = grid
        self.sigma_coord = sigma_coord
        self.config = config or FCCGPrimitiveEquationConfig()
        self._target_mass = None

        self.fc_config = build_fc_config(
            d=self.config.fc_d,
            C=self.config.fc_C,
            degree=self.config.fc_degree,
            div_damp_2=self.config.div_damp_2,
            div_damp_4=self.config.div_damp_4,
        )

        for name in ("edge_blend_uv", "edge_blend_T", "edge_blend_p_s"):
            value = float(getattr(self.config, name))
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")

    def tendencies(
        self,
        state: HydrostaticState,
        physics_tendency: HydrostaticTendencies | None = None,
    ) -> HydrostaticTendencies:
        return fc_cgrid_hydrostatic_tendencies(
            state, self.grid, self.sigma_coord,
            self.fc_config, self.config, physics_tendency,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: HydrostaticState, dt: float) -> HydrostaticState:
        def tendency_fn(s):
            tend = fc_cgrid_hydrostatic_tendencies(
                s, self.grid, self.sigma_coord,
                self.fc_config, self.config,
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
            from legoesm.atmosphere.dynamics.primitive_eq_fc import (
                _apply_fc_hydro_edge_blend,
            )
            state_new = _apply_fc_hydro_edge_blend(
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
            tend = fc_cgrid_hydrostatic_tendencies(
                s, self.grid, self.sigma_coord,
                self.fc_config, self.config, phys,
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
            from legoesm.atmosphere.dynamics.primitive_eq_fc import (
                _apply_fc_hydro_edge_blend,
            )
            state_new = _apply_fc_hydro_edge_blend(
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
