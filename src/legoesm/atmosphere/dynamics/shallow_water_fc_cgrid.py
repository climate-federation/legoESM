"""FC-Gram Shallow Water with divergence damping on the cubed-sphere.

Combines FC spectral operators with C-grid-style divergence damping.
This is the FC analog of shallow_water_cgrid.py.

References
----------
- Lyon & Bruno (2010): FC-Gram methods
- Lin (2004): FV3 divergence damping strategy
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import ShallowWaterState, ShallowWaterTendencies
from legoesm.core.operators_fc import (
    FCOperatorConfig,
    build_fc_config,
    fc_gradient_x,
    fc_gradient_y,
    fc_curl_z,
    fc_flux_divergence,
    fc_hyperdiffusion,
    fc_divergence_damping,
)
from legoesm.core.conservation import (
    apply_conservation_fixer,
    zero_mean_tendency,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.timestepping.ssp_rk3 import ssp_rk3_step
from legoesm.timestepping.ssp_rk34 import ssp_rk34_step
from legoesm.timestepping.ssp_rk54 import ssp_rk54_step
from legoesm import constants


class FCCGShallowWaterConfig(NamedTuple):
    """Configuration for the FC + div-damping shallow-water model."""
    g: float = constants.g
    div_damp_2: float = 0.0
    div_damp_4: float = 0.0
    hyperdiff_coeff: float = 0.0
    use_conservation_fixer: bool = True
    fix_mass: bool = True
    fix_energy: bool = True
    time_integrator: str = "ssp_rk3"
    fc_d: int = 2
    fc_C: int = 4
    fc_degree: int = 5


def fc_cgrid_shallow_water_tendencies(
    state: ShallowWaterState,
    grid: CubedSphereGrid,
    fc_config: FCOperatorConfig,
    config: FCCGShallowWaterConfig = FCCGShallowWaterConfig(),
) -> ShallowWaterTendencies:
    """Compute FC + div-damping shallow water tendencies.

    Parameters
    ----------
    state : ShallowWaterState
    grid : CubedSphereGrid
    fc_config : FCOperatorConfig
    config : FCCGShallowWaterConfig

    Returns
    -------
    ShallowWaterTendencies
    """
    h = state.h
    u = state.u
    v = state.v
    h_s = state.h_s
    g = config.g

    # Mass continuity
    dh_dt_data = fc_flux_divergence(h.data, u.data, v.data, grid, fc_config)
    dh_dt_data = zero_mean_tendency(dh_dt_data, grid)

    # Vorticity
    zeta = fc_curl_z(u.data, v.data, grid, fc_config)
    abs_vor = zeta + grid.f

    # Bernoulli gradient
    KE = 0.5 * (u.data**2 + v.data**2)
    B = KE + g * (h.data + h_s.data)
    dB_dx = fc_gradient_x(B, grid, fc_config)
    dB_dy = fc_gradient_y(B, grid, fc_config)

    # Vector-invariant momentum
    du_dt_data = abs_vor * v.data - dB_dx
    dv_dt_data = -abs_vor * u.data - dB_dy

    # Divergence damping (FC spectral — single exchange)
    du_damp, dv_damp = fc_divergence_damping(
        u.data, v.data, grid, fc_config)
    du_dt_data = du_dt_data + du_damp
    dv_dt_data = dv_dt_data + dv_damp

    # Hyperdiffusion
    if config.hyperdiff_coeff > 0:
        du_dt_data = du_dt_data + fc_hyperdiffusion(
            u.data, grid, fc_config, config.hyperdiff_coeff)
        dv_dt_data = dv_dt_data + fc_hyperdiffusion(
            v.data, grid, fc_config, config.hyperdiff_coeff)

    dims = h.dims
    return ShallowWaterTendencies(
        dh_dt=Field(data=dh_dt_data, name="dh_dt", dims=dims, units="m/s"),
        du_dt=Field(data=du_dt_data, name="du_dt", dims=dims, units="m/s^2"),
        dv_dt=Field(data=dv_dt_data, name="dv_dt", dims=dims, units="m/s^2"),
    )


class FCCGShallowWaterModel:
    """FC + divergence-damping shallow water model on the cubed-sphere.

    Parameters
    ----------
    grid : CubedSphereGrid
    config : FCCGShallowWaterConfig, optional
    """

    def __init__(
        self,
        grid: CubedSphereGrid,
        config: FCCGShallowWaterConfig | None = None,
    ):
        self.grid = grid
        self.config = config or FCCGShallowWaterConfig()
        self.fc_config = build_fc_config(
            d=self.config.fc_d,
            C=self.config.fc_C,
            degree=self.config.fc_degree,
            div_damp_2=self.config.div_damp_2,
            div_damp_4=self.config.div_damp_4,
        )

    def tendencies(self, state: ShallowWaterState) -> ShallowWaterTendencies:
        return fc_cgrid_shallow_water_tendencies(
            state, self.grid, self.fc_config, self.config,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: ShallowWaterState, dt: float) -> ShallowWaterState:
        def tendency_fn(s):
            tend = fc_cgrid_shallow_water_tendencies(
                s, self.grid, self.fc_config, self.config,
            )
            return ShallowWaterState(
                h=s.h.replace(data=tend.dh_dt.data),
                u=s.u.replace(data=tend.du_dt.data),
                v=s.v.replace(data=tend.dv_dt.data),
                h_s=s.h_s.replace(data=jnp.zeros_like(s.h_s.data)),
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

        if self.config.use_conservation_fixer:
            state_new = apply_conservation_fixer(
                state_new, state, self.grid,
                fix_mass=self.config.fix_mass,
                fix_energy=self.config.fix_energy,
                g=self.config.g,
            )

        return state_new

    def integrate(
        self,
        state: ShallowWaterState,
        duration: float,
        dt: float,
        save_every: int = 1,
    ) -> tuple[ShallowWaterState, list[ShallowWaterState]]:
        n_steps = int(duration / dt)
        trajectory = [state]
        for i in range(n_steps):
            state = self.step(state, dt)
            if (i + 1) % save_every == 0:
                trajectory.append(state)
        return state, trajectory
