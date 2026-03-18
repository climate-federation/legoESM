"""FC-Gram Shallow Water Equations on the cubed-sphere.

Uses FC spectral operators for all horizontal derivatives,
replacing both centered differences and PPM transport.

    dh/dt = -fc_flux_divergence(h, u, v)    [FC mass continuity]
    du/dt =  (zeta + f) * v - dB/dx + D_u   [vector-invariant momentum]
    dv/dt = -(zeta + f) * u - dB/dy + D_v

References
----------
- Lyon & Bruno (2010): FC-Gram methods
- Lin (2004): FV3 dynamical core (equation structure)
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
)
from legoesm.core.conservation import (
    apply_conservation_fixer,
    zero_mean_tendency,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import IntegrationMixin
from legoesm import constants


class FCShallowWaterConfig(NamedTuple):
    """Configuration for the FC shallow-water model."""
    g: float = constants.g
    hyperdiff_coeff: float = 0.0
    use_conservation_fixer: bool = True
    fix_mass: bool = True
    fix_energy: bool = True
    time_integrator: str = "ssp_rk3"
    fc_d: int = 2
    fc_C: int = 4
    fc_degree: int = 5


def fc_shallow_water_tendencies(
    state: ShallowWaterState,
    grid: CubedSphereGrid,
    fc_config: FCOperatorConfig,
    config: FCShallowWaterConfig = FCShallowWaterConfig(),
) -> ShallowWaterTendencies:
    """Compute tendencies for the FC shallow water equations.

    Parameters
    ----------
    state : ShallowWaterState
    grid : CubedSphereGrid
    fc_config : FCOperatorConfig
    config : FCShallowWaterConfig

    Returns
    -------
    ShallowWaterTendencies
    """
    h = state.h
    u = state.u
    v = state.v
    h_s = state.h_s
    g = config.g

    # Mass continuity: FC spectral flux divergence
    dh_dt_data = fc_flux_divergence(h.data, u.data, v.data, grid, fc_config)
    dh_dt_data = zero_mean_tendency(dh_dt_data, grid)

    # Relative vorticity: FC spectral curl
    zeta = fc_curl_z(u.data, v.data, grid, fc_config)
    abs_vor = zeta + grid.f

    # Bernoulli function
    KE = 0.5 * (u.data**2 + v.data**2)
    B = KE + g * (h.data + h_s.data)

    # FC spectral gradients
    dB_dx = fc_gradient_x(B, grid, fc_config)
    dB_dy = fc_gradient_y(B, grid, fc_config)

    # Vector-invariant momentum
    du_dt_data = abs_vor * v.data - dB_dx
    dv_dt_data = -abs_vor * u.data - dB_dy

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


class FCShallowWaterModel(IntegrationMixin):
    """FC-Gram shallow water model on the cubed-sphere.

    Parameters
    ----------
    grid : CubedSphereGrid
    config : FCShallowWaterConfig, optional
    """

    def __init__(
        self,
        grid: CubedSphereGrid,
        config: FCShallowWaterConfig | None = None,
    ):
        self.grid = grid
        self.config = config or FCShallowWaterConfig()
        self.fc_config = build_fc_config(
            d=self.config.fc_d,
            C=self.config.fc_C,
            degree=self.config.fc_degree,
        )

    def tendencies(self, state: ShallowWaterState) -> ShallowWaterTendencies:
        return fc_shallow_water_tendencies(
            state, self.grid, self.fc_config, self.config,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: ShallowWaterState, dt: float) -> ShallowWaterState:
        def tendency_fn(s):
            tend = fc_shallow_water_tendencies(
                s, self.grid, self.fc_config, self.config,
            )
            return ShallowWaterState(
                h=s.h.replace(data=tend.dh_dt.data),
                u=s.u.replace(data=tend.du_dt.data),
                v=s.v.replace(data=tend.dv_dt.data),
                h_s=s.h_s.replace(data=jnp.zeros_like(s.h_s.data)),
            )

        state_new = dispatch_integrator(
            state, tendency_fn, dt, self.config.time_integrator,
        )

        if self.config.use_conservation_fixer:
            state_new = apply_conservation_fixer(
                state_new, state, self.grid,
                fix_mass=self.config.fix_mass,
                fix_energy=self.config.fix_energy,
                g=self.config.g,
            )

        return state_new
