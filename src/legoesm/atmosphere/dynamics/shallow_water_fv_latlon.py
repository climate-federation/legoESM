"""Shallow Water Equations on the lat-lon grid (FV framework).

Uses centered divergence for mass transport (consistent with the centered
gradient and vorticity operators in the momentum equation) and
vector-invariant form for momentum. Includes a Fourier polar filter
for CFL stability near the poles.

    dh/dt = -div(h*u, h*v)                       [centered mass continuity]
    du/dt =  (zeta + f) * v - dB/dx + D_u        [vector-invariant momentum]
    dv/dt = -(zeta + f) * u - dB/dy + D_v

Note: PPM transport is used in the PE and CE lat-lon models for scalar
fields (T, p_s, theta, rho, tracers) where it does not directly feed
back into the momentum equation. For the SW mass equation on an A-grid,
centered divergence avoids energy-inconsistent feedback between mass and
momentum operators that would otherwise cause exponential v-velocity growth.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import ShallowWaterState, ShallowWaterTendencies
from legoesm.core.operators_latlon import (
    curl_z, divergence, gradient_x, gradient_y, hyperdiffusion,
)
from legoesm.core.conservation import (
    apply_conservation_fixer,
    zero_mean_tendency,
)
from legoesm.grids.latlon import LatLonGrid
from legoesm.grids.polar_filter import compute_polar_filter_mask, fourier_filter
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import IntegrationMixin
from legoesm import constants


class FVShallowWaterLatLonConfig(NamedTuple):
    """Configuration for the FV shallow-water model on a lat-lon grid."""
    g: float = constants.g
    hyperdiff_coeff: float = 0.0
    use_conservation_fixer: bool = True
    fix_mass: bool = True
    fix_energy: bool = True
    time_integrator: str = "ssp_rk3"
    use_limiter: bool = True
    use_polar_filter: bool = True
    polar_filter_cutoff_deg: float = 60.0
    polar_filter_max_wave_speed: float = 300.0


def fv_shallow_water_tendencies_latlon(
    state: ShallowWaterState,
    grid: LatLonGrid,
    config: FVShallowWaterLatLonConfig = FVShallowWaterLatLonConfig(),
    polar_filter_mask: jnp.ndarray | None = None,
) -> ShallowWaterTendencies:
    """Compute tendencies for the FV shallow water equations on a lat-lon grid.

    Parameters
    ----------
    state : ShallowWaterState
    grid : LatLonGrid
    config : FVShallowWaterLatLonConfig
    polar_filter_mask : jax.Array, optional
        Precomputed polar filter mask. If None, no polar filtering.

    Returns
    -------
    ShallowWaterTendencies
    """
    h = state.h
    u = state.u
    v = state.v
    h_s = state.h_s
    g = config.g

    # --- Mass continuity: centered divergence (consistent with momentum) ---
    hu = h.replace(data=h.data * u.data)
    hv = h.replace(data=h.data * v.data)
    dh_dt_data = -divergence(hu, hv, grid).data
    dh_dt_data = zero_mean_tendency(dh_dt_data, grid)

    # --- Relative vorticity ---
    zeta = curl_z(u, v, grid).data

    # --- Absolute vorticity ---
    abs_vor = zeta + grid.f

    # --- Bernoulli function: B = K + g*(h + h_s) ---
    kinetic_energy = 0.5 * (u.data**2 + v.data**2)
    bernoulli_data = kinetic_energy + g * (h.data + h_s.data)

    # Centered gradients (consistent with 2nd-order vorticity operator)
    dims = h.dims
    B_field = Field(data=bernoulli_data, name="B", dims=dims, units="m^2/s^2")
    dB_dx_data = gradient_x(B_field, grid).data
    dB_dy_data = gradient_y(B_field, grid).data

    # --- Vector-invariant momentum equations ---
    du_dt_data = abs_vor * v.data - dB_dx_data
    dv_dt_data = -abs_vor * u.data - dB_dy_data

    # --- Hyperdiffusion on velocity only ---
    if config.hyperdiff_coeff > 0:
        diff_u = hyperdiffusion(u, grid, config.hyperdiff_coeff)
        diff_v = hyperdiffusion(v, grid, config.hyperdiff_coeff)
        du_dt_data = du_dt_data + diff_u.data
        dv_dt_data = dv_dt_data + diff_v.data

    # --- Polar filter all tendencies ---
    if config.use_polar_filter and polar_filter_mask is not None:
        dh_dt_data = fourier_filter(dh_dt_data, grid, polar_filter_mask)
        du_dt_data = fourier_filter(du_dt_data, grid, polar_filter_mask)
        dv_dt_data = fourier_filter(dv_dt_data, grid, polar_filter_mask)

    dh_dt = Field(data=dh_dt_data, name="dh_dt", dims=dims, units="m/s")
    du_dt = Field(data=du_dt_data, name="du_dt", dims=dims, units="m/s^2")
    dv_dt = Field(data=dv_dt_data, name="dv_dt", dims=dims, units="m/s^2")

    return ShallowWaterTendencies(dh_dt=dh_dt, du_dt=du_dt, dv_dt=dv_dt)


class FVShallowWaterLatLonModel(IntegrationMixin):
    """FV3-style shallow water model on the lat-lon grid.

    Parameters
    ----------
    grid : LatLonGrid
    config : FVShallowWaterLatLonConfig, optional
    dt : float, optional
        Time step for polar filter mask precomputation.
    """

    def __init__(
        self,
        grid: LatLonGrid,
        config: FVShallowWaterLatLonConfig | None = None,
        dt: float = 600.0,
    ):
        import warnings
        warnings.warn(
            "FVShallowWaterLatLonModel (A-grid) is deprecated. Use the "
            "cubed-sphere or icosahedral shallow-water models instead. "
            "See #115.",
            FutureWarning, stacklevel=2,
        )
        self.grid = grid
        self.config = config or FVShallowWaterLatLonConfig()

        # Precompute polar filter mask
        if self.config.use_polar_filter:
            self.polar_filter_mask = compute_polar_filter_mask(
                grid,
                dt=dt,
                max_wave_speed=self.config.polar_filter_max_wave_speed,
                cutoff_lat_deg=self.config.polar_filter_cutoff_deg,
            )
        else:
            self.polar_filter_mask = None

    def tendencies(
        self,
        state: ShallowWaterState,
    ) -> ShallowWaterTendencies:
        """Compute tendencies (pure function wrapper)."""
        return fv_shallow_water_tendencies_latlon(
            state, self.grid, self.config, self.polar_filter_mask,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: ShallowWaterState, dt: float) -> ShallowWaterState:
        """Advance one time step using SSP-RK3 with PPM transport."""
        from legoesm.core.precision import cast_pytree
        state = cast_pytree(state, None, "compute")

        def tendency_fn(s):
            tend = fv_shallow_water_tendencies_latlon(
                s, self.grid, self.config, self.polar_filter_mask,
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

        # Apply conservation fixers
        if self.config.use_conservation_fixer:
            state_new = apply_conservation_fixer(
                state_new, state, self.grid,
                fix_mass=self.config.fix_mass,
                fix_energy=self.config.fix_energy,
                g=self.config.g,
            )

        return cast_pytree(state_new, None, "storage")

