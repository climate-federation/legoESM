"""FV3-Style Shallow Water Equations on the lat-lon grid.

Uses PPM reconstruction for mass transport (unsplit), and vector-invariant
form for momentum (same as centered). Includes a Fourier polar filter
for CFL stability near the poles.

    dh/dt = fv_flux_divergence_latlon(h, u, v)  [FV mass continuity]
    du/dt =  (zeta + f) * v - dB/dx + D_u       [vector-invariant momentum]
    dv/dt = -(zeta + f) * u - dB/dy + D_v

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
from legoesm.core.state import ShallowWaterState, ShallowWaterTendencies
from legoesm.core.operators_latlon import curl_z, hyperdiffusion
from legoesm.core.operators_fv_latlon import (
    fv_flux_divergence_latlon,
    fv_gradient_lon,
    fv_gradient_lat,
)
from legoesm.core.conservation import (
    apply_conservation_fixer_latlon,
    zero_mean_tendency_latlon,
)
from legoesm.grids.latlon import LatLonGrid
from legoesm.grids.polar_filter import compute_polar_filter_mask, fourier_filter
from legoesm.timestepping.ssp_rk3 import ssp_rk3_step
from legoesm.timestepping.ssp_rk34 import ssp_rk34_step
from legoesm.timestepping.ssp_rk54 import ssp_rk54_step
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

    # --- Mass continuity: dh/dt via FV PPM transport ---
    dh_dt_data = fv_flux_divergence_latlon(
        h.data, u.data, v.data, grid,
        limiter=config.use_limiter,
    )
    dh_dt_data = zero_mean_tendency_latlon(dh_dt_data, grid)

    # --- Relative vorticity ---
    zeta = curl_z(u, v, grid).data

    # --- Absolute vorticity ---
    abs_vor = zeta + grid.f

    # --- Bernoulli function: B = K + g*(h + h_s) ---
    kinetic_energy = 0.5 * (u.data**2 + v.data**2)
    bernoulli_data = kinetic_energy + g * (h.data + h_s.data)

    # PPM-compatible gradients
    dB_dx_data = fv_gradient_lon(bernoulli_data, grid)
    dB_dy_data = fv_gradient_lat(bernoulli_data, grid)

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

    dims = h.dims

    dh_dt = Field(data=dh_dt_data, name="dh_dt", dims=dims, units="m/s")
    du_dt = Field(data=du_dt_data, name="du_dt", dims=dims, units="m/s^2")
    dv_dt = Field(data=dv_dt_data, name="dv_dt", dims=dims, units="m/s^2")

    return ShallowWaterTendencies(dh_dt=dh_dt, du_dt=du_dt, dv_dt=dv_dt)


class FVShallowWaterLatLonModel:
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
        """Advance one time step using SSP-RK3 with PPM transport.

        Parameters
        ----------
        state : ShallowWaterState
        dt : float
            Time step [seconds].

        Returns
        -------
        ShallowWaterState
        """
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

        # Apply conservation fixers
        if self.config.use_conservation_fixer:
            state_new = apply_conservation_fixer_latlon(
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
        """Integrate forward for a given duration.

        Parameters
        ----------
        state : ShallowWaterState
        duration : float
            Total integration time [seconds].
        dt : float
            Time step [seconds].
        save_every : int
            Save state every N steps.

        Returns
        -------
        final_state : ShallowWaterState
        trajectory : list of ShallowWaterState
        """
        n_steps = int(duration / dt)
        trajectory = [state]

        for i in range(n_steps):
            state = self.step(state, dt)
            if (i + 1) % save_every == 0:
                trajectory.append(state)

        return state, trajectory

    def integrate_scan(
        self,
        state: ShallowWaterState,
        n_steps: int,
        dt: float,
    ) -> tuple[ShallowWaterState, ShallowWaterState]:
        """Integrate using jax.lax.scan (differentiable, JIT-friendly).

        Parameters
        ----------
        state : ShallowWaterState
        n_steps : int
        dt : float

        Returns
        -------
        final_state : ShallowWaterState
        trajectory : ShallowWaterState
            All intermediate states (each leaf: (n_steps, n_lat, n_lon)).
        """
        def scan_fn(state, _):
            new_state = self.step(state, dt)
            return new_state, new_state

        final_state, trajectory = jax.lax.scan(
            scan_fn, state, jnp.arange(n_steps)
        )
        return final_state, trajectory
