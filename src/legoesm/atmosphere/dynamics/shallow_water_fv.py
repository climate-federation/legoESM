"""FV3-Style Shallow Water Equations on the cubed-sphere.

Uses PPM reconstruction with Lin-Rood directional splitting for mass
transport, and vector-invariant form for momentum (same as centered).

    dh/dt = fv_flux_divergence(h, u, v)    [FV mass continuity]
    du/dt =  (zeta + f) * v - dB/dx + D_u  [vector-invariant momentum]
    dv/dt = -(zeta + f) * u - dB/dy + D_v

The FV mass equation uses PPM + directional splitting to eliminate
cube-edge artifacts. Momentum uses the same vector-invariant form
as the centered scheme (only scalar gradients cross face boundaries).

References
----------
- Lin & Rood (1996): Multidimensional Flux-Form Semi-Lagrangian Transport
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
from legoesm.core.operators import (
    gradient_x,
    gradient_y,
    curl_z,
    hyperdiffusion,
)
from legoesm.core.operators_fv import fv_flux_divergence
from legoesm.core.conservation import (
    apply_conservation_fixer,
    zero_mean_tendency,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.timestepping.ssp_rk3 import ssp_rk3_step
from legoesm.timestepping.ssp_rk34 import ssp_rk34_step
from legoesm.timestepping.ssp_rk54 import ssp_rk54_step
from legoesm import constants


class FVShallowWaterConfig(NamedTuple):
    """Configuration for the FV shallow-water model."""
    g: float = constants.g
    hyperdiff_coeff: float = 0.0     # Only on u,v (PPM handles h dissipation)
    use_conservation_fixer: bool = True
    fix_mass: bool = True
    fix_energy: bool = True
    time_integrator: str = "ssp_rk3"
    use_limiter: bool = True


def fv_shallow_water_tendencies(
    state: ShallowWaterState,
    grid: CubedSphereGrid,
    config: FVShallowWaterConfig = FVShallowWaterConfig(),
    dt: float = 600.0,
    x_first: bool = True,
) -> ShallowWaterTendencies:
    """Compute tendencies for the FV shallow water equations.

    Mass uses PPM + Lin-Rood directional splitting.
    Momentum uses vector-invariant form (same as centered).

    Parameters
    ----------
    state : ShallowWaterState
    grid : CubedSphereGrid
    config : FVShallowWaterConfig
    dt : float
        Time step for operator splitting.
    x_first : bool
        Direction ordering for operator splitting.

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
    dh_dt_data = fv_flux_divergence(
        h.data, u.data, v.data, grid, dt,
        limiter=config.use_limiter, x_first=x_first,
    )
    # Conservation cleanup
    dh_dt_data = zero_mean_tendency(dh_dt_data, grid)

    dh_dt = Field(
        data=dh_dt_data,
        name="dh_dt", dims=h.dims, units="m/s",
    )

    # --- Relative vorticity: zeta = dv/dx - du/dy ---
    zeta = curl_z(u, v, grid).data

    # --- Absolute vorticity: zeta + f ---
    abs_vor = zeta + grid.f

    # --- Bernoulli function: B = K + g*(h + h_s) ---
    kinetic_energy = 0.5 * (u.data**2 + v.data**2)
    bernoulli_data = kinetic_energy + g * (h.data + h_s.data)
    bernoulli = Field(data=bernoulli_data, name="bernoulli",
                      dims=h.dims, units="m^2/s^2", staggering="cell")
    dB_dx = gradient_x(bernoulli, grid)
    dB_dy = gradient_y(bernoulli, grid)

    # --- Vector-invariant momentum equations ---
    du_dt_data = abs_vor * v.data - dB_dx.data
    dv_dt_data = -abs_vor * u.data - dB_dy.data

    # --- Hyperdiffusion on velocity only ---
    if config.hyperdiff_coeff > 0:
        diff_u = hyperdiffusion(u, grid, config.hyperdiff_coeff)
        diff_v = hyperdiffusion(v, grid, config.hyperdiff_coeff)
        du_dt_data = du_dt_data + diff_u.data
        dv_dt_data = dv_dt_data + diff_v.data

    du_dt = Field(data=du_dt_data, name="du_dt", dims=u.dims, units="m/s^2")
    dv_dt = Field(data=dv_dt_data, name="dv_dt", dims=v.dims, units="m/s^2")

    return ShallowWaterTendencies(dh_dt=dh_dt, du_dt=du_dt, dv_dt=dv_dt)


class FVShallowWaterModel:
    """FV3-style shallow water model on the cubed-sphere.

    Same interface as ShallowWaterModel but uses PPM + Lin-Rood
    for mass transport instead of centered differences.

    Parameters
    ----------
    grid : CubedSphereGrid
    config : FVShallowWaterConfig, optional
    """

    def __init__(
        self,
        grid: CubedSphereGrid,
        config: FVShallowWaterConfig | None = None,
    ):
        self.grid = grid
        self.config = config or FVShallowWaterConfig()
        self._step_count = 0

    def tendencies(
        self,
        state: ShallowWaterState,
        dt: float = 600.0,
        x_first: bool = True,
    ) -> ShallowWaterTendencies:
        """Compute tendencies (pure function wrapper)."""
        return fv_shallow_water_tendencies(
            state, self.grid, self.config, dt, x_first
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: ShallowWaterState, dt: float) -> ShallowWaterState:
        """Advance one time step using SSP-RK3 with PPM transport.

        Alternates x_first each step for directional symmetry.

        Parameters
        ----------
        state : ShallowWaterState
        dt : float
            Time step [seconds].

        Returns
        -------
        ShallowWaterState
        """
        # Alternate splitting direction for symmetry
        x_first = (self._step_count % 2 == 0)
        self._step_count += 1

        def tendency_fn(s):
            tend = fv_shallow_water_tendencies(
                s, self.grid, self.config, dt, x_first
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
            All intermediate states (each leaf: (n_steps, 6, n, n)).
        """
        def scan_fn(state, _):
            new_state = self.step(state, dt)
            return new_state, new_state

        final_state, trajectory = jax.lax.scan(
            scan_fn, state, jnp.arange(n_steps)
        )
        return final_state, trajectory
