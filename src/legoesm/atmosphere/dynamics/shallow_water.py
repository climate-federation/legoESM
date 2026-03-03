"""Shallow Water Equations on the cubed-sphere.

The rotating shallow water equations in **vector-invariant form**:

    dh/dt = -div(h * v)                          [mass continuity]
    du/dt =  (zeta + f) * v - dB/dx + D_u        [x-momentum]
    dv/dt = -(zeta + f) * u - dB/dy + D_v        [y-momentum]

where:
    h      = fluid depth
    u, v   = velocity components (grid-aligned)
    zeta   = relative vorticity (dv/dx - du/dy)
    f      = Coriolis parameter (2*Omega*sin(lat))
    B      = Bernoulli function = K + g*(h + h_s)
    K      = kinetic energy per unit mass = 0.5*(u^2 + v^2)
    g      = gravitational acceleration
    h_s    = surface topography
    D_u, D_v = diffusion/hyperdiffusion terms

The vector-invariant form avoids explicit momentum advection, which:
- Eliminates the need for vector halo exchange in advection operators
- Conserves energy and potential enstrophy in the continuous limit
- Only requires scalar gradients (of B) across face boundaries

References
----------
- Williamson et al. (1992): A standard test set for numerical approximations
  to the shallow water equations in spherical geometry.
- Sadourny (1975): The dynamics of finite-difference models of the
  shallow water equations.
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
    divergence,
    curl_z,
    advect_upwind,
    advect_centered,
    hyperdiffusion,
)
from legoesm.core.conservation import apply_conservation_fixer
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.timestepping.ssp_rk3 import ssp_rk3_step
from legoesm.timestepping.ssp_rk54 import ssp_rk54_step
from legoesm import constants


class ShallowWaterConfig(NamedTuple):
    """Configuration for the shallow-water model."""
    g: float = constants.g                  # Gravitational acceleration [m/s^2]
    hyperdiff_coeff: float = 0.0            # Hyperdiffusion coefficient [m^4/s]
    use_conservation_fixer: bool = True      # Apply mass/energy fixers
    fix_mass: bool = True
    fix_energy: bool = True
    use_upwind_advection: bool = True       # Use upwind (True) or centered (False)
    time_integrator: str = "ssp_rk3"        # "ssp_rk3" or "ssp_rk54"/"ssp45"


def shallow_water_tendencies(
    state: ShallowWaterState,
    grid: CubedSphereGrid,
    config: ShallowWaterConfig = ShallowWaterConfig(),
) -> ShallowWaterTendencies:
    """Compute tendencies for the shallow water equations.

    Uses the vector-invariant form:
        du/dt =  (zeta + f) * v - dB/dx
        dv/dt = -(zeta + f) * u - dB/dy
    where B = K + g*(h + h_s) is the Bernoulli function and
    K = 0.5*(u^2 + v^2) is the kinetic energy per unit mass.

    This form avoids explicit momentum advection, which:
    - Eliminates vector halo issues (only scalar gradients of B)
    - Conserves energy and enstrophy in the continuous limit

    The vorticity zeta = dv/dx - du/dy is computed via curl_z(), which
    uses pad_halo_vector for correct velocity rotation at face boundaries.

    This is a pure function: state in, tendencies out.
    Fully compatible with jax.grad, jax.jit, jax.vmap.

    Parameters
    ----------
    state : ShallowWaterState
        Current state (h, u, v, h_s).
    grid : CubedSphereGrid
        The cubed-sphere grid.
    config : ShallowWaterConfig
        Model configuration.

    Returns
    -------
    ShallowWaterTendencies : Time derivatives (dh/dt, du/dt, dv/dt).
    """
    h = state.h
    u = state.u
    v = state.v
    h_s = state.h_s
    g = config.g

    # --- Mass continuity: dh/dt = -div(h*u, h*v) ---
    hu = h * u
    hv = h * v
    dh_dt = Field(
        data=-divergence(hu, hv, grid).data,
        name="dh_dt", dims=h.dims, units="m/s",
    )

    # --- Relative vorticity: zeta = dv/dx - du/dy ---
    # curl_z uses pad_halo_vector for correct velocity rotation at faces
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
    # du/dt =  (zeta + f) * v - dB/dx
    # dv/dt = -(zeta + f) * u - dB/dy
    du_dt_data = abs_vor * v.data - dB_dx.data
    dv_dt_data = -abs_vor * u.data - dB_dy.data

    # --- Hyperdiffusion (scale-selective damping) ---
    # Applied to all prognostic fields (u, v, h) to drain spurious energy
    # from grid-scale oscillations, especially at face boundaries.
    if config.hyperdiff_coeff > 0:
        diff_u = hyperdiffusion(u, grid, config.hyperdiff_coeff)
        diff_v = hyperdiffusion(v, grid, config.hyperdiff_coeff)
        diff_h = hyperdiffusion(h, grid, config.hyperdiff_coeff)
        du_dt_data = du_dt_data + diff_u.data
        dv_dt_data = dv_dt_data + diff_v.data
        dh_dt = Field(data=dh_dt.data + diff_h.data,
                       name="dh_dt", dims=h.dims, units="m/s")

    du_dt = Field(data=du_dt_data, name="du_dt", dims=u.dims, units="m/s^2")
    dv_dt = Field(data=dv_dt_data, name="dv_dt", dims=v.dims, units="m/s^2")

    return ShallowWaterTendencies(dh_dt=dh_dt, du_dt=du_dt, dv_dt=dv_dt)


class ShallowWaterModel:
    """Shallow water model on the cubed-sphere.

    This is the main user-facing class for Milestone 1.

    Parameters
    ----------
    grid : CubedSphereGrid
        The computational grid.
    config : ShallowWaterConfig, optional
        Model configuration.

    Example
    -------
    >>> grid = create_cubed_sphere(48)
    >>> model = ShallowWaterModel(grid)
    >>> state = williamson_test2(grid)
    >>> state_24h = model.integrate(state, duration=86400, dt=600)
    """

    def __init__(
        self,
        grid: CubedSphereGrid,
        config: ShallowWaterConfig | None = None,
    ):
        self.grid = grid
        self.config = config or ShallowWaterConfig()

    def tendencies(self, state: ShallowWaterState) -> ShallowWaterTendencies:
        """Compute tendencies (pure function wrapper)."""
        return shallow_water_tendencies(state, self.grid, self.config)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: ShallowWaterState, dt: float) -> ShallowWaterState:
        """Advance one time step using SSP-RK3.

        Parameters
        ----------
        state : ShallowWaterState
            Current state.
        dt : float
            Time step [seconds].

        Returns
        -------
        ShallowWaterState : State after one time step.
        """
        def tendency_fn(s):
            tend = shallow_water_tendencies(s, self.grid, self.config)
            # Return same pytree structure as state for tree_map compatibility.
            # Field metadata (name, dims, units) must match the state's fields
            # so that jax.tree.map(f, state, tendencies) works correctly.
            return ShallowWaterState(
                h=s.h.replace(data=tend.dh_dt.data),
                u=s.u.replace(data=tend.du_dt.data),
                v=s.v.replace(data=tend.dv_dt.data),
                h_s=s.h_s.replace(data=jnp.zeros_like(s.h_s.data)),
            )

        integrator = self.config.time_integrator.lower()
        if integrator in ("ssp_rk54", "ssp54", "ssp45", "rk54"):
            state_new = ssp_rk54_step(state, tendency_fn, dt)
        elif integrator in ("ssp_rk3", "ssp3", "rk3"):
            state_new = ssp_rk3_step(state, tendency_fn, dt)
        else:
            raise ValueError(f"Unsupported time_integrator={self.config.time_integrator!r}")

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
            Initial state.
        duration : float
            Total integration time [seconds].
        dt : float
            Time step [seconds].
        save_every : int
            Save state every N steps.

        Returns
        -------
        final_state : ShallowWaterState
            Final state.
        trajectory : list of ShallowWaterState
            Saved states at intervals.
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

        This is the preferred method for gradient computation.
        Returns the full trajectory as stacked pytree leaves.

        Parameters
        ----------
        state : ShallowWaterState
            Initial state.
        n_steps : int
            Number of time steps.
        dt : float
            Time step [seconds].

        Returns
        -------
        final_state : ShallowWaterState
            State after n_steps.
        trajectory : ShallowWaterState
            All intermediate states (each leaf shape: (n_steps, 6, n, n)).
        """
        def scan_fn(state, _):
            new_state = self.step(state, dt)
            return new_state, new_state

        final_state, trajectory = jax.lax.scan(
            scan_fn, state, jnp.arange(n_steps)
        )
        return final_state, trajectory
