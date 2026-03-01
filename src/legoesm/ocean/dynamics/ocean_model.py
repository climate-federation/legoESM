"""Ocean model class with split-explicit barotropic/baroclinic stepping.

Uses a proper ocean split-explicit scheme:
1. Compute 3D baroclinic tendencies (slow mode)
2. Update tracers (T, S) and 3D velocity with slow tendency
3. Run barotropic substeps (forward-backward) for free surface
4. Reconcile 3D velocities with updated barotropic mode

The barotropic solver is NOT wrapped inside RK3 stages (unlike the
atmospheric split-explicit scheme) because the barotropic solver
updates the SAME variables (eta, u, v) as the slow tendency. Wrapping
it in RK3 creates inconsistencies from the convex blending of
barotropic-updated states with un-updated states.

Public API: state_new = model.step(state, dt)
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.vertical import OceanZStarCoordinate
from legoesm.ocean.state import OceanState, OceanConfig
from legoesm.ocean.dynamics.ocean_pe import ocean_baroclinic_tendencies
from legoesm.ocean.dynamics.barotropic import barotropic_substeps


class OceanModel:
    """Boussinesq hydrostatic ocean model on the cubed-sphere.

    Uses split-explicit time stepping:
    - Baroclinic (3D) tendencies computed once per step.
    - Forward-backward barotropic substeps for free surface.

    Parameters
    ----------
    grid : CubedSphereGrid
        Horizontal grid.
    z_coord : OceanZStarCoordinate
        Vertical coordinate.
    config : OceanConfig, optional
        Model configuration. Defaults to OceanConfig().

    Example
    -------
    >>> grid = create_cubed_sphere(48)
    >>> z_coord = create_ocean_z_star(50)
    >>> model = OceanModel(grid, z_coord)
    >>> state = rest_state_ocean(grid, z_coord)
    >>> state_new = model.step(state, dt=3600.0)
    """

    def __init__(
        self,
        grid: CubedSphereGrid,
        z_coord: OceanZStarCoordinate,
        config: OceanConfig | None = None,
    ):
        self.grid = grid
        self.z_coord = z_coord
        self.config = config or OceanConfig()

    def tendencies(self, state: OceanState):
        """Compute baroclinic tendencies (pure function wrapper)."""
        return ocean_baroclinic_tendencies(
            state, self.grid, self.z_coord, self.config,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: OceanState, dt: float) -> OceanState:
        """Advance one time step using split-explicit stepping.

        1. Compute baroclinic (slow) tendencies at current state.
        2. Update T, S with slow tendency (forward Euler).
        3. Update 3D u, v with slow tendency (forward Euler).
        4. Run barotropic substeps to update eta and reconcile u, v.
        5. Apply conservation fixers.

        Parameters
        ----------
        state : OceanState
            Current state.
        dt : float
            Time step [seconds].

        Returns
        -------
        OceanState : State after one time step.
        """
        # --- 1. Baroclinic tendencies ---
        tend = ocean_baroclinic_tendencies(
            state, self.grid, self.z_coord, self.config,
        )

        # --- 2. Update tracers (forward Euler) ---
        T_new = state.T.data + dt * tend.dT_dt.data
        S_new = state.S.data + dt * tend.dS_dt.data

        # --- 3. Update 3D velocity with slow tendency ---
        # Planetary Coriolis (f*v) is NOT in the baroclinic tendency — it
        # is applied at the barotropic substep level for stability. The
        # tendency here contains only relative vorticity, PGF, advection,
        # and mixing.
        u_new = state.u.data + dt * tend.du_dt.data
        v_new = state.v.data + dt * tend.dv_dt.data

        state_mid = state._replace(
            u=state.u.replace(data=u_new),
            v=state.v.replace(data=v_new),
            T=state.T.replace(data=T_new),
            S=state.S.replace(data=S_new),
        )

        # --- 4. Barotropic substeps ---
        # eta is updated ONLY here (not by the slow tendency).
        # The slow u/v tendency is already in state_mid, so F_slow=0
        # in the barotropic solver to avoid double-counting.
        dt_s = dt / self.config.n_barotropic_substeps
        state_new = barotropic_substeps(
            state_mid, state_mid,  # slow_tend unused (F_slow=0)
            dt_s, self.config.n_barotropic_substeps,
            self.grid, self.z_coord, self.config,
        )

        # --- 5. Conservation fixers ---
        if self.config.use_conservation_fixer:
            from legoesm.ocean.conservation import ocean_conservation_fixer
            state_new = ocean_conservation_fixer(
                state_new, state, self.grid, self.z_coord, self.config,
            )

        return state_new

    def integrate(
        self,
        state: OceanState,
        duration: float,
        dt: float,
        save_every: int = 1,
    ) -> tuple[OceanState, list[OceanState]]:
        """Integrate forward for a given duration.

        Parameters
        ----------
        state : OceanState
            Initial state.
        duration : float
            Total integration time [seconds].
        dt : float
            Time step [seconds].
        save_every : int
            Save state every N steps.

        Returns
        -------
        final_state : OceanState
        trajectory : list of OceanState
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
        state: OceanState,
        n_steps: int,
        dt: float,
    ) -> tuple[OceanState, OceanState]:
        """Integrate using jax.lax.scan (differentiable, JIT-friendly).

        Parameters
        ----------
        state : OceanState
            Initial state.
        n_steps : int
            Number of time steps.
        dt : float
            Time step [seconds].

        Returns
        -------
        final_state : OceanState
        trajectory : OceanState (stacked, each leaf shape (n_steps, ...))
        """
        def scan_fn(state, _):
            new_state = self.step(state, dt)
            return new_state, new_state

        final_state, trajectory = jax.lax.scan(
            scan_fn, state, xs=None, length=n_steps,
        )
        return final_state, trajectory
