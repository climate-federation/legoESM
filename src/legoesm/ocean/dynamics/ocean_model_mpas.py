"""MPAS ocean model on Voronoi meshes.

Orchestrates split-explicit time stepping: baroclinic (slow) tendencies
computed by :func:`mpas_ocean_baroclinic_tendencies`, then barotropic
(fast) substeps update the free surface and reconcile velocities.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import MPASOceanState, MPASOceanTendencies
from legoesm.grids.voronoi import VoronoiMesh
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_layer_thickness
from legoesm.ocean.dynamics.ocean_pe_mpas import mpas_ocean_baroclinic_tendencies
from legoesm.ocean.dynamics.barotropic_mpas import (
    barotropic_substeps_mpas,
    reconcile_3d_velocity,
)
from legoesm.ocean.conservation_mpas import mpas_ocean_conservation_fixer
from legoesm.ocean.freshwater import FreshwaterForcing, freshwater_eta_tendency


class MPASOceanModel:
    """MPAS ocean model with split-explicit time stepping.

    Parameters
    ----------
    mesh : VoronoiMesh
    z_coord : OceanZStarCoordinate
    config : MPASOceanConfig or None
    """

    def __init__(
        self,
        mesh: VoronoiMesh,
        z_coord: OceanZStarCoordinate,
        config: MPASOceanConfig | None = None,
    ):
        self.mesh = mesh
        self.z_coord = z_coord
        self.config = config or MPASOceanConfig()

    def tendencies(
        self,
        state: MPASOceanState,
        freshwater: FreshwaterForcing | None = None,
    ) -> MPASOceanTendencies:
        """Compute baroclinic tendencies."""
        return mpas_ocean_baroclinic_tendencies(
            state, self.mesh, self.z_coord, self.config,
            freshwater=freshwater,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: MPASOceanState,
        dt: float,
        freshwater: FreshwaterForcing | None = None,
    ) -> MPASOceanState:
        """Advance one full timestep (baroclinic + barotropic).

        Parameters
        ----------
        state : MPASOceanState
        dt : float
            Baroclinic timestep [s].
        freshwater : FreshwaterForcing or None
            Freshwater forcing (P, E, runoff, ice). If None, no freshwater.

        Returns
        -------
        MPASOceanState
        """
        config = self.config
        mesh = self.mesh
        z_coord = self.z_coord
        mask = state.land_mask.data

        # 1. Compute baroclinic tendencies
        tend = self.tendencies(state, freshwater=freshwater)

        # 2. Update tracers (forward Euler)
        T_new = state.T.data + dt * tend.dT_dt.data
        S_new = state.S.data + dt * tend.dS_dt.data

        # Mask land
        T_new = T_new * mask[:, jnp.newaxis]
        S_new = S_new * mask[:, jnp.newaxis]

        # 3. Update 3D velocity with baroclinic tendency
        u_baro = state.u.data + dt * tend.du_dt.data

        # 4. Barotropic substeps
        n_sub = config.n_barotropic_substeps
        dt_baro = dt / n_sub

        # Compute slow forcing for barotropic from baroclinic tendency
        # Depth-averaged tendency from 3D momentum
        h_k = compute_layer_thickness(
            state.eta.data, state.H_bathy.data, z_coord,
            min_water_column_m=config.min_water_column_m,
        )
        c1 = mesh.cellsOnEdge[0]
        c2 = mesh.cellsOnEdge[1]
        h_e_k = 0.5 * (h_k[c1] + h_k[c2])  # (nEdges, nlev)
        H_e = jnp.sum(h_e_k, axis=1)  # (nEdges,)
        F_slow = jnp.sum(tend.du_dt.data * h_e_k, axis=1) / jnp.maximum(H_e, 1e-10)

        # Create intermediate state with updated tracers for barotropic
        state_for_baro = state._replace(
            u=state.u.replace(data=u_baro),
        )

        # Freshwater mass flux for barotropic continuity equation
        F_slow_eta = None
        if freshwater is not None and config.freshwater_closure != "none":
            F_slow_eta = freshwater_eta_tendency(freshwater, config.rho_0) * mask

        eta_new, u_bar_new = barotropic_substeps_mpas(
            state_for_baro, mesh, z_coord, config, dt_baro, n_sub, F_slow,
            F_slow_eta=F_slow_eta,
        )

        # 5. Reconcile 3D velocity
        H_total = jnp.maximum(state.eta.data + state.H_bathy.data,
                              config.min_water_column_m)
        H_e_old = 0.5 * (H_total[c1] + H_total[c2])
        u_bar_old = jnp.sum(state.u.data * h_e_k, axis=1) / jnp.maximum(H_e_old, 1e-10)

        u_3d_new = reconcile_3d_velocity(
            u_baro, u_bar_old, u_bar_new, h_k, mesh, mask,
        )

        # Mask land edges
        edge_mask = mask[c1] * mask[c2]
        u_3d_new = u_3d_new * edge_mask[:, jnp.newaxis]

        state_new = MPASOceanState(
            u=state.u.replace(data=u_3d_new),
            T=state.T.replace(data=T_new),
            S=state.S.replace(data=S_new),
            eta=state.eta.replace(data=eta_new * mask),
            H_bathy=state.H_bathy,
            land_mask=state.land_mask,
        )

        # 6. Conservation fixers
        if config.use_conservation_fixer:
            state_new = mpas_ocean_conservation_fixer(
                state_new, state, mesh, z_coord, config,
            )

        return state_new

    def integrate(
        self,
        state: MPASOceanState,
        duration: float,
        dt: float,
        save_every: int = 1,
    ):
        """Forward integration.

        Parameters
        ----------
        state : MPASOceanState
        duration : float
            Total integration time [s].
        dt : float
            Timestep [s].
        save_every : int
            Save trajectory every N steps.

        Returns
        -------
        (final_state, trajectory)
        """
        n_steps = int(duration / dt)
        trajectory = []

        for i in range(n_steps):
            state = self.step(state, dt)
            if (i + 1) % save_every == 0:
                trajectory.append(state)

        return state, trajectory

    def integrate_scan(
        self,
        state: MPASOceanState,
        n_steps: int,
        dt: float,
    ):
        """Differentiable integration via jax.lax.scan.

        Parameters
        ----------
        state : MPASOceanState
        n_steps : int
        dt : float

        Returns
        -------
        (final_state, trajectory)
        """
        def scan_fn(carry, _):
            s = self.step(carry, dt)
            return s, s

        final, trajectory = jax.lax.scan(scan_fn, state, None, length=n_steps)
        return final, trajectory
