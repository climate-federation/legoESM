"""Lat-lon C-grid FV ocean model with split-explicit barotropic/baroclinic stepping.

Arakawa C-grid staggering eliminates the 2*dx checkerboard null space
that plagues collocated A-grid formulations at moderate resolutions.

  u -> lon interfaces  (n_lat, n_lon+1, nlev)
  v -> lat interfaces  (n_lat+1, n_lon, nlev)
  eta, T, S -> cell centers (n_lat, n_lon [, nlev])

Uses the same split-explicit scheme as the A-grid LatLonOceanModel:
1. Compute 3D baroclinic tendencies (slow mode)
2. Update tracers (T, S) and 3D velocity with slow tendency
3. Run barotropic substeps (forward-backward) for free surface
4. Apply conservation fixers

Public API: state_new = model.step(state, dt)
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from legoesm.core.precision import cast_pytree
from legoesm.grids.latlon import LatLonGrid
from legoesm.ocean.vertical import OceanZStarCoordinate
from legoesm.ocean.state import (
    LatLonCGridOceanState,
    LatLonCGridOceanConfig,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    latlon_cgrid_ocean_baroclinic_tendencies,
)
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    barotropic_substeps_latlon_cgrid,
)


class LatLonCGridOceanModel:
    """Boussinesq hydrostatic ocean model on a C-grid latitude-longitude grid.

    Uses split-explicit time stepping with compact-stencil C-grid operators
    for pressure gradient and divergence.

    Parameters
    ----------
    grid : LatLonGrid
        Horizontal grid.
    z_coord : OceanZStarCoordinate
        Vertical coordinate.
    config : LatLonCGridOceanConfig, optional
        Model configuration.

    Example
    -------
    >>> grid = create_latlon_grid(90, 180)
    >>> z_coord = create_ocean_z_star(50)
    >>> model = LatLonCGridOceanModel(grid, z_coord)
    >>> state = rest_state_latlon_cgrid_ocean(grid, z_coord)
    >>> state_new = model.step(state, dt=3600.0)
    """

    def __init__(
        self,
        grid: LatLonGrid,
        z_coord: OceanZStarCoordinate,
        config: LatLonCGridOceanConfig | None = None,
    ):
        self.grid = grid
        self.z_coord = z_coord
        self.config = config or LatLonCGridOceanConfig()
        self._validate_config(self.config)

        if self.config.physics is not None:
            from legoesm.ocean.physics.combined import make_ocean_physics
            self._physics_fn = make_ocean_physics(self.config.physics)
        else:
            self._physics_fn = None

    @staticmethod
    def _validate_config(config: LatLonCGridOceanConfig) -> None:
        """Validate configuration ranges."""
        nonnegative = {
            "A_h": config.A_h,
            "K_h": config.K_h,
            "A_v": config.A_v,
            "K_v": config.K_v,
            "hyperdiff_coeff": config.hyperdiff_coeff,
            "barotropic_diffusion_alpha": config.barotropic_diffusion_alpha,
        }
        for name, value in nonnegative.items():
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value!r}")

        if config.n_barotropic_substeps < 1:
            raise ValueError(
                f"n_barotropic_substeps must be >= 1, got "
                f"{config.n_barotropic_substeps!r}",
            )
        if config.barotropic_diffusion_dt_ref <= 0.0:
            raise ValueError(
                f"barotropic_diffusion_dt_ref must be > 0, got "
                f"{config.barotropic_diffusion_dt_ref!r}",
            )
        if config.min_water_column_m <= 0.0:
            raise ValueError(
                f"min_water_column_m must be > 0, got "
                f"{config.min_water_column_m!r}",
            )

    def tendencies(self, state: LatLonCGridOceanState, surface_forcing=None):
        """Compute baroclinic tendencies."""
        return latlon_cgrid_ocean_baroclinic_tendencies(
            state, self.grid, self.z_coord, self.config,
            physics_fn=self._physics_fn,
            surface_forcing=surface_forcing,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: LatLonCGridOceanState, dt: float,
             surface_forcing=None) -> LatLonCGridOceanState:
        """Advance one time step using split-explicit stepping.

        Parameters
        ----------
        state : LatLonCGridOceanState
        dt : float
            Time step [seconds].
        surface_forcing : OceanSurfaceForcing or None

        Returns
        -------
        LatLonCGridOceanState
        """
        state = cast_pytree(state, None, "compute")

        u_mask_3d = state.u_mask.data[..., jnp.newaxis]
        v_mask_3d = state.v_mask.data[..., jnp.newaxis]
        mask_3d = state.land_mask.data[..., jnp.newaxis]

        # 1. Baroclinic tendencies
        tend = self.tendencies(state, surface_forcing)

        # 2. Update tracers
        T_new = state.T.data + dt * tend.dT_dt.data
        S_new = state.S.data + dt * tend.dS_dt.data

        # 3. Update 3D velocity with slow tendency
        u_new = state.u.data + dt * tend.du_dt.data
        v_new = state.v.data + dt * tend.dv_dt.data

        state_mid = state._replace(
            u=state.u.replace(data=u_new * u_mask_3d),
            v=state.v.replace(data=v_new * v_mask_3d),
            T=state.T.replace(data=T_new * mask_3d),
            S=state.S.replace(data=S_new * mask_3d),
        )

        # 4. Barotropic substeps
        dt_s = dt / self.config.n_barotropic_substeps
        state_new = barotropic_substeps_latlon_cgrid(
            state_mid, dt_s, self.config.n_barotropic_substeps,
            self.grid, self.z_coord, self.config,
        )

        return cast_pytree(state_new, None, "storage")

    def integrate(
        self,
        state: LatLonCGridOceanState,
        duration: float,
        dt: float,
        save_every: int = 1,
    ) -> tuple[LatLonCGridOceanState, list[LatLonCGridOceanState]]:
        """Integrate forward for a given duration.

        Parameters
        ----------
        state : LatLonCGridOceanState
        duration : float
            Total integration time [seconds].
        dt : float
            Time step [seconds].
        save_every : int
            Save state every N steps.

        Returns
        -------
        final_state, trajectory
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt!r}")
        if duration < 0.0:
            raise ValueError(f"duration must be >= 0, got {duration!r}")
        if save_every < 1:
            raise ValueError(f"save_every must be >= 1, got {save_every!r}")

        n_steps = int(duration / dt)
        if duration > 0.0 and n_steps < 1:
            raise ValueError(
                f"zero steps; increase duration or reduce dt "
                f"(duration={duration!r}, dt={dt!r})",
            )

        trajectory = [state]
        for i in range(n_steps):
            state = self.step(state, dt)
            if (i + 1) % save_every == 0:
                trajectory.append(state)

        return state, trajectory

    def integrate_scan(
        self,
        state: LatLonCGridOceanState,
        n_steps: int,
        dt: float,
    ) -> tuple[LatLonCGridOceanState, LatLonCGridOceanState]:
        """Integrate using jax.lax.scan (differentiable).

        Parameters
        ----------
        state : LatLonCGridOceanState
        n_steps : int
        dt : float

        Returns
        -------
        final_state, trajectory (stacked)
        """
        def scan_fn(state, _):
            new_state = self.step(state, dt)
            return new_state, new_state

        final_state, trajectory = jax.lax.scan(
            scan_fn, state, xs=None, length=n_steps,
        )
        return final_state, trajectory
