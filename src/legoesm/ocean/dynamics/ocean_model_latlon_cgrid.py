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
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    compute_layer_thickness,
)
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


def _forward_backward_coriolis_3d(
    u: jnp.ndarray,
    v: jnp.ndarray,
    dt: float,
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    config: LatLonCGridOceanConfig,
    u_mask: jnp.ndarray,
    v_mask: jnp.ndarray,
    land_mask: jnp.ndarray,
    eta: jnp.ndarray,
    H_bathy: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply forward-backward (Matsuno) Coriolis to baroclinic perturbation velocity.

    The forward-backward scheme is unconditionally neutral for the
    inertial oscillation: the amplification factor is exactly 1,
    unlike forward Euler which amplifies by sqrt(1 + (f*dt)^2).

    Steps:
      1. Compute perturbation velocity u' = u - U_bar, v' = v - V_bar
      2. Forward:  u'_new = u' + dt * f_u * avg(v'  -> u-points)
      3. Backward: v'_new = v' - dt * f_v * avg(u'_new -> v-points)
      4. Reconstruct: u_new = u'_new + U_bar, v_new = v'_new + V_bar

    Parameters
    ----------
    u : (n_lat, n_lon+1, nlev) full 3D u velocity
    v : (n_lat+1, n_lon, nlev) full 3D v velocity
    dt : time step [s]
    grid : LatLonGrid
    z_coord : OceanZStarCoordinate
    config : LatLonCGridOceanConfig
    u_mask : (n_lat, n_lon+1)
    v_mask : (n_lat+1, n_lon)
    land_mask : (n_lat, n_lon)
    eta, H_bathy : (n_lat, n_lon) for layer thickness computation

    Returns
    -------
    u_new, v_new : updated velocities with Coriolis applied
    """
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    nlev = u.shape[-1]

    u_mask_3d = u_mask[..., jnp.newaxis]
    v_mask_3d = v_mask[..., jnp.newaxis]

    # --- Layer thickness at face points for depth averaging ---
    min_water_col = jnp.asarray(config.min_water_column_m, dtype=eta.dtype)
    eta_floor = min_water_col - H_bathy
    eta_safe = jnp.maximum(eta, eta_floor) * land_mask

    h_k = compute_layer_thickness(
        eta_safe, H_bathy, z_coord,
        min_water_column_m=config.min_water_column_m,
    )

    # h at u-faces
    h_west = jnp.roll(h_k, 1, axis=1)
    h_u = 0.5 * (h_west + h_k)
    h_u = jnp.concatenate([h_u, h_u[:, 0:1, :]], axis=1)

    # h at v-faces
    h_v_interior = 0.5 * (h_k[:-1] + h_k[1:])
    zero_h = jnp.zeros((1, n_lon, nlev), dtype=h_k.dtype)
    h_v = jnp.concatenate([zero_h, h_v_interior, zero_h], axis=0)

    # --- Depth-averaged velocity (barotropic component) ---
    H_u = jnp.maximum(jnp.sum(h_u, axis=-1), min_water_col)
    U_bar = jnp.sum(u * h_u, axis=-1) / H_u * u_mask

    H_v = jnp.maximum(jnp.sum(h_v, axis=-1), min_water_col)
    V_bar = jnp.sum(v * h_v, axis=-1) / H_v * v_mask

    # --- Perturbation velocity ---
    u_prime = (u - U_bar[..., jnp.newaxis]) * u_mask_3d
    v_prime = (v - V_bar[..., jnp.newaxis]) * v_mask_3d

    # --- Coriolis parameter at face points ---
    f_cell = grid.f.astype(u.dtype)

    # f at u-points: average of flanking cells
    f_u = 0.5 * (jnp.roll(f_cell, 1, axis=1) + f_cell)
    f_u = jnp.concatenate([f_u, f_u[:, 0:1]], axis=1)  # (n_lat, n_lon+1)

    # f at v-points: average of flanking cells
    f_v_interior = 0.5 * (f_cell[:-1] + f_cell[1:])
    f_v = jnp.concatenate(
        [f_cell[0:1], f_v_interior, f_cell[-1:]], axis=0,
    )  # (n_lat+1, n_lon)

    # --- Forward step: update u' using old v' ---
    # Average v' to u-points (Sadourny 4-point average)
    v_west = jnp.roll(v_prime, 1, axis=1)
    v_at_u = 0.25 * (v_prime[:-1] + v_prime[1:]
                      + v_west[:-1] + v_west[1:])
    v_at_u = jnp.concatenate([v_at_u, v_at_u[:, 0:1, :]], axis=1)

    u_prime_new = (u_prime + dt * f_u[:, :, jnp.newaxis] * v_at_u) * u_mask_3d

    # --- Backward step: update v' using NEW u' ---
    # Average u'_new to v-points (Sadourny 4-point average)
    u_at_v_interior = 0.25 * (
        u_prime_new[:-1, :-1] + u_prime_new[:-1, 1:]
        + u_prime_new[1:, :-1] + u_prime_new[1:, 1:]
    )
    zero_row = jnp.zeros((1, n_lon, nlev), dtype=u.dtype)
    u_at_v = jnp.concatenate([zero_row, u_at_v_interior, zero_row], axis=0)

    v_prime_new = (v_prime - dt * f_v[:, :, jnp.newaxis] * u_at_v) * v_mask_3d

    # --- Reconstruct full velocity ---
    u_new = u_prime_new + U_bar[..., jnp.newaxis]
    v_new = v_prime_new + V_bar[..., jnp.newaxis]

    return u_new, v_new


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

        # 1. Baroclinic tendencies (non-Coriolis)
        tend = self.tendencies(state, surface_forcing)

        # 2. Update tracers
        T_new = state.T.data + dt * tend.dT_dt.data
        S_new = state.S.data + dt * tend.dS_dt.data

        # 3. Update 3D velocity with non-Coriolis tendency
        u_star = state.u.data + dt * tend.du_dt.data
        v_star = state.v.data + dt * tend.dv_dt.data

        # 4. Forward-backward Coriolis on perturbation velocity
        #
        # Forward Euler Coriolis amplifies inertial oscillations by
        # sqrt(1 + (f*dt)^2) per step, causing blowup at high latitudes.
        # Forward-backward (Matsuno) stepping is unconditionally neutral:
        #   u' += dt * f * v'_at_u          (forward: old v')
        #   v' -= dt * f * u'_new_at_v      (backward: new u')
        # This matches the barotropic solver's Coriolis treatment.
        u_star, v_star = _forward_backward_coriolis_3d(
            u_star, v_star, dt, self.grid, self.z_coord, self.config,
            state.u_mask.data, state.v_mask.data, state.land_mask.data,
            state.eta.data, state.H_bathy.data,
        )

        state_mid = state._replace(
            u=state.u.replace(data=u_star * u_mask_3d),
            v=state.v.replace(data=v_star * v_mask_3d),
            T=state.T.replace(data=T_new * mask_3d),
            S=state.S.replace(data=S_new * mask_3d),
        )

        # 5. Barotropic substeps
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
