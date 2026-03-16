"""Lat-lon FV ocean model with split-explicit barotropic/baroclinic stepping.

Uses the same split-explicit scheme as OceanModel (cubed-sphere):
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

from legoesm.grids.latlon import LatLonGrid
from legoesm.ocean.vertical import OceanZStarCoordinate
from legoesm.ocean.state import LatLonOceanState, LatLonOceanConfig
from legoesm.ocean.dynamics.ocean_pe_latlon import latlon_ocean_baroclinic_tendencies
from legoesm.ocean.dynamics.barotropic_latlon import barotropic_substeps_latlon


class LatLonOceanModel:
    """Boussinesq hydrostatic ocean model on a latitude-longitude grid.

    Uses split-explicit time stepping with PPM finite-volume transport.

    Parameters
    ----------
    grid : LatLonGrid
        Horizontal grid.
    z_coord : OceanZStarCoordinate
        Vertical coordinate.
    config : LatLonOceanConfig, optional
        Model configuration.

    Example
    -------
    >>> grid = create_latlon_grid(90, 180)
    >>> z_coord = create_ocean_z_star(50)
    >>> model = LatLonOceanModel(grid, z_coord)
    >>> state = rest_state_latlon_ocean(grid, z_coord)
    >>> state_new = model.step(state, dt=3600.0)
    """

    def __init__(
        self,
        grid: LatLonGrid,
        z_coord: OceanZStarCoordinate,
        config: LatLonOceanConfig | None = None,
    ):
        self.grid = grid
        self.z_coord = z_coord
        self.config = config or LatLonOceanConfig()
        self._validate_config(self.config)

        if self.config.physics is not None:
            from legoesm.ocean.physics.combined import make_ocean_physics
            self._physics_fn = make_ocean_physics(self.config.physics)
        else:
            self._physics_fn = None

    @staticmethod
    def _validate_config(config: LatLonOceanConfig) -> None:
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
                f"min_water_column_m must be > 0, got {config.min_water_column_m!r}",
            )

    def _assert_runtime_invariants(self, state: LatLonOceanState) -> None:
        """Host-side runtime checks."""
        mask = state.land_mask.data
        wet = mask > 0.5

        finite_ok = bool(
            jnp.all(jnp.isfinite(state.u.data))
            & jnp.all(jnp.isfinite(state.v.data))
            & jnp.all(jnp.isfinite(state.T.data))
            & jnp.all(jnp.isfinite(state.S.data))
            & jnp.all(jnp.isfinite(state.eta.data))
        )
        if not finite_ok:
            raise FloatingPointError(
                "LatLon ocean runtime check failed: non-finite state detected",
            )

        if bool(jnp.any(wet)):
            water_col = state.eta.data + state.H_bathy.data
            min_wc = float(jnp.min(jnp.where(wet, water_col, jnp.inf)))
            if min_wc < self.config.min_water_column_m:
                raise ValueError(
                    f"Water column too small: {min_wc:.6g} m < "
                    f"{self.config.min_water_column_m:.6g} m",
                )

            eta_abs = float(jnp.max(jnp.abs(jnp.where(wet, state.eta.data, 0.0))))
            if eta_abs > self.config.max_abs_eta_m:
                raise ValueError(
                    f"|eta| exceeded threshold: {eta_abs:.6g} m > "
                    f"{self.config.max_abs_eta_m:.6g} m",
                )

            T_ocean = jnp.where(wet[..., jnp.newaxis], state.T.data, jnp.nan)
            T_min, T_max = float(jnp.nanmin(T_ocean)), float(jnp.nanmax(T_ocean))
            if T_min < self.config.temperature_min_c or T_max > self.config.temperature_max_c:
                raise ValueError(
                    f"Temperature out of bounds: [{T_min:.3f}, {T_max:.3f}] C",
                )

            S_ocean = jnp.where(wet[..., jnp.newaxis], state.S.data, jnp.nan)
            S_min, S_max = float(jnp.nanmin(S_ocean)), float(jnp.nanmax(S_ocean))
            if S_min < self.config.salinity_min_psu or S_max > self.config.salinity_max_psu:
                raise ValueError(
                    f"Salinity out of bounds: [{S_min:.3f}, {S_max:.3f}] PSU",
                )

    def tendencies(self, state: LatLonOceanState):
        """Compute baroclinic tendencies."""
        return latlon_ocean_baroclinic_tendencies(
            state, self.grid, self.z_coord, self.config,
            physics_fn=self._physics_fn,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: LatLonOceanState, dt: float) -> LatLonOceanState:
        """Advance one time step using split-explicit stepping.

        Parameters
        ----------
        state : LatLonOceanState
        dt : float
            Time step [seconds].

        Returns
        -------
        LatLonOceanState
        """
        # 1. Baroclinic tendencies
        tend = self.tendencies(state)

        # 2. Update tracers
        T_new = state.T.data + dt * tend.dT_dt.data
        S_new = state.S.data + dt * tend.dS_dt.data

        # 3. Update 3D velocity with slow tendency
        u_new = state.u.data + dt * tend.du_dt.data
        v_new = state.v.data + dt * tend.dv_dt.data

        state_mid = state._replace(
            u=state.u.replace(data=u_new),
            v=state.v.replace(data=v_new),
            T=state.T.replace(data=T_new),
            S=state.S.replace(data=S_new),
        )

        # 4. Barotropic substeps
        dt_s = dt / self.config.n_barotropic_substeps
        state_new = barotropic_substeps_latlon(
            state_mid, dt_s, self.config.n_barotropic_substeps,
            self.grid, self.z_coord, self.config,
        )

        # 5. Conservation fixers
        if self.config.use_conservation_fixer:
            from legoesm.ocean.conservation_latlon import (
                latlon_ocean_conservation_fixer,
            )
            state_new = latlon_ocean_conservation_fixer(
                state_new, state, self.grid, self.z_coord, self.config,
            )

        return state_new

    def step_checked(
        self, state: LatLonOceanState, dt: float,
    ) -> LatLonOceanState:
        """Advance one timestep with optional runtime checks."""
        state_new = self.step(state, dt)
        if self.config.enable_runtime_checks:
            self._assert_runtime_invariants(state_new)
        return state_new

    def integrate(
        self,
        state: LatLonOceanState,
        duration: float,
        dt: float,
        save_every: int = 1,
    ) -> tuple[LatLonOceanState, list[LatLonOceanState]]:
        """Integrate forward for a given duration.

        Parameters
        ----------
        state : LatLonOceanState
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
        step_fn = (
            self.step_checked if self.config.enable_runtime_checks else self.step
        )

        for i in range(n_steps):
            state = step_fn(state, dt)
            if (i + 1) % save_every == 0:
                trajectory.append(state)

        return state, trajectory

    def integrate_scan(
        self,
        state: LatLonOceanState,
        n_steps: int,
        dt: float,
    ) -> tuple[LatLonOceanState, LatLonOceanState]:
        """Integrate using jax.lax.scan (differentiable).

        Parameters
        ----------
        state : LatLonOceanState
        n_steps : int
        dt : float

        Returns
        -------
        final_state, trajectory (stacked)
        """
        if self.config.enable_runtime_checks:
            raise ValueError(
                "integrate_scan does not support runtime checks.",
            )

        def scan_fn(state, _):
            new_state = self.step(state, dt)
            return new_state, new_state

        final_state, trajectory = jax.lax.scan(
            scan_fn, state, xs=None, length=n_steps,
        )
        return final_state, trajectory
