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

Discretization options:
- "centered" (default): centered finite differences (ocean_pe.py)
- "finite_volume": FV PPM transport + FV-consistent w-diagnosis +
  divergence damping (ocean_pe_fv.py)
- "fc_gram": FC-Gram spectral operators (ocean_pe_fc.py)
- "fc_gram_cgrid": FC-Gram + divergence damping (ocean_pe_fc_cgrid.py)

The finite-volume PPM option for tracer transport is controlled by
OceanConfig.use_fv_tracer_transport within the centered discretization.

Public API: state_new = model.step(state, dt)
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.core.operators_fv_cubed import (
    face_boundary_weight,
    edge_blend_scalar,
    edge_blend_scalar_3d,
    edge_blend_vector_3d,
)
from legoesm.ocean.vertical import OceanZStarCoordinate
from legoesm.ocean.state import OceanState, OceanConfig
from legoesm.ocean.dynamics.ocean_pe import ocean_baroclinic_tendencies
from legoesm.ocean.dynamics.barotropic import barotropic_substeps

OCEAN_DISCRETIZATIONS = ["centered", "finite_volume", "fc_gram", "fc_gram_cgrid", "cdgrid"]


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
    discretization : str, optional
        Horizontal discretization: "centered" (default), "fc_gram",
        or "fc_gram_cgrid".
    fc_config : FCOperatorConfig, optional
        FC-Gram operator config (required for fc_gram/fc_gram_cgrid).

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
        discretization: str = "centered",
        fc_config=None,
    ):
        if discretization not in OCEAN_DISCRETIZATIONS:
            raise ValueError(
                f"Unknown ocean discretization {discretization!r}. "
                f"Options: {OCEAN_DISCRETIZATIONS}",
            )

        self.grid = grid
        self.z_coord = z_coord
        self.config = config or OceanConfig()
        self.discretization = discretization
        self._validate_config(self.config)
        self._eb_weight = None

        # Build FC config if needed
        if discretization in ("fc_gram", "fc_gram_cgrid"):
            if fc_config is None:
                from legoesm.core.operators_fc import build_fc_config
                fc_config = build_fc_config(d=2, C=4, degree=5)
            self._fc_config = fc_config
        else:
            self._fc_config = None

        # Build C-D grid if needed
        if discretization == "cdgrid":
            from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
            self._cdgrid = create_cubed_sphere_cdgrid(grid)
        else:
            self._cdgrid = None

        # Build physics function if configured
        if self.config.physics is not None:
            from legoesm.ocean.physics.combined import make_ocean_physics
            self._physics_fn = make_ocean_physics(self.config.physics)
        else:
            self._physics_fn = None

        # Optional FV face-edge continuity relaxation.
        if (
            self.discretization == "finite_volume"
            and self.config.edge_blend_strength > 0.0
            and self.config.edge_blend_depth > 0
        ):
            self._eb_weight = face_boundary_weight(
                self.grid.n,
                self.config.edge_blend_depth,
                self.config.edge_blend_strength,
            )

    @staticmethod
    def _validate_config(config: OceanConfig) -> None:
        """Validate configuration ranges early (fail fast)."""
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
                "n_barotropic_substeps must be >= 1, got "
                f"{config.n_barotropic_substeps!r}",
            )
        if config.barotropic_diffusion_dt_ref <= 0.0:
            raise ValueError(
                "barotropic_diffusion_dt_ref must be > 0, got "
                f"{config.barotropic_diffusion_dt_ref!r}",
            )
        if config.min_water_column_m <= 0.0:
            raise ValueError(
                "min_water_column_m must be > 0, got "
                f"{config.min_water_column_m!r}",
            )
        if config.max_abs_eta_m <= 0.0:
            raise ValueError(f"max_abs_eta_m must be > 0, got {config.max_abs_eta_m!r}")
        if config.temperature_min_c > config.temperature_max_c:
            raise ValueError(
                "temperature_min_c must be <= temperature_max_c, got "
                f"{config.temperature_min_c!r} > {config.temperature_max_c!r}",
            )
        if config.salinity_min_psu > config.salinity_max_psu:
            raise ValueError(
                "salinity_min_psu must be <= salinity_max_psu, got "
                f"{config.salinity_min_psu!r} > {config.salinity_max_psu!r}",
            )
        if not (0.0 <= config.edge_blend_strength <= 1.0):
            raise ValueError(
                "edge_blend_strength must be in [0, 1], got "
                f"{config.edge_blend_strength!r}",
            )
        if config.edge_blend_depth < 0:
            raise ValueError(
                "edge_blend_depth must be >= 0, got "
                f"{config.edge_blend_depth!r}",
            )

    def _assert_runtime_invariants(self, state: OceanState) -> None:
        """Host-side runtime checks for debugging/regression hardening."""
        mask = state.land_mask.data
        wet = mask > 0.5
        land = ~wet

        finite_ok = bool(
            jnp.all(jnp.isfinite(state.u.data))
            & jnp.all(jnp.isfinite(state.v.data))
            & jnp.all(jnp.isfinite(state.T.data))
            & jnp.all(jnp.isfinite(state.S.data))
            & jnp.all(jnp.isfinite(state.eta.data))
            & jnp.all(jnp.isfinite(state.H_bathy.data))
        )
        if not finite_ok:
            raise FloatingPointError("Ocean runtime check failed: non-finite state detected")

        water_col = state.eta.data + state.H_bathy.data
        min_water_col = float(
            jnp.min(jnp.where(wet, water_col, jnp.inf))
        ) if bool(jnp.any(wet)) else float("inf")
        if min_water_col < self.config.min_water_column_m:
            raise ValueError(
                "Ocean runtime check failed: water column too small. "
                f"min(eta+H_bathy)={min_water_col:.6g} m, "
                f"threshold={self.config.min_water_column_m:.6g} m",
            )

        eta_abs = float(
            jnp.max(jnp.abs(jnp.where(wet, state.eta.data, 0.0)))
        ) if bool(jnp.any(wet)) else 0.0
        if eta_abs > self.config.max_abs_eta_m:
            raise ValueError(
                "Ocean runtime check failed: |eta| exceeded threshold. "
                f"max|eta|={eta_abs:.6g} m, threshold={self.config.max_abs_eta_m:.6g} m",
            )

        if bool(jnp.any(wet)):
            T_ocean = jnp.where(wet[..., jnp.newaxis], state.T.data, jnp.nan)
            T_min = float(jnp.nanmin(T_ocean))
            T_max = float(jnp.nanmax(T_ocean))
            if T_min < self.config.temperature_min_c or T_max > self.config.temperature_max_c:
                raise ValueError(
                    "Ocean runtime check failed: temperature out of bounds. "
                    f"range=[{T_min:.3f}, {T_max:.3f}] C, "
                    f"bounds=[{self.config.temperature_min_c:.3f}, "
                    f"{self.config.temperature_max_c:.3f}] C",
                )

            S_ocean = jnp.where(wet[..., jnp.newaxis], state.S.data, jnp.nan)
            S_min = float(jnp.nanmin(S_ocean))
            S_max = float(jnp.nanmax(S_ocean))
            if S_min < self.config.salinity_min_psu or S_max > self.config.salinity_max_psu:
                raise ValueError(
                    "Ocean runtime check failed: salinity out of bounds. "
                    f"range=[{S_min:.3f}, {S_max:.3f}] PSU, "
                    f"bounds=[{self.config.salinity_min_psu:.3f}, "
                    f"{self.config.salinity_max_psu:.3f}] PSU",
                )

        if bool(jnp.any(land)):
            land_3d = jnp.broadcast_to(land[..., jnp.newaxis], state.u.data.shape)
            max_land_u = float(jnp.max(jnp.abs(jnp.where(land_3d, state.u.data, 0.0))))
            max_land_v = float(jnp.max(jnp.abs(jnp.where(land_3d, state.v.data, 0.0))))
            max_land_eta = float(jnp.max(jnp.abs(jnp.where(land, state.eta.data, 0.0))))
            if max(max_land_u, max_land_v, max_land_eta) > 1.0e-8:
                raise ValueError(
                    "Ocean runtime check failed: land cells are not zero. "
                    f"max(|u_land|,|v_land|,|eta_land|)="
                    f"{max(max_land_u, max_land_v, max_land_eta):.3e}",
                )

    def tendencies(self, state: OceanState):
        """Compute baroclinic tendencies (pure function wrapper)."""
        return self._compute_tendencies(state)

    def _compute_tendencies(self, state: OceanState):
        """Dispatch to the appropriate tendency function."""
        if self.discretization == "centered":
            return ocean_baroclinic_tendencies(
                state, self.grid, self.z_coord, self.config,
                physics_fn=self._physics_fn,
            )
        elif self.discretization == "finite_volume":
            from legoesm.ocean.dynamics.ocean_pe_fv import (
                ocean_baroclinic_tendencies_fv,
            )
            return ocean_baroclinic_tendencies_fv(
                state, self.grid, self.z_coord, self.config,
                physics_fn=self._physics_fn,
            )
        elif self.discretization == "fc_gram":
            from legoesm.ocean.dynamics.ocean_pe_fc import (
                ocean_baroclinic_tendencies_fc,
            )
            return ocean_baroclinic_tendencies_fc(
                state, self.grid, self.z_coord,
                self._fc_config, self.config,
                physics_fn=self._physics_fn,
            )
        elif self.discretization == "fc_gram_cgrid":
            from legoesm.ocean.dynamics.ocean_pe_fc_cgrid import (
                ocean_baroclinic_tendencies_fc_cgrid,
            )
            return ocean_baroclinic_tendencies_fc_cgrid(
                state, self.grid, self.z_coord,
                self._fc_config, self.config,
                physics_fn=self._physics_fn,
            )
        elif self.discretization == "cdgrid":
            from legoesm.ocean.dynamics.ocean_pe_cdgrid import (
                ocean_baroclinic_tendencies_cdgrid,
            )
            return ocean_baroclinic_tendencies_cdgrid(
                state, self.grid, self.z_coord,
                self._cdgrid, self.config,
                physics_fn=self._physics_fn,
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
        tend = self._compute_tendencies(state)

        # --- 2. Update tracers (forward Euler) ---
        T_new = state.T.data + dt * tend.dT_dt.data
        S_new = state.S.data + dt * tend.dS_dt.data

        # --- 3. Update 3D velocity with slow tendency ---
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
        # Slow u/v tendency is already included in state_mid.
        dt_s = dt / self.config.n_barotropic_substeps
        state_new = barotropic_substeps(
            state_mid,
            dt_s, self.config.n_barotropic_substeps,
            self.grid, self.z_coord, self.config,
        )

        # FV cube-edge continuity relaxation (optional).
        if self._eb_weight is not None:
            mask = state_new.land_mask.data
            mask_3d = mask[..., jnp.newaxis]

            u_new, v_new = edge_blend_vector_3d(
                state_new.u.data, state_new.v.data, self.grid, self._eb_weight,
            )
            T_new = edge_blend_scalar_3d(
                state_new.T.data, self.grid, self._eb_weight,
            )
            S_new = edge_blend_scalar_3d(
                state_new.S.data, self.grid, self._eb_weight,
            )
            eta_new = edge_blend_scalar(
                state_new.eta.data, self.grid, self._eb_weight,
            )

            state_new = state_new._replace(
                u=state_new.u.replace(data=u_new * mask_3d),
                v=state_new.v.replace(data=v_new * mask_3d),
                T=state_new.T.replace(data=T_new * mask_3d),
                S=state_new.S.replace(data=S_new * mask_3d),
                eta=state_new.eta.replace(data=eta_new * mask),
            )

        # --- 5. Conservation fixers ---
        if self.config.use_conservation_fixer:
            from legoesm.ocean.conservation import ocean_conservation_fixer
            state_new = ocean_conservation_fixer(
                state_new, state, self.grid, self.z_coord, self.config,
            )

        return state_new

    def step_checked(self, state: OceanState, dt: float) -> OceanState:
        """Advance one timestep and optionally apply host-side runtime checks."""
        state_new = self.step(state, dt)
        if self.config.enable_runtime_checks:
            self._assert_runtime_invariants(state_new)
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
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt!r}")
        if duration < 0.0:
            raise ValueError(f"duration must be >= 0, got {duration!r}")
        if save_every < 1:
            raise ValueError(f"save_every must be >= 1, got {save_every!r}")

        n_steps = int(duration / dt)
        if duration > 0.0 and n_steps < 1:
            raise ValueError(
                "integration has zero steps; increase duration or reduce dt "
                f"(duration={duration!r}, dt={dt!r})",
            )
        trajectory = [state]
        step_fn = self.step_checked if self.config.enable_runtime_checks else self.step

        for i in range(n_steps):
            state = step_fn(state, dt)
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
        if self.config.enable_runtime_checks:
            raise ValueError(
                "integrate_scan does not support host-side runtime checks. "
                "Use integrate() or disable enable_runtime_checks.",
            )
        if n_steps < 0:
            raise ValueError(f"n_steps must be >= 0, got {n_steps!r}")
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt!r}")

        def scan_fn(state, _):
            new_state = self.step(state, dt)
            return new_state, new_state

        final_state, trajectory = jax.lax.scan(
            scan_fn, state, xs=None, length=n_steps,
        )
        return final_state, trajectory
