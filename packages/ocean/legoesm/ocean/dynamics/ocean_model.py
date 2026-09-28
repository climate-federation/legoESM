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

Supported discretization: ``"cdgrid"`` (C-D grid finite volume, the only
cubed-sphere implementation).

Deprecated aliases that silently map to ``"cdgrid"``:
``"centered"``, ``"finite_volume"``, ``"fv"``.
These emit ``DeprecationWarning`` and will be removed in a future release.

Public API: state_new = model.step(state, dt)
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.ocean.vertical import OceanZStarCoordinate
from legoesm.ocean.state import OceanState, OceanConfig
from legoesm.core.precision import cast_pytree
from legoesm.ocean.dynamics.barotropic import barotropic_substeps
from legoesm.ocean.dynamics.ocean_pe_cdgrid import ocean_baroclinic_tendencies_cdgrid
from legoesm.ocean.dynamics.barotropic_cgrid import (
    barotropic_substeps_cgrid, barotropic_substeps_fv3sw,
    barotropic_substeps_fv3edge,
)
from legoesm.ocean.conservation import ocean_conservation_fixer
from legoesm.ocean.physics.combined import make_ocean_physics

OCEAN_DISCRETIZATIONS = ["cdgrid"]

# Legacy discretization name mapping.  The Fourier-continuation A-grid backend
# ("fc_gram"/"fc_gram_cgrid") has been REMOVED — all cubed-sphere ocean
# discretizations are the FV3 C-D grid (the atmosphere-matching staggering).
_LEGACY_DISCRETIZATION_MAP = {
    "centered": "cdgrid",
    "finite_volume": "cdgrid",
    "fv": "cdgrid",
}


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
        Horizontal discretization. Only "cdgrid" (FV3 C-D grid) is supported.
        Legacy names ("centered", "finite_volume", "fv") are accepted with a
        deprecation warning.

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
        discretization: str = "cdgrid",
        cdgrid=None,
    ):
        # Map legacy discretization names with deprecation warning
        if discretization in _LEGACY_DISCRETIZATION_MAP:
            import warnings
            canonical = _LEGACY_DISCRETIZATION_MAP[discretization]
            warnings.warn(
                f"Ocean discretization {discretization!r} is deprecated; "
                f"use {canonical!r} instead. All cubed-sphere ocean "
                f"discretizations map to the C-D grid implementation.",
                DeprecationWarning,
                stacklevel=2,
            )
            discretization = canonical

        if discretization not in OCEAN_DISCRETIZATIONS:
            raise ValueError(
                f"Unknown ocean discretization {discretization!r}. "
                f"Options: {OCEAN_DISCRETIZATIONS}",
            )

        self.grid = grid
        self.z_coord = z_coord
        self.config = config or OceanConfig()
        self.discretization = "cdgrid"  # Only C-D grid supported
        self._validate_config(self.config)

        # Build C-D grid.  For single-face panels, pass a pre-built cdgrid
        # via create_cubed_sphere_panel(return_cdgrid=True).
        if cdgrid is not None:
            self._cdgrid = cdgrid
        else:
            self._cdgrid = create_cubed_sphere_cdgrid(grid)

        # FV3-faithful barotropic: route the free-surface mode through the
        # validated cube shallow-water core (vector-invariant absolute-vorticity
        # flux + SSP-RK3 + divergence damping/hyperdiffusion).  Built once (the
        # SW model's step is jitted on a static ``self``).  See
        # ``barotropic_substeps_fv3sw`` and ``fv3_faithful.md``.
        self._sw_baro_model = None
        if self.config.barotropic_staggering in ("fv3sw", "fv3edge"):
            # Resolve the SHARED shallow-water barotropic core by name from the
            # foundational registry.  The atmosphere SW dycore registers its
            # "fv3sw" (corner-staggered C-D grid) and "fv3edge" (FV3 edge-
            # staggered, algorithmically faithful) builders on import, so the
            # ocean reuses the validated SW dycore WITHOUT importing the
            # atmosphere component — the SW-core sharing flows through the
            # registry (mass-fixer disabled in the builder: the ocean's masked
            # conservation fixer handles the wet domain).
            from legoesm.registry import (
                SW_BAROTROPIC_REGISTRY,
                UnknownRegistryEntryError,
            )
            try:
                _build_sw = SW_BAROTROPIC_REGISTRY.get(
                    self.config.barotropic_staggering
                )
            except UnknownRegistryEntryError as exc:
                raise UnknownRegistryEntryError(
                    f"No {self.config.barotropic_staggering!r} shallow-water "
                    "barotropic-core provider is registered.  Import the FV3 SW "
                    "core provider before building a cube ocean with this "
                    "barotropic_staggering, e.g. `import "
                    "legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid`."
                ) from exc
            self._sw_baro_model = _build_sw(grid, self._cdgrid, self.config)

        # Build physics function if configured
        if self.config.physics is not None:
            self._physics_fn = make_ocean_physics(self.config.physics)
        else:
            self._physics_fn = None

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
        if config.barotropic_staggering not in (
            "a_grid", "c_grid", "fv3sw", "fv3edge",
        ):
            raise ValueError(
                "barotropic_staggering must be 'a_grid', 'c_grid', 'fv3sw' or "
                f"'fv3edge', got {config.barotropic_staggering!r}",
            )

    def _assert_runtime_invariants(self, state: OceanState) -> None:
        """Host-side runtime checks for debugging/regression hardening.

        All reductions are fused into a single ``jnp.stack`` and pulled
        to host with one ``np.asarray`` call so enabling
        ``enable_runtime_checks`` costs one GPU→host sync per step
        instead of 11.
        """
        u = state.u.data
        v = state.v.data
        T = state.T.data
        S = state.S.data
        eta = state.eta.data
        H_bathy = state.H_bathy.data
        mask = state.land_mask.data

        wet = mask > 0.5
        land = ~wet
        wet3 = wet[..., jnp.newaxis]
        land3 = jnp.broadcast_to(land[..., jnp.newaxis], u.shape)
        any_wet = jnp.any(wet)
        any_land = jnp.any(land)
        water_col = eta + H_bathy

        finite_ok = (
            jnp.all(jnp.isfinite(u))
            & jnp.all(jnp.isfinite(v))
            & jnp.all(jnp.isfinite(T))
            & jnp.all(jnp.isfinite(S))
            & jnp.all(jnp.isfinite(eta))
            & jnp.all(jnp.isfinite(H_bathy))
        )

        # Mask reductions so the safe scalars can be shipped together.
        T_wet = jnp.where(wet3, T, jnp.nan)
        S_wet = jnp.where(wet3, S, jnp.nan)
        eta_wet = jnp.where(wet, eta, 0.0)
        wc_wet = jnp.where(wet, water_col, jnp.inf)

        u_land = jnp.where(land3, u, 0.0)
        v_land = jnp.where(land3, v, 0.0)
        eta_land = jnp.where(land, eta, 0.0)

        _stats = jnp.stack([
            finite_ok.astype(eta.dtype),
            any_wet.astype(eta.dtype),
            any_land.astype(eta.dtype),
            jnp.min(wc_wet).astype(eta.dtype),
            jnp.max(jnp.abs(eta_wet)).astype(eta.dtype),
            jnp.nanmin(T_wet).astype(eta.dtype),
            jnp.nanmax(T_wet).astype(eta.dtype),
            jnp.nanmin(S_wet).astype(eta.dtype),
            jnp.nanmax(S_wet).astype(eta.dtype),
            jnp.max(jnp.abs(u_land)).astype(eta.dtype),
            jnp.max(jnp.abs(v_land)).astype(eta.dtype),
            jnp.max(jnp.abs(eta_land)).astype(eta.dtype),
        ])
        host = np.asarray(_stats)
        finite_ok_h = bool(host[0] > 0.5)
        any_wet_h = bool(host[1] > 0.5)
        any_land_h = bool(host[2] > 0.5)
        min_water_col = float(host[3]) if any_wet_h else float("inf")
        eta_abs = float(host[4]) if any_wet_h else 0.0
        T_min = float(host[5]) if any_wet_h else float("nan")
        T_max = float(host[6]) if any_wet_h else float("nan")
        S_min = float(host[7]) if any_wet_h else float("nan")
        S_max = float(host[8]) if any_wet_h else float("nan")
        max_land_u = float(host[9])
        max_land_v = float(host[10])
        max_land_eta = float(host[11])

        if not finite_ok_h:
            raise FloatingPointError("Ocean runtime check failed: non-finite state detected")
        if min_water_col < self.config.min_water_column_m:
            raise ValueError(
                "Ocean runtime check failed: water column too small. "
                f"min(eta+H_bathy)={min_water_col:.6g} m, "
                f"threshold={self.config.min_water_column_m:.6g} m",
            )
        if eta_abs > self.config.max_abs_eta_m:
            raise ValueError(
                "Ocean runtime check failed: |eta| exceeded threshold. "
                f"max|eta|={eta_abs:.6g} m, threshold={self.config.max_abs_eta_m:.6g} m",
            )
        if any_wet_h and (
            T_min < self.config.temperature_min_c
            or T_max > self.config.temperature_max_c
        ):
            raise ValueError(
                "Ocean runtime check failed: temperature out of bounds. "
                f"range=[{T_min:.3f}, {T_max:.3f}] C, "
                f"bounds=[{self.config.temperature_min_c:.3f}, "
                f"{self.config.temperature_max_c:.3f}] C",
            )
        if any_wet_h and (
            S_min < self.config.salinity_min_psu
            or S_max > self.config.salinity_max_psu
        ):
            raise ValueError(
                "Ocean runtime check failed: salinity out of bounds. "
                f"range=[{S_min:.3f}, {S_max:.3f}] PSU, "
                f"bounds=[{self.config.salinity_min_psu:.3f}, "
                f"{self.config.salinity_max_psu:.3f}] PSU",
            )
        if any_land_h and max(max_land_u, max_land_v, max_land_eta) > 1.0e-8:
            raise ValueError(
                "Ocean runtime check failed: land cells are not zero. "
                f"max(|u_land|,|v_land|,|eta_land|)="
                f"{max(max_land_u, max_land_v, max_land_eta):.3e}",
            )

    def tendencies(self, state: OceanState, surface_forcing=None, dt=None):
        """Compute baroclinic tendencies (pure function wrapper)."""
        return self._compute_tendencies(state, surface_forcing, dt)

    def _compute_tendencies(self, state: OceanState, surface_forcing=None,
                            dt=None):
        """Compute baroclinic tendencies on the C-D grid backend."""
        return ocean_baroclinic_tendencies_cdgrid(
            state, self.grid, self.z_coord,
            self._cdgrid, self.config,
            physics_fn=self._physics_fn,
            surface_forcing=surface_forcing,
            dt=dt,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: OceanState, dt: float,
             surface_forcing=None) -> OceanState:
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
        surface_forcing : OceanSurfaceForcing or None
            External atmospheric forcing (SW, heat, wind, freshwater).

        Returns
        -------
        OceanState : State after one time step.
        """
        state = cast_pytree(state, None, "compute")

        if getattr(self.config, "baroclinic_rk3", False):
            # --- 1-3. Baroclinic explicit update via 3-stage SSP-RK3 ---
            # Forward-Euler (else branch) cannot carry a realistic WOA cold-start
            # (violent geostrophic adjustment -> grid-scale blowup); SSP-RK3 on the
            # slow (baroclinic) u,v,T,S update closes that gap (mirrors the lat-lon
            # tripole RK3 fix). eta is held fixed through the stages — updated ONLY
            # by the barotropic substeps below, exactly as the Euler path. 3x cost.
            def _bc(st):
                return self._compute_tendencies(st, surface_forcing, dt)

            def _upd(st, k, a):  # st + a*dt*k for u,v,T,S
                return st._replace(
                    u=st.u.replace(data=st.u.data + a * dt * k.du_dt.data),
                    v=st.v.replace(data=st.v.data + a * dt * k.dv_dt.data),
                    T=st.T.replace(data=st.T.data + a * dt * k.dT_dt.data),
                    S=st.S.replace(data=st.S.data + a * dt * k.dS_dt.data),
                )

            def _comb(sa, sb, wa, wb):  # wa*sa + wb*sb per field
                return sa._replace(
                    u=sa.u.replace(data=wa * sa.u.data + wb * sb.u.data),
                    v=sa.v.replace(data=wa * sa.v.data + wb * sb.v.data),
                    T=sa.T.replace(data=wa * sa.T.data + wb * sb.T.data),
                    S=sa.S.replace(data=wa * sa.S.data + wb * sb.S.data),
                )

            s1 = _upd(state, _bc(state), 1.0)
            s2 = _comb(state, _upd(s1, _bc(s1), 1.0), 0.75, 0.25)
            state_mid = _comb(state, _upd(s2, _bc(s2), 1.0),
                              1.0 / 3.0, 2.0 / 3.0)
        else:
            # --- 1. Baroclinic tendencies ---
            tend = self._compute_tendencies(state, surface_forcing, dt)

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
        # eta is updated ONLY by the barotropic solver's continuity equation.
        # Do NOT pre-apply tend.deta_dt here — the barotropic solver computes
        # the same depth-integrated flux divergence, so pre-applying would
        # double-count the eta tendency.
        # Slow u/v tendency is already included in state_mid.
        dt_s = dt / self.config.n_barotropic_substeps
        if self.config.barotropic_staggering == "fv3edge":
            state_new = barotropic_substeps_fv3edge(
                state_mid,
                dt_s, self.config.n_barotropic_substeps,
                self.grid, self._cdgrid, self.z_coord, self.config,
                self._sw_baro_model,
            )
        elif self.config.barotropic_staggering == "fv3sw":
            state_new = barotropic_substeps_fv3sw(
                state_mid,
                dt_s, self.config.n_barotropic_substeps,
                self.grid, self._cdgrid, self.z_coord, self.config,
                self._sw_baro_model,
            )
        elif self.config.barotropic_staggering == "c_grid":
            state_new = barotropic_substeps_cgrid(
                state_mid,
                dt_s, self.config.n_barotropic_substeps,
                self.grid, self._cdgrid, self.z_coord, self.config,
            )
        else:
            state_new = barotropic_substeps(
                state_mid,
                dt_s, self.config.n_barotropic_substeps,
                self.grid, self.z_coord, self.config,
            )

        # --- 5. Conservation fixers ---
        if self.config.use_conservation_fixer:
            state_new = ocean_conservation_fixer(
                state_new, state, self.grid, self.z_coord, self.config,
            )

        # --- 5b. Velocity ceiling (STATIC config gate) ---
        # Pragmatic stabiliser for the quasi-uniform cube: marginal seas
        # (Med/Aegean/Gulf) are sub-grid even at 1/4deg (the ORCA tripole refines
        # coasts, the cube cannot) -> their sharp WOA fronts spin up uncarriable
        # >10 m/s jets that blow the cold-start. Clipping |u|,|v| to a physical
        # ceiling bounds those spikes (open ocean |u|<ceiling is untouched), so the
        # run is stable for a CAVEATED open-ocean comparison (the capped marginal-sea
        # cells are non-physical -- like the lat-lon Arctic caveat). Documented band-aid
        # (docs/ocean/experiments/cubed_sphere_pgf_stability.md: clipping prevents NaN).
        if self.config.velocity_ceiling > 0.0:
            vmax = self.config.velocity_ceiling
            state_new = state_new._replace(
                u=state_new.u.replace(
                    data=jnp.clip(state_new.u.data, -vmax, vmax)),
                v=state_new.v.replace(
                    data=jnp.clip(state_new.v.data, -vmax, vmax)),
            )

        # ``allow_downcast=True`` is required here so the output state
        # matches the user-provided input dtype.  Without it the
        # ``cast_pytree(..., "compute")`` upcast at the top of ``step``
        # silently ratchets fp32 inputs to fp64 outputs under
        # ``JAX_ENABLE_X64``, breaking ``jax.lax.scan`` carry-dtype
        # invariants and any downstream consumer that expects
        # the storage-precision policy to be respected.
        return cast_pytree(state_new, None, "storage", allow_downcast=True)

    def step_checked(self, state: OceanState, dt: float,
                     surface_forcing=None) -> OceanState:
        """Advance one timestep and optionally apply host-side runtime checks."""
        state_new = self.step(state, dt, surface_forcing)
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
