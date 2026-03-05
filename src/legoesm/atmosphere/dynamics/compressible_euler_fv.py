"""FV3-Style Non-Hydrostatic Compressible Euler Equations on the cubed-sphere.

Uses unsplit PPM for:
- Theta horizontal advection (replaces centered gradient)
- Continuity horizontal divergence (replaces centered divergence)
- Tracer advection (replaces centered gradient)

Momentum remains in vector-invariant form (same as centered).
Acoustic substeps (vertical) are unchanged.

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
from legoesm.core.state import NonHydrostaticState, NonHydrostaticTendencies
from legoesm.core.operators_3d import (
    vorticity_3d,
    gradient_x_3d,
    gradient_y_3d,
    hyperdiffusion_3d,
    vertical_advection_height,
    fv_flux_divergence_3d,
    fv_scalar_advection_3d,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.vertical import HeightCoordinate, TerrainMetric
from legoesm.timestepping.split_explicit import (
    split_explicit_step,
    SplitExplicitConfig,
)
from legoesm.atmosphere.dynamics.edge_blending import (
    blend_scalar_cube_edges,
    blend_vector_cube_edges,
)
from legoesm.atmosphere.dynamics.compressible_euler import (
    compute_exner_perturbation,
    _sponge_profile,
    acoustic_substeps,
    acoustic_substeps_semi_implicit,
)
from legoesm.atmosphere.physics.thermodynamics import sanitize_theta_rho
from legoesm import constants


class FVCompressibleEulerConfig(NamedTuple):
    """Configuration for the FV non-hydrostatic compressible Euler model."""
    g: float = constants.g
    hyperdiff_coeff: float = 0.0
    hyperdiff_rho_coeff: float = 0.0
    hyperdiff_w_coeff: float = 0.0
    sponge_width: float = 10000.0
    sponge_coeff: float = 0.05
    n_acoustic_substeps: int = 6
    small_earth_factor: float = 1.0
    use_coriolis: bool = True
    semi_implicit_acoustic: bool = False
    outer_integrator: str = "ssp_rk3"
    use_limiter: bool = True
    edge_blend_uv: float = 0.0
    edge_blend_w: float = 0.0
    edge_blend_theta: float = 0.0
    edge_blend_rho: float = 0.0
    edge_blend_tracers: float = 0.0
    edge_blend_width: int = 1
    fix_mass: bool = False
    anchor_mass_to_initial: bool = False


def _apply_fv_nh_edge_blend(
    state: NonHydrostaticState,
    grid: CubedSphereGrid,
    config: FVCompressibleEulerConfig,
) -> NonHydrostaticState:
    """Apply variable-specific cubed-sphere edge blending."""
    u_data = state.u.data
    v_data = state.v.data
    w_data = state.w.data
    theta_data = state.theta_prime.data
    rho_data = state.rho_prime.data
    tracers_data = state.tracers.data

    if config.edge_blend_uv > 0.0:
        u_data, v_data = blend_vector_cube_edges(
            u_data, v_data, grid.cos_angle, grid.sin_angle,
            config.edge_blend_uv, width=config.edge_blend_width,
        )
    if config.edge_blend_w > 0.0:
        w_data = blend_scalar_cube_edges(
            w_data, config.edge_blend_w, width=config.edge_blend_width
        )
    if config.edge_blend_theta > 0.0:
        theta_data = blend_scalar_cube_edges(
            theta_data, config.edge_blend_theta, width=config.edge_blend_width
        )
    if config.edge_blend_rho > 0.0:
        rho_data = blend_scalar_cube_edges(
            rho_data, config.edge_blend_rho, width=config.edge_blend_width
        )
    if config.edge_blend_tracers > 0.0 and tracers_data.ndim == 5 and tracers_data.shape[-1] > 0:
        tracers_data = blend_scalar_cube_edges(
            tracers_data, config.edge_blend_tracers, width=config.edge_blend_width
        )

    return state._replace(
        u=state.u.replace(data=u_data),
        v=state.v.replace(data=v_data),
        w=state.w.replace(data=w_data),
        theta_prime=state.theta_prime.replace(data=theta_data),
        rho_prime=state.rho_prime.replace(data=rho_data),
        tracers=state.tracers.replace(data=tracers_data),
    )


def fv_compressible_euler_slow_tendencies(
    state: NonHydrostaticState,
    grid: CubedSphereGrid,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
    config: FVCompressibleEulerConfig,
    physics_tendency: NonHydrostaticTendencies | None = None,
) -> NonHydrostaticTendencies:
    """Compute slow (advective) tendencies using FV transport.

    Differences from centered:
    - Theta horizontal advection: FV scalar advection (PPM)
    - Continuity: FV flux divergence (PPM)
    - Tracer advection: FV scalar advection (PPM)
    - Momentum: unchanged (vector-invariant)
    - Acoustic substeps: unchanged (vertical only)
    """
    u = state.u.data
    v = state.v.data
    w = state.w.data
    theta_p = state.theta_prime.data
    rho_p = state.rho_prime.data
    tracers = state.tracers.data

    c_p = constants.c_pd
    rho_0 = height_coord.rho_ref
    theta_0 = height_coord.theta_ref
    dz = height_coord.dz
    dz_half = height_coord.dz_half
    J = terrain_metric.jacobian

    theta_total, rho_total = sanitize_theta_rho(
        theta_0 + theta_p,
        rho_0 + rho_p,
    )

    # --- 1. Exner perturbation and horizontal PGF ---
    pi_prime = compute_exner_perturbation(rho_p, theta_p, height_coord)
    dpi_dx = gradient_x_3d(pi_prime, grid)
    dpi_dy = gradient_y_3d(pi_prime, grid)

    # --- 2. Vorticity and Coriolis ---
    zeta = vorticity_3d(u, v, grid)
    abs_vor = (zeta + grid.f[..., None]) if config.use_coriolis else zeta

    # --- 3. Kinetic energy gradient ---
    K = 0.5 * (u**2 + v**2)
    dK_dx = gradient_x_3d(K, grid)
    dK_dy = gradient_y_3d(K, grid)

    # --- 4. Horizontal momentum (vector-invariant) ---
    du_dt = abs_vor * v - dK_dx - c_p * theta_total * dpi_dx
    dv_dt = -abs_vor * u - dK_dy - c_p * theta_total * dpi_dy

    # --- 5. Vertical advection of u, v ---
    du_dt = du_dt + vertical_advection_height(u, w, dz, dz_half, J)
    dv_dt = dv_dt + vertical_advection_height(v, w, dz, dz_half, J)

    # --- 6. Theta equation: FV horizontal advection ---
    dtheta_p_dt = fv_scalar_advection_3d(
        theta_total, u, v, grid,
        limiter=config.use_limiter,
    )

    # --- 7. Continuity: FV flux divergence ---
    drho_p_dt = fv_flux_divergence_3d(
        rho_total, u, v, grid,
        limiter=config.use_limiter,
    )

    # --- 8. Tracer advection via FV ---
    n_tracers = tracers.shape[-1] if tracers.ndim > 3 else 0
    if n_tracers > 0:
        tracers_t = jnp.moveaxis(tracers, -1, 0)

        def _single_tracer_tendency(q):
            horiz_adv_q = fv_scalar_advection_3d(
                q, u, v, grid,
                limiter=config.use_limiter,
            )
            vert_adv_q = vertical_advection_height(q, w, dz, dz_half, J)
            return horiz_adv_q + vert_adv_q

        dtracers_dt_t = jax.vmap(_single_tracer_tendency)(tracers_t)
        dtracers_dt = jnp.moveaxis(dtracers_dt_t, 0, -1)
    else:
        dtracers_dt = jnp.zeros_like(tracers)

    # --- 9. Hyperdiffusion ---
    if config.hyperdiff_coeff > 0:
        du_dt = du_dt + hyperdiffusion_3d(u, grid, config.hyperdiff_coeff)
        dv_dt = dv_dt + hyperdiffusion_3d(v, grid, config.hyperdiff_coeff)
        dtheta_p_dt = dtheta_p_dt + hyperdiffusion_3d(
            theta_p, grid, config.hyperdiff_coeff
        )
    if config.hyperdiff_rho_coeff > 0:
        drho_p_dt = drho_p_dt + hyperdiffusion_3d(
            rho_p, grid, config.hyperdiff_rho_coeff
        )

    # --- 10. Sponge layer ---
    sponge = _sponge_profile(
        height_coord.z_full, height_coord.H,
        config.sponge_width, config.sponge_coeff,
    )
    du_dt = du_dt - sponge * u
    dv_dt = dv_dt - sponge * v
    dtheta_p_dt = dtheta_p_dt - sponge * theta_p

    sponge_half = _sponge_profile(
        height_coord.z_half, height_coord.H,
        config.sponge_width, config.sponge_coeff,
    )

    # --- 11. w tendency (slow: horizontal advection) ---
    w_full = 0.5 * (w[..., :-1] + w[..., 1:])
    dw_dx = gradient_x_3d(w_full, grid)
    dw_dy = gradient_y_3d(w_full, grid)
    horiz_adv_w = -(u * dw_dx + v * dw_dy)

    horiz_adv_w_half = jnp.zeros_like(w)
    horiz_adv_w_half = horiz_adv_w_half.at[..., 1:-1].set(
        0.5 * (horiz_adv_w[..., :-1] + horiz_adv_w[..., 1:])
    )

    dw_dt = horiz_adv_w_half - sponge_half * w
    if config.hyperdiff_w_coeff > 0:
        dw_dt = dw_dt + hyperdiffusion_3d(
            w, grid, config.hyperdiff_w_coeff
        )

    # --- 12. Physics ---
    if physics_tendency is not None:
        du_dt = du_dt + physics_tendency.du_dt.data
        dv_dt = dv_dt + physics_tendency.dv_dt.data
        dw_dt = dw_dt + physics_tendency.dw_dt.data
        dtheta_p_dt = dtheta_p_dt + physics_tendency.dtheta_prime_dt.data
        drho_p_dt = drho_p_dt + physics_tendency.drho_prime_dt.data
        dtracers_dt = dtracers_dt + physics_tendency.dtracers_dt.data

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    dims_tr = ("face", "x", "y", "level", "tracer")

    return NonHydrostaticTendencies(
        du_dt=Field(data=du_dt, name="du_dt", dims=dims_3d, units="m/s^2"),
        dv_dt=Field(data=dv_dt, name="dv_dt", dims=dims_3d, units="m/s^2"),
        dw_dt=Field(data=dw_dt, name="dw_dt", dims=dims_w, units="m/s^2"),
        dtheta_prime_dt=Field(
            data=dtheta_p_dt, name="dtheta_prime_dt", dims=dims_3d, units="K/s",
        ),
        drho_prime_dt=Field(
            data=drho_p_dt, name="drho_prime_dt", dims=dims_3d, units="kg/m^3/s",
        ),
        dphis_dt=Field(
            data=jnp.zeros_like(state.phis.data),
            name="dphis_dt", dims=dims_2d, units="m^2/s^3",
        ),
        dtracers_dt=Field(
            data=dtracers_dt, name="dtracers_dt", dims=dims_tr, units="1/s",
        ),
    )


class FVCompressibleEulerModel:
    """FV3-style non-hydrostatic compressible Euler model.

    Same interface as CompressibleEulerModel but uses PPM + Lin-Rood
    for theta advection, continuity, and tracer transport.

    Parameters
    ----------
    grid : CubedSphereGrid
    height_coord : HeightCoordinate
    terrain_metric : TerrainMetric
    config : FVCompressibleEulerConfig, optional
    """

    def __init__(
        self,
        grid: CubedSphereGrid,
        height_coord: HeightCoordinate,
        terrain_metric: TerrainMetric,
        config: FVCompressibleEulerConfig | None = None,
    ):
        self.height_coord = height_coord
        self.terrain_metric = terrain_metric
        self.config = config or FVCompressibleEulerConfig()
        self._target_mass = None

        if self.config.small_earth_factor != 1.0:
            from legoesm.grids.cubed_sphere import apply_small_earth_scaling
            grid = apply_small_earth_scaling(grid, self.config.small_earth_factor)
        self.grid = grid

        for name in ("hyperdiff_coeff", "hyperdiff_rho_coeff", "hyperdiff_w_coeff"):
            value = float(getattr(self.config, name))
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value!r}")
        for name in (
            "edge_blend_uv", "edge_blend_w", "edge_blend_theta",
            "edge_blend_rho", "edge_blend_tracers",
        ):
            value = float(getattr(self.config, name))
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")
        if int(self.config.edge_blend_width) < 1:
            raise ValueError(
                f"edge_blend_width must be >= 1, got {self.config.edge_blend_width!r}"
            )

    def tendencies(
        self,
        state: NonHydrostaticState,
        physics_tendency: NonHydrostaticTendencies | None = None,
    ) -> NonHydrostaticTendencies:
        """Compute slow tendencies."""
        return fv_compressible_euler_slow_tendencies(
            state, self.grid, self.height_coord, self.terrain_metric,
            self.config, physics_tendency,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: NonHydrostaticState, dt: float) -> NonHydrostaticState:
        """Advance one time step using split-explicit RK3 with FV transport."""
        # Build a centered-style config for the acoustic substeps
        # (they reuse the existing acoustic_substeps function)
        from legoesm.atmosphere.dynamics.compressible_euler import CompressibleEulerConfig
        acoustic_cfg = CompressibleEulerConfig(
            g=self.config.g,
            n_acoustic_substeps=self.config.n_acoustic_substeps,
            semi_implicit_acoustic=self.config.semi_implicit_acoustic,
        )

        se_config = SplitExplicitConfig(
            n_substeps=self.config.n_acoustic_substeps,
            outer_integrator=self.config.outer_integrator,
        )

        def slow_tendency_fn(s):
            tend = fv_compressible_euler_slow_tendencies(
                s, self.grid, self.height_coord, self.terrain_metric,
                self.config,
            )
            return NonHydrostaticState(
                u=s.u.replace(data=tend.du_dt.data),
                v=s.v.replace(data=tend.dv_dt.data),
                w=s.w.replace(data=tend.dw_dt.data),
                theta_prime=s.theta_prime.replace(data=tend.dtheta_prime_dt.data),
                rho_prime=s.rho_prime.replace(data=tend.drho_prime_dt.data),
                phis=s.phis.replace(data=jnp.zeros_like(s.phis.data)),
                tracers=s.tracers.replace(data=tend.dtracers_dt.data),
            )

        def acoustic_update_fn(s, slow_tend, dt_s, n_sub, cfg):
            if self.config.semi_implicit_acoustic:
                return acoustic_substeps_semi_implicit(
                    s, slow_tend, dt_s, n_sub, cfg,
                    self.height_coord, self.terrain_metric, acoustic_cfg,
                )
            return acoustic_substeps(
                s, slow_tend, dt_s, n_sub, cfg,
                self.height_coord, self.terrain_metric, acoustic_cfg,
            )

        state_new = split_explicit_step(
            state, slow_tendency_fn, acoustic_update_fn, dt, se_config,
        )

        if (
            self.config.edge_blend_uv > 0.0
            or self.config.edge_blend_w > 0.0
            or self.config.edge_blend_theta > 0.0
            or self.config.edge_blend_rho > 0.0
            or self.config.edge_blend_tracers > 0.0
        ):
            state_new = _apply_fv_nh_edge_blend(state_new, self.grid, self.config)

        if self.config.fix_mass:
            from legoesm.core.conservation import (
                fix_mass_nonhydrostatic, compute_nh_dry_mass,
            )
            if self.config.anchor_mass_to_initial and self._target_mass is None:
                self._target_mass = compute_nh_dry_mass(
                    state.rho_prime.data, self.height_coord,
                    self.terrain_metric, self.grid,
                )
            target = self._target_mass if self.config.anchor_mass_to_initial else (
                compute_nh_dry_mass(
                    state.rho_prime.data, self.height_coord,
                    self.terrain_metric, self.grid,
                )
            )
            state_new = fix_mass_nonhydrostatic(
                state_new, target, self.height_coord,
                self.terrain_metric, self.grid,
            )

        return state_new

    @partial(jax.jit, static_argnums=(0, 3))
    def step_with_physics(
        self,
        state: NonHydrostaticState,
        dt: float,
        physics_fn=None,
    ) -> NonHydrostaticState:
        """Advance one time step with physics forcing."""
        from legoesm.atmosphere.dynamics.compressible_euler import CompressibleEulerConfig
        acoustic_cfg = CompressibleEulerConfig(
            g=self.config.g,
            n_acoustic_substeps=self.config.n_acoustic_substeps,
            semi_implicit_acoustic=self.config.semi_implicit_acoustic,
        )

        se_config = SplitExplicitConfig(
            n_substeps=self.config.n_acoustic_substeps,
            outer_integrator=self.config.outer_integrator,
        )

        def slow_tendency_fn(s):
            phys = None
            if physics_fn is not None:
                phys = physics_fn(
                    s, self.grid, self.height_coord, self.terrain_metric,
                )
            tend = fv_compressible_euler_slow_tendencies(
                s, self.grid, self.height_coord, self.terrain_metric,
                self.config, phys,
            )
            return NonHydrostaticState(
                u=s.u.replace(data=tend.du_dt.data),
                v=s.v.replace(data=tend.dv_dt.data),
                w=s.w.replace(data=tend.dw_dt.data),
                theta_prime=s.theta_prime.replace(data=tend.dtheta_prime_dt.data),
                rho_prime=s.rho_prime.replace(data=tend.drho_prime_dt.data),
                phis=s.phis.replace(data=jnp.zeros_like(s.phis.data)),
                tracers=s.tracers.replace(data=tend.dtracers_dt.data),
            )

        def acoustic_update_fn(s, slow_tend, dt_s, n_sub, cfg):
            if self.config.semi_implicit_acoustic:
                return acoustic_substeps_semi_implicit(
                    s, slow_tend, dt_s, n_sub, cfg,
                    self.height_coord, self.terrain_metric, acoustic_cfg,
                )
            return acoustic_substeps(
                s, slow_tend, dt_s, n_sub, cfg,
                self.height_coord, self.terrain_metric, acoustic_cfg,
            )

        state_new = split_explicit_step(
            state, slow_tendency_fn, acoustic_update_fn, dt, se_config,
        )

        if (
            self.config.edge_blend_uv > 0.0
            or self.config.edge_blend_w > 0.0
            or self.config.edge_blend_theta > 0.0
            or self.config.edge_blend_rho > 0.0
            or self.config.edge_blend_tracers > 0.0
        ):
            state_new = _apply_fv_nh_edge_blend(state_new, self.grid, self.config)

        if self.config.fix_mass:
            from legoesm.core.conservation import (
                fix_mass_nonhydrostatic, compute_nh_dry_mass,
            )
            if self.config.anchor_mass_to_initial and self._target_mass is None:
                self._target_mass = compute_nh_dry_mass(
                    state.rho_prime.data, self.height_coord,
                    self.terrain_metric, self.grid,
                )
            target = self._target_mass if self.config.anchor_mass_to_initial else (
                compute_nh_dry_mass(
                    state.rho_prime.data, self.height_coord,
                    self.terrain_metric, self.grid,
                )
            )
            state_new = fix_mass_nonhydrostatic(
                state_new, target, self.height_coord,
                self.terrain_metric, self.grid,
            )

        return state_new

    def integrate(
        self,
        state: NonHydrostaticState,
        duration: float,
        dt: float,
        save_every: int = 1,
        physics_fn=None,
    ) -> tuple[NonHydrostaticState, list[NonHydrostaticState]]:
        """Integrate forward for a given duration."""
        n_steps = int(duration / dt)
        trajectory = [state]

        for i in range(n_steps):
            if physics_fn is not None:
                state = self.step_with_physics(state, dt, physics_fn)
            else:
                state = self.step(state, dt)
            if (i + 1) % save_every == 0:
                trajectory.append(state)

        return state, trajectory
