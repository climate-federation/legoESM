"""Consistent FV Non-Hydrostatic Compressible Euler on the cubed-sphere.

Uses a compatible face-based discretization where rho, theta, tracers,
and horizontal momentum all share the same FV interface flux layer:

- Theta advection: FV scalar advection (PPM)
- Continuity (rho): FV flux divergence (PPM)
- Tracer advection: FV scalar advection (PPM)
- Momentum: vector-invariant form with centered PGF
- Divergence damping: selective damping using FV-consistent divergence
- Acoustic substeps: unchanged (vertical only)

The divergence damping uses the SAME FV divergence operator as the
mass/scalar transport, ensuring discrete compatibility and preventing
cube-imprinted computational modes.

References
----------
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core (FV3)
- Harris & Lin (2013): A Two-Way Nested Global-Regional Dynamical Core
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
from legoesm.core.operators_fv_cubed import (
    fv_divergence_damping_3d as _fv_divergence_damping_3d,
    face_boundary_weight,
    edge_blend_scalar_3d,
    edge_blend_vector_3d,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.vertical import HeightCoordinate, TerrainMetric
from legoesm.timestepping.split_explicit import (
    split_explicit_step,
    SplitExplicitConfig,
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
    """Configuration for the consistent FV non-hydrostatic CE model.

    Divergence damping is the primary mechanism for controlling
    cube-imprinted computational modes. Edge blending provides
    localized Laplacian smoothing near face boundaries.
    """
    g: float = constants.g
    div_damp_2: float = 0.0       # 2nd-order divergence damping [m²/s]
    div_damp_4: float = 0.0       # 4th-order divergence damping [m⁴/s]
    hyperdiff_coeff: float = 0.0
    hyperdiff_rho_coeff: float = 0.0
    hyperdiff_w_coeff: float = 0.0
    edge_blend_strength: float = 0.25  # Face-boundary blend (0=off)
    edge_blend_depth: int = 2          # Rows to blend near each edge
    sponge_width: float = 10000.0
    sponge_coeff: float = 0.05
    n_acoustic_substeps: int = 6
    small_earth_factor: float = 1.0
    use_coriolis: bool = True
    semi_implicit_acoustic: bool = False
    outer_integrator: str = "ssp_rk3"
    use_limiter: bool = True
    fix_mass: bool = False
    anchor_mass_to_initial: bool = False


def fv_compressible_euler_slow_tendencies(
    state: NonHydrostaticState,
    grid: CubedSphereGrid,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
    config: FVCompressibleEulerConfig,
    physics_tendency: NonHydrostaticTendencies | None = None,
) -> NonHydrostaticTendencies:
    """Compute slow (advective) tendencies using consistent FV transport.

    Key consistency: all scalar transport AND divergence damping use the
    same FV interface flux layer.

    Parameters
    ----------
    state : NonHydrostaticState
    grid : CubedSphereGrid
    height_coord : HeightCoordinate
    terrain_metric : TerrainMetric
    config : FVCompressibleEulerConfig
    physics_tendency : NonHydrostaticTendencies, optional

    Returns
    -------
    NonHydrostaticTendencies
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

    # --- 9. Divergence damping (primary stabilization) ---
    if config.div_damp_2 > 0 or config.div_damp_4 > 0:
        du_damp, dv_damp = _fv_divergence_damping_3d(
            u, v, grid, config.div_damp_2, config.div_damp_4,
        )
        du_dt = du_dt + du_damp
        dv_dt = dv_dt + dv_damp

    # --- 10. Hyperdiffusion (secondary) ---
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

    # --- 11. Sponge layer ---
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

    # --- 12. w tendency (slow: horizontal advection) ---
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

    # --- 13. Physics ---
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
    """Consistent FV non-hydrostatic compressible Euler model.

    Uses PPM transport for theta/rho/tracers, FV-consistent
    divergence damping, and edge blending near face boundaries.

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

        cfg = self.config
        if cfg.edge_blend_strength > 0 and cfg.edge_blend_depth > 0:
            self._eb_weight = face_boundary_weight(
                grid.n, cfg.edge_blend_depth, cfg.edge_blend_strength,
            )
        else:
            self._eb_weight = None

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

    @partial(jax.jit, static_argnums=(0, 3), donate_argnums=(1,))
    def step(self, state: NonHydrostaticState, dt: float, physics_fn=None) -> NonHydrostaticState:
        """Advance one time step using split-explicit RK3 with FV transport."""
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

        # Edge blending: localized smoothing near face boundaries
        if self._eb_weight is not None:
            u_new, v_new = edge_blend_vector_3d(
                state_new.u.data, state_new.v.data,
                self.grid, self._eb_weight,
            )
            w_new = edge_blend_scalar_3d(
                state_new.w.data, self.grid, self._eb_weight,
            )
            theta_new = edge_blend_scalar_3d(
                state_new.theta_prime.data, self.grid, self._eb_weight,
            )
            rho_new = edge_blend_scalar_3d(
                state_new.rho_prime.data, self.grid, self._eb_weight,
            )
            state_new = NonHydrostaticState(
                u=state_new.u.replace(data=u_new),
                v=state_new.v.replace(data=v_new),
                w=state_new.w.replace(data=w_new),
                theta_prime=state_new.theta_prime.replace(data=theta_new),
                rho_prime=state_new.rho_prime.replace(data=rho_new),
                phis=state_new.phis,
                tracers=state_new.tracers,
            )

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

    def step_with_physics(self, state, dt, physics_fn=None):
        """Backward-compatible wrapper for step() with physics."""
        return self.step(state, dt, physics_fn=physics_fn)

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
            state = self.step(state, dt, physics_fn=physics_fn)
            if (i + 1) % save_every == 0:
                trajectory.append(state)

        return state, trajectory

    def integrate_scan(
        self,
        state: NonHydrostaticState,
        n_steps: int,
        dt: float,
    ) -> tuple[NonHydrostaticState, NonHydrostaticState]:
        def scan_fn(state, _):
            new_state = self.step(state, dt)
            return new_state, new_state
        final_state, trajectory = jax.lax.scan(
            scan_fn, state, jnp.arange(n_steps)
        )
        return final_state, trajectory
