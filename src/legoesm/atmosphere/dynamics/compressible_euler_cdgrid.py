"""FV3-inspired C-D grid Non-Hydrostatic Compressible Euler on the cubed-sphere.

**Fidelity status: stabilized research path, not a faithful FV3 port.**

Uses the same C-D grid discretisation as the shallow water and PE solvers:

* D-grid winds (cell corners) are prognostic (internal conversion from cell-centre).
* C-grid velocities (cell edges) are diagnosed for mass/scalar transport.
* Vorticity from circulation (exact on D-grid).
* Bernoulli/pressure gradient via Arakawa-Lamb at D-grid corners.
* Scalar transport (theta, rho, tracers) via C-grid upwind mass flux.
* Acoustic substeps for vertically propagating sound waves.

**State is stored at cell centres** for compatibility with the existing physics
infrastructure. Velocities are converted to D-grid for momentum computation
and back to cell-centre for output. A faithful FV3 NH path would store winds
on D-grid edges throughout and require a physics coupler that consumes D-grid
winds.

References
----------
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState, NonHydrostaticTendencies
from legoesm.core.operators_3d import (
    gradient_x_3d,
    gradient_y_3d,
    hyperdiffusion_3d,
    vertical_advection_height,
)
from legoesm.core.operators_cdgrid import (
    dgrid_to_cgrid,
    dgrid_to_center_vector,
    dgrid_vorticity,
    cgrid_mass_flux_divergence,
    cgrid_divergence,
    _arakawa_lamb_gradient,
    _interp_center_to_corner,
    _interp_corner_to_center,
    _laplacian_dgrid,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.cubed_sphere_cdgrid import (
    CubedSphereCDGrid,
    create_cubed_sphere_cdgrid,
)
from legoesm.grids.vertical import HeightCoordinate, TerrainMetric
from legoesm.timestepping.integration import IntegrationMixin
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


class CDGridCompressibleEulerConfig(NamedTuple):
    """Configuration for the C-D grid non-hydrostatic CE model.

    This is the recommended cubed-sphere non-hydrostatic solver for
    production AMIP/CMIP simulations.  Uses FV3-style C-D grid staggering
    (Lin 2004, Putman & Lin 2007) which eliminates the Hollingsworth-Kallberg
    instability that affects cell-centre solvers.
    """
    g: float = constants.g
    A_h: float = 0.0              # Laplacian viscosity [m^2/s]
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
    fix_mass: bool = False
    anchor_mass_to_initial: bool = False
    acoustic_off_centering: float = 0.0   # Off-centering beta for acoustic damping
                                          # 0.0 = centered, 0.1 = recommended for long runs


def cdgrid_compressible_euler_slow_tendencies(
    state: NonHydrostaticState,
    grid: CubedSphereGrid,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
    cdgrid: CubedSphereCDGrid,
    config: CDGridCompressibleEulerConfig,
    physics_tendency: NonHydrostaticTendencies | None = None,
) -> NonHydrostaticTendencies:
    """Compute slow (advective) tendencies using C-D grid operators.

    Parameters
    ----------
    state : NonHydrostaticState
    grid : CubedSphereGrid
    height_coord : HeightCoordinate
    terrain_metric : TerrainMetric
    cdgrid : CubedSphereCDGrid
    config : CDGridCompressibleEulerConfig
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
        theta_0 + theta_p, rho_0 + rho_p,
    )

    # --- 1. Exner perturbation and horizontal PGF ---
    pi_prime = compute_exner_perturbation(rho_p, theta_p, height_coord)

    # --- 2. Convert to D-grid ---
    u_d = _interp_center_to_corner(u, cdgrid)
    v_d = _interp_center_to_corner(v, cdgrid)

    # --- 3. C-grid velocities ---
    u_c, v_c = dgrid_to_cgrid(u_d, v_d, cdgrid)

    # --- 4. Vorticity and Coriolis ---
    zeta = dgrid_vorticity(u_d, v_d, cdgrid)
    if config.use_coriolis:
        abs_vor = zeta + cdgrid.base.f[..., None]
    else:
        abs_vor = zeta

    # --- 5. KE at cell centres from D-grid (orthogonal basis) ---
    u_cc, v_cc = dgrid_to_center_vector(u_d, v_d)
    K = 0.5 * (u_cc ** 2 + v_cc ** 2)

    # --- 6. Gradients at D-grid corners ---
    dK_dx, dK_dy_perp = _arakawa_lamb_gradient(K, cdgrid)
    dpi_dx, dpi_dy_perp = _arakawa_lamb_gradient(pi_prime, cdgrid)

    # --- 7. D-grid momentum tendencies ---
    abs_vor_corner = _interp_center_to_corner(abs_vor, cdgrid)
    theta_corner = _interp_center_to_corner(theta_total, cdgrid)

    du_d_dt = abs_vor_corner * v_d - dK_dx - c_p * theta_corner * dpi_dx
    dv_d_dt = -abs_vor_corner * u_d - dK_dy_perp - c_p * theta_corner * dpi_dy_perp

    # Laplacian viscosity
    if config.A_h > 0:
        du_d_dt = du_d_dt + config.A_h * _laplacian_dgrid(u_d, cdgrid)
        dv_d_dt = dv_d_dt + config.A_h * _laplacian_dgrid(v_d, cdgrid)

    # --- 8. Convert back to cell-centre ---
    du_dt = _interp_corner_to_center(du_d_dt)
    dv_dt = _interp_corner_to_center(dv_d_dt)

    # --- 9. Vertical advection of u, v ---
    du_dt = du_dt + vertical_advection_height(u, w, dz, dz_half, J)
    dv_dt = dv_dt + vertical_advection_height(v, w, dz, dz_half, J)

    # --- 10. Theta equation: advective form -v·∇θ ---
    # The θ equation uses advective form (not divergence/flux form) because
    # θ is NOT a conserved density — it satisfies dθ/dt = 0, not ∂(ρθ)/∂t = -∇·(ρθv).
    # Advective form = flux divergence + θ·div(v):  -v·∇θ = -∇·(θv) + θ∇·v
    div_v = cgrid_divergence(u_c, v_c, cdgrid)
    dtheta_p_dt = (cgrid_mass_flux_divergence(theta_total, u_c, v_c, cdgrid)
                   + theta_total * div_v)

    # --- 11. Continuity: C-grid upwind mass flux (divergence form) ---
    drho_p_dt = cgrid_mass_flux_divergence(rho_total, u_c, v_c, cdgrid)

    # --- 12. Tracer advection (advective form) ---
    n_tracers = tracers.shape[-1] if tracers.ndim > 3 else 0
    if n_tracers > 0:
        # Fold the tracer axis into the level axis so
        # ``cgrid_mass_flux_divergence`` runs ONE ``pad_halo_4d`` for all
        # tracers instead of n_tracers separate halo exchanges under
        # vmap-over-tracers.  The operator's 4D path passes the trailing
        # axis through passively (PPM operates on (i, j) only).
        n_face, n_i, n_j, nlev_t, _ = tracers.shape
        tracers_flat = tracers.reshape(n_face, n_i, n_j, nlev_t * n_tracers)
        # u_c, v_c, div_v are the same for every tracer; broadcast across
        # the combined level/tracer trailing axis.  ``tracers_flat`` reshape
        # interleaves levels and tracers as
        # ``[lev0/trc0, lev0/trc1, ..., lev1/trc0, ...]``, so we use
        # ``jnp.repeat`` (each level value duplicated n_tracers times) rather
        # than ``jnp.tile`` (which would concatenate the whole array and
        # mis-align tracer ↔ level).
        if n_tracers == 1:
            u_c_b, v_c_b, div_v_b = u_c, v_c, div_v
        else:
            u_c_b = jnp.repeat(u_c, n_tracers, axis=-1)
            v_c_b = jnp.repeat(v_c, n_tracers, axis=-1)
            div_v_b = jnp.repeat(div_v, n_tracers, axis=-1)
        flux_flat = cgrid_mass_flux_divergence(
            tracers_flat, u_c_b, v_c_b, cdgrid,
        )
        horiz_flat = flux_flat + tracers_flat * div_v_b
        horiz = horiz_flat.reshape(n_face, n_i, n_j, nlev_t, n_tracers)

        # Vertical advection per-tracer — local stencil along axis -1, no
        # halo cost.  vmap over the trailing axis so JAX produces one
        # batched kernel rather than n_tracers separate ones.
        def _vert_one(q):
            return vertical_advection_height(q, w, dz, dz_half, J)

        vert = jax.vmap(_vert_one, in_axes=-1, out_axes=-1)(tracers)
        dtracers_dt = horiz + vert
    else:
        dtracers_dt = jnp.zeros_like(tracers)

    # --- 13. Hyperdiffusion ---
    # Stack (u, v, theta_p) along a trailing axis and fold into the level
    # dim so a single ``hyperdiffusion_3d`` (∇⁴ = ∇²∇², two pad_halo_4d
    # MPI exchanges) handles all three fields, replacing the prior 3
    # sequential calls that each issued their own halo pads.  rho_p uses
    # a different coefficient (hyperdiff_rho_coeff), so it stays separate.
    if config.hyperdiff_coeff > 0:
        n_face_h, n_i_h, n_j_h, nlev_h = u.shape
        hyper_stack = jnp.stack(
            [u, v, theta_p], axis=-1,
        )  # (6, n, n, nlev, 3)
        hyper_flat = hyper_stack.reshape(n_face_h, n_i_h, n_j_h, nlev_h * 3)
        hyper_out_flat = hyperdiffusion_3d(
            hyper_flat, grid, config.hyperdiff_coeff,
        )
        hyper_out = hyper_out_flat.reshape(n_face_h, n_i_h, n_j_h, nlev_h, 3)
        du_dt = du_dt + hyper_out[..., 0]
        dv_dt = dv_dt + hyper_out[..., 1]
        dtheta_p_dt = dtheta_p_dt + hyper_out[..., 2]
    if config.hyperdiff_rho_coeff > 0:
        drho_p_dt = drho_p_dt + hyperdiffusion_3d(
            rho_p, grid, config.hyperdiff_rho_coeff,
        )

    # --- 14. Sponge layer ---
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

    # --- 15. w tendency (slow: horizontal advection) ---
    w_full = 0.5 * (w[..., :-1] + w[..., 1:])
    dw_dx = gradient_x_3d(w_full, grid)
    dw_dy = gradient_y_3d(w_full, grid)
    horiz_adv_w = -(u * dw_dx + v * dw_dy)

    # Pad zero at top/bottom interfaces (rigid BC).  Single Pad HLO op
    # replaces alloc-zeros + scatter.
    pad_axes_w = ((0, 0),) * (w.ndim - 1)
    horiz_adv_w_half = jnp.pad(
        0.5 * (horiz_adv_w[..., :-1] + horiz_adv_w[..., 1:]),
        (*pad_axes_w, (1, 1)),
    )

    dw_dt = horiz_adv_w_half - sponge_half * w
    if config.hyperdiff_w_coeff > 0:
        dw_dt = dw_dt + hyperdiffusion_3d(w, grid, config.hyperdiff_w_coeff)

    # --- 16. Physics ---
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


class CDGridCompressibleEulerModel(IntegrationMixin):
    """FV3-style C-D grid non-hydrostatic compressible Euler model.

    Parameters
    ----------
    grid : CubedSphereGrid
    height_coord : HeightCoordinate
    terrain_metric : TerrainMetric
    config : CDGridCompressibleEulerConfig, optional
    """

    def __init__(
        self,
        grid: CubedSphereGrid,
        height_coord: HeightCoordinate,
        terrain_metric: TerrainMetric,
        config: CDGridCompressibleEulerConfig | None = None,
    ):
        self.height_coord = height_coord
        self.terrain_metric = terrain_metric
        self.config = config or CDGridCompressibleEulerConfig()
        self._target_mass = None

        if self.config.small_earth_factor != 1.0:
            from legoesm.grids.cubed_sphere import apply_small_earth_scaling
            grid = apply_small_earth_scaling(grid, self.config.small_earth_factor)
        self.grid = grid
        self.cdgrid = create_cubed_sphere_cdgrid(grid)

        # Acoustic CFL check at construction time (outside JIT)
        from legoesm.core.cfl import estimate_min_dx_cubed_sphere
        import logging as _logging
        _ce_logger = _logging.getLogger("legoesm.compressible_euler")
        dx_min = estimate_min_dx_cubed_sphere(
            grid.n, getattr(grid, 'radius', 6.371229e6),
        )
        c_sound = float(jnp.sqrt(
            constants.c_pd / constants.c_vd * constants.R_d * 300.0
        ))
        self._dx_min = dx_min
        self._c_sound = c_sound
        _ce_logger.info(
            f"  Acoustic check: dx_min={dx_min/1000:.0f}km, c_s={c_sound:.0f}m/s, "
            f"n_substeps={self.config.n_acoustic_substeps}"
        )

    def tendencies(
        self,
        state: NonHydrostaticState,
        physics_tendency: NonHydrostaticTendencies | None = None,
    ) -> NonHydrostaticTendencies:
        return cdgrid_compressible_euler_slow_tendencies(
            state, self.grid, self.height_coord, self.terrain_metric,
            self.cdgrid, self.config, physics_tendency,
        )

    def step(self, state: NonHydrostaticState, dt: float, physics_fn=None) -> NonHydrostaticState:
        """Advance one time step using split-explicit RK3 with C-D grid transport.

        This non-jitted wrapper precomputes target mass outside the JIT
        boundary, then delegates to the jitted ``_step_jitted``.
        """
        # Precompute target mass outside JIT boundary (host-side only).
        if (self.config.fix_mass
                and self.config.anchor_mass_to_initial
                and self._target_mass is None):
            from legoesm.core.conservation import compute_nh_dry_mass
            self._target_mass = compute_nh_dry_mass(
                state.rho_prime.data, self.height_coord,
                self.terrain_metric, self.grid,
            )
        return self._step_jitted(state, dt, physics_fn=physics_fn)

    @partial(jax.jit, static_argnums=(0, 3))
    def _step_jitted(self, state: NonHydrostaticState, dt: float, physics_fn=None) -> NonHydrostaticState:
        """JIT-compiled step core."""
        from legoesm.atmosphere.dynamics.compressible_euler import CompressibleEulerConfig
        acoustic_cfg = CompressibleEulerConfig(
            g=self.config.g,
            n_acoustic_substeps=self.config.n_acoustic_substeps,
            semi_implicit_acoustic=self.config.semi_implicit_acoustic,
            acoustic_off_centering=self.config.acoustic_off_centering,
        )

        se_config = SplitExplicitConfig(
            n_substeps=self.config.n_acoustic_substeps,
            outer_integrator=self.config.outer_integrator,
        )

        def slow_tendency_fn(s):
            phys = None
            if physics_fn is not None:
                _phys_result = physics_fn(
                    s, self.grid, self.height_coord, self.terrain_metric,
                )
                phys = _phys_result[0] if type(_phys_result) is tuple else _phys_result
            tend = cdgrid_compressible_euler_slow_tendencies(
                s, self.grid, self.height_coord, self.terrain_metric,
                self.cdgrid, self.config, phys,
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

        if self.config.fix_mass:
            from legoesm.core.conservation import (
                fix_mass_nonhydrostatic, compute_nh_dry_mass,
            )
            # _target_mass is precomputed in step() outside the JIT boundary.
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
        return self.step(state, dt, physics_fn=physics_fn)

    # integrate() and integrate_scan() inherited from IntegrationMixin
