"""Non-hydrostatic compressible Euler equations on MPAS Voronoi meshes.

Solves the fully compressible Euler equations using TRiSK operators on
C-grids with height-based terrain-following (z*) coordinates and
reference-state subtraction.

Equations:
    du/dt   = F_pv - grad(KE) - c_p θ grad(π') + visc + vert_adv + sponge
    dw/dt   = -c_p θ (1/J) ∂π'/∂z* + g θ'/θ₀ + sponge
    dθ'/dt  = -v·∇θ - (w/J) ∂θ/∂z* + diffusion + sponge
    dρ'/dt  = -(1/J)[div_h(J ρ v_h) + ∂(ρ w)/∂z*]

Time integration: split-explicit (SSP-RK3 outer + forward-backward
acoustic substeps for fast sound/gravity waves).

References
----------
- Skamarock & Klemp (2008): A Time-Split Nonhydrostatic Atmospheric Model.
- Skamarock et al. (2012): MPAS-Atmosphere. Mon. Wea. Rev., 140, 3090-3105.
- Ringler et al. (2010). J. Comput. Phys., 229(9), 3065-3090.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import MPASNonHydrostaticState, MPASNonHydrostaticTendencies
from legoesm.core.operators_voronoi import (
    divergence_cell,
    gradient_edge,
    kinetic_energy_cell,
    potential_vorticity_vertex,
    pv_flux_energy_conserving,
    pv_flux_enstrophy_conserving,
    edge_thickness,
    cell_to_edge_avg,
    vector_laplacian_del2,
    vector_laplacian_del4,
)
from legoesm.grids.voronoi import VoronoiMesh
from legoesm.grids.vertical import HeightCoordinate, TerrainMetric
from legoesm.atmosphere.dynamics.compressible_euler import (
    compute_exner_perturbation,
    _sponge_profile,
)
from legoesm.atmosphere.physics.thermodynamics import sanitize_theta_rho
from legoesm.timestepping.split_explicit import (
    split_explicit_step,
    SplitExplicitConfig,
)
from legoesm import constants


class MPASCompressibleEulerConfig(NamedTuple):
    """Configuration for the MPAS non-hydrostatic compressible Euler model."""
    g: float = constants.g
    nu_del2: float = 0.0           # del2 viscosity for u [m²/s]
    nu_del4: float = 0.0           # del4 viscosity for u [m⁴/s]
    K_h: float = 0.0               # scalar diffusion [m²/s]
    pv_scheme: str = "energy"      # "energy" or "enstrophy"
    sponge_width: float = 10000.0  # sponge layer width from top [m]
    sponge_coeff: float = 0.05     # max Rayleigh damping rate [1/s]
    n_acoustic_substeps: int = 6
    use_coriolis: bool = True
    outer_integrator: str = "ssp_rk3"


# ============================================================================
# Slow tendency computation
# ============================================================================

def mpas_compressible_euler_slow_tendencies(
    state: MPASNonHydrostaticState,
    mesh: VoronoiMesh,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
    config: MPASCompressibleEulerConfig,
    physics_tendency: MPASNonHydrostaticTendencies | None = None,
) -> MPASNonHydrostaticTendencies:
    """Compute slow (advective) tendencies for the MPAS compressible Euler equations.

    These are evaluated once per RK3 stage and held constant during
    acoustic substeps.
    """
    u_3d = state.u.data           # (nEdges, nlev)
    w = state.w.data              # (nCells, nlev+1)
    theta_p = state.theta_prime.data  # (nCells, nlev)
    rho_p = state.rho_prime.data      # (nCells, nlev)
    tracers = state.tracers.data      # (nCells, nlev, n_tracers)

    c_p = constants.c_pd
    rho_0 = height_coord.rho_ref        # (nlev,)
    theta_0 = height_coord.theta_ref    # (nlev,)
    dz = height_coord.dz                # (nlev,)
    dz_half = height_coord.dz_half      # (nlev-1,)
    J = terrain_metric.jacobian         # (nCells,)

    c1 = mesh.cellsOnEdge[0]  # (nEdges,)
    c2 = mesh.cellsOnEdge[1]
    nlev = theta_p.shape[-1]

    theta_total, rho_total = sanitize_theta_rho(
        theta_0 + theta_p, rho_0 + rho_p,
    )

    # --- 1. Exner perturbation and horizontal pressure gradient ---
    pi_prime = compute_exner_perturbation(rho_p, theta_p, height_coord)

    # --- Per-level momentum and continuity ---
    def _level_slow_tend(carry, k):
        u_k = u_3d[:, k]              # (nEdges,)
        theta_k = theta_total[:, k]   # (nCells,)
        rho_k = rho_total[:, k]       # (nCells,)
        pi_k = pi_prime[:, k]         # (nCells,)

        # KE at cells
        ke = kinetic_energy_cell(u_k, mesh)

        # Horizontal Exner gradient at edges
        grad_pi = gradient_edge(pi_k, mesh)  # (nEdges,)

        # Theta at edges for PGF
        theta_e = cell_to_edge_avg(theta_k, mesh)

        # KE gradient at edges
        grad_ke = gradient_edge(ke, mesh)

        # PV flux (Coriolis + vorticity)
        # Use rho*dz as thickness proxy for mass-weighted PV
        h_proxy = rho_k * dz[k]  # (nCells,)
        f_v = mesh.fVertex if config.use_coriolis else jnp.zeros_like(mesh.fVertex)
        q_v = potential_vorticity_vertex(u_k, h_proxy, f_v, mesh)

        if config.pv_scheme == "enstrophy":
            pv_flux = pv_flux_enstrophy_conserving(u_k, h_proxy, q_v, mesh)
        else:
            pv_flux = pv_flux_energy_conserving(u_k, h_proxy, q_v, mesh)

        # Momentum tendency
        du_dt_k = pv_flux - grad_ke - c_p * theta_e * grad_pi

        # Viscosity
        if config.nu_del2 > 0:
            du_dt_k = du_dt_k + config.nu_del2 * vector_laplacian_del2(u_k, mesh)
        if config.nu_del4 > 0:
            du_dt_k = du_dt_k + config.nu_del4 * vector_laplacian_del4(u_k, mesh)

        # Horizontal divergence of rho*u for continuity
        rho_e = cell_to_edge_avg(rho_k, mesh)  # (nEdges,)
        div_rho_v = divergence_cell(rho_e * u_k, mesh)  # (nCells,)

        # Horizontal theta advection: -v·∇θ
        grad_theta = gradient_edge(theta_k, mesh)
        # Reconstruct v·∇θ at cells via flux form
        theta_e_adv = cell_to_edge_avg(theta_k, mesh)
        div_u_theta = divergence_cell(u_k * theta_e_adv, mesh)
        div_u = divergence_cell(u_k, mesh)
        horiz_adv_theta = -(div_u_theta - theta_k * div_u)

        return carry, (du_dt_k, div_rho_v, horiz_adv_theta)

    _, (du_dt_all, div_rho_v_all, horiz_adv_theta_all) = jax.lax.scan(
        _level_slow_tend, None, jnp.arange(nlev),
    )
    du_dt_3d = jnp.moveaxis(du_dt_all, 0, -1)          # (nEdges, nlev)
    div_rho_v_3d = jnp.moveaxis(div_rho_v_all, 0, -1)  # (nCells, nlev)
    dtheta_p_dt = jnp.moveaxis(horiz_adv_theta_all, 0, -1)

    # Horizontal continuity: drho'/dt = -div_h(rho * v)
    drho_p_dt = -div_rho_v_3d

    # --- 2. Vertical advection of u ---
    # Interpolate w from cells to edges, then apply vertical advection
    w_edge = 0.5 * (w[c1] + w[c2])  # (nEdges, nlev+1)
    J_edge = 0.5 * (J[c1] + J[c2])  # (nEdges,)
    du_dt_3d = du_dt_3d + _vertical_advection_height_1d(
        u_3d, w_edge, dz, dz_half, J_edge,
    )

    # --- 3. Sponge layer ---
    sponge = _sponge_profile(
        height_coord.z_full, height_coord.H,
        config.sponge_width, config.sponge_coeff,
    )
    du_dt_3d = du_dt_3d - sponge * u_3d
    dtheta_p_dt = dtheta_p_dt - sponge * theta_p

    sponge_half = _sponge_profile(
        height_coord.z_half, height_coord.H,
        config.sponge_width, config.sponge_coeff,
    )

    # --- 4. w tendency (slow part: horizontal advection + sponge) ---
    # Interpolate w to full levels, compute horizontal advection, map back
    w_full = 0.5 * (w[:, :-1] + w[:, 1:])  # (nCells, nlev)

    # Horizontal advection of w: -v·∇w ≈ -(div(u*w) - w*div(u))
    def _w_horiz_adv(k):
        w_k = w_full[:, k]
        w_e = 0.5 * (w_k[c1] + w_k[c2])
        u_k = u_3d[:, k]
        div_uw = divergence_cell(u_k * w_e, mesh)
        div_u = divergence_cell(u_k, mesh)
        return -(div_uw - w_k * div_u)

    _, w_adv_all = jax.lax.scan(
        lambda c, k: (c, _w_horiz_adv(k)), None, jnp.arange(nlev),
    )
    w_adv_full = jnp.moveaxis(w_adv_all, 0, -1)  # (nCells, nlev)

    # Map back to half levels by averaging
    dw_dt = jnp.zeros_like(w)
    dw_dt = dw_dt.at[:, 1:-1].set(
        0.5 * (w_adv_full[:, :-1] + w_adv_full[:, 1:])
    )
    dw_dt = dw_dt - sponge_half * w

    # --- 5. Tracer advection ---
    n_tracers = tracers.shape[-1] if tracers.ndim > 1 else 0
    if n_tracers > 0:
        dtracers_dt = _tracer_tendencies(
            tracers, u_3d, w, dz, dz_half, J, mesh, c1, c2,
        )
    else:
        dtracers_dt = jnp.zeros_like(tracers)

    # --- 6. Scalar diffusion on theta ---
    if config.K_h > 0:
        def _diff_theta(k):
            grad_th = gradient_edge(theta_p[:, k], mesh)
            return divergence_cell(grad_th, mesh)
        _, diff_all = jax.lax.scan(
            lambda c, k: (c, _diff_theta(k)), None, jnp.arange(nlev),
        )
        dtheta_p_dt = dtheta_p_dt + config.K_h * jnp.moveaxis(diff_all, 0, -1)

    # --- 7. Add physics tendencies ---
    if physics_tendency is not None:
        du_dt_3d = du_dt_3d + physics_tendency.du_dt.data
        dw_dt = dw_dt + physics_tendency.dw_dt.data
        dtheta_p_dt = dtheta_p_dt + physics_tendency.dtheta_prime_dt.data
        drho_p_dt = drho_p_dt + physics_tendency.drho_prime_dt.data
        dtracers_dt = dtracers_dt + physics_tendency.dtracers_dt.data

    return MPASNonHydrostaticTendencies(
        du_dt=Field(data=du_dt_3d, name="du_dt",
                    dims=("nEdges", "nlev"), units="m/s²"),
        dw_dt=Field(data=dw_dt, name="dw_dt",
                    dims=("nCells", "nlev_half"), units="m/s²"),
        dtheta_prime_dt=Field(data=dtheta_p_dt, name="dtheta_prime_dt",
                              dims=("nCells", "nlev"), units="K/s"),
        drho_prime_dt=Field(data=drho_p_dt, name="drho_prime_dt",
                            dims=("nCells", "nlev"), units="kg/m³/s"),
        dphis_dt=Field(data=jnp.zeros_like(state.phis.data),
                       name="dphis_dt", dims=("nCells",), units="m²/s³"),
        dtracers_dt=Field(data=dtracers_dt, name="dtracers_dt",
                          dims=("nCells", "nlev", "tracer"), units="1/s"),
    )


def _vertical_advection_height_1d(field_3d, w, dz, dz_half, J):
    """Vertical advection for height-based coords: -(w/J) df/dz*.

    Parameters
    ----------
    field_3d : (n, nlev)
    w : (n, nlev+1) at interfaces
    dz : (nlev,)
    dz_half : (nlev-1,)
    J : (n,) or scalar

    Returns
    -------
    (n, nlev) tendency
    """
    nlev = field_3d.shape[-1]
    # w at full levels
    w_full = 0.5 * (w[:, :-1] + w[:, 1:])

    # Vertical gradient at full levels (centered)
    df_dz = jnp.zeros_like(field_3d)
    if nlev > 2:
        inner = (field_3d[:, :-2] - field_3d[:, 2:]) / (dz_half[:-1] + dz_half[1:])
        df_dz = df_dz.at[:, 1:-1].set(inner)

    J_col = J[:, None] if J.ndim == 1 else J
    return -w_full / J_col * df_dz


def _tracer_tendencies(tracers, u_3d, w, dz, dz_half, J, mesh, c1, c2):
    """Horizontal + vertical advection of tracers."""
    nCells, nlev, n_tracers = tracers.shape

    def _single_tracer(q_3d):
        # Horizontal advection per level
        def _horiz_adv(k):
            q_k = q_3d[:, k]
            u_k = u_3d[:, k]
            q_e = 0.5 * (q_k[c1] + q_k[c2])
            div_uq = divergence_cell(u_k * q_e, mesh)
            div_u = divergence_cell(u_k, mesh)
            return -(div_uq - q_k * div_u)

        _, horiz_all = jax.lax.scan(
            lambda c, k: (c, _horiz_adv(k)), None, jnp.arange(nlev),
        )
        horiz = jnp.moveaxis(horiz_all, 0, -1)

        # Vertical advection
        vert = _vertical_advection_height_1d(q_3d, w, dz, dz_half, J)
        return horiz + vert

    # vmap over tracers
    tracers_t = jnp.moveaxis(tracers, -1, 0)  # (n_tracers, nCells, nlev)
    dt_t = jax.vmap(_single_tracer)(tracers_t)
    return jnp.moveaxis(dt_t, 0, -1)  # (nCells, nlev, n_tracers)


# ============================================================================
# Acoustic substeps
# ============================================================================

def mpas_acoustic_substeps(
    state: MPASNonHydrostaticState,
    slow_tend: MPASNonHydrostaticTendencies,
    dt_s: float,
    n_substeps: int,
    config: SplitExplicitConfig,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
    euler_config: MPASCompressibleEulerConfig,
) -> MPASNonHydrostaticState:
    """Forward-backward acoustic substeps on MPAS.

    Each substep:
    1. Forward: update w using vertical Exner gradient + buoyancy
    2. Backward: update rho' using vertical divergence
    3. Backward: update theta' using vertical advection by w
    """
    g = euler_config.g
    c_p = constants.c_pd
    dz = height_coord.dz
    theta_0 = height_coord.theta_ref
    rho_0 = height_coord.rho_ref
    J = terrain_metric.jacobian  # (nCells,)

    w = state.w.data              # (nCells, nlev+1)
    theta_p = state.theta_prime.data
    rho_p = state.rho_prime.data
    nlev = theta_p.shape[-1]

    def substep_body(i, carry):
        w_c, theta_p_c, rho_p_c = carry

        theta_total, rho_total = sanitize_theta_rho(
            theta_0 + theta_p_c, rho_0 + rho_p_c,
        )

        # --- Forward: update w ---
        pi_p = compute_exner_perturbation(rho_p_c, theta_p_c, height_coord)

        # Exner gradient at half levels
        dpi_dz_inner = (pi_p[:, :-1] - pi_p[:, 1:]) / (
            0.5 * (dz[:-1] + dz[1:])
        )

        theta_half_inner = 0.5 * (theta_total[:, :-1] + theta_total[:, 1:])
        theta_p_half = 0.5 * (theta_p_c[:, :-1] + theta_p_c[:, 1:])
        theta_0_half = 0.5 * (theta_0[:-1] + theta_0[1:])
        buoyancy = g * theta_p_half / theta_0_half

        dw_dt_inner = (
            -c_p * theta_half_inner * dpi_dz_inner / J[:, None]
            + buoyancy
        )

        w_new = w_c.at[:, 1:-1].set(
            w_c[:, 1:-1] + dt_s * dw_dt_inner
        )

        # --- Backward: update rho' ---
        rho_half = 0.5 * (rho_total[:, :-1] + rho_total[:, 1:])
        rho_w = jnp.zeros_like(w_new)
        rho_w = rho_w.at[:, 1:-1].set(rho_half * w_new[:, 1:-1])

        vert_div = (rho_w[:, :-1] - rho_w[:, 1:]) / dz
        vert_div = vert_div / J[:, None]

        rho_p_new = rho_p_c - dt_s * vert_div

        # --- Backward: update theta' ---
        w_full = 0.5 * (w_new[:, :-1] + w_new[:, 1:])
        dtheta_dz = jnp.zeros_like(theta_total)
        if nlev > 2:
            dz_half_val = height_coord.dz_half
            dz_centered = dz_half_val[:-1] + dz_half_val[1:]
            inner_grad = (theta_total[:, :-2] - theta_total[:, 2:]) / dz_centered
            dtheta_dz = dtheta_dz.at[:, 1:-1].set(inner_grad)

        theta_p_new = theta_p_c - dt_s * w_full / J[:, None] * dtheta_dz

        return (w_new, theta_p_new, rho_p_new)

    w_final, theta_p_final, rho_p_final = jax.lax.fori_loop(
        0, n_substeps, substep_body, (w, theta_p, rho_p),
    )

    return MPASNonHydrostaticState(
        u=state.u,
        w=state.w.replace(data=w_final),
        theta_prime=state.theta_prime.replace(data=theta_p_final),
        rho_prime=state.rho_prime.replace(data=rho_p_final),
        phis=state.phis,
        tracers=state.tracers,
    )


# ============================================================================
# Model class
# ============================================================================

class MPASCompressibleEulerModel:
    """Non-hydrostatic compressible Euler model on an MPAS Voronoi mesh.

    Parameters
    ----------
    mesh : VoronoiMesh
    height_coord : HeightCoordinate
    terrain_metric : TerrainMetric
    config : MPASCompressibleEulerConfig, optional
    """

    def __init__(
        self,
        mesh: VoronoiMesh,
        height_coord: HeightCoordinate,
        terrain_metric: TerrainMetric,
        config: MPASCompressibleEulerConfig | None = None,
    ):
        self.mesh = mesh
        self.height_coord = height_coord
        self.terrain_metric = terrain_metric
        self.config = config or MPASCompressibleEulerConfig()

    def tendencies(
        self,
        state: MPASNonHydrostaticState,
        physics_tendency: MPASNonHydrostaticTendencies | None = None,
    ) -> MPASNonHydrostaticTendencies:
        return mpas_compressible_euler_slow_tendencies(
            state, self.mesh, self.height_coord, self.terrain_metric,
            self.config, physics_tendency,
        )

    @partial(jax.jit, static_argnums=(0, 3))
    def step(
        self,
        state: MPASNonHydrostaticState,
        dt: float,
        physics_fn=None,
    ) -> MPASNonHydrostaticState:
        """Advance one time step using split-explicit RK3.

        Parameters
        ----------
        state : MPASNonHydrostaticState
        dt : float
        physics_fn : callable, optional
            Function (state, mesh, height_coord, terrain_metric) -> tendencies.
        """
        se_config = SplitExplicitConfig(
            n_substeps=self.config.n_acoustic_substeps,
            outer_integrator=self.config.outer_integrator,
        )

        def slow_tendency_fn(s):
            phys = None
            if physics_fn is not None:
                _phys_result = physics_fn(
                    s, self.mesh, self.height_coord, self.terrain_metric,
                )
                phys = _phys_result[0] if type(_phys_result) is tuple else _phys_result
            tend = mpas_compressible_euler_slow_tendencies(
                s, self.mesh, self.height_coord, self.terrain_metric,
                self.config, phys,
            )
            return MPASNonHydrostaticState(
                u=s.u.replace(data=tend.du_dt.data),
                w=s.w.replace(data=tend.dw_dt.data),
                theta_prime=s.theta_prime.replace(data=tend.dtheta_prime_dt.data),
                rho_prime=s.rho_prime.replace(data=tend.drho_prime_dt.data),
                phis=s.phis.replace(data=jnp.zeros_like(s.phis.data)),
                tracers=s.tracers.replace(data=tend.dtracers_dt.data),
            )

        def acoustic_update_fn(s, slow_tend, dt_s, n_sub, cfg):
            return mpas_acoustic_substeps(
                s, slow_tend, dt_s, n_sub, cfg,
                self.height_coord, self.terrain_metric, self.config,
            )

        return split_explicit_step(
            state, slow_tendency_fn, acoustic_update_fn, dt, se_config,
        )

    def integrate(self, state, duration, dt, save_every=1, physics_fn=None):
        n_steps = int(duration / dt)
        trajectory = [state]
        for i in range(n_steps):
            state = self.step(state, dt, physics_fn=physics_fn)
            if (i + 1) % save_every == 0:
                trajectory.append(state)
        return state, trajectory

    def integrate_scan(self, state, n_steps, dt):
        def scan_fn(state, _):
            new_state = self.step(state, dt)
            return new_state, new_state
        final_state, trajectory = jax.lax.scan(
            scan_fn, state, jnp.arange(n_steps),
        )
        return final_state, trajectory
