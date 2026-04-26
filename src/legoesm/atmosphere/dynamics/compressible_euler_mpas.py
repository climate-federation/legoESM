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
    # 3D-native operators (loop-free per-level computation).
    divergence_cell_3d,
    gradient_edge_3d,
    kinetic_energy_cell_3d,
    potential_vorticity_vertex_3d,
    pv_flux_energy_conserving_3d,
    pv_flux_enstrophy_conserving_3d,
    cell_to_edge_avg_3d,
    vector_laplacian_del2_3d,
    vector_laplacian_del4_3d,
)
from legoesm.grids.voronoi import VoronoiMesh
from legoesm.grids.vertical import HeightCoordinate, TerrainMetric
from legoesm.atmosphere.dynamics.compressible_euler import (
    compute_exner_perturbation,
    _sponge_profile,
)
from legoesm.atmosphere.physics.thermodynamics import sanitize_theta_rho
from legoesm.timestepping.integration import IntegrationMixin
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

    # --- All-levels momentum and continuity (3D-native) ---
    # ``operators_voronoi`` exposes ``*_3d`` variants that broadcast
    # over the trailing level axis natively, so the previous per-level
    # ``jax.lax.scan`` + 3 ``jnp.moveaxis`` round-trip is redundant.

    # KE and Exner gradient at edges
    ke_3d = kinetic_energy_cell_3d(u_3d, mesh)             # (nCells, nlev)
    grad_pi_3d = gradient_edge_3d(pi_prime, mesh)          # (nEdges, nlev)

    # Theta at edges for PGF
    theta_e_3d = cell_to_edge_avg_3d(theta_total, mesh)    # (nEdges, nlev)

    # KE gradient at edges
    grad_ke_3d = gradient_edge_3d(ke_3d, mesh)             # (nEdges, nlev)

    # PV flux (Coriolis + vorticity).  Use rho*dz as thickness proxy
    # for mass-weighted PV.
    h_proxy_3d = rho_total * dz[None, :]                   # (nCells, nlev)
    f_v = mesh.fVertex if config.use_coriolis else jnp.zeros_like(mesh.fVertex)
    q_v_3d = potential_vorticity_vertex_3d(u_3d, h_proxy_3d, f_v, mesh)

    if config.pv_scheme == "enstrophy":
        pv_flux_3d = pv_flux_enstrophy_conserving_3d(
            u_3d, h_proxy_3d, q_v_3d, mesh,
        )
    else:
        pv_flux_3d = pv_flux_energy_conserving_3d(
            u_3d, h_proxy_3d, q_v_3d, mesh,
        )

    # Momentum tendency
    du_dt_3d = pv_flux_3d - grad_ke_3d - c_p * theta_e_3d * grad_pi_3d

    # Viscosity
    if config.nu_del2 > 0:
        du_dt_3d = du_dt_3d + config.nu_del2 * vector_laplacian_del2_3d(
            u_3d, mesh,
        )
    if config.nu_del4 > 0:
        du_dt_3d = du_dt_3d + config.nu_del4 * vector_laplacian_del4_3d(
            u_3d, mesh,
        )

    # Horizontal divergences batched.  ``divergence_cell_3d`` does a
    # gather (``u_edge_3d[edgesOnCell]``) + weighted reduce on the
    # leading axis only — the trailing nlev axis is purely passive.
    # Stack the three (nEdges, nlev) flux inputs along a new trailing
    # axis to (nEdges, nlev, 3), fold to (nEdges, nlev*3), call
    # ``divergence_cell_3d`` once on the thicker tensor, then unfold.
    # 3 divergence calls → 1.
    rho_e_3d = cell_to_edge_avg_3d(rho_total, mesh)        # (nEdges, nlev)
    n_edges_d, nlev_d = u_3d.shape
    _div_inputs = jnp.stack(
        [rho_e_3d * u_3d, u_3d * theta_e_3d, u_3d], axis=-1,
    )  # (nEdges, nlev, 3)
    _div_outputs = divergence_cell_3d(
        _div_inputs.reshape(n_edges_d, nlev_d * 3), mesh,
    ).reshape(-1, nlev_d, 3)
    div_rho_v_3d = _div_outputs[..., 0]
    div_u_theta_3d = _div_outputs[..., 1]
    div_u_3d = _div_outputs[..., 2]

    # Horizontal theta advection (advective form): -(div(u·θ) - θ · div(u)).
    dtheta_p_dt = -(div_u_theta_3d - theta_total * div_u_3d)

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
    # Interpolate w to full levels, compute horizontal advection, map back.
    w_full = 0.5 * (w[:, :-1] + w[:, 1:])  # (nCells, nlev)

    # Horizontal advection of w in advective form (3D-native):
    #   -v·∇w ≈ -(div(u*w) - w·div(u))
    # ``div_u_3d`` was already computed for the theta advection block;
    # reuse it instead of recomputing nlev redundant copies.
    w_e_3d = 0.5 * (w_full[c1] + w_full[c2])              # (nEdges, nlev)
    div_uw_3d = divergence_cell_3d(u_3d * w_e_3d, mesh)   # (nCells, nlev)
    w_adv_full = -(div_uw_3d - w_full * div_u_3d)         # (nCells, nlev)

    # Map back to half levels by averaging.  ``dw_dt`` zero at top/bottom
    # interfaces (rigid BC); single Pad HLO op replaces alloc-zeros +
    # scatter.
    dw_dt = jnp.pad(
        0.5 * (w_adv_full[:, :-1] + w_adv_full[:, 1:]),
        ((0, 0), (1, 1)),
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

    # --- 6. Scalar diffusion on theta (3D-native) ---
    if config.K_h > 0:
        grad_th_3d = gradient_edge_3d(theta_p, mesh)         # (nEdges, nlev)
        diff_3d = divergence_cell_3d(grad_th_3d, mesh)       # (nCells, nlev)
        dtheta_p_dt = dtheta_p_dt + config.K_h * diff_3d

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

    # Vertical gradient at full levels (centred); zero at top/bottom
    # (no ghost cells).  Single Pad HLO op replaces alloc-zeros + scatter.
    if nlev > 2:
        inner = (field_3d[:, :-2] - field_3d[:, 2:]) / (dz_half[:-1] + dz_half[1:])
        df_dz = jnp.pad(inner, ((0, 0), (1, 1)))
    else:
        df_dz = jnp.zeros_like(field_3d)

    J_col = J[:, None] if J.ndim == 1 else J
    return -w_full / J_col * df_dz


def _tracer_tendencies(tracers, u_3d, w, dz, dz_half, J, mesh, c1, c2):
    """Horizontal + vertical advection of tracers.

    Fold the tracer axis into the level axis so the cell→edge gather and
    ``divergence_cell_3d`` operate on a single thicker (nCells, nlev*n_tracers)
    field — one HLO graph for all tracers instead of n_tracers vmap'd graphs.
    """
    nCells, nlev_t, n_tracers = tracers.shape
    div_u_3d = divergence_cell_3d(u_3d, mesh)             # (nCells, nlev)

    # Horizontal advection (advective form): -(div(u*q) - q*div(u)).
    # Reshape so cell→edge gather and divergence run once on all tracers.
    tracers_flat = tracers.reshape(nCells, nlev_t * n_tracers)
    q_e_flat = 0.5 * (tracers_flat[c1] + tracers_flat[c2])  # (nEdges, nlev*n_tracers)
    # Multiply by u_3d via reshape→multiply→reshape so u_3d (nEdges, nlev)
    # broadcasts against the tracer axis without materializing a tile.
    n_edges = q_e_flat.shape[0]
    q_e = q_e_flat.reshape(n_edges, nlev_t, n_tracers)
    flux = u_3d[..., None] * q_e                              # (nEdges, nlev, n_tracers)
    flux_flat = flux.reshape(n_edges, nlev_t * n_tracers)
    div_uq_flat = divergence_cell_3d(flux_flat, mesh)         # (nCells, nlev*n_tracers)
    div_uq = div_uq_flat.reshape(nCells, nlev_t, n_tracers)
    horiz = -(div_uq - tracers * div_u_3d[..., None])

    # Vertical advection — local stencil along axis -1, no halo cost.
    # ``_vertical_advection_height_1d`` hard-codes axis 1 as nlev for 2D
    # input, so vmap over the trailing tracer axis to get one batched
    # kernel rather than a Python-unrolled loop.
    def _vert_one(q):
        return _vertical_advection_height_1d(q, w, dz, dz_half, J)

    vert = jax.vmap(_vert_one, in_axes=-1, out_axes=-1)(tracers)
    return horiz + vert


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
        # ``rho_w`` zero at top/bottom (rigid BC); single Pad HLO op
        # replaces alloc-zeros + scatter.
        rho_half = 0.5 * (rho_total[:, :-1] + rho_total[:, 1:])
        rho_w = jnp.pad(rho_half * w_new[:, 1:-1], ((0, 0), (1, 1)))

        vert_div = (rho_w[:, :-1] - rho_w[:, 1:]) / dz
        vert_div = vert_div / J[:, None]

        rho_p_new = rho_p_c - dt_s * vert_div

        # --- Backward: update theta' ---
        w_full = 0.5 * (w_new[:, :-1] + w_new[:, 1:])
        if nlev > 2:
            dz_half_val = height_coord.dz_half
            dz_centered = dz_half_val[:-1] + dz_half_val[1:]
            inner_grad = (theta_total[:, :-2] - theta_total[:, 2:]) / dz_centered
            dtheta_dz = jnp.pad(inner_grad, ((0, 0), (1, 1)))
        else:
            dtheta_dz = jnp.zeros_like(theta_total)

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

class MPASCompressibleEulerModel(IntegrationMixin):
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

    # integrate() and integrate_scan() inherited from IntegrationMixin
