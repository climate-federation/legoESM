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
from legoesm.core.conservation import (
    compute_nh_dry_mass_mpas,
    fix_mass_nonhydrostatic_mpas,
)
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
from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    compute_exner_perturbation,
    sponge_profile,
)
from legoesm.atmosphere.physics.thermodynamics import sanitize_theta_rho
from legoesm.timestepping.integration import (
    IntegrationMixin,
    refuse_unthreaded_stateful_physics,
)
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
    fix_mass: bool = False                 # iter-8: opt-in anchored dry-mass fixer
    anchor_mass_to_initial: bool = False   # iter-8: snapshot target on first step


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

    # KE, Exner, and (optional) theta-prime gradients at edges.  All
    # are cell-centered (nCells, nlev) fields, and ``gradient_edge_3d``
    # treats the trailing axis as a passive batch (the cellsOnEdge
    # gather operates on the leading nCells axis only).  Stack the
    # fields along a new trailing axis, fold to (nCells, nlev*K), call
    # ``gradient_edge_3d`` once, then unfold and slice.  When K_h > 0
    # we add ``theta_p`` to the batch so its scalar-diffusion gradient
    # is computed in the same pass — Loop 128 extension of Loop 111.
    ke_3d = kinetic_energy_cell_3d(u_3d, mesh)             # (nCells, nlev)
    n_cells_g = pi_prime.shape[0]
    if config.K_h > 0:
        _grad_stack = jnp.stack(
            [pi_prime, ke_3d, theta_p], axis=-1,
        )  # (nCells, nlev, 3)
        _grad_flat = gradient_edge_3d(
            _grad_stack.reshape(n_cells_g, nlev * 3), mesh,
        ).reshape(-1, nlev, 3)
        grad_pi_3d = _grad_flat[..., 0]
        grad_ke_3d = _grad_flat[..., 1]
        grad_th_3d = _grad_flat[..., 2]
    else:
        _grad_stack = jnp.stack([pi_prime, ke_3d], axis=-1)
        _grad_flat = gradient_edge_3d(
            _grad_stack.reshape(n_cells_g, nlev * 2), mesh,
        ).reshape(-1, nlev, 2)
        grad_pi_3d = _grad_flat[..., 0]
        grad_ke_3d = _grad_flat[..., 1]
        grad_th_3d = None

    # Cell-to-edge averaging for theta, rho, and w_full — all three are
    # (nCells, nlev) inputs and ``cell_to_edge_avg_3d`` is a pure
    # ``cellsOnEdge`` gather + average (passive on the trailing axis).
    # Stack and fold so a single gather computes all three edge
    # averages.  3 calls → 1.  ``w_full`` is the half-to-full level
    # average of ``w``; computing it here lets us share its
    # cell-to-edge gather with theta/rho.  ``rho_e_3d`` and ``w_e_3d``
    # are consumed in the divergence batching below.
    n_cells_e = theta_total.shape[0]
    w_full = 0.5 * (w[:, :-1] + w[:, 1:])              # (nCells, nlev)
    _te_stack = jnp.stack(
        [theta_total, rho_total, w_full], axis=-1,
    )  # (nCells, nlev, 3)
    _te_edge = cell_to_edge_avg_3d(
        _te_stack.reshape(n_cells_e, nlev * 3), mesh,
    ).reshape(-1, nlev, 3)
    theta_e_3d = _te_edge[..., 0]
    rho_e_3d = _te_edge[..., 1]
    w_e_3d = _te_edge[..., 2]

    # PV flux (Coriolis + vorticity).  Use rho*dz as thickness proxy
    # for mass-weighted PV.  ``rho_e_3d`` is already produced by the
    # batched cell-to-edge gather above, so the matching edge-thickness
    # proxy is just ``rho_e_3d * dz`` — pass it via ``h_edge_3d=`` to
    # skip the internal ``edge_thickness_3d`` (one redundant
    # ``cellsOnEdge`` gather).
    h_proxy_3d = rho_total * dz[None, :]                   # (nCells, nlev)
    h_proxy_edge_3d = rho_e_3d * dz[None, :]               # (nEdges, nlev)
    f_v = mesh.fVertex if config.use_coriolis else jnp.zeros_like(mesh.fVertex)
    q_v_3d = potential_vorticity_vertex_3d(u_3d, h_proxy_3d, f_v, mesh)

    if config.pv_scheme == "energy":
        pv_flux_3d = pv_flux_energy_conserving_3d(
            u_3d, h_proxy_3d, q_v_3d, mesh,
            h_edge_3d=h_proxy_edge_3d,
        )
    elif config.pv_scheme == "enstrophy":
        pv_flux_3d = pv_flux_enstrophy_conserving_3d(
            u_3d, h_proxy_3d, q_v_3d, mesh,
            h_edge_3d=h_proxy_edge_3d,
        )
    else:
        raise ValueError(
            f"Unknown pv_scheme {config.pv_scheme!r}; "
            "expected one of: 'energy', 'enstrophy'."
        )

    # Momentum tendency
    du_dt_3d = pv_flux_3d - grad_ke_3d - c_p * theta_e_3d * grad_pi_3d

    # Viscosity — share the inner del2 between the explicit del2 and
    # the biharmonic del4 when both are active.  Same Loop 135 exploit
    # as the latlon ocean K_h+K_bih sharing — saves one full
    # ``vector_laplacian_del2_3d`` (1 div + 1 curl + 1 grad +
    # 1 tangential-curl difference) per RHS evaluation.
    if config.nu_del2 > 0 and config.nu_del4 > 0:
        _del2_u = vector_laplacian_del2_3d(u_3d, mesh)
        du_dt_3d = du_dt_3d + config.nu_del2 * _del2_u
        du_dt_3d = du_dt_3d - config.nu_del4 * vector_laplacian_del2_3d(
            _del2_u, mesh,
        )
    elif config.nu_del2 > 0:
        du_dt_3d = du_dt_3d + config.nu_del2 * vector_laplacian_del2_3d(
            u_3d, mesh,
        )
    elif config.nu_del4 > 0:
        du_dt_3d = du_dt_3d + config.nu_del4 * vector_laplacian_del4_3d(
            u_3d, mesh,
        )

    # Horizontal divergences batched.  ``divergence_cell_3d`` does a
    # gather (``u_edge_3d[edgesOnCell]``) + weighted reduce on the
    # leading axis only — the trailing nlev axis is purely passive.
    # Stack the four (nEdges, nlev) flux inputs along a new trailing
    # axis to (nEdges, nlev, 4), fold to (nEdges, nlev*4), call
    # ``divergence_cell_3d`` once on the thicker tensor, then unfold.
    # 4 divergence calls → 1.  ``w_e_3d`` is now produced by the
    # batched cell-to-edge gather above (Loop 154), so we can plug it
    # directly into the divergence batch.
    #
    # Loop 186 — when ``K_h > 0`` the scalar-diffusion step also needs
    # ``divergence_cell_3d(grad_th_3d)`` (theta diffusion).  Append it
    # to the trailing axis as a 5th passive batch entry — same gather
    # + weighted reduce, no extra halo cost — saving one full
    # ``divergence_cell_3d`` call per RHS evaluation.  Same exploit as
    # Loop 156 for the prescribed-wind tracer transport.
    n_edges_d, nlev_d = u_3d.shape
    # Terrain-following z* (J = dz/dz*, horizontally varying): continuity
    # is conservative in the pseudo-density J·ρ (docstring eq:
    # dρ'/dt = -(1/J)[div_h(J ρ v_h) + ∂(ρ w)/∂z*]).  J is averaged
    # cell→edge exactly like ρ (2-pt cellsOnEdge mean, mirroring
    # ``cell_to_edge_avg_3d``); J ≡ 1 (flat terrain) is bit-identical.
    J_edge = 0.5 * (J[c1] + J[c2])  # (nEdges,)
    _div_inputs = jnp.stack(
        [J_edge[:, None] * rho_e_3d * u_3d, u_3d * theta_e_3d, u_3d,
         u_3d * w_e_3d], axis=-1,
    )  # (nEdges, nlev, 4)
    _div_inputs_flat = _div_inputs.reshape(n_edges_d, nlev_d * 4)
    if config.K_h > 0:
        _div_inputs_flat = jnp.concatenate(
            [_div_inputs_flat, grad_th_3d], axis=-1,
        )  # (nEdges, nlev*4 + nlev) = (nEdges, nlev*5)
    _div_outputs_flat = divergence_cell_3d(_div_inputs_flat, mesh)
    _div_outputs = _div_outputs_flat[:, : nlev_d * 4].reshape(
        -1, nlev_d, 4,
    )
    div_rho_v_3d = _div_outputs[..., 0]
    div_u_theta_3d = _div_outputs[..., 1]
    div_u_3d = _div_outputs[..., 2]
    div_uw_3d = _div_outputs[..., 3]
    if config.K_h > 0:
        # 5th slice: ``divergence_cell_3d(grad_th_3d)`` — consumed below
        # in section 6 ("Scalar diffusion on theta").
        _diff_th_3d = _div_outputs_flat[:, nlev_d * 4:]  # (nCells, nlev)
    else:
        _diff_th_3d = None

    # Horizontal theta advection (advective form): -(div(u·θ) - θ · div(u)).
    dtheta_p_dt = -(div_u_theta_3d - theta_total * div_u_3d)

    # Horizontal continuity in z*: drho'/dt = -(1/J) div_h(J rho v).
    # The vertical leg carries its 1/J in ``mpas_acoustic_substeps``;
    # both legs now apply the metric consistently (net outflow of the
    # J-weighted flux lowers rho — sign per divergence_cell_3d > 0 for
    # outflow).
    drho_p_dt = -div_rho_v_3d / J[:, None]

    # --- 2. Vertical advection of u ---
    # Interpolate w from cells to edges, then apply vertical advection
    # (``J_edge`` hoisted above the continuity flux batch).
    w_edge = 0.5 * (w[c1] + w[c2])  # (nEdges, nlev+1)
    du_dt_3d = du_dt_3d + _vertical_advection_height_1d(
        u_3d, w_edge, dz, dz_half, J_edge,
    )

    # --- 3. Sponge layer ---
    sponge = sponge_profile(
        height_coord.z_full, height_coord.H,
        config.sponge_width, config.sponge_coeff,
    )
    du_dt_3d = du_dt_3d - sponge * u_3d
    dtheta_p_dt = dtheta_p_dt - sponge * theta_p

    sponge_half = sponge_profile(
        height_coord.z_half, height_coord.H,
        config.sponge_width, config.sponge_coeff,
    )

    # --- 4. w tendency (slow part: horizontal advection + sponge) ---
    # ``div_uw_3d`` was already computed via the batched divergence
    # block above.  Use it together with ``div_u_3d`` to assemble the
    # horizontal w-advection in advective form: -v·∇w ≈ -(div(u*w) -
    # w·div(u)).
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
        # ``_diff_th_3d`` is the 5th slice of the merged
        # ``divergence_cell_3d`` batch above (Loop 186) — its
        # ``divergence_cell_3d(grad_th_3d, mesh)`` call has already
        # been folded in at zero extra ``edgesOnCell`` gather cost.
        dtheta_p_dt = dtheta_p_dt + config.K_h * _diff_th_3d

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

    # Batch ``div(u)`` (used by the advective-form correction) with the
    # per-tracer ``div(u*q)`` flux divergences into one
    # ``divergence_cell_3d`` call by concatenating along the trailing
    # axis.  ``edgesOnCell`` is gathered once and the
    # ``sign·dvEdge`` weighting applies uniformly.  ``n_tracers + 1``
    # divergences → 1.  Same exploit as Loop 156 for the prescribed-wind
    # tracer transport.
    _u_and_flux = jnp.concatenate(
        [u_3d, flux_flat], axis=-1,
    )  # (nEdges, nlev*(1 + n_tracers))
    _u_flux_div = divergence_cell_3d(_u_and_flux, mesh)
    div_u_3d = _u_flux_div[:, :nlev_t]
    div_uq_flat = _u_flux_div[:, nlev_t:]
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
        # iter-8: lazy fp64 dry-mass snapshot for anchor-to-initial.
        self._target_mass: jax.Array | None = None

    def reset_target_mass(self) -> None:
        """Clear the anchored mass target (iter-18; see iter-4 SW twin)."""
        self._target_mass = None

    def set_target_mass(self, target_mass) -> None:
        """Explicitly set the anchored mass target (iter-19)."""
        self._target_mass = target_mass

    def compute_dry_mass(self, state) -> jax.Array:
        """Global dry mass ``∫ J · (rho_ref + rho_prime) · dz · dA`` (fp64)."""
        return compute_nh_dry_mass_mpas(
            state.rho_prime.data, self.height_coord,
            self.terrain_metric, self.mesh,
        )

    def tendencies(
        self,
        state: MPASNonHydrostaticState,
        physics_tendency: MPASNonHydrostaticTendencies | None = None,
    ) -> MPASNonHydrostaticTendencies:
        return mpas_compressible_euler_slow_tendencies(
            state, self.mesh, self.height_coord, self.terrain_metric,
            self.config, physics_tendency,
        )

    def step(
        self,
        state: MPASNonHydrostaticState,
        dt: float,
        physics_fn=None,
    ) -> MPASNonHydrostaticState:
        """Outer wrapper: snapshots dry mass on first call when
        ``anchor_mass_to_initial`` is on (fp64, outside JIT)."""
        # NH MPAS does not thread a PhysicsState carry — refuse a stateful
        # make_physics(model_type="nonhydrostatic") fn rather than silently
        # reseed its prognostic fields every step (#405/#413).
        refuse_unthreaded_stateful_physics(
            physics_fn, None, where="NH MPAS step()")
        target_mass = self._target_mass
        if (self.config.fix_mass
                and self.config.anchor_mass_to_initial
                and target_mass is None):
            target_mass = self.compute_dry_mass(state)
            if not isinstance(target_mass, jax.core.Tracer):
                # Designed eager path: cache the concrete t=0 mass so
                # later segments keep anchoring to the same constant.
                self._target_mass = target_mass
            # Traced path (step() inside an OUTER jit/grad/scan): NEVER
            # cache — a tracer stored on self leaks into the next trace
            # (UnexpectedTracerError; gh-417, same class as the
            # primitive_eq_cdgrid A1-gate bug).  Thread the per-call
            # pre-step mass instead (telescoping fixer semantics).
        return self._step_jit(state, dt, physics_fn, target_mass)

    @partial(jax.jit, static_argnums=(0, 3))
    def _step_jit(
        self,
        state: MPASNonHydrostaticState,
        dt: float,
        physics_fn=None,
        target_mass=None,
    ) -> MPASNonHydrostaticState:
        """Advance one time step using split-explicit RK3.

        Parameters
        ----------
        state : MPASNonHydrostaticState
        dt : float
        physics_fn : callable, optional
            Function (state, mesh, height_coord, terrain_metric) -> tendencies.
        target_mass : jax.Array | None
            Anchored dry mass; passed by ``step()``.  Skipped when fix_mass
            is off OR ``anchor_mass_to_initial`` is off.
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

        state_new = split_explicit_step(
            state, slow_tendency_fn, acoustic_update_fn, dt, se_config,
        )

        # iter-8: anchored dry-mass fixer (opt-in via fix_mass +
        # anchor_mass_to_initial).  Uniform additive correction to
        # ``rho_prime``; preserves rho gradients (same property as
        # cubed-sphere ``fix_mass_nonhydrostatic``).
        if (self.config.fix_mass and target_mass is not None):
            state_new = fix_mass_nonhydrostatic_mpas(
                state_new, target_mass,
                self.height_coord, self.terrain_metric, self.mesh,
            )

        return state_new

    # integrate() and integrate_scan() inherited from IntegrationMixin
