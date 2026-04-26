"""Hydrostatic Primitive Equations on MPAS Voronoi meshes.

Implements the vector-invariant form of the hydrostatic PE using the
TRiSK discretization (Ringler et al. 2010) on the C-grid.  Reuses
the existing Voronoi operators and shape-agnostic vertical coordinate
functions (sigma and hybrid) from the core library.

Equations (per level k):
    du/dt = F_pv - grad(B_k) - R_d T_k grad(ln p_s) + visc + vert_adv
    dT/dt = -v·∇T - σ̇ ∂T/∂σ + κ T ω/p
    dp_s/dt = -(1/σ_range) ∫ div(p_s v) dσ

where B_k = KE + Φ_k (Bernoulli function), F_pv is the potential-
vorticity flux, and σ̇ is diagnosed from the continuity equation.

References
----------
- Ringler, T. D., et al. (2010). J. Comput. Phys., 229(9), 3065-3090.
- Skamarock, W. C., et al. (2012). Mon. Wea. Rev., 140, 3090-3105.
- Simmons & Burridge (1981). Mon. Wea. Rev., 109, 758-766.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.precision import cast_pytree

from legoesm.core.field import Field
from legoesm.core.state import MPASHydrostaticState, MPASHydrostaticTendencies
from legoesm.core.operators_voronoi import (
    # 2D operators used for surface-pressure-only fields (ln_ps, p_s).
    divergence_cell,
    gradient_edge,
    cell_to_edge_avg,
    # Batched 3D operators — single gather for all levels.
    divergence_cell_3d,
    gradient_edge_3d,
    kinetic_energy_cell_3d,
    potential_vorticity_vertex_3d,
    pv_flux_energy_conserving_3d,
    pv_flux_enstrophy_conserving_3d,
    vector_laplacian_del2_3d,
    vector_laplacian_del4_3d,
    cell_to_edge_avg_3d,
    edge_thickness_3d,
    apvm_correction_3d,
)
from legoesm.grids.voronoi import VoronoiMesh
from legoesm.grids.vertical import (
    SigmaCoordinate,
    HybridSigmaPressureCoordinate,
    pressure_from_sigma,
    pressure_from_hybrid,
    dp_from_hybrid,
    compute_geopotential,
    compute_geopotential_hybrid,
    compute_sigma_dot,
    compute_mass_flux_hybrid,
    vertical_advection,
    vertical_advection_hybrid,
    compute_pressure_velocity,
    compute_omega_hybrid,
)
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import IntegrationMixin
from legoesm import constants


class MPASPrimitiveEquationConfig(NamedTuple):
    """Configuration for the MPAS hydrostatic primitive equation model."""
    g: float = constants.g
    nu_del2: float = 0.0          # del2 viscosity [m²/s]
    nu_del4: float = 0.0          # del4 viscosity [m⁴/s]
    nu_del4_ps: float = 0.0       # del4 diffusion for surface pressure [m⁴/s]
    K_h: float = 0.0              # scalar diffusion [m²/s]
    T_min: float = 50.0           # temperature floor [K]
    p_floor: float = 100.0        # pressure floor [Pa] for adiabatic heating (limits 1/p)
    pv_scheme: str = "energy"     # "energy" or "enstrophy"
    apvm_scale: float = 0.0       # APVM upwinding (0 = off)
    fix_mass: bool = True
    time_integrator: str = "ssp_rk3"


# ============================================================================
# Tendency computation
# ============================================================================

def mpas_hydrostatic_tendencies(
    state: MPASHydrostaticState,
    mesh: VoronoiMesh,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    config: MPASPrimitiveEquationConfig = MPASPrimitiveEquationConfig(),
    physics_tendency: MPASHydrostaticTendencies | None = None,
    dt: float = 0.0,
) -> MPASHydrostaticTendencies:
    """Compute tendencies for the hydrostatic PE on an MPAS mesh.

    Parameters
    ----------
    state : MPASHydrostaticState
    mesh : VoronoiMesh
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
    config : MPASPrimitiveEquationConfig
    physics_tendency : MPASHydrostaticTendencies, optional
    dt : float
        Time step (needed for APVM correction).

    Returns
    -------
    MPASHydrostaticTendencies
    """
    _hybrid = isinstance(sigma_coord, HybridSigmaPressureCoordinate)

    u_3d = state.u.data        # (nEdges, nlev)
    T_3d = state.T.data        # (nCells, nlev)
    p_s = state.p_s.data       # (nCells,)
    phis = state.phis.data     # (nCells,)

    R_d = constants.R_d
    kappa = constants.kappa
    nlev = T_3d.shape[-1]
    p_s = jnp.clip(p_s, config.p_floor, 2.0e6)

    # --- 1. Pressure at full levels ---
    if _hybrid:
        p_full = pressure_from_hybrid(sigma_coord, p_s)   # (nCells, nlev)
        dp = dp_from_hybrid(sigma_coord, p_s)
    else:
        p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)

    # --- 2. Geopotential ---
    if _hybrid:
        Phi = compute_geopotential_hybrid(T_3d, p_s, sigma_coord, phis)
    else:
        Phi = compute_geopotential(T_3d, p_s, sigma_coord, phis)

    # --- 3. Precompute ln(p_s) gradient at edges ---
    ln_ps = jnp.log(p_s)
    grad_ln_ps = gradient_edge(ln_ps, mesh)  # (nEdges,)

    # --- Batched 3D tendencies (single gather for all levels) ---
    # Kinetic energy at all levels
    ke_3d = kinetic_energy_cell_3d(u_3d, mesh)  # (nCells, nlev)

    # Bernoulli function: KE + Phi
    bernoulli_3d = ke_3d + Phi  # (nCells, nlev)

    # Bernoulli gradient at edges — when scalar diffusion is active
    # (``K_h > 0``), batch ``grad(B)`` together with ``grad(T)`` along a
    # trailing axis so a single ``gradient_edge_3d`` (one ``cellsOnEdge``
    # gather + one finite difference) serves both.  Same passive
    # trailing-axis exploit as the MPAS divergence_cell_3d batches
    # below.
    if config.K_h > 0:
        n_cells_BT = bernoulli_3d.shape[0]
        nlev_BT = bernoulli_3d.shape[-1]
        _BT_stack = jnp.stack(
            [bernoulli_3d, T_3d], axis=-1,
        )  # (nCells, nlev, 2)
        _grad_BT = gradient_edge_3d(
            _BT_stack.reshape(n_cells_BT, nlev_BT * 2), mesh,
        ).reshape(-1, nlev_BT, 2)
        grad_B_3d = _grad_BT[..., 0]
        grad_T_3d_pre = _grad_BT[..., 1]
    else:
        grad_B_3d = gradient_edge_3d(bernoulli_3d, mesh)  # (nEdges, nlev)
        grad_T_3d_pre = None

    # Pressure gradient correction: R_d * T_edge * grad_eta(ln p)
    # In sigma coords: grad_eta(ln p) = grad(ln p_s).
    # In hybrid coords: grad_eta(ln p) = (B*p_s/p) * grad(ln p_s).
    if _hybrid:
        # Batch three cell-to-edge gathers (T_3d, p_full, dp) into one
        # call.  ``cell_to_edge_avg_3d`` is a pure ``cellsOnEdge``
        # gather + average — the trailing axis is purely passive.
        # ``edge_thickness_3d(dp)`` is just ``cell_to_edge_avg_3d(dp)``,
        # so folding ``dp`` into this batch lets the hybrid-path
        # ``dp_edge_3d`` reuse the gather instead of issuing a
        # standalone call later.  Same exploit as Loop 154 for MPAS CE.
        nlev_te = T_3d.shape[-1]
        _Tpd_stack = jnp.stack([T_3d, p_full, dp], axis=-1)  # (nCells, nlev, 3)
        _Tpd_edge = cell_to_edge_avg_3d(
            _Tpd_stack.reshape(_Tpd_stack.shape[0], nlev_te * 3), mesh,
        ).reshape(-1, nlev_te, 3)
        T_edge_3d = _Tpd_edge[..., 0]
        p_full_edge = _Tpd_edge[..., 1]
        dp_edge_3d = _Tpd_edge[..., 2]  # consumed in the divergence batch below
        p_s_edge_scalar = cell_to_edge_avg(p_s, mesh)    # (nEdges,)
        B_full = sigma_coord.B_full  # (nlev,)
        hybrid_factor_edge = B_full * p_s_edge_scalar[:, None] / jnp.maximum(p_full_edge, 1e-10)
        pg_corr_3d = R_d * T_edge_3d * grad_ln_ps[:, None] * hybrid_factor_edge
    else:
        T_edge_3d = cell_to_edge_avg_3d(T_3d, mesh)  # (nEdges, nlev)
        dp_edge_3d = None
        pg_corr_3d = R_d * T_edge_3d * grad_ln_ps[:, None]  # (nEdges, nlev)

    # PV flux: h_proxy = dp/g (pressure thickness)
    if _hybrid:
        h_proxy_3d = dp / config.g  # (nCells, nlev)
    else:
        h_proxy_3d = p_s[:, None] * sigma_coord.dsigma[None, :] / config.g

    q_v_3d = potential_vorticity_vertex_3d(u_3d, h_proxy_3d, mesh.fVertex, mesh)

    if config.apvm_scale > 0:
        q_v_3d = apvm_correction_3d(q_v_3d, u_3d, mesh, config.apvm_scale * dt)

    if config.pv_scheme == "enstrophy":
        pv_flux_3d = pv_flux_enstrophy_conserving_3d(u_3d, h_proxy_3d, q_v_3d, mesh)
    else:
        pv_flux_3d = pv_flux_energy_conserving_3d(u_3d, h_proxy_3d, q_v_3d, mesh)

    # Momentum tendency
    du_dt_3d = -grad_B_3d - pg_corr_3d + pv_flux_3d  # (nEdges, nlev)

    # Viscosity — when both del2 and del4 are active, the biharmonic
    # ``vector_laplacian_del4_3d`` is defined as
    # ``-vector_laplacian_del2_3d(vector_laplacian_del2_3d(u))``, so
    # the *inner* del2 is identical to the explicit del2 viscosity.
    # Compute it once and reuse — saves one full
    # ``vector_laplacian_del2_3d`` call (1 div + 1 curl + 1 grad +
    # 1 tangential-curl difference) per RHS evaluation.  Same exploit
    # as Loop 135 for the latlon ocean K_h+K_bih sharing.
    if config.nu_del2 > 0 and config.nu_del4 > 0:
        _del2_u = vector_laplacian_del2_3d(u_3d, mesh)
        du_dt_3d = du_dt_3d + config.nu_del2 * _del2_u
        du_dt_3d = du_dt_3d - config.nu_del4 * vector_laplacian_del2_3d(_del2_u, mesh)
    elif config.nu_del2 > 0:
        du_dt_3d = du_dt_3d + config.nu_del2 * vector_laplacian_del2_3d(u_3d, mesh)
    elif config.nu_del4 > 0:
        du_dt_3d = du_dt_3d + config.nu_del4 * vector_laplacian_del4_3d(u_3d, mesh)

    # Batched divergences.  ``divergence_cell_3d`` shares the same
    # MPAS edgesOnCell gather + reduce on the leading edge axis (the
    # trailing nlev axis is purely passive), so all the divergences
    # the dycore needs at this stage can fold into a single call:
    #
    #   * div(u)                 — continuity / sigma-dot closure
    #   * div(u * T_edge)        — temperature flux divergence
    #   * div(u * ln_ps_edge)    — v·∇(ln p_s) thermodynamic correction
    #   * div(u * dp_edge)       — hybrid layer-mass continuity
    #                              (only when ``_hybrid``)
    #   * div(grad_T)            — K_h scalar Laplacian (only when ``K_h > 0``)
    #
    # Pull ``ln_ps_edge`` and (when hybrid) ``u*dp_edge_3d`` and (when
    # K_h > 0) ``grad_T_3d_pre`` up into the batch so the standalone
    # divergence calls that previously fired later in the function are
    # eliminated.
    flux_T_3d = u_3d * T_edge_3d  # (nEdges, nlev)
    ln_ps_edge = cell_to_edge_avg(ln_ps, mesh)  # (nEdges,)
    flux_lnps_3d = u_3d * ln_ps_edge[:, None]   # (nEdges, nlev)
    n_edges_d, nlev_d = u_3d.shape

    _div_input_list = [u_3d, flux_T_3d, flux_lnps_3d]
    _idx_u, _idx_uT, _idx_ulnps = 0, 1, 2
    _idx_udp = -1
    _idx_gradT = -1
    if _hybrid:
        _div_input_list.append(u_3d * dp_edge_3d)
        _idx_udp = len(_div_input_list) - 1
    if config.K_h > 0:
        _div_input_list.append(grad_T_3d_pre)
        _idx_gradT = len(_div_input_list) - 1

    _n_div = len(_div_input_list)
    _div_inputs = jnp.stack(_div_input_list, axis=-1)  # (nEdges, nlev, K)
    _div_outputs = divergence_cell_3d(
        _div_inputs.reshape(n_edges_d, nlev_d * _n_div), mesh,
    ).reshape(-1, nlev_d, _n_div)
    div_3d = _div_outputs[..., _idx_u]
    div_uT_3d = _div_outputs[..., _idx_uT]
    div_flux_lnps = _div_outputs[..., _idx_ulnps]
    if _hybrid:
        div_dp_3d_pre = _div_outputs[..., _idx_udp]
    if config.K_h > 0:
        _div_grad_T = _div_outputs[..., _idx_gradT]
    horiz_adv_T_3d = -div_uT_3d + T_3d * div_3d  # (nCells, nlev)

    # Scalar diffusion — ``grad_T_3d`` and its divergence were already
    # computed in the batched blocks above; reuse the cached results.
    if config.K_h > 0:
        horiz_adv_T_3d = horiz_adv_T_3d + config.K_h * _div_grad_T

    # --- 4. Surface pressure tendency and vertical velocity ---
    if _hybrid:
        # Hybrid closure on MPAS:
        #   B_range * dp_s/dt = -sum_k div(dp_k * v_k)
        # ``div_dp_3d_pre`` (i.e. ``div(u * dp_edge_3d)``) was already
        # produced by the batched divergence block above (Loop 155),
        # so reuse it instead of issuing a standalone divergence call.
        div_dp_3d = div_dp_3d_pre  # (nCells, nlev)
        dp_s_dt = -jnp.sum(div_dp_3d, axis=-1) / sigma_coord.B_range

        mass_flux = compute_mass_flux_hybrid(div_3d, p_s, sigma_coord)
        vert_adv_T = vertical_advection_hybrid(T_3d, mass_flux, p_s, sigma_coord)
        omega = compute_omega_hybrid(mass_flux, p_s, dp_s_dt, sigma_coord)
    else:
        dsigma = sigma_coord.dsigma
        sigma_top = sigma_coord.sigma_half[0]
        sigma_range = 1.0 - sigma_top

        D_total = jnp.sum(div_3d * dsigma, axis=-1)  # (nCells,)
        dp_s_dt = -p_s * D_total / sigma_range

        sigma_dot = compute_sigma_dot(div_3d, sigma_coord)
        vert_adv_T = vertical_advection(T_3d, sigma_dot, sigma_coord)
        omega = compute_pressure_velocity(sigma_dot, p_s, dp_s_dt, sigma_coord)

    # Surface pressure hyperdiffusion: -nu * del2(del2(p_s))
    if config.nu_del4_ps > 0:
        del2_ps = divergence_cell(gradient_edge(p_s, mesh), mesh)
        del4_ps = divergence_cell(gradient_edge(del2_ps, mesh), mesh)
        dp_s_dt = dp_s_dt - config.nu_del4_ps * del4_ps

    # Vertical advection of u: approximate via edge-averaged sigma-dot
    if _hybrid:
        vert_adv_u = _vertical_advection_edge(u_3d, mass_flux, sigma_coord, mesh, hybrid=True, p_s=p_s)
    else:
        vert_adv_u = _vertical_advection_edge(u_3d, sigma_dot, sigma_coord, mesh, hybrid=False)

    du_dt_3d = du_dt_3d + vert_adv_u

    # --- 5. Thermodynamic equation ---
    # Adiabatic heating: κ·T·ω/p
    p_adiab = jnp.maximum(p_full, config.p_floor)
    adiabatic = kappa * T_3d * omega / p_adiab

    # v·∇(ln p_s) at cells: div(u * ln_ps_edge) - ln_ps * div(u).
    # ``div_flux_lnps`` was already computed via the batched divergence
    # block above; just combine it with ``div_3d`` here.
    v_grad_lnps = div_flux_lnps - ln_ps[:, None] * div_3d  # (nCells, nlev)

    # In hybrid coords: grad_eta(ln p) = (B*p_s/p) * grad(ln p_s),
    # so the adiabatic correction needs the same factor.
    if _hybrid:
        v_grad_lnps = v_grad_lnps * (sigma_coord.B_full * p_s[:, None] / p_adiab)
    adiabatic = adiabatic + kappa * T_3d * v_grad_lnps

    dT_dt_3d = horiz_adv_T_3d + vert_adv_T + adiabatic

    # --- 6. Add physics tendencies ---
    if physics_tendency is not None:
        du_dt_3d = du_dt_3d + physics_tendency.du_dt.data
        dT_dt_3d = dT_dt_3d + physics_tendency.dT_dt.data
        dp_s_dt = dp_s_dt + physics_tendency.dp_s_dt.data

    return MPASHydrostaticTendencies(
        du_dt=Field(data=du_dt_3d, name="du_dt",
                    dims=("nEdges", "nlev"), units="m/s²"),
        dT_dt=Field(data=dT_dt_3d, name="dT_dt",
                    dims=("nCells", "nlev"), units="K/s"),
        dp_s_dt=Field(data=dp_s_dt, name="dp_s_dt",
                      dims=("nCells",), units="Pa/s"),
        dphis_dt=Field(data=jnp.zeros_like(phis), name="dphis_dt",
                       dims=("nCells",), units="m²/s³"),
    )


def _vertical_advection_edge(
    u_3d, vert_vel, sigma_coord, mesh, hybrid=False, p_s=None,
):
    """Vertical advection of edge-based velocity.

    Interpolates the vertical velocity (sigma-dot or mass flux) from
    cells to edges, then applies the standard vertical advection.
    """
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]

    if hybrid:
        # mass_flux is (nCells, nlev+1), average to edges
        mass_flux_edge = 0.5 * (vert_vel[c1] + vert_vel[c2])
        p_s_edge = 0.5 * (p_s[c1] + p_s[c2])
        return vertical_advection_hybrid(u_3d, mass_flux_edge, p_s_edge, sigma_coord)
    else:
        # sigma_dot is (nCells, nlev+1), average to edges
        sigma_dot_edge = 0.5 * (vert_vel[c1] + vert_vel[c2])
        return vertical_advection(u_3d, sigma_dot_edge, sigma_coord)


# ============================================================================
# Model class
# ============================================================================

class MPASPrimitiveEquationModel(IntegrationMixin):
    """Hydrostatic primitive equation model on an MPAS Voronoi mesh.

    Parameters
    ----------
    mesh : VoronoiMesh
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
    config : MPASPrimitiveEquationConfig, optional
    """

    def __init__(
        self,
        mesh: VoronoiMesh,
        sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
        config: MPASPrimitiveEquationConfig | None = None,
    ):
        self.mesh = mesh
        self.sigma_coord = sigma_coord
        self.config = config or MPASPrimitiveEquationConfig()

    def tendencies(
        self,
        state: MPASHydrostaticState,
        dt: float = 0.0,
        physics_tendency: MPASHydrostaticTendencies | None = None,
    ) -> MPASHydrostaticTendencies:
        return mpas_hydrostatic_tendencies(
            state, self.mesh, self.sigma_coord, self.config,
            physics_tendency=physics_tendency, dt=dt,
        )

    @partial(jax.jit, static_argnums=(0, 3))
    def step(
        self,
        state: MPASHydrostaticState,
        dt: float,
        physics_fn=None,
    ) -> MPASHydrostaticState:
        """Advance one time step.

        Parameters
        ----------
        state : MPASHydrostaticState
        dt : float
        physics_fn : callable, optional
            Function (state, mesh, sigma_coord) -> MPASHydrostaticTendencies.

        Returns
        -------
        MPASHydrostaticState
        """
        state = cast_pytree(state, None, "compute")

        def tendency_fn(s):
            phys = None
            if physics_fn is not None:
                _phys_result = physics_fn(s, self.mesh, self.sigma_coord)
                phys = _phys_result[0] if type(_phys_result) is tuple else _phys_result
            tend = mpas_hydrostatic_tendencies(
                s, self.mesh, self.sigma_coord, self.config,
                physics_tendency=phys, dt=dt,
            )
            return MPASHydrostaticState(
                u=s.u.replace(data=tend.du_dt.data),
                T=s.T.replace(data=tend.dT_dt.data),
                p_s=s.p_s.replace(data=tend.dp_s_dt.data),
                phis=s.phis.replace(data=jnp.zeros_like(s.phis.data)),
            )

        state_new = dispatch_integrator(
            state, tendency_fn, dt, self.config.time_integrator,
        )

        # Temperature floor
        if self.config.T_min > 0:
            T_clipped = jnp.maximum(state_new.T.data, self.config.T_min)
            state_new = state_new._replace(
                T=state_new.T.replace(data=T_clipped),
            )

        if self.config.fix_mass:
            state_new = _fix_mass_mpas_hydro(state_new, state, self.mesh)

        return cast_pytree(state_new, None, "storage")

    # integrate() and integrate_scan() inherited from IntegrationMixin


def _fix_mass_mpas_hydro(state_new, state_old, mesh):
    """Fix mass conservation: uniform additive correction to p_s.

    The 3 sums (mass_old, mass_new, total_area) are computed locally
    and reduced together — this collapses 3 MPI allreduces into 1
    when the MPAS mesh is sharded across ranks.  ``total_area`` is
    constant per mesh; reduce it alongside the masses to keep the
    helper signature simple, and rely on XLA constant-folding for
    the case where it can.
    """
    area = mesh.areaCell
    local = jnp.stack([
        jnp.sum(state_old.p_s.data * area),
        jnp.sum(state_new.p_s.data * area),
        jnp.sum(area),
    ])
    if jax.process_count() > 1:
        from legoesm.parallel.reductions import global_sum_mpi
        local = global_sum_mpi(local)
    mass_old, mass_new, total_area = local[0], local[1], local[2]
    correction = (mass_old - mass_new) / total_area
    p_s_fixed = state_new.p_s.replace(data=state_new.p_s.data + correction)
    return state_new._replace(p_s=p_s_fixed)
