"""Implicit Crank-Nicolson barotropic free-surface solver (MPAS Voronoi).

Mirrors :mod:`barotropic_implicit_latlon_cgrid` but on a Voronoi C-grid
with TRiSK operators (Ringler et al. 2010).  Replaces the explicit
substep loop with a single-step implicit treatment of the depth-
integrated free-surface gravity wave equations.  Eliminates the
chequerboard contamination of the depth-mean velocity by construction:
the implicit Helmholtz operator has a small, well-defined null space, so
the TRiSK rotational null branch (Thuburn 2008; Ringler et al. 2010 §6)
that grew monotonically over 5 yr in the explicit run cannot accumulate.

Equations (depth-averaged, z-star reference column held over the step):

    ∂η/∂t = -∇·(H · u_bar)                  [continuity, cell-centered]
    ∂u_bar/∂t = -g·∇η + f·v_t(u_bar) + R_u  [momentum, edge-normal]

Scheme — Crank-Nicolson with Heun (trapezoidal) Coriolis predictor:

    Predictor (Heun on Coriolis, OLD η gradient):
        v_t_old = tangential_velocity(u_n, mesh)
        u_star  = u_n + dt·[-g·∇η_n + f·v_t_old + F_slow_u]
        v_t_star = tangential_velocity(u_star, mesh)
        u_pred  = u_n + dt·[-g·∇η_n + f·½(v_t_old + v_t_star) + F_slow_u]

    Time-averaged predicted transport (continuity weight θ_eta):
        u_avg = (1-θ_eta)·u_n + θ_eta·u_pred

    Elliptic Helmholtz solve for η^{n+1} (PGF weight θ_pgf):
        [I - θ_eta·θ_pgf·dt²·g·∇·(H·∇)] η^{n+1}
            = η_n  -  dt·∇·(H · u_avg)
                   -  θ_eta·θ_pgf·dt²·g·∇·(H·∇η_n)
                   +  dt·F_slow_eta

    Corrector (apply gradient delta from PGF):
        u^{n+1} = u_pred - θ_pgf·dt·g·(∇η^{n+1} - ∇η_n)

θ_eta = θ_pgf = 0.55 by default (slightly past Crank-Nicolson — gives
unconditional gravity-wave damping while staying close to 2nd-order
accurate; standard MITgcm/MPAS-O choice).

Coriolis treatment differs from the lat-lon C-grid path:
the lat-lon solver uses Forward-Backward (Matsuno) which directly
couples staggered (U, V) edges; on a Voronoi mesh tangential velocity
is reconstructed from neighboring normal velocities (Thuburn 2009
weights), so we use a Heun predictor-corrector instead. This is the
same scheme the explicit substep loop already uses (see
``barotropic_substeps_mpas`` line 221+).

Mass conservation:
the operator ``A(η) = η - coeff · div(H · grad η)`` is built from the
TRiSK ``divergence_cell`` and ``gradient_edge``, which are FV-adjoint:

    Σ_c A_c · div_c(F) · φ_c = -Σ_e F_e · grad_e(φ) · dv_e · dc_e  +  bdry

This makes ``A`` symmetric in the area-weighted cell inner product, so
PCG converges rapidly with a Jacobi preconditioner.

Differentiability:
``jax.scipy.sparse.linalg.cg`` is differentiable through implicit-
function-theorem custom-VJP.

Tracer transport consistency:
returns ``Hu_avg = H_e_old · [(1-θ)·u_n + θ·u_{n+1}]`` — time-averaged
edge transport using the OLD edge thickness throughout, ensuring
``div(Hu_avg) = (η_old − η_new)/dt`` exactly (required for flux-form
tracer conservation on partial cells).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.grids.voronoi import VoronoiMesh
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_layer_thickness
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.core.operators_voronoi import (
    divergence_cell,
    gradient_edge,
    tangential_velocity,
    vector_laplacian_del2,
    vector_laplacian_del4,
    edge_thickness as _edge_avg,
)
from legoesm.ocean.dynamics.eta_floor import (
    clamp_and_redistribute as _clamp_redistribute,
)
from legoesm.ocean.dynamics.mpas_fill import fill_land_cells_mpas
from legoesm.ocean.dynamics.mpas_partial_cell_helpers import min_cell_to_edge
from legoesm.ocean.vertical import OceanPartialCellCoordinate


def _depth_average_to_edges(
    u_3d: jnp.ndarray,
    h_k: jnp.ndarray,
    H_e: jnp.ndarray,
    mesh: VoronoiMesh,
    *,
    partial_cells: bool = False,
) -> jnp.ndarray:
    """Thickness-weighted depth-average of edge-normal velocity.

    On partial cells, the per-level edge thickness MUST use the min-rule
    (MITgcm hFacZ) so the depth-mean matches what the baroclinic step's
    ``F_slow_u`` is computed against (see ``ocean_pe_mpas.py:186``).
    Using a centered ``0.5*(h[c1]+h[c2])`` here lets phantom transport
    leak through a step edge — drives the seamount rest-state explosion.
    """
    if partial_cells:
        h_e_k = min_cell_to_edge(h_k, mesh)
    else:
        c1 = mesh.cellsOnEdge[0]
        c2 = mesh.cellsOnEdge[1]
        h_e_k = 0.5 * (h_k[c1] + h_k[c2])  # (nEdges, nlev)
    Hu = jnp.sum(u_3d * h_e_k, axis=1)
    return Hu / jnp.maximum(H_e, 1.0e-10)


def _edge_H_min_rule(
    h_k: jnp.ndarray,
    mesh: VoronoiMesh,
    min_water_col,
) -> jnp.ndarray:
    """Column-sum of min-rule per-level edge thickness — the partial-cell
    analog of ``_edge_avg(eta+H_bathy)``.

    On partial cells, the barotropic Helmholtz operator must use this
    flux-closure H_e (matching the baroclinic step's H_e) so that
    ``c² = g·H_min`` is the correct gravity-wave speed across step
    edges and the predictor-corrector eta/u_bar update stays
    consistent with the slow-forcing F_slow_u.
    """
    h_e_k = min_cell_to_edge(h_k, mesh)
    return jnp.maximum(jnp.sum(h_e_k, axis=1), min_water_col)


def _make_helmholtz(
    H_e: jnp.ndarray,
    coeff: jnp.ndarray,
    mesh: VoronoiMesh,
    mask: jnp.ndarray,
    edge_mask: jnp.ndarray,
):
    """Return ``A_op(η) = η - coeff · div(H · grad η)`` (cell-centred Helmholtz).

    Built from the TRiSK FV gradient + FV divergence (Ringler et al.
    2010 Eqs. 21-22).  Symmetric in the area-weighted cell inner
    product, positive-definite for ``coeff ≥ 0`` and ``H ≥ 0``.

    The land-cell Neumann fill on η before gradient_edge ensures the
    operator is consistent with the same fill used in the explicit
    substep — both feed gradient_edge a coastline-smoothed field rather
    than the raw (ocean → 0 → ocean) jump.
    """
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    H_e_face = H_e * edge_mask

    def A_op(eta_in: jnp.ndarray) -> jnp.ndarray:
        eta_m = eta_in * mask
        eta_filled = fill_land_cells_mpas(eta_m, mask, c1, c2)
        grad = gradient_edge(eta_filled, mesh)
        flux = H_e_face * grad
        div_grad = divergence_cell(flux, mesh) * mask
        return (eta_m - coeff * div_grad) * mask

    return A_op


def _make_diag_preconditioner(
    H_e: jnp.ndarray,
    coeff: jnp.ndarray,
    mesh: VoronoiMesh,
    mask: jnp.ndarray,
    edge_mask: jnp.ndarray,
):
    """Jacobi preconditioner for the Voronoi Helmholtz.

    Diagonal of ``A(η) = η - coeff · div(H · grad η)``:

        A_diag(c) = 1 + coeff · (1/A_c) · Σ_{e∈∂c} H_e · dv_e / dc_e

    (See module docstring for the derivation.  Uses scatter-gather over
    ``edgesOnCell`` to stay JIT-compatible.)
    """
    eoc = mesh.edgesOnCell  # (maxEdges, nCells)
    mask_eoc = (eoc >= 0).astype(H_e.dtype)
    eoc_safe = jnp.maximum(eoc, 0)

    H_g = (H_e * edge_mask)[eoc_safe]   # (maxEdges, nCells)
    dv_g = mesh.dvEdge[eoc_safe]
    dc_g = mesh.dcEdge[eoc_safe]

    diag_off = jnp.sum(
        H_g * dv_g / dc_g * mask_eoc, axis=0,
    ) / mesh.areaCell
    diag = 1.0 + coeff * diag_off
    inv_diag = jnp.where(
        mask > 0.5, 1.0 / jnp.maximum(diag, 1.0e-30), 0.0,
    )

    def M_inv(r: jnp.ndarray) -> jnp.ndarray:
        return r * inv_diag

    return M_inv


def barotropic_implicit_mpas(
    state,
    mesh: VoronoiMesh,
    z_coord: OceanZStarCoordinate,
    config: MPASOceanConfig,
    dt: float,
    F_slow_eta=None,
    F_slow_u=None,
    *,
    return_residual: bool = False,
):
    """Single-step implicit free-surface solver (MPAS Voronoi C-grid).

    Parameters
    ----------
    state : MPASOceanState
        State with 3D velocity already updated by baroclinic tendency.
    mesh : VoronoiMesh
    z_coord : OceanZStarCoordinate
    config : MPASOceanConfig
    dt : float
        Full baroclinic timestep [s] — no substepping.
    F_slow_eta : jax.Array or None, shape (nCells,)
    F_slow_u : jax.Array or None, shape (nEdges,)
        Depth-mean of the 3D baroclinic tendency with the planetary-
        Coriolis contribution EXCLUDED (see ``ocean_pe_mpas.py``).

    Returns
    -------
    eta_new : (nCells,)
    u_bar_new : (nEdges,)
    Hu_avg : (nEdges,)
        Time-averaged depth-integrated edge transport, used by the
        tracer-flux step in :class:`MPASOceanModel` (matches the
        :func:`barotropic_substeps_mpas` interface).

    When ``return_residual=True`` (static; default ``False``) a fourth
    value ``rel_residual`` is appended — the (rank-local) relative
    Helmholtz residual diagnostic of the stock-CG solve.  MPAS runs stock
    CG only (single-rank; the distributed PCG is deferred — see the
    Step-4 note), so this residual is NOT a global reduction.  Log /
    assert it OUTSIDE the JIT; never branch the compiled step on it.
    """
    # FAIL-FAST at ENTRY, before any state-derived JAX work (codex
    # 2026-06-11 MINOR: the previous placement built the predictor/RHS
    # first).  Predicate (codex CRITICAL): ``is_distributed()`` alone
    # NEVER fires on the Voronoi MPI path — it checks the global halo
    # backend, which ``initialize_voronoi_mpi`` does not arm; the
    # mpi4py world size trips on any real multi-rank launch.  This
    # solver would otherwise run a SILENT rank-local stock CG + a
    # rank-local mass projection (see TODO(distributed-mpas-pcg)).
    from legoesm.core.operators import is_distributed as _is_distributed
    from legoesm.parallel.reductions import mpi_world_size as _world
    if _is_distributed() or _world() > 1:
        raise NotImplementedError(
            "MPAS barotropic_solver='implicit_cn' is single-rank only: "
            "the stock-CG solve and its mass projection are rank-local "
            "and would silently diverge under MPI.  Use "
            "barotropic_solver='explicit_substep' for distributed MPAS "
            "runs (see TODO(distributed-mpas-pcg) in "
            "barotropic_implicit_mpas.py)."
        )
    g = jnp.asarray(config.g)
    mask = state.land_mask.data
    H_bathy = state.H_bathy.data
    eta_old = state.eta.data
    u_3d = state.u.data

    eta_dtype = eta_old.dtype
    g = g.astype(eta_dtype)
    dt_t = jnp.asarray(dt, dtype=eta_dtype)
    min_water_col = jnp.asarray(config.min_water_column_m, dtype=eta_dtype)

    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    edge_mask = mask[c1] * mask[c2]

    eta_floor = (min_water_col - H_bathy) * mask
    eta_old = jnp.maximum(eta_old, eta_floor) * mask

    if F_slow_eta is None:
        F_slow_eta = jnp.zeros_like(eta_old)
    else:
        F_slow_eta = F_slow_eta.astype(eta_dtype)
    if F_slow_u is None:
        F_slow_u = jnp.zeros((u_3d.shape[0],), dtype=eta_dtype)
    else:
        F_slow_u = F_slow_u.astype(eta_dtype)

    # ----- Step 1: depth-average u to edges -----------------------------
    h_k_old = compute_layer_thickness(
        eta_old, H_bathy, z_coord,
        min_water_column_m=config.min_water_column_m,
    )  # (nCells, nlev)
    # Partial-cell H_e: use the min-rule column-sum of per-level edge
    # thickness so the Helmholtz operator's gravity-wave speed matches
    # the flux closure at step edges (consistent with the baroclinic
    # step's H_e at ocean_pe_mpas.py:186).  On legacy z-star (every
    # column full) min-rule equals 0.5*(H[c1]+H[c2]) bit-exactly.
    partial_cells = isinstance(z_coord, OceanPartialCellCoordinate)
    if partial_cells:
        H_e_old = _edge_H_min_rule(h_k_old, mesh, min_water_col).astype(eta_dtype)
    else:
        H_total_old = jnp.maximum(eta_old + H_bathy, min_water_col)
        H_e_old = _edge_avg(H_total_old, mesh)
    u_bar_old = _depth_average_to_edges(
        u_3d, h_k_old, H_e_old, mesh, partial_cells=partial_cells,
    ).astype(eta_dtype)

    # ----- Step 2: predictor (Heun on Coriolis, OLD η gradient) ---------
    eta_filled_old = fill_land_cells_mpas(eta_old, mask, c1, c2)
    grad_eta_old = gradient_edge(eta_filled_old, mesh).astype(eta_dtype)
    f_e = mesh.fEdge.astype(eta_dtype)

    v_t_old = tangential_velocity(u_bar_old, mesh)
    u_star = (
        u_bar_old + dt_t * (-g * grad_eta_old + f_e * v_t_old + F_slow_u)
    ) * edge_mask
    v_t_star = tangential_velocity(u_star, mesh)
    u_pred = (
        u_bar_old
        + dt_t * (
            -g * grad_eta_old
            + f_e * 0.5 * (v_t_old + v_t_star)
            + F_slow_u
        )
    ) * edge_mask

    # ----- Step 3: Helmholtz RHS ----------------------------------------
    theta_eta = jnp.asarray(
        config.barotropic_implicit_theta_eta, dtype=eta_dtype,
    )
    theta_pgf = jnp.asarray(
        config.barotropic_implicit_theta_pgf, dtype=eta_dtype,
    )
    coeff = theta_eta * theta_pgf * dt_t * dt_t * g

    u_avg_pred = ((1.0 - theta_eta) * u_bar_old + theta_eta * u_pred) * edge_mask

    flux_HU_pred = H_e_old * u_avg_pred * edge_mask
    div_HU_pred = divergence_cell(flux_HU_pred, mesh) * mask
    div_HU_pred = div_HU_pred.astype(eta_dtype)

    flux_eta_old = H_e_old * grad_eta_old * edge_mask
    div_grad_eta_old = divergence_cell(flux_eta_old, mesh) * mask
    div_grad_eta_old = div_grad_eta_old.astype(eta_dtype)

    rhs = (
        eta_old
        - dt_t * div_HU_pred
        - coeff * div_grad_eta_old
        + dt_t * F_slow_eta * mask
    ) * mask

    # ----- Step 4: PCG solve --------------------------------------------
    # MPAS stays on stock ``jax.scipy`` CG (single-rank only).  The
    # distributed fixed-iteration PCG that the lat-lon C-grid solver uses
    # (``barotropic_common.solve_helmholtz_implicit``) is NOT yet wired up
    # for MPAS because the Voronoi barotropic path lacks the two pieces a
    # multi-rank iterated solve needs, and adding them is real
    # infrastructure rather than a shared-helper reuse:
    #
    #   1. Halo-in-matvec.  ``A_op`` -> ``gradient_edge(phi_cell)`` only
    #      indexes ``phi_cell[cellsOnEdge]`` locally; it does NOT exchange
    #      ghost-cell values.  The explicit substep loop has no halo
    #      exchange either.  In a fixed-iteration PCG ``p``/``eta`` change
    #      every iteration, so the ghost cells would go stale after the
    #      first matvec.  A correct distributed MPAS PCG must call a
    #      Voronoi cell halo-exchange inside ``A_op`` before
    #      ``fill_land_cells_mpas``/``gradient_edge``.
    #   2. Owned-cell reductions.  Voronoi local meshes hold owned + halo
    #      cells (``voronoi_mpi.make_voronoi_partition_layout`` exposes
    #      ``owned_mask_cells``).  PCG dot products, the residual, and the
    #      mass projection would double-count ghost cells unless every
    #      global SUM is masked to owned cells.
    #
    # The lat-lon C-grid band decomposition has neither problem (its
    # ``A_op`` pre-pads through the backend-dispatched halo, and cell rows
    # partition without overlap so there are no ghost cells in the
    # reduction).  Distributing the MPAS PCG is therefore deferred —
    # TODO(distributed-mpas-pcg): thread a Voronoi cell-halo exchange into
    # ``A_op`` and an ``owned_cell_mask`` into ``solve_helmholtz_implicit``
    # /``_global_dot_batch``, then mirror the lat-lon dispatch here.  The
    # MPAS ocean MPI path is currently forward-only for AD
    # (``voronoi_mpi.py``), so implicit_cn under MPI is unsupported until
    # then; single-rank stock CG is unchanged and fully differentiable.
    # (np>1 refusal is at function ENTRY — see top of this function.)
    A_op = _make_helmholtz(H_e_old, coeff, mesh, mask, edge_mask)
    M_inv = _make_diag_preconditioner(H_e_old, coeff, mesh, mask, edge_mask)

    pcg_tol = jnp.asarray(
        config.barotropic_implicit_pcg_tol, dtype=eta_dtype,
    )
    pcg_maxiter = int(config.barotropic_implicit_pcg_maxiter)
    eta_new, _info = jax.scipy.sparse.linalg.cg(
        A_op, rhs, x0=eta_old, tol=pcg_tol, maxiter=pcg_maxiter, M=M_inv,
    )
    eta_new = eta_new * mask
    # Single-rank only (see Step-4 note): the area-weighted sums are
    # rank-local, which is exact because there is exactly one rank.  When
    # distributed MPAS lands, these must become owned-cell-masked global
    # SUMs (NOT a bare ``batch_allreduce_mpi``, which would double-count
    # halo cells).  Accumulate in ``ocean_diagnostics`` precision so the
    # huge-area ``target - actual`` cancellation survives in f32.
    from legoesm.core.precision import cast as _cast
    _M = "ocean_diagnostics"
    _area_cell = mesh.areaCell.astype(eta_dtype)
    _area_acc = _cast(_area_cell, _M, "accumulate")
    _wa = _area_acc * _cast(mask, _M, "accumulate")
    _ocean_area = jnp.sum(_wa)
    _target_mass = jnp.sum(_cast(rhs, _M, "accumulate") * _area_acc)
    _actual_mass = jnp.sum(_cast(eta_new, _M, "accumulate") * _area_acc)
    _correction = (_target_mass - _actual_mass) / jnp.maximum(_ocean_area, 1e-30)
    eta_new = (eta_new + _correction.astype(eta_dtype) * mask) * mask
    # Residual diagnostic for the single-rank stock-CG path (uniform
    # return shape with the lat-lon solver's ``return_residual``).
    # RANK-LOCAL ON PURPOSE: do NOT route through the lat-lon helper's
    # ``_global_dot_batch`` (which would fire a bare ``batch_allreduce_mpi``
    # under MPI and double-count Voronoi halo cells — the exact reason the
    # MPAS solver is single-rank only here).  Plain ``jnp.sum`` is exact
    # for the single rank this path runs on.  Computed AFTER the floor
    # clamp below so it reflects the ACTUAL returned eta.

    # Mass-conserving floor clamp (safety net for extreme transients;
    # in normal operation this is a no-op since the PCG converges to
    # well-resolved η).
    eta_new = _clamp_redistribute(eta_new, eta_floor, mask, mesh.areaCell)

    # Residual diagnostic of the FINAL eta (post projection + clamp).
    # Rank-local (single-rank path); ``stop_gradient`` keeps it out of
    # reverse mode.
    _rr = jnp.sum((rhs - A_op(eta_new)) ** 2)
    _bb = jnp.sum(rhs ** 2)
    _solve_diag_rel = jax.lax.stop_gradient(
        jnp.sqrt(_rr / jnp.maximum(_bb, jnp.asarray(1.0e-30, dtype=eta_dtype)))
    )

    # ----- Step 5: corrector for u_bar using new η gradient delta ------
    eta_filled_new = fill_land_cells_mpas(eta_new, mask, c1, c2)
    grad_eta_new = gradient_edge(eta_filled_new, mesh).astype(eta_dtype)
    delta_grad = grad_eta_new - grad_eta_old

    u_bar_new = (u_pred - theta_pgf * dt_t * g * delta_grad) * edge_mask

    # Barotropic-mode lateral viscosity on u_bar — damps modes that
    # have ∇·(H·u_bar)≈0 (so the Helmholtz solve doesn't see them) and
    # f·v_t cancellations near step edges (so the predictor-corrector
    # doesn't damp them either).  On flat bottom the implicit Helmholtz
    # is sufficient (project_mpas_barotropic_noise.md, 5-yr σ plateau);
    # on partial-cell ETOPO the topographic step edges energize a
    # rotational u_bar null mode that grows e-folding ~5 days
    # (project_mpas_etopo_instability.md).  Mirrors the explicit-substep
    # path (barotropic_mpas.py:272) and the lat-lon Follow-up C
    # recommendation (docs/issues/barotropic_mode_noise.md §"Residual").
    A_baro_visc = jnp.asarray(
        getattr(config, "barotropic_u_viscosity", 0.0), dtype=eta_dtype,
    )
    # Per-edge equatorial-boost factor — same mechanism as 3D A_h.
    # Damps the equatorial f→0 u_baro mode that the implicit-CN
    # solver's Coriolis predictor-corrector cannot catch.  See
    # project_mpas_etopo_instability.md §"equatorial mode".
    _eq_boost = jnp.asarray(
        getattr(config, "equatorial_visc_boost", 0.0), dtype=eta_dtype,
    )
    _cos2 = jnp.cos(mesh.latEdge.astype(eta_dtype)) ** 2
    _lat_factor = 1.0 + _eq_boost * _cos2  # (nEdges,)
    if config.barotropic_u_viscosity > 0.0:
        lap_u = vector_laplacian_del2(u_bar_new, mesh).astype(eta_dtype)
        u_bar_new = (
            u_bar_new + dt_t * A_baro_visc * _lat_factor * lap_u
        ) * edge_mask

    # Biharmonic ∇⁴ damping on u_bar — preferred over harmonic for the
    # partial-cell rotational null mode (dycore-expert review 2026-05-03).
    # Scale-selective: damps grid-scale much harder than mesoscale, so
    # safe to use at production strength.  ``vector_laplacian_del4``
    # returns ``-∇²(∇²u)`` so adding ``+dt·K·del4`` gives stable decay.
    K_baro_bih = jnp.asarray(
        getattr(config, "barotropic_u_biharmonic", 0.0), dtype=eta_dtype,
    )
    if config.barotropic_u_biharmonic > 0.0:
        del4_u = vector_laplacian_del4(u_bar_new, mesh).astype(eta_dtype)
        u_bar_new = (u_bar_new + dt_t * K_baro_bih * del4_u) * edge_mask

    # Bottom drag enters via F_slow_u (depth-mean of the 3D bottom-cell
    # drag set in ocean_pe_mpas.py); applying it again here would
    # double-count. The lat-lon implicit solver omits it for the same
    # reason. The explicit-substep solver still double-counts — tracked
    # as an open issue.

    # ----- Step 6: time-averaged transport for tracer flux --------------
    # Use H_e_old consistently so that div(Hu_avg) = (eta_old - eta_new)/dt
    # exactly — required for flux-form tracer conservation.  The Helmholtz
    # solve used H_e_old throughout, so the transport average must too.
    # Recomputing an end-of-step edge thickness and using it here instead
    # introduced a theta * div((H_e_new - H_e_old) * u_new) error that broke
    # salt conservation on partial cells (shallow cells lost 0.003 PSU in
    # 30 days), so no end-of-step H_e is formed.
    Hu_avg = H_e_old * (
        (1.0 - theta_eta) * u_bar_old + theta_eta * u_bar_new
    ) * edge_mask

    if return_residual:
        return eta_new, u_bar_new, Hu_avg, _solve_diag_rel
    return eta_new, u_bar_new, Hu_avg
