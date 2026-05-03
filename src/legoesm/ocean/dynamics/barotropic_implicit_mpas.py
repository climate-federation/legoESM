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
returns ``Hu_avg = ½·(H_e_old·u_n + H_e_new·u_{n+1})`` — trapezoidal-
rule estimator of the time-integrated edge transport, matching the
explicit substep's box-averaged ``Hu_avg`` interface for the tracer
correction step in :class:`MPASOceanModel`.
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
    """
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

    # Mass-conserving floor clamp (safety net for extreme transients;
    # in normal operation this is a no-op since the PCG converges to
    # well-resolved η).
    eta_new = _clamp_redistribute(eta_new, eta_floor, mask, mesh.areaCell)

    # ----- Step 5: corrector for u_bar using new η gradient delta ------
    eta_filled_new = fill_land_cells_mpas(eta_new, mask, c1, c2)
    grad_eta_new = gradient_edge(eta_filled_new, mesh).astype(eta_dtype)
    delta_grad = grad_eta_new - grad_eta_old

    u_bar_new = (u_pred - theta_pgf * dt_t * g * delta_grad) * edge_mask

    # Bottom drag enters via F_slow_u (depth-mean of the 3D bottom-cell
    # drag set in ocean_pe_mpas.py); applying it again here would
    # double-count. The lat-lon implicit solver omits it for the same
    # reason. The explicit-substep solver still double-counts — tracked
    # as an open issue.

    # ----- Step 6: time-averaged transport for tracer flux --------------
    # Same partial-cell H_e convention as step 1 — use min-rule so the
    # tracer-flux divergence matches the η evolution exactly.
    if partial_cells:
        h_k_new = compute_layer_thickness(
            eta_new, H_bathy, z_coord,
            min_water_column_m=config.min_water_column_m,
        )
        H_e_new = _edge_H_min_rule(h_k_new, mesh, min_water_col).astype(eta_dtype)
    else:
        H_total_new = jnp.maximum(eta_new + H_bathy, min_water_col)
        H_e_new = _edge_avg(H_total_new, mesh)
    Hu_avg = (
        (1.0 - theta_eta) * H_e_old * u_bar_old
        + theta_eta * H_e_new * u_bar_new
    ) * edge_mask

    return eta_new, u_bar_new, Hu_avg
