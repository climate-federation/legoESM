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
function-theorem custom-VJP, but its transpose solve re-uses the SAME
operator, assuming Euclidean symmetry — and ``A`` is symmetric only in
the AREA-WEIGHTED cell inner product (measured Euclidean asymmetry
1.4e-2 on the level-2 icosahedral mesh, area-weighted ~2e-15), so the
stock VJP silently biased every reverse-mode gradient through the
free-surface solve (forward always correct).  Fixed backward-only via
``jax.custom_vjp`` in :func:`solve_helmholtz_freesurface_mpas`: the
primal CG call is verbatim (forward bit-identical) and the adjoint
applies the TRUE transpose ``A^{-T} = W·A^{-1}·W^{-1}`` with
``W = diag(areaCell)`` — the exact mirror of the lat-lon C-grid fix in
``barotropic_implicit_latlon_cgrid.solve_helmholtz_freesurface`` and a
sibling of the rigid-lid seam-reduced adjoint (commit 1e370679).

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


def _helmholtz_apply_mpas(
    eta_in: jnp.ndarray,
    H_e: jnp.ndarray,
    coeff: jnp.ndarray,
    mesh: VoronoiMesh,
    mask: jnp.ndarray,
    edge_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Apply ``A(η) = η - coeff · div(H · grad η)`` (cell-centred Helmholtz).

    Parametric form so the custom VJP in
    :func:`solve_helmholtz_freesurface_mpas` can take exact cotangents
    w.r.t. the operator parameters (``θ̄ = -(∂_θ A(θ)·x)ᵀ·λ``) via
    ``jax.vjp`` without closure-capturing tracers (scan-lowering safe;
    mirrors ``barotropic_implicit_latlon_cgrid._helmholtz_apply``).

    ``fill_land_cells_mpas`` is deliberately NOT applied here, and its
    absence is value-preserving rather than a behaviour change.
    ``gradient_edge`` is the two-cell stencil
    ``grad(e) = (phi[c2(e)] - phi[c1(e)]) / dcEdge`` and
    ``edge_mask = mask[c1]*mask[c2]`` is zero on every edge with a land
    endpoint; the fill only alters land cells, so any edge whose gradient
    could see an altered value carries zero flux. The filled field cannot
    reach the output. It was removed because this operator is the inner
    matvec of the barotropic PCG — 60 applications per step — and the fill
    costs several scatter-add passes in each one, which measured as dead
    work in the lane's strong-scaling plateau. The routine is still used
    where its output IS consumed: ``init_mpas``, ``ocean_pe_mpas``, and the
    eta_old/eta_new fills elsewhere in this file.
    """
    H_e_face = H_e * edge_mask
    eta_m = eta_in * mask
    grad = gradient_edge(eta_m, mesh)
    flux = H_e_face * grad
    div_grad = divergence_cell(flux, mesh) * mask
    return (eta_m - coeff * div_grad) * mask


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
    product (NOT the Euclidean one — the divergence carries 1/areaCell;
    see :func:`solve_helmholtz_freesurface_mpas`), positive-definite for
    ``coeff ≥ 0`` and ``H ≥ 0``.

    The land-cell Neumann fill on η before gradient_edge ensures the
    operator is consistent with the same fill used in the explicit
    substep — both feed gradient_edge a coastline-smoothed field rather
    than the raw (ocean → 0 → ocean) jump.
    """

    def A_op(eta_in: jnp.ndarray) -> jnp.ndarray:
        return _helmholtz_apply_mpas(eta_in, H_e, coeff, mesh, mask, edge_mask)

    return A_op


def _helmholtz_inv_diag_mpas(
    H_e: jnp.ndarray,
    coeff: jnp.ndarray,
    mesh: VoronoiMesh,
    mask: jnp.ndarray,
    edge_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Inverse diagonal of the Voronoi Helmholtz (Jacobi preconditioner).

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
    return jnp.where(
        mask > 0.5, 1.0 / jnp.maximum(diag, 1.0e-30), 0.0,
    )


def _make_diag_preconditioner(
    H_e: jnp.ndarray,
    coeff: jnp.ndarray,
    mesh: VoronoiMesh,
    mask: jnp.ndarray,
    edge_mask: jnp.ndarray,
):
    """Return ``M(r) = diag⁻¹ · r`` Jacobi preconditioner (see
    :func:`_helmholtz_inv_diag_mpas`)."""
    inv_diag = _helmholtz_inv_diag_mpas(H_e, coeff, mesh, mask, edge_mask)

    def M_inv(r: jnp.ndarray) -> jnp.ndarray:
        return r * inv_diag

    return M_inv


def solve_helmholtz_freesurface_mpas(
    rhs: jnp.ndarray,
    x0: jnp.ndarray,
    H_e: jnp.ndarray,
    coeff: jnp.ndarray,
    mask: jnp.ndarray,
    edge_mask: jnp.ndarray,
    inv_diag: jnp.ndarray,
    mesh: VoronoiMesh,
    *,
    tol,
    maxiter: int,
) -> jnp.ndarray:
    """Preconditioned-CG Helmholtz solve with the area-weighted adjoint.

    Exact mirror of
    ``barotropic_implicit_latlon_cgrid.solve_helmholtz_freesurface`` on
    the Voronoi mesh — read that docstring for the full derivation and
    the probe/ test references.  Summary:

    - Forward: byte-identical to the stock ``jax.scipy.sparse.linalg.cg``
      call (the pre-fix path, verbatim primal inside ``jax.custom_vjp``).
    - ``A`` is self-adjoint only in ``<x,y>_W = Σ x·y·areaCell``
      (``Aᵀ = W·A·W⁻¹``); stock cg's VJP re-solves with ``A`` assuming
      Euclidean symmetry → biased reverse-mode gradients.  The custom
      bwd applies the true transpose

          λ = A⁻ᵀ·η̄ = w̃ · cg(A, η̄·mask/w̃)·mask,
          w̃ = areaCell/max(areaCell)

      (the constant rescale cancels exactly by linearity — verified
      bitwise in x64 — and keeps the adjoint rhs at O(η̄); raw ~1e12 m²
      cell areas push float32 CG residual norms toward subnormal, the
      config-dependent NaN gradient seen while shipping this fix, which
      w̃ removes entirely at zero cost).
    - Operator-parameter cotangents are exact IFT,
      ``θ̄ = -(∂_θ A(θ)·x)ᵀ·λ`` via ``jax.vjp`` of
      :func:`_helmholtz_apply_mpas` at the converged solution, for
      θ ∈ {H_e, coeff, mask, edge_mask}.
    - ``x0`` and ``inv_diag`` get exact zero cotangents (IFT; stock-cg
      behaviour for the Krylov guess).
    - CLOSURE RULE: fwd/bwd closures capture only ``mesh``/``maxiter``;
      every traced array AND ``tol`` (array-valued in the production
      caller, created inside the traced step body) is an explicit
      custom_vjp argument — a closure-captured constant array leaks as a
      tracer under grad-of-scan ("No constant handler for
      DynamicJaxprTracer").
    - ``custom_vjp`` does not support forward-mode AD: ``jax.jvp``
      through this solve raises (documented limitation; remedy = paired
      ``custom_jvp`` on the same operator).

    Verified against a dense ground truth (level-2 icosahedral mesh,
    explicit matrix, ``Sᵀ`` solve) in
    tests/ocean/unit/test_freesurface_helmholtz_adjoint_mpas.py,
    including the weighting-slip mutations.
    """

    def _forward_cg(rhs_in, x0_in, H_e_in, coeff_in,
                    mask_in, edge_mask_in, inv_diag_in, tol_in):
        # The pre-fix forward, verbatim.
        A_op = _make_helmholtz(H_e_in, coeff_in, mesh, mask_in, edge_mask_in)

        def M_inv(r):
            return r * inv_diag_in

        eta_sol, _info = jax.scipy.sparse.linalg.cg(
            A_op, rhs_in, x0=x0_in, tol=tol_in, maxiter=maxiter, M=M_inv,
        )
        return eta_sol

    @jax.custom_vjp
    def _cg_area_adjoint(rhs_in, x0_in, H_e_in, coeff_in,
                         mask_in, edge_mask_in, inv_diag_in, tol_in):
        return _forward_cg(rhs_in, x0_in, H_e_in, coeff_in,
                           mask_in, edge_mask_in, inv_diag_in, tol_in)

    def _cg_fwd(rhs_in, x0_in, H_e_in, coeff_in,
                mask_in, edge_mask_in, inv_diag_in, tol_in):
        eta_sol = _forward_cg(rhs_in, x0_in, H_e_in, coeff_in,
                              mask_in, edge_mask_in, inv_diag_in, tol_in)
        res = (eta_sol, H_e_in, coeff_in, mask_in, edge_mask_in,
               inv_diag_in, tol_in)
        return eta_sol, res

    def _cg_bwd(res, eta_bar):
        (eta_sol, H_e_in, coeff_in, mask_in, edge_mask_in,
         inv_diag_in, tol_in) = res
        A_op = _make_helmholtz(H_e_in, coeff_in, mesh, mask_in, edge_mask_in)

        def M_inv(r):
            return r * inv_diag_in

        area_w = mesh.areaCell.astype(eta_bar.dtype)
        w_rel = area_w / jnp.max(area_w)
        eta_bar_m = eta_bar * mask_in
        lam_hat, _info = jax.scipy.sparse.linalg.cg(
            A_op, eta_bar_m / w_rel, x0=jnp.zeros_like(eta_bar_m),
            tol=tol_in, maxiter=maxiter, M=M_inv,
        )
        lam = w_rel * lam_hat * mask_in

        _, vjp_params = jax.vjp(
            lambda He, c, m, em: _helmholtz_apply_mpas(
                eta_sol, He, c, mesh, m, em,
            ),
            H_e_in, coeff_in, mask_in, edge_mask_in,
        )
        dH_e, dcoeff, dmask, dedge_mask = vjp_params(lam)

        return (
            lam,                          # rhs
            jnp.zeros_like(lam),          # x0: exact zero (IFT, stock-cg)
            -dH_e, -dcoeff,               # operator parameters
            -dmask, -dedge_mask,          # masks
            jnp.zeros_like(inv_diag_in),  # preconditioner: zero (IFT)
            jnp.zeros_like(tol_in),       # solver knob: zero (IFT)
        )

    _cg_area_adjoint.defvjp(_cg_fwd, _cg_bwd)

    return _cg_area_adjoint(rhs, x0, H_e, coeff, mask, edge_mask,
                            inv_diag, jnp.asarray(tol))


def barotropic_implicit_mpas(
    state,
    mesh: VoronoiMesh,
    z_coord: OceanZStarCoordinate,
    config: MPASOceanConfig,
    dt: float,
    F_slow_eta=None,
    F_slow_u=None,
    halo_refresh=None,
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
    value ``rel_residual`` is appended — the relative Helmholtz residual
    diagnostic of the FINAL eta.  Single-rank (stock-CG branch): a
    rank-local ``jnp.sum`` (exact for one rank).  Distributed (the entry
    dispatch below, armed by ``initialize_voronoi_mpi``): an owned-cell-
    masked GLOBAL reduction (one batched allreduce; halo rows excluded so
    Voronoi ghost cells are not double-counted).  Log / assert it OUTSIDE
    the JIT; never branch the compiled step on it.
    """
    # Distributed dispatch at ENTRY (resolves TODO(distributed-mpas-pcg)):
    # when ``initialize_voronoi_mpi`` has armed a partition layout, the
    # solve runs the shared fixed-M PCG with (a) a cell-halo exchange
    # composed into every A_op application — the local TRiSK stencil
    # then sees fresh ghost-ring values each iteration — and (b)
    # owned-cell-masked area-weighted dots (``dot_weight``), so the
    # global reductions never double-count halo cells.  A multi-rank
    # launch WITHOUT the layout stays a loud refusal: the stock-CG
    # fallback would silently run rank-local (codex CRITICAL,
    # 2026-06-11 — ``is_distributed()`` alone never fires here, hence
    # the mpi4py world-size predicate).
    from legoesm.parallel.reductions import mpi_world_size as _world
    from legoesm.parallel.voronoi_mpi import get_matching_voronoi_layout
    # Mesh-matched accessor (codex MAJOR): a stale layout from another
    # mesh must not hijack this solve into the distributed branch.
    _vlayout = get_matching_voronoi_layout(mesh)
    # SPMD (shard_map) lane: no MPI layout, but the refresh object carries
    # the owned mask + psum reducer + cell exchange (voronoi_spmd_ocean).
    _hr_owned = getattr(halo_refresh, "owned_mask_cells", None)
    _dist = _vlayout is not None or _hr_owned is not None
    if _vlayout is None and _hr_owned is None and _world() > 1:
        raise NotImplementedError(
            "MPAS barotropic_solver='implicit_cn' under MPI requires the "
            "Voronoi partition layout (call initialize_voronoi_mpi and "
            "build the model on layout.local_mesh); without it the "
            "stock-CG solve and its mass projection would silently run "
            "rank-local.  Use 'explicit_substep' otherwise."
        )
    # Preconditioner selection: validated on the static config string here,
    # after the multi-rank refusal above so that guard keeps firing first.
    _pcg_precond = str(config.barotropic_implicit_pcg_precond)
    if _pcg_precond not in ("jacobi", "poly", "gpoly"):
        raise ValueError(
            f"Unknown barotropic PCG preconditioner variant {_pcg_precond!r}: "
            "config.barotropic_implicit_pcg_precond must be one of "
            "'jacobi', 'poly' or 'gpoly'"
        )
    if int(config.barotropic_implicit_pcg_poly_sweeps) < 1:
        raise ValueError(
            "config.barotropic_implicit_pcg_poly_sweeps must be >= 1, got "
            f"{int(config.barotropic_implicit_pcg_poly_sweeps)}"
        )
    _gpoly = _pcg_precond == "gpoly"
    if _gpoly and _vlayout is not None:
        raise ValueError(
            "barotropic_implicit_pcg_precond='gpoly' is implemented on the "
            "SPMD (shard_map) lane only (a single device keeps the stock "
            "CG solve to tolerance); the MPI Voronoi layout cannot run it — "
            "select 'poly' there (with barotropic_implicit_pcg_fixed_iters=20, "
            "the count poly was validated at).")
    if _gpoly and _hr_owned is not None:
        _hd = getattr(halo_refresh, "halo_depth", None)
        _k = int(config.barotropic_implicit_pcg_poly_sweeps)
        # K-1 one-ring sweeps must leave z exact on owned + ring 1; the
        # layout holds depth + 2 rings (see halo_depth_for_config).
        if _hd is None or int(_hd) < _k - 2:
            raise ValueError(
                f"barotropic_implicit_pcg_precond='gpoly' with {_k} sweeps needs "
                f"a layout halo_depth of >= {_k - 2}; the SPMD layout has {_hd}. "
                "Build it with halo_depth=halo_depth_for_config(config).")
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
    # [stage-halo I0] u_bar_old was depth-averaged from the post-Coriolis
    # u (its ring consumed 2 tangential hops) and F_slow_u's ring carries
    # the neighbor rank's masked-wrong tendency depth-means; the Heun
    # predictor consumes 1-2 more tangential hops of both.  One packed
    # edge message re-arms them (solver-internal matvecs already exchange
    # per iteration — audit table stage 4').
    # [stage-halo I1] the step-3 flux divergence of the eta gradient needs a
    # refreshed gradient ring (stage audit).  grad_eta_old does not depend on
    # the I0-refreshed fields, so it rides in the same message; the predictor
    # keeps the unrefreshed copy.
    eta_filled_old = fill_land_cells_mpas(eta_old, mask, c1, c2,
                                          mesh.edgesOnCell, mesh.nEdgesOnCell)
    grad_eta_old = gradient_edge(eta_filled_old, mesh).astype(eta_dtype)
    grad_eta_for_div = grad_eta_old
    if halo_refresh is not None:
        u_bar_old, F_slow_u, grad_eta_for_div = halo_refresh.edges(
            u_bar_old, F_slow_u, grad_eta_old)
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

    flux_eta_old = H_e_old * grad_eta_for_div * edge_mask
    div_grad_eta_old = divergence_cell(flux_eta_old, mesh) * mask
    div_grad_eta_old = div_grad_eta_old.astype(eta_dtype)

    rhs = (
        eta_old
        - dt_t * div_HU_pred
        - coeff * div_grad_eta_old
        + dt_t * F_slow_eta * mask
    ) * mask

    # ----- Step 4: PCG solve --------------------------------------------
    # Two dispatch legs, selected at ENTRY (see the distributed-dispatch
    # block at the top of this function, which resolved the historical
    # TODO(distributed-mpas-pcg)):
    #
    #   - single-rank / no layout: the stock-CG custom-VJP solver below;
    #   - Voronoi partition layout armed: the shared distributed fixed-M
    #     PCG (``_vlayout is not None`` branch) with a cell-halo exchange
    #     composed into every ``A_op`` application and owned-cell-masked
    #     area-weighted dots — the two pieces (halo-in-matvec,
    #     owned-cell reductions) a multi-rank iterated solve needs.
    #
    # MERGE COMPOSITION (PR #394 × MPI-scaling refactor): the single-rank
    # solve routes through :func:`solve_helmholtz_freesurface_mpas` —
    # forward = the stock preconditioned CG VERBATIM (``jax.custom_vjp``
    # inlines the primal, bit-identical), reverse mode = the TRUE
    # area-weighted transpose (stock cg's symmetry-reusing VJP biased the
    # free-surface gradients; commit-1e370679 sibling).  ``A_op`` is
    # still built here for the post-solve residual diagnostic; the
    # custom-VJP solver rebuilds the identical operator internally so its
    # adjoint can take exact parameter cotangents.
    A_op = _make_helmholtz(H_e_old, coeff, mesh, mask, edge_mask)
    # Jacobi diagonal (fed to the custom-VJP solver as an explicit
    # argument — closure-captured tracers fail at scan lowering).
    inv_diag = _helmholtz_inv_diag_mpas(H_e_old, coeff, mesh, mask, edge_mask)

    if _dist:
        # ---- Distributed fixed-M PCG (shared solver) ----------------
        # The local TRiSK A_op is correct on OWNED cells provided its
        # input carries fresh ghost values — compose one cell-halo
        # exchange per application (one message round per PCG
        # iteration, static collective schedule) — or, for
        # "single_reduce_deep", exchange (r, s) once every certified-ring
        # count of iterations and recompute the halo.  Dots are owned-
        # masked AND area-weighted: owned-masking removes the halo
        # double-count in the allreduce; the area weight is the inner
        # product in which this FV Helmholtz is self-adjoint (required
        # by the single_reduce recurrences, harmless for standard).
        # AD: this path is differentiated straight THROUGH the unrolled
        # fixed-M loop (sendrecv-VJP + allreduce-SUM dots), so the
        # weighted-transpose concern the single-rank custom-VJP solver
        # addresses does not arise here — the unrolled adjoint is exact
        # by construction.
        from legoesm.ocean.dynamics.barotropic_common import (
            precision_aware_rel_tol,
            solve_helmholtz_implicit,
        )
        if _vlayout is not None:
            from legoesm.parallel.halo_exchange_voronoi import (
                VoronoiHaloExchange,
            )
            _exchanger = VoronoiHaloExchange(_vlayout.partition, backend="mpi")
            _exchange_cells = _exchanger.exchange_cell_field
            _owned = _vlayout.owned_mask_cells.astype(eta_dtype)
        else:
            def _exchange_cells(f):
                return halo_refresh.cells(f)[0]
            _owned = _hr_owned.astype(eta_dtype)

        def A_op_dist(eta_in: jnp.ndarray) -> jnp.ndarray:
            return A_op(_exchange_cells(eta_in))

        _pcg_variant = str(config.barotropic_implicit_pcg_variant)
        _deep_halo = None
        if _pcg_variant == "single_reduce_deep":
            # jacobi implies not gpoly, so this branch is the distributed one
            if _pcg_precond != "jacobi":
                raise NotImplementedError(
                    "barotropic_implicit_pcg_variant='single_reduce_deep' needs "
                    "barotropic_implicit_pcg_precond='jacobi' (got "
                    f"{_pcg_precond!r}): the preconditioner must be pointwise; "
                    "select pcg_variant='single_reduce' for the polynomial ones.")
            if _vlayout is not None:
                _rings = int(_vlayout.complete_cell_rings)
                _exch_many = (halo_refresh.cells if halo_refresh is not None
                              else lambda *fs: tuple(_exchange_cells(f) for f in fs))
            else:
                _rings = int(halo_refresh.complete_cell_rings)
                _exch_many = halo_refresh.cells
            if _rings < 1:
                raise NotImplementedError(
                    "barotropic_implicit_pcg_variant='single_reduce_deep' needs a "
                    f"layout certifying >= 1 complete cell ring (got {_rings}).")
            _deep_halo = (_exch_many, _owned, _rings)

        _x0_solve = eta_old
        if _gpoly:
            # GLOBAL Neumann polynomial (no device-boundary coupling
            # dropped), evaluated redundantly on the halo. The one-ring
            # Helmholtz erodes one valid ring per application, so with r
            # fresh on the layout's depth+2 rings the K-1 sweeps leave z
            # valid on owned + ring 1 when depth >= K-2; p = z + beta p
            # stays valid there and A p is exact on owned cells, which is
            # all the owned-weighted dots and x/r updates read. One
            # exchange per iteration (of r, inside M_inv) replaces the
            # exchange of p inside A_op. Coefficients and the warm start
            # are refreshed once so the halo rings carry OWNER values (the
            # outer ring's local diagonal and edges see missing neighbours).
            (_H_g, _em_g), (_mask_g, _inv_g, _x0_solve) = halo_refresh.both(
                (H_e_old, edge_mask), (mask, inv_diag, eta_old))
            _A_g = _make_helmholtz(_H_g, coeff, mesh, _mask_g, _em_g)
            _gsweeps = int(config.barotropic_implicit_pcg_poly_sweeps)
            _gw = 2.0 / 3.0

            def A_op_dist(eta_in: jnp.ndarray) -> jnp.ndarray:  # noqa: F811
                return _A_g(eta_in)

            def _M_inv_dist(r: jnp.ndarray) -> jnp.ndarray:
                r = _exchange_cells(r)
                z = _gw * _inv_g * r
                for _ in range(_gsweeps - 1):
                    z = z + _gw * _inv_g * (r - _A_g(z))
                return z
        elif _pcg_precond == "jacobi":
            def _M_inv_dist(r: jnp.ndarray) -> jnp.ndarray:
                return r * inv_diag
        else:
            _poly_sweeps = int(config.barotropic_implicit_pcg_poly_sweeps)
            _neumann_w = 2.0 / 3.0

            def _M_inv_dist(r: jnp.ndarray) -> jnp.ndarray:
                # SPD polynomial in the device-local block of A: TRUE
                # diagonal (inv_diag), off-diagonals restricted to
                # owned-owned couplings by zeroing the halo BEFORE the
                # operator.  Applies A_op, never A_op_dist, so no halo
                # exchange is composed in; pure jnp, reverse mode goes
                # straight through.
                z = _neumann_w * inv_diag * r * _owned
                for _ in range(_poly_sweeps - 1):
                    r_local = A_op(z) * _owned
                    z = z + _neumann_w * inv_diag * (r - r_local) * _owned
                return z

        _w_dots = _owned * mesh.areaCell.astype(eta_dtype) * mask
        eta_new, _solve_diag = solve_helmholtz_implicit(
            # deep halo: the LOCAL operator; the solver does the exchanges
            A_op if _deep_halo is not None else A_op_dist,
            rhs, _M_inv_dist, _x0_solve,
            distributed=True,
            fixed_iters=int(config.barotropic_implicit_pcg_fixed_iters),
            # f32-safe acceptance tolerance (f64 unchanged); the fixed-iter
            # PCG runs a static count, so this only floors the converged
            # diagnostic.
            residual_tol=precision_aware_rel_tol(
                config.barotropic_implicit_pcg_residual_tol, eta_dtype,
            ),
            stock_cg_tol=config.barotropic_implicit_pcg_tol,
            stock_cg_maxiter=int(config.barotropic_implicit_pcg_maxiter),
            pcg_variant=_pcg_variant,
            dot_weight=_w_dots,
            deep_halo=_deep_halo,
        )
        # Refresh the halo ring of the solution before downstream
        # stencils consume it.
        eta_new = _exchange_cells(eta_new) * mask
    else:
        # f32: floor the 1e-10 rel-tol to the f32-reachable value so stock CG
        # stops at convergence rather than maxiter (f64 unchanged).
        from legoesm.ocean.dynamics.barotropic_common import (
            precision_aware_rel_tol as _precision_aware_rel_tol,
        )
        pcg_tol = _precision_aware_rel_tol(
            config.barotropic_implicit_pcg_tol, eta_dtype,
        )
        pcg_maxiter = int(config.barotropic_implicit_pcg_maxiter)
        # Forward = stock preconditioned CG, bit-identical; reverse mode
        # uses the TRUE (area-weighted) transpose — stock cg's
        # symmetry-reusing VJP biased free-surface gradients
        # (commit-1e370679 sibling).
        eta_new = solve_helmholtz_freesurface_mpas(
            rhs, eta_old, H_e_old, coeff, mask, edge_mask, inv_diag, mesh,
            tol=pcg_tol, maxiter=pcg_maxiter,
        )
        eta_new = eta_new * mask
    # Mass projection: area-weighted sums.  Single-rank: plain local
    # sums (exact).  Distributed: OWNED-masked partial sums + ONE
    # batched global allreduce (a bare allreduce of unmasked local sums
    # would double-count halo cells).  Accumulate in
    # ``ocean_diagnostics`` precision so the huge-area ``target -
    # actual`` cancellation survives in f32.
    from legoesm.core.precision import cast as _cast
    _M = "ocean_diagnostics"
    _area_cell = mesh.areaCell.astype(eta_dtype)
    _area_acc = _cast(_area_cell, _M, "accumulate")
    _wa = _area_acc * _cast(mask, _M, "accumulate")
    if _dist:
        _owned_acc = _cast(_owned, _M, "accumulate")
        _wa = _wa * _owned_acc
        _area_proj = _area_acc * _owned_acc
    else:
        _area_proj = _area_acc
    _ocean_area_l = jnp.sum(_wa)
    _target_mass_l = jnp.sum(_cast(rhs, _M, "accumulate") * _area_proj)
    _actual_mass_l = jnp.sum(_cast(eta_new, _M, "accumulate") * _area_proj)
    if _dist:
        _gsum = getattr(halo_refresh, "global_sum", None)
        if _gsum is None:
            from legoesm.parallel.reductions import batch_allreduce_mpi
            _gsum = batch_allreduce_mpi
        _ocean_area, _target_mass, _actual_mass = _gsum(
            [_ocean_area_l, _target_mass_l, _actual_mass_l],
        )
    else:
        _ocean_area = _ocean_area_l
        _target_mass = _target_mass_l
        _actual_mass = _actual_mass_l
    _correction = (_target_mass - _actual_mass) / jnp.maximum(_ocean_area, 1e-30)
    eta_new = (eta_new + _correction.astype(eta_dtype) * mask) * mask
    # Residual diagnostic (uniform return shape with the lat-lon solver's
    # ``return_residual``), computed AFTER the floor clamp below so it
    # reflects the ACTUAL returned eta.  Single-rank: plain ``jnp.sum``
    # (exact for one rank).  Distributed: owned-masked sums + ONE batched
    # allreduce — never the lat-lon helper's ``_global_dot_batch``, whose
    # bare reduction would double-count Voronoi halo cells.

    # Mass-conserving floor clamp (safety net for extreme transients;
    # in normal operation this is a no-op since the PCG converges to
    # well-resolved η).
    # ``eta_floor_clamp_iters`` was declared on the config but never read
    # here (the explicit-substep path honours it); the bench's
    # --eta-clamp-iters knob was inert on this solver.  Default 3 is the
    # value that was hard-wired, so nothing changes unless it is set.
    _clamp_iters = int(config.eta_floor_clamp_iters)
    if _dist:
        eta_new = _clamp_redistribute(
            eta_new, eta_floor, mask, mesh.areaCell, _clamp_iters,
            owned_weight=_owned, force_global=True,
        )
    else:
        eta_new = _clamp_redistribute(
            eta_new, eta_floor, mask, mesh.areaCell, _clamp_iters,
        )

    # Residual diagnostic of the FINAL eta (post projection + clamp).
    # Rank-local (single-rank path); ``stop_gradient`` keeps it out of
    # reverse mode.
    _res_vec = rhs - A_op(eta_new)
    if _dist:
        # Owned-masked global residual (eta_new's halo ring was
        # refreshed above, so the local A_op is exact on owned cells;
        # halo rows are excluded from the sums and the two squared
        # norms ride one batched allreduce).
        _rr_l = jnp.sum(_owned * _res_vec**2)
        _bb_l = jnp.sum(_owned * rhs**2)
        _rr, _bb = _gsum([_rr_l, _bb_l])
    else:
        _rr = jnp.sum(_res_vec**2)
        _bb = jnp.sum(rhs**2)
    _solve_diag_rel = jax.lax.stop_gradient(
        jnp.sqrt(_rr / jnp.maximum(_bb, jnp.asarray(1.0e-30, dtype=eta_dtype)))
    )

    # ----- Step 5: corrector for u_bar using new η gradient delta ------
    eta_filled_new = fill_land_cells_mpas(eta_new, mask, c1, c2,
                                          mesh.edgesOnCell, mesh.nEdgesOnCell)
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
    # recommendation (docs/dev-notes/issues/barotropic_mode_noise.md §"Residual").
    A_baro_visc = jnp.asarray(config.barotropic_u_viscosity, dtype=eta_dtype)
    # Per-edge equatorial-boost factor — SAME Gaussian mechanism as the 3D A_h
    # path (ocean_pe_mpas), guarded identically on the STATIC config float so
    # boost<=0 keeps _lat_factor==1 (no boost) and never evaluates the Gaussian
    # (σ=0 would give 0*NaN at an exact-equator edge).  A tight Gaussian in
    # latitude (σ = equatorial_visc_sigma_deg), NOT the old cos²(lat) which
    # overdamped real mid-latitude flow.  Damps the equatorial f→0 u_baro mode
    # the implicit-CN Coriolis predictor-corrector cannot catch.  See
    # project_mpas_etopo_instability.md §"equatorial mode".
    _eq_boost = config.equatorial_visc_boost
    if _eq_boost > 0:
        if config.equatorial_visc_sigma_deg <= 0:
            raise ValueError(
                "equatorial_visc_sigma_deg must be > 0 when "
                "equatorial_visc_boost > 0 (Gaussian width divides latEdge; "
                f"got {config.equatorial_visc_sigma_deg})."
            )
        _sigma_rad = jnp.radians(config.equatorial_visc_sigma_deg)
        _gauss = jnp.exp(
            -0.5 * (mesh.latEdge.astype(eta_dtype) / _sigma_rad) ** 2
        )
        _lat_factor = 1.0 + _eq_boost * _gauss  # (nEdges,)
    else:
        _lat_factor = 1.0
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
    K_baro_bih = jnp.asarray(config.barotropic_u_biharmonic, dtype=eta_dtype)
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
