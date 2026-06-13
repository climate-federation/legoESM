"""Implicit Crank-Nicolson barotropic free-surface solver (lat-lon C-grid).

Replaces the explicit substep loop with a single-step implicit treatment
of the depth-integrated free-surface gravity wave equations.  Eliminates
the chequerboard mode by construction: the implicit Helmholtz operator
has a well-defined, small null space, so the C-grid Coriolis null-mode
checkerboard cannot accumulate (issue: barotropic_mode_noise).

Equations (depth-averaged, z-star reference column held constant over
the timestep):

    ∂η/∂t = -∇·(H · U_bar)             [continuity]
    ∂U_bar/∂t = -g·∇η + f·V_bar + R_u  [zonal momentum]
    ∂V_bar/∂t = -g·∇η - f·U_bar + R_v  [meridional momentum]

Scheme — Crank-Nicolson with Forward-Backward Coriolis predictor:

    Predictor (FB Coriolis, OLD η gradient):
        U* = U^n + dt·[-g·∇η^n + f·V^n + F_slow_u]
        V* = V^n + dt·[-g·∇η^n - f·U* + F_slow_v]   (FB: uses NEW U*)

    Time-averaged predicted transport (continuity weight θ_eta):
        U_avg* = (1-θ_eta)·U^n + θ_eta·U*
        V_avg* = (1-θ_eta)·V^n + θ_eta·V*

    Elliptic Helmholtz solve for η^{n+1} (PGF weight θ_pgf):
        [I - θ_eta·θ_pgf·dt²·g·∇·(H·∇)] η^{n+1}
            = η^n  -  dt·∇·(H · U_avg*)
                   -  θ_eta·θ_pgf·dt²·g·∇·(H·∇η^n)
                   +  dt·F_slow_eta

    Corrector (apply gradient delta from PGF):
        δη   = η^{n+1} - η^n
        U^{n+1} = U* - θ_pgf·dt·g·∂_x δη
        V^{n+1} = V* - θ_pgf·dt·g·∂_y δη

θ_eta = θ_pgf = 0.55 by default (slightly past Crank-Nicolson — gives
unconditional gravity-wave damping while staying close to 2nd-order
accurate; standard choice in MITgcm and MPAS-O).

Mass conservation: per-step mass change equals
    Σ (η^{n+1} - η^n)·area = dt · Σ F_slow_eta · area · mask
exactly, because the Helmholtz operator is constructed from the
mass-conserving FV divergence and the gradient is the FV adjoint.

Differentiability: ``jax.scipy.sparse.linalg.cg`` is differentiable
through implicit-function-theorem custom-VJP, but its transpose solve
re-uses the SAME operator, assuming Euclidean symmetry.  The FV Helmholtz
operator here is self-adjoint only in the AREA-WEIGHTED inner product
(measured Euclidean asymmetry up to ~0.16 at production resolution;
area-weighted ~3e-15), so the stock VJP silently biased every
reverse-mode gradient through the free-surface solve (per-solve AD/FD
0.991 at 18x36; forward always correct).  Fixed backward-only via
``jax.custom_vjp`` in :func:`solve_helmholtz_freesurface`: the primal CG
call is verbatim (forward bit-identical) and the adjoint applies the TRUE
transpose ``A^{-T} = W.A^{-1}.W^{-1}`` with ``W = diag(area)`` — the
sibling of the rigid-lid seam-reduced adjoint (commit 1e370679).

Tracer transport consistency: returns ``(Hu_avg, Hv_avg)`` =
``H_old · [(1-θ)·U^n + θ·U^{n+1}]`` — time-averaged face transport using
the OLD face thickness, ensuring ``div(Hu_avg) = (η_old − η_new)/dt``
exactly (required for flux-form tracer conservation on partial cells).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.grids.latlon import LatLonGrid
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_layer_thickness
from legoesm.ocean.state import LatLonCGridOceanState, LatLonCGridOceanConfig
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    fold_is_local,
    divergence_cgrid,
    fold_vface_row,
    gradient_x_cgrid,
    gradient_y_cgrid,
    is_tripolar,
    pad_ns_zero,
)
from legoesm.grids.halo_latlon import zero_polar_lat_ends as _zero_polar_lat_ends
from legoesm.ocean.dynamics.eta_floor import (
    clamp_and_redistribute as _clamp_redistribute,
)


def _depth_average_to_faces(
    u_3d: jnp.ndarray,
    v_3d: jnp.ndarray,
    h_k: jnp.ndarray,
    min_water_col: jnp.ndarray,
    u_mask: jnp.ndarray,
    v_mask: jnp.ndarray,
    grid=None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Depth-average 3D velocity to C-grid face points (thickness-weighted).

    Partial-cells aware: at faces between cells with different
    ``bottom_level`` (one column has water at level k, the other
    doesn't), the face thickness contribution must be zero — naively
    averaging ``0.5 * (h_active + 0)`` would otherwise double-count
    depth and skew the depth-integrated transport.  We multiply by a
    3D face-active mask derived from ``h_k > 0``.  For legacy z\\*
    columns where every cell has ``h_k > 0`` (full cells everywhere),
    the mask is 1 and the result is bit-exact unchanged.
    """
    # Use ``min(h_left, h_right)`` instead of the arithmetic mean.  This
    # is the physically correct face thickness — the partial cell is
    # the constraint on flux capacity at the face.  Identical to the
    # arithmetic mean when adjacent cells have the same thickness
    # (flat bottom, full cells on both sides) → backwards-compat
    # bit-exact for the dominant legacy use case.  Naturally zero at
    # active-vs-inactive partial-cell faces (one side has h=0).
    h_u_inner = jnp.minimum(jnp.roll(h_k, 1, axis=1), h_k)
    h_u = jnp.concatenate([h_u_inner, h_u_inner[:, 0:1, :]], axis=1)
    H_u = jnp.maximum(jnp.sum(h_u, axis=-1), min_water_col)
    U_bar = jnp.sum(u_3d * h_u, axis=-1) / H_u * u_mask

    # Cell-pad-first (PR357 Bug-2 pattern): pad the cell thickness so the
    # v-face min at a partition cut uses the neighbour rank's adjacent
    # column (MPI halo).  On tripolar grids the fold face connects cell
    # (fold_j, i) with its fold partner (fold_j, perm_T[i]); the seam is
    # overwritten only on the rank that owns it.
    h_k_pad = pad_ns_zero(h_k)
    h_v = jnp.minimum(h_k_pad[:-1], h_k_pad[1:])
    h_v = _zero_polar_lat_ends(h_v)
    if fold_is_local(grid):
        north_row = jnp.minimum(h_k[-1:], fold_vface_row(h_k, grid))
        h_v = jnp.concatenate([h_v[:-1], north_row], axis=0)
    _v_pair = jnp.sum(jnp.stack([h_v, v_3d * h_v], axis=-1), axis=-2)
    H_v = jnp.maximum(_v_pair[..., 0], min_water_col)
    V_bar = _v_pair[..., 1] / H_v * v_mask

    return U_bar, V_bar


def _h_total_at_faces(
    h_k: jnp.ndarray,
    min_water_col: jnp.ndarray,
    mask: jnp.ndarray,
    grid=None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Total water-column thickness ``H = sum_k(h_k)`` at u/v faces.

    Uses ``min(H_left, H_right)`` rather than the arithmetic mean —
    the gravity-wave at the face propagates through the shared wet
    depth (the shallower of two columns), so this is the physically
    correct face H for the Helmholtz operator.  When adjacent cells
    have the same H_bathy (flat bottom), min == average → bit-exact
    backwards-compat preserved.

    The cell-center total is computed as ``sum_k(h_k)``, NOT as
    ``eta + H_bathy``.  This is mathematically equivalent but bit-
    consistent with the per-level mass flux pipeline that uses
    ``min_cell_to_uface(h_k)`` and ``H_u_old = sum_k(h_u_old)``.
    Using ``eta + H_bathy`` here would drift by O(1e-4 m) from
    ``sum_k(h_k)`` due to the float-precision residue in
    ``create_ocean_z_star``'s ``dz_ref`` construction, breaking the
    Hallberg-Adcroft 2009 column-sum identity.

    Multiplied by the *cell* mask so that dry-face transport is zero.
    """
    H_total = jnp.maximum(jnp.sum(h_k, axis=-1), min_water_col) * mask

    H_u_inner = jnp.minimum(jnp.roll(H_total, 1, axis=1), H_total)
    H_u = jnp.concatenate([H_u_inner, H_u_inner[:, 0:1]], axis=1)

    # Cell-pad-first (PR357 Bug-2 pattern): pad the cell column total so the
    # v-face min at a partition cut uses the neighbour rank's adjacent
    # column (MPI halo); the fold seam is overwritten only on the rank that
    # owns it.
    H_total_pad = pad_ns_zero(H_total)
    H_v = jnp.minimum(H_total_pad[:-1], H_total_pad[1:])
    H_v = _zero_polar_lat_ends(H_v)
    if fold_is_local(grid):
        north_row = jnp.minimum(H_total[-1:], fold_vface_row(H_total, grid))
        H_v = jnp.concatenate([H_v[:-1], north_row], axis=0)

    return H_u, H_v


def _helmholtz_apply(
    eta_in: jnp.ndarray,
    H_u: jnp.ndarray,
    H_v: jnp.ndarray,
    coeff: jnp.ndarray,
    grid: LatLonGrid,
    mask: jnp.ndarray,
    u_mask: jnp.ndarray,
    v_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Apply ``A(η) = η - coeff · ∇·(H·∇η)`` (cell-centred Helmholtz).

    Parametric form of the operator so the custom VJP in
    :func:`solve_helmholtz_freesurface` can take exact cotangents w.r.t.
    the operator parameters (``θ̄ = -(∂_θ A(θ)·x)ᵀ·λ``) via ``jax.vjp``
    without closure-capturing tracers (scan-lowering safe; see the
    rigid-lid precedent's CLOSURE RULE, commit 1e370679).
    """
    H_u_face = H_u * u_mask
    H_v_face = H_v * v_mask
    _dt = eta_in.dtype
    eta_m = eta_in * mask
    grad_x = gradient_x_cgrid(eta_m, grid).astype(_dt)
    grad_y = gradient_y_cgrid(eta_m, grid).astype(_dt)
    flux_x = H_u_face * grad_x
    flux_y = H_v_face * grad_y
    div_grad = divergence_cgrid(
        flux_x, flux_y, grid, u_mask=u_mask, v_mask=v_mask,
    ).astype(_dt)
    return (eta_m - coeff * div_grad) * mask


def _make_helmholtz(
    H_u: jnp.ndarray,
    H_v: jnp.ndarray,
    coeff: jnp.ndarray,
    grid: LatLonGrid,
    mask: jnp.ndarray,
    u_mask: jnp.ndarray,
    v_mask: jnp.ndarray,
):
    """Return ``A_op(η) = η - coeff · ∇·(H·∇η)`` (cell-centred Helmholtz).

    Built from FV gradient + FV divergence, so the operator is symmetric
    positive-definite (for ``coeff ≥ 0`` and ``H ≥ 0``) — but ONLY in the
    area-weighted inner product ``<x,y> = Σ x·y·area`` (the divergence
    carries 1/area); as a Euclidean matrix it is NOT symmetric.  Stock
    ``jax.scipy`` cg forward-solves it fine; its symmetry-reusing VJP does
    not (see :func:`solve_helmholtz_freesurface`).
    """

    def A_op(eta_in: jnp.ndarray) -> jnp.ndarray:
        return _helmholtz_apply(
            eta_in, H_u, H_v, coeff, grid, mask, u_mask, v_mask,
        )

    return A_op


def _helmholtz_inv_diag(
    H_u: jnp.ndarray,
    H_v: jnp.ndarray,
    coeff: jnp.ndarray,
    grid: LatLonGrid,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Inverse diagonal of the Helmholtz operator (Jacobi preconditioner).

    The diagonal of ``A_op`` at cell (j,i):
        A_diag(j,i) = 1 + coeff · [
              H_u(j,i)/dx_u(j)² + H_u(j,i+1)/dx_u(j)²
            + H_v(j,i)·dx_v(j)/(dy²·A_cell) + H_v(j+1,i)·dx_v(j+1)/(dy²·A_cell)
        ] · (correctly weighted by face lengths and 1/A)

    Specifically, expanding the FV ∇·(H·∇η) at cell (j,i):
        ∇·(H·∇η)(j,i) = (1/A_cell(j,i)) · {
            H_u(j,i+1)·dy_u·(η(j,i+1)-η(j,i))/dx_u(j)
          - H_u(j,i)  ·dy_u·(η(j,i)  -η(j,i-1))/dx_u(j)
          + H_v(j+1,i)·dx_v(j+1)·(η(j+1,i)-η(j,i))/dy_v
          - H_v(j,i)  ·dx_v(j)  ·(η(j,i)  -η(j-1,i))/dy_v
        }
    The diagonal coefficient on η(j,i) is therefore:
        - (H_u(j,i+1) + H_u(j,i)) · dy_u / (dx_u(j) · A_cell(j,i))
        - (H_v(j+1,i)·dx_v(j+1) + H_v(j,i)·dx_v(j)) / (dy_v · A_cell(j,i))
    """
    area = grid.area                                     # (n_lat, n_lon)
    inv_area = 1.0 / area

    H_u_E = H_u[:, 1:]   # east face of cell j: u-face (j, i+1)
    H_u_W = H_u[:, :-1]  # west face of cell j: u-face (j, i)
    H_v_N = H_v[1:, :]   # north face of cell j: v-face (j+1, i)
    H_v_S = H_v[:-1, :]  # south face of cell j: v-face (j, i)

    if is_tripolar(grid):
        # Tripolar: use full per-face 2D metrics so the preconditioner
        # captures the longitude variation of cell sizes in the bipolar
        # cap.  Taking only column 0 (as the previous version did) gives
        # an unrepresentative preconditioner that makes PCG diverge.
        dy_u_E = grid.dy_u[:, 1:]    # (n_lat, n_lon)
        dy_u_W = grid.dy_u[:, :-1]
        dx_u_E = grid.dx_u[:, 1:]
        dx_u_W = grid.dx_u[:, :-1]
        dx_v_N = grid.dx_v[1:, :]    # (n_lat, n_lon)
        dx_v_S = grid.dx_v[:-1, :]
        dy_v_N = grid.dy_v[1:, :]
        dy_v_S = grid.dy_v[:-1, :]

        diag_zonal = (
            H_u_E * dy_u_E / jnp.maximum(dx_u_E, 1.0e-30)
            + H_u_W * dy_u_W / jnp.maximum(dx_u_W, 1.0e-30)
        ) * inv_area
        diag_merid = (
            H_v_N * dx_v_N / jnp.maximum(dy_v_N, 1.0e-30)
            + H_v_S * dx_v_S / jnp.maximum(dy_v_S, 1.0e-30)
        ) * inv_area
    else:
        # Regular lat-lon or Mercator: variable-dy safe.
        R = grid.radius
        dlon = grid.dlon
        cos_lat_c = grid.cos_lat
        dx_u = R * dlon * cos_lat_c                      # (n_lat,)

        # u-face meridional extent: cell row j's height.
        dy_h = grid.dy * 0.5                             # (n_lat,)
        dy_u = dy_h                                      # alias

        # v-face cell-centre-to-cell-centre distance: depends on row pair.
        dy_v_int = 0.5 * (dy_h[1:] + dy_h[:-1])          # (n_lat-1,)
        dy_v_face = jnp.pad(dy_v_int, (1, 1), mode='edge')  # (n_lat+1,)

        lat = grid.lat
        lat_v_int = 0.5 * (lat[:-1] + lat[1:])
        lat_v = jnp.concatenate([
            jnp.array([-jnp.pi / 2], dtype=lat.dtype),
            lat_v_int,
            jnp.array([jnp.pi / 2], dtype=lat.dtype),
        ])
        cos_lat_v = jnp.cos(lat_v)                      # (n_lat+1,)
        dx_v = R * cos_lat_v * dlon                      # (n_lat+1,)

        diag_zonal = (
            (H_u_E + H_u_W) * dy_u[:, None] / dx_u[:, None]
        ) * inv_area
        diag_merid = (
            H_v_N * dx_v[1:, None] / dy_v_face[1:, None]
            + H_v_S * dx_v[:-1, None] / dy_v_face[:-1, None]
        ) * inv_area

    # Diagonal of the Helmholtz operator A = I - coeff·∇·(H·∇):
    # diag(A) = 1 + coeff · (diag_zonal + diag_merid)
    diag = 1.0 + coeff * (diag_zonal + diag_merid)
    # On land cells, A_op forces output to zero (mask kills), so M⁻¹=0 there.
    return jnp.where(mask > 0.5, 1.0 / jnp.maximum(diag, 1.0e-30), 0.0)


def _make_diag_preconditioner(
    H_u: jnp.ndarray,
    H_v: jnp.ndarray,
    coeff: jnp.ndarray,
    grid: LatLonGrid,
    mask: jnp.ndarray,
):
    """Return ``M(r) = diag⁻¹ · r`` Jacobi preconditioner for the Helmholtz."""
    inv_diag = _helmholtz_inv_diag(H_u, H_v, coeff, grid, mask)

    def M_inv(r: jnp.ndarray) -> jnp.ndarray:
        return r * inv_diag.astype(r.dtype)

    return M_inv


def solve_helmholtz_freesurface(
    rhs: jnp.ndarray,
    x0: jnp.ndarray,
    H_u: jnp.ndarray,
    H_v: jnp.ndarray,
    coeff: jnp.ndarray,
    mask: jnp.ndarray,
    u_mask: jnp.ndarray,
    v_mask: jnp.ndarray,
    inv_diag: jnp.ndarray,
    grid: LatLonGrid,
    *,
    tol,
    maxiter: int,
) -> jnp.ndarray:
    """Preconditioned-CG Helmholtz solve with the area-weighted adjoint.

    Forward: byte-identical to the plain ``jax.scipy.sparse.linalg.cg``
    call on ``A = _make_helmholtz(...)`` with the Jacobi preconditioner
    ``r * inv_diag`` (the pre-fix path, verbatim — ``jax.custom_vjp``
    inlines the primal transparently; sha256 A/B verified over 6 full
    model steps on the variable-bathymetry implicit_cn config).

    Area-weighted adjoint (custom VJP)
    ----------------------------------
    ``A`` is built from the FV gradient and the FV divergence; the
    divergence carries ``1/area``, so ``A`` is self-adjoint only in the
    area-weighted inner product ``<x,y>_W = Σ x·y·area``:
    ``W·A = Aᵀ·W`` with ``W = diag(area)``, i.e. ``Aᵀ = W·A·W⁻¹`` —
    measured Euclidean relative asymmetry 3.9e-2 at 18x36 (up to ~0.16 at
    production resolution), area-weighted ~3e-15
    (tests/ocean/unit/test_freesurface_helmholtz_adjoint.py).

    ``jax.scipy.sparse.linalg.cg``'s VJP (``lax.custom_linear_solve`` with
    ``transpose_solve = solve``) re-solves with ``A`` itself, assuming
    Euclidean symmetry, so every reverse-mode gradient through the
    free-surface solve was biased — per-solve AD/FD 0.9909 at 18x36,
    eps-stable over 3 decades of FD step; dense 6x8 ground truth shows the
    stock rhs-cotangent off by 4.7e-3 elementwise (forward always
    correct).  Same bug class as the rigid-lid seam adjoint (commit
    1e370679), different W.

    Fix (backward-only): the adjoint solve applies the TRUE transpose

        λ = A⁻ᵀ·η̄ = W·A⁻¹·W⁻¹·η̄   →   λ = w̃ · cg(A, η̄·mask/w̃)·mask

    with ``w̃ = area/max(area)`` (the constant rescale cancels exactly by
    linearity; raw ~1e11 m² areas underflow the float32 CG residual
    norms), reusing the SAME preconditioned CG (A is what CG likes; only
    the rhs is conjugated by W).  The weighting side matters: putting
    ``w̃`` on the rhs and ``1/w̃`` on the solution (the classic slip) is
    wrong by ~1e-2 against the dense ground truth — pinned by the
    mutation test.
    Operator-parameter cotangents are exact IFT:
    ``θ̄ = -(∂_θ A(θ)·x)ᵀ·λ`` via ``jax.vjp`` of
    :func:`_helmholtz_apply` at the converged solution ``x``, for
    θ ∈ {H_u, H_v, coeff, mask, u_mask, v_mask}.  ``x0`` keeps the exact
    zero cotangent (stock-cg behaviour; the Krylov guess does not affect
    the converged solution — IFT).  ``inv_diag`` (preconditioner) gets a
    zero cotangent for the same reason.  Dense 6x8 end-to-end gradients
    (rhs, H_u, H_v, coeff) match the differentiable dense-solve ground
    truth to ~6e-14 (probe:
    .physics-validator/helmholtz_adjoint_review/probe1_dense_ground_truth.py).

    CLOSURE RULE (scan compatibility, from the rigid-lid precedent): the
    fwd/bwd closures capture only ``grid``/``maxiter`` (concrete /
    static).  Every traced array (rhs, x0, H, coeff, masks, inv_diag) AND
    ``tol`` is an explicit custom_vjp argument carried through residuals —
    a tracer captured in the bwd closure fails at scan lowering ("No
    constant handler for DynamicJaxprTracer").  ``tol`` is array-valued in
    the production caller (``jnp.asarray(config..., dtype=eta_dtype)``,
    created INSIDE the traced step body), so even though it is constant it
    leaks as a tracer across the custom_vjp boundary under
    ``grad``-of-``scan`` if closure-captured — pinned by
    test_grad_under_scan_two_steps_finite.

    Known limitation: ``custom_vjp`` does not support forward-mode AD, so
    ``jax.jvp``/``jacfwd`` through this solve raises (stock ``cg``
    supported it).  Reverse mode is the end-to-end design goal and no repo
    path uses forward-mode through the ocean step; if one ever does, add a
    paired ``custom_jvp`` solving the tangent system with the same
    operator (the tangent solve needs no transpose, so plain CG on ``A``
    is exact there).
    """

    def _forward_cg(rhs_in, x0_in, H_u_in, H_v_in, coeff_in,
                    mask_in, u_mask_in, v_mask_in, inv_diag_in, tol_in):
        # The pre-fix forward, verbatim.
        A_op = _make_helmholtz(
            H_u_in, H_v_in, coeff_in, grid, mask_in, u_mask_in, v_mask_in,
        )

        def M_inv(r):
            return r * inv_diag_in.astype(r.dtype)

        eta_sol, _info = jax.scipy.sparse.linalg.cg(
            A_op, rhs_in, x0=x0_in, tol=tol_in, maxiter=maxiter, M=M_inv,
        )
        return eta_sol

    @jax.custom_vjp
    def _cg_area_adjoint(rhs_in, x0_in, H_u_in, H_v_in, coeff_in,
                         mask_in, u_mask_in, v_mask_in, inv_diag_in, tol_in):
        return _forward_cg(rhs_in, x0_in, H_u_in, H_v_in, coeff_in,
                           mask_in, u_mask_in, v_mask_in, inv_diag_in,
                           tol_in)

    def _cg_fwd(rhs_in, x0_in, H_u_in, H_v_in, coeff_in,
                mask_in, u_mask_in, v_mask_in, inv_diag_in, tol_in):
        eta_sol = _forward_cg(rhs_in, x0_in, H_u_in, H_v_in, coeff_in,
                              mask_in, u_mask_in, v_mask_in, inv_diag_in,
                              tol_in)
        res = (eta_sol, H_u_in, H_v_in, coeff_in,
               mask_in, u_mask_in, v_mask_in, inv_diag_in, tol_in)
        return eta_sol, res

    def _cg_bwd(res, eta_bar):
        (eta_sol, H_u_in, H_v_in, coeff_in,
         mask_in, u_mask_in, v_mask_in, inv_diag_in, tol_in) = res
        A_op = _make_helmholtz(
            H_u_in, H_v_in, coeff_in, grid, mask_in, u_mask_in, v_mask_in,
        )

        def M_inv(r):
            return r * inv_diag_in.astype(r.dtype)

        # λ = A⁻ᵀ·η̄ = W·A⁻¹·W⁻¹·η̄ — the area conjugation is the fix
        # (re-solving with A directly, what stock cg's VJP does, IS the
        # bug).  Mask the cotangent: the land components of the output
        # are x0 pass-through with no dependence on rhs/θ.
        # W is normalized by max(area): a CONSTANT rescale cancels
        # exactly by linearity (c·w̃·A⁻¹(η̄/(c·w̃)) = w̃·A⁻¹(η̄/w̃);
        # verified bitwise to ~5e-16 in x64) but keeps the adjoint rhs at
        # O(η̄) instead of O(η̄/1e11).  Raw cell areas (~1e10-1e12 m²)
        # push the float32 CG residual norms toward subnormal — observed
        # as a NaN gradient while shipping this fix on the production
        # config; w̃ is the unconditionally-conditioned choice (the
        # specific NaN is config-dependent, but w̃ removes the failure
        # mode entirely at zero cost).
        area_w = grid.area.astype(eta_bar.dtype)
        w_rel = area_w / jnp.max(area_w)
        eta_bar_m = eta_bar * mask_in
        lam_hat, _info = jax.scipy.sparse.linalg.cg(
            A_op, eta_bar_m / w_rel, x0=jnp.zeros_like(eta_bar_m),
            tol=tol_in, maxiter=maxiter, M=M_inv,
        )
        lam = w_rel * lam_hat * mask_in

        # Exact operator-parameter cotangents: θ̄ = -(∂_θ A(θ)·x)ᵀ·λ at
        # the converged solution (IFT), in one jax.vjp of the parametric
        # apply.  Mask cotangents are propagated for completeness (land
        # topology is not a physical control, but correctness is free).
        _, vjp_params = jax.vjp(
            lambda Hu, Hv, c, m, um, vm: _helmholtz_apply(
                eta_sol, Hu, Hv, c, grid, m, um, vm,
            ),
            H_u_in, H_v_in, coeff_in, mask_in, u_mask_in, v_mask_in,
        )
        dH_u, dH_v, dcoeff, dmask, du_mask, dv_mask = vjp_params(lam)

        return (
            lam,                          # rhs
            jnp.zeros_like(lam),          # x0: exact zero (IFT, stock-cg)
            -dH_u, -dH_v, -dcoeff,        # operator parameters
            -dmask, -du_mask, -dv_mask,   # masks
            jnp.zeros_like(inv_diag_in),  # preconditioner: zero (IFT)
            jnp.zeros_like(tol_in),       # solver knob: zero (IFT)
        )

    _cg_area_adjoint.defvjp(_cg_fwd, _cg_bwd)

    return _cg_area_adjoint(rhs, x0, H_u, H_v, coeff,
                            mask, u_mask, v_mask, inv_diag,
                            jnp.asarray(tol))


def barotropic_implicit_latlon_cgrid(
    state: LatLonCGridOceanState,
    dt: float,
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    config: LatLonCGridOceanConfig,
    F_slow_eta: jnp.ndarray | None = None,
    F_slow_u: jnp.ndarray | None = None,
    F_slow_v: jnp.ndarray | None = None,
    *,
    return_residual: bool = False,
) -> tuple[LatLonCGridOceanState, tuple[jnp.ndarray, jnp.ndarray]]:
    """Single-step implicit free-surface solver (lat-lon C-grid).

    Returns
    -------
    (state_new, (Hu_avg, Hv_avg))
        Same return signature as ``barotropic_substeps_latlon_cgrid``.
        When ``return_residual=True`` (static; default ``False`` keeps
        the hot path unchanged) the returned tuple is instead
        ``(state_new, (Hu_avg, Hv_avg), rel_residual)`` where
        ``rel_residual`` is the GLOBAL relative Helmholtz residual
        ``sqrt(global(r·r)/global(rhs·rhs))`` — a DIAGNOSTIC for the
        fixed-iteration distributed PCG.  Log / assert it OUTSIDE the
        JIT; never branch the compiled step on it.

    Notes
    -----
    The barotropic timestep ``dt`` is the *full baroclinic* timestep —
    no substepping is needed because the gravity-wave terms are treated
    implicitly.  CFL on the explicit terms (Coriolis + slow forcing) is
    only ``f·dt`` and ``|U|·dt/dx``, both ≪ 1 at typical parameters.
    """
    g = jnp.asarray(config.g)
    H_bathy = state.H_bathy.data
    mask = state.land_mask.data
    u_mask = state.u_mask.data
    v_mask = state.v_mask.data
    eta_old = state.eta.data
    u_3d = state.u.data
    v_3d = state.v.data

    eta_dtype = eta_old.dtype
    # Cast all arrays to eta's dtype so that the CG while_loop and
    # downstream lax.scan carry types stay in a single precision.
    # Without this, H_bathy / u_3d / v_3d (float64 under x64) promote
    # the Helmholtz operator output, breaking the CG solver's
    # while_loop carry-type invariant.
    g = g.astype(eta_dtype)
    dt_t = jnp.asarray(dt, dtype=eta_dtype)
    min_water_col = jnp.asarray(config.min_water_column_m, dtype=eta_dtype)
    H_bathy = H_bathy.astype(eta_dtype)
    mask = mask.astype(eta_dtype)
    u_mask = u_mask.astype(eta_dtype)
    v_mask = v_mask.astype(eta_dtype)
    u_3d = u_3d.astype(eta_dtype)
    v_3d = v_3d.astype(eta_dtype)

    if F_slow_eta is None:
        F_slow_eta = jnp.zeros_like(eta_old)
    else:
        F_slow_eta = F_slow_eta.astype(eta_dtype)
    if F_slow_u is None:
        F_slow_u = jnp.zeros((eta_old.shape[0], eta_old.shape[1] + 1),
                             dtype=eta_dtype)
    else:
        F_slow_u = F_slow_u.astype(eta_dtype)
    if F_slow_v is None:
        F_slow_v = jnp.zeros((eta_old.shape[0] + 1, eta_old.shape[1]),
                             dtype=eta_dtype)
    else:
        F_slow_v = F_slow_v.astype(eta_dtype)

    eta_floor = min_water_col - H_bathy
    eta_old = jnp.maximum(eta_old, eta_floor) * mask

    # ----- Step 1: depth-average u/v to face barotropic velocity ---------
    h_k_old = compute_layer_thickness(
        eta_old, H_bathy, z_coord,
        min_water_column_m=config.min_water_column_m,
    ).astype(eta_dtype)
    U_old, V_old = _depth_average_to_faces(
        u_3d, v_3d, h_k_old, min_water_col, u_mask, v_mask, grid,
    )

    # ----- Step 2: face total depth from eta_old -------------------------
    # Use sum(h_k_old) for cell-center total, not eta+H_bathy: ensures
    # bit-consistency with the per-level mass flux pipeline that builds
    # H_u_old as sum_k(min_cell_to_uface(h_k_old)).  This is required
    # for the Hallberg-Adcroft 2009 column-sum identity.
    H_u_old, H_v_old = _h_total_at_faces(
        h_k_old, min_water_col, mask, grid,
    )

    # ----- Step 3: Coriolis face values ---------------------------------
    if hasattr(grid, "f_u") and hasattr(grid, "f_v"):
        f_u = grid.f_u.astype(eta_dtype)
        f_v = grid.f_v.astype(eta_dtype)
    else:
        f_cell = grid.f.astype(eta_dtype)
        f_u = 0.5 * (jnp.roll(f_cell, 1, axis=1) + f_cell)
        f_u = jnp.concatenate([f_u, f_u[:, 0:1]], axis=1)
        f_v_int = 0.5 * (f_cell[:-1] + f_cell[1:])
        f_v = jnp.concatenate([f_cell[0:1], f_v_int, f_cell[-1:]], axis=0)

    # ----- Step 4: predictor (FB Coriolis, OLD eta gradient) -----------
    grad_x_eta_old = gradient_x_cgrid(eta_old, grid).astype(eta_dtype)
    grad_y_eta_old = gradient_y_cgrid(eta_old, grid).astype(eta_dtype)

    # V at u-points (4-pt average over surrounding v-faces)
    V_west = jnp.roll(V_old, 1, axis=1)
    V_at_u = 0.25 * (V_old[:-1] + V_old[1:] + V_west[:-1] + V_west[1:])
    V_at_u = jnp.concatenate([V_at_u, V_at_u[:, 0:1]], axis=1)

    U_pred = (
        U_old + dt_t * (-g * grad_x_eta_old + f_u * V_at_u + F_slow_u)
    ) * u_mask

    # U_pred at v-points (4-pt average) for FB Coriolis on V, via the
    # shared cell-pad-first helper (partition-cut faces average the
    # neighbour rank's true U row; wall/fold conventions bit-identical
    # to the old interior-then-pad_ns_vector_u serial path).
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        interp_u_to_vface_4pt,
    )
    U_pred_at_v = interp_u_to_vface_4pt(U_pred, grid)

    V_pred = (
        V_old + dt_t * (-g * grad_y_eta_old - f_v * U_pred_at_v + F_slow_v)
    ) * v_mask

    # ----- Step 5: build elliptic RHS and solve ------------------------
    theta_eta = jnp.asarray(config.barotropic_implicit_theta_eta, dtype=eta_dtype)
    theta_pgf = jnp.asarray(config.barotropic_implicit_theta_pgf, dtype=eta_dtype)
    coeff = theta_eta * theta_pgf * dt_t * dt_t * g

    # Time-averaged predicted transport for continuity
    U_avg_pred = ((1.0 - theta_eta) * U_old + theta_eta * U_pred) * u_mask
    V_avg_pred = ((1.0 - theta_eta) * V_old + theta_eta * V_pred) * v_mask

    flux_u_pred = H_u_old * U_avg_pred
    flux_v_pred = H_v_old * V_avg_pred
    div_HU_pred = divergence_cgrid(
        flux_u_pred, flux_v_pred, grid, u_mask=u_mask, v_mask=v_mask,
    ).astype(eta_dtype)

    # ∇·(H · ∇η_old) for the eta_old gradient term in the RHS
    flux_eta_old_x = H_u_old * grad_x_eta_old * u_mask
    flux_eta_old_y = H_v_old * grad_y_eta_old * v_mask
    div_grad_eta_old = divergence_cgrid(
        flux_eta_old_x, flux_eta_old_y, grid, u_mask=u_mask, v_mask=v_mask,
    ).astype(eta_dtype)

    rhs = (
        eta_old
        - dt_t * div_HU_pred
        - coeff * div_grad_eta_old
        + dt_t * F_slow_eta * mask
    ) * mask

    # Jacobi preconditioner diagonal + Helmholtz operator.  ``A_op``
    # feeds the distributed-PCG branch and the residual diagnostics; on
    # the single-rank branch the operator is REBUILT inside the
    # custom-VJP solver so its adjoint can take exact parameter
    # cotangents (see solve_helmholtz_freesurface) — same
    # ``_make_helmholtz`` numerics, so the two instances are identical.
    inv_diag = _helmholtz_inv_diag(
        H_u_old, H_v_old, coeff, grid, mask,
    )
    A_op = _make_helmholtz(
        H_u_old, H_v_old, coeff, grid, mask, u_mask, v_mask,
    )

    def M_inv(r: jnp.ndarray) -> jnp.ndarray:
        return r * inv_diag.astype(r.dtype)

    # Solve dispatch — MERGE COMPOSITION of the area-weighted-adjoint fix
    # (PR #394 branch, commit 625a5610) with the MPI-scaling refactor
    # (``barotropic_common.solve_helmholtz_implicit``).  Static Python
    # branch, so only one solver is traced:
    #
    # * single-rank, no force_pcg → :func:`solve_helmholtz_freesurface`.
    #   Forward = the stock preconditioned ``jax.scipy`` CG VERBATIM
    #   (``jax.custom_vjp`` inlines the primal — bit-identical to the
    #   ``solve_helmholtz_implicit`` stock branch this routes around);
    #   reverse mode applies the TRUE area-weighted transpose
    #   ``A⁻ᵀ = W·A⁻¹·W⁻¹`` — stock cg's symmetry-reusing VJP biased
    #   every free-surface gradient by ~1% (per-solve AD/FD 0.9909;
    #   commit-1e370679 sibling).  Tolerance is on relative residual;
    #   1e-10 keeps the per-step mass error below the float precision of
    #   the integral.
    # * distributed or force_pcg → ``barotropic_common``'s UNROLLED
    #   fixed-iteration PCG (static fori_loop, uniform MPI collective
    #   schedule, no deadlock).  Differentiated STRAIGHT THROUGH — A_op's
    #   halo sendrecv-VJP + the allreduce-SUM dots are AD-safe, so those
    #   gradients are correct by construction and need no custom
    #   transpose.  (``custom_linear_solve`` is impossible there: the
    #   halo ``_sendrecv_vjp`` custom_vjp has no transpose rule — see
    #   barotropic_common's module note.)
    from legoesm.core.operators import is_distributed as _is_distributed
    from legoesm.ocean.dynamics.barotropic_common import (
        HelmholtzSolveDiagnostics,
        global_rel_residual as _rel_resid,
        solve_helmholtz_implicit,
    )
    _area_eta = grid.area.astype(eta_dtype)
    _residual_tol = config.barotropic_implicit_pcg_residual_tol
    # ``force_pcg`` selects the fixed-M PCG body even single-rank
    # (solver-matched parity references + the faster-single-rank
    # option, job 8458701); its global dots reduce locally when not
    # multi-process, so the flag is safe pre-arming.
    _use_pcg = _is_distributed() or bool(config.barotropic_implicit_force_pcg)
    if not _use_pcg:
        pcg_tol = jnp.asarray(
            config.barotropic_implicit_pcg_tol, dtype=eta_dtype,
        )
        eta_new = solve_helmholtz_freesurface(
            rhs, eta_old, H_u_old, H_v_old, coeff, mask, u_mask, v_mask,
            inv_diag, grid, tol=pcg_tol,
            maxiter=int(config.barotropic_implicit_pcg_maxiter),
        )
        # Same diagnostic contract as the solve_helmholtz_implicit stock
        # branch (global rel-residual + converged flag), so the
        # runtime-check consumer below is branch-uniform.
        _rel = _rel_resid(A_op, eta_new, rhs)
        _solve_diag = HelmholtzSolveDiagnostics(
            rel_residual=_rel, converged=_rel <= _residual_tol,
        )
    else:
        eta_new, _solve_diag = solve_helmholtz_implicit(
            A_op, rhs, M_inv, eta_old,
            distributed=True,
            fixed_iters=int(config.barotropic_implicit_pcg_fixed_iters),
            residual_tol=_residual_tol,
            stock_cg_tol=config.barotropic_implicit_pcg_tol,
            stock_cg_maxiter=int(config.barotropic_implicit_pcg_maxiter),
            pcg_variant=str(config.barotropic_implicit_pcg_variant),
            # W-inner-product weight for the single_reduce recurrences
            # (masked cell area — the dot in which this FV Helmholtz is
            # self-adjoint; ignored by the standard body).
            dot_weight=_area_eta * mask,
        )
    # Debug-gated consumer for the solve diagnostic (codex 2026-06-11
    # MAJOR: ``_solve_diag`` was computed and silently discarded on the
    # production path — an under-converged fixed-M solve would advance
    # the model with a bad eta).  Prints ONLY on non-convergence; the
    # production-default path (enable_runtime_checks=False) is
    # unchanged.  ``return_residual=True`` callers keep the loud
    # outside-JIT handling.
    if bool(config.enable_runtime_checks):
        jax.lax.cond(
            _solve_diag.converged,
            lambda _r: None,
            lambda _r: jax.debug.print(
                "WARNING barotropic_implicit_latlon_cgrid: fixed-M PCG "
                "NOT converged (rel_residual={r:.3e} > tol) — raise "
                "barotropic_implicit_pcg_fixed_iters or check "
                "conditioning.", r=_r,
            ),
            _solve_diag.rel_residual,
        )
    eta_new = eta_new * mask

    # Global mass conservation correction.  The CG solve minimizes the
    # L2 residual but does not guarantee that sum(r * area) = 0 — the
    # area-weighted integral of the residual can have a small nonzero
    # bias that accumulates over 10⁵-10⁶ steps.  Project out the global
    # mean drift so that sum(eta_new * area) = sum(rhs * area) exactly.
    # This is standard practice (MOM6, NEMO, MITgcm).
    #
    # BLOCKER-1 fix: the three area-weighted sums MUST be GLOBAL under
    # MPI.  A rank-local ``jnp.sum`` would apply a different per-rank
    # mean correction to each sub-domain — distorting the spatial eta
    # field per-rank (the outer ``fix_eta_drift`` global projection can
    # only fix the global mean, not undo per-rank spatial offsets, and
    # may be disabled).  Use the same MPI-aware batched SUM that the
    # adjacent ``clamp_and_redistribute`` (eta_floor) uses.
    #
    # Accumulate in ``ocean_diagnostics`` precision (f64 even when the
    # state runs at f32) — ``target_mass - actual_mass`` is a huge-area
    # cancellation that would lose the whole correction signal in f32
    # (mirrors the outer ``fix_eta_drift`` cast in ocean_model_latlon).
    from legoesm.parallel.reductions import (
        batch_allreduce_mpi as _batch_allreduce_mpi,
        is_multi_process as _is_multi_process,
    )
    from legoesm.core.precision import cast as _cast
    _M = "ocean_diagnostics"
    _area_acc = _cast(_area_eta, _M, "accumulate")
    _mask_acc = _cast(mask, _M, "accumulate")
    _wa = _area_acc * _mask_acc
    _ocean_area_l = jnp.sum(_wa)
    _target_mass_l = jnp.sum(_cast(rhs, _M, "accumulate") * _area_acc)
    _actual_mass_l = jnp.sum(_cast(eta_new, _M, "accumulate") * _area_acc)
    if _is_multi_process():
        _ocean_area, _target_mass, _actual_mass = _batch_allreduce_mpi(
            [_ocean_area_l, _target_mass_l, _actual_mass_l], op="sum",
        )
    else:
        _ocean_area, _target_mass, _actual_mass = (
            _ocean_area_l, _target_mass_l, _actual_mass_l,
        )
    _correction = (_target_mass - _actual_mass) / jnp.maximum(_ocean_area, 1e-30)
    eta_new = (eta_new + _correction.astype(eta_dtype) * mask) * mask

    # Floor clamp (mass-conserving redistribution).  In normal operation
    # this never fires; it is a safety net for extreme transients.
    eta_new = _clamp_redistribute(
        eta_new, eta_floor, mask, grid.area.astype(eta_dtype),
    )

    # ----- Step 6: corrector for U,V using new eta gradient delta ------
    grad_x_eta_new = gradient_x_cgrid(eta_new, grid).astype(eta_dtype)
    grad_y_eta_new = gradient_y_cgrid(eta_new, grid).astype(eta_dtype)
    delta_grad_x = grad_x_eta_new - grad_x_eta_old
    delta_grad_y = grad_y_eta_new - grad_y_eta_old

    U_new = (U_pred - theta_pgf * dt_t * g * delta_grad_x) * u_mask
    V_new = (V_pred - theta_pgf * dt_t * g * delta_grad_y) * v_mask

    # ----- Step 7: time-averaged transport for tracer step --------------
    # H_u_new uses sum(h_k_new) — bit-consistent with the per-level
    # mass flux pipeline.  See comment in Step 2.
    h_k_new = compute_layer_thickness(
        eta_new, H_bathy, z_coord,
        min_water_column_m=config.min_water_column_m,
    )
    H_u_new, H_v_new = _h_total_at_faces(
        h_k_new, min_water_col, mask, grid,
    )
    # Use H_old consistently so that div(Hu_avg) = (eta_old - eta_new)/dt
    # exactly — required for flux-form tracer conservation.  The Helmholtz
    # solve used H_old throughout; using H_new here breaks the discrete
    # continuity closure on partial cells.
    Hu_avg = H_u_old * (
        (1.0 - theta_eta) * U_old + theta_eta * U_new
    ) * u_mask
    Hv_avg = H_v_old * (
        (1.0 - theta_eta) * V_old + theta_eta * V_new
    ) * v_mask

    # ----- Step 8: update 3D velocity (preserve baroclinic structure) --
    u_baro_old = U_old[..., jnp.newaxis]
    v_baro_old = V_old[..., jnp.newaxis]
    u_prime = u_3d - u_baro_old
    v_prime = v_3d - v_baro_old
    # Partial-cells: at faces where one cell is below-seafloor (h_k=0)
    # and the other has water, the 2D u_mask is 1 (both columns are
    # ocean cells in 2D), but u_new at the inactive level should be
    # zero.  Use a 3D face-active mask derived from h_k > 0.  For
    # legacy z* (all h_k > 0), this mask is 1 everywhere → bit-exact.
    h_active_3d = (h_k_old > 0).astype(u_3d.dtype)
    u_active_3d_inner = h_active_3d * jnp.roll(h_active_3d, 1, axis=1)
    u_active_3d = jnp.concatenate(
        [u_active_3d_inner, u_active_3d_inner[:, 0:1, :]], axis=1,
    )
    # Cell-pad-first (PR357 Bug-2 pattern): pad the 3D cell active mask so
    # the v-face active flag at a partition cut is the product of the two
    # adjacent cells across the cut (MPI halo), not a zeroed wall.  The fold
    # seam is overwritten only on the rank that owns it.
    h_active_pad = pad_ns_zero(h_active_3d)
    v_active_3d = h_active_pad[:-1] * h_active_pad[1:]
    v_active_3d = _zero_polar_lat_ends(v_active_3d)
    if fold_is_local(grid):
        north_3d = h_active_3d[-1:] * h_active_3d[-1:, grid.fold.perm_T, :]
        v_active_3d = jnp.concatenate(
            [v_active_3d[:-1], north_3d], axis=0,
        )
    u_new_3d = (
        (u_prime + U_new[..., jnp.newaxis])
        * u_mask[..., jnp.newaxis] * u_active_3d
    )
    v_new_3d = (
        (v_prime + V_new[..., jnp.newaxis])
        * v_mask[..., jnp.newaxis] * v_active_3d
    )

    state_new = state._replace(
        eta=state.eta.replace(data=eta_new),
        u=state.u.replace(data=u_new_3d),
        v=state.v.replace(data=v_new_3d),
    )
    if return_residual:
        # Report the residual of the ACTUAL returned eta (after the mass
        # projection + floor clamp), not the raw-solve residual — those
        # post-steps add a uniform offset / redistribute mass, so the
        # contract diagnostic must describe ``state_new.eta``.  Recompute
        # via the same A_op; SUM-only + stop_gradient inside the helper
        # (``_rel_resid`` imported at the solve dispatch above).
        rel_final = _rel_resid(A_op, eta_new, rhs)
        return state_new, (Hu_avg, Hv_avg), rel_final
    return state_new, (Hu_avg, Hv_avg)
