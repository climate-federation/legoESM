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
    coriolis_cgrid_energy_conserving,
    fold_is_local,
    north_fold_mask,
    apply_north_fold,
    divergence_cgrid,
    fold_vface_row,
    gradient_x_cgrid,
    gradient_y_cgrid,
    is_tripolar,
    reads_stored_vface_metric,
    pad_ns_zero,
    vface_zonal_cos_lat,
)
from legoesm.grids.halo_latlon import zero_polar_lat_ends as _zero_polar_lat_ends
from legoesm.ocean.dynamics.eta_floor import (
    clamp_and_redistribute as _clamp_redistribute,
)
from legoesm.ocean.dynamics.ocean_tendency_common import (
    depth_average_to_faces,
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
    # Barotropic-mean face velocity (#517 item 1: shared
    # depth_average_to_faces; floor = min_water_col → bit-identical).
    # U_bar was open-coded as TWO separate sums → fused=False (byte-identity).
    U_bar = depth_average_to_faces(u_3d, h_u, u_mask, min_water_col,
                                   fused=False)

    # Cell-pad-first (PR357 Bug-2 pattern): pad the cell thickness so the
    # v-face min at a partition cut uses the neighbour rank's adjacent
    # column (MPI halo).  On tripolar grids the fold face connects cell
    # (fold_j, i) with its fold partner (fold_j, perm_T[i]); the seam is
    # overwritten only on the rank that owns it.
    h_k_pad = pad_ns_zero(h_k)
    h_v = jnp.minimum(h_k_pad[:-1], h_k_pad[1:])
    h_v = _zero_polar_lat_ends(h_v)
    nmask = north_fold_mask(grid)
    if fold_is_local(grid) or nmask is not None:
        north_row = jnp.minimum(h_k[-1:], fold_vface_row(h_k, grid))
        # Data-dependent fold application (SPMD-safe: the seam is masked,
        # not sliced, so non-owner shards are untouched under shard_map).
        h_v = apply_north_fold(h_v, north_row, grid, north_mask=nmask)
    # #517 shared helper, fused=True == the fused-stack sum topology
    # (byte-identical to the open-coded stack/sum this replaces).
    V_bar = depth_average_to_faces(v_3d, h_v, v_mask, min_water_col)

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
    nmask = north_fold_mask(grid)
    if fold_is_local(grid) or nmask is not None:
        north_row = jnp.minimum(H_total[-1:], fold_vface_row(H_total, grid))
        H_v = apply_north_fold(H_v, north_row, grid, north_mask=nmask)

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


def _helmholtz_coupling_pieces(
    H_u: jnp.ndarray,
    H_v: jnp.ndarray,
    grid: LatLonGrid,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """FV coupling magnitudes of ``∇·(H·∇)`` at each cell (positive).

    Returns ``(zonal_E, zonal_W, merid_sum)`` — the per-cell positive
    coupling coefficients to the east/west zonal neighbours and the
    summed meridional couplings, such that
    ``diag(A) = 1 + coeff·(zonal_E + zonal_W + merid_sum)`` and the
    zonal off-diagonals of ``A`` are ``-coeff·zonal_E`` /
    ``-coeff·zonal_W``.  Serves the ZONAL-LINE preconditioner;
    :func:`_helmholtz_inv_diag` keeps its own verbatim expression (its
    fp grouping is baked into every existing implicit_cn trajectory —
    rewiring through these pieces would regroup the arithmetic and
    perturb bit-exact parity lanes).  The two are tied by a mechanical
    consistency test (``test_zonal_line_preconditioner.py``): these
    pieces must reconstruct ``1/inv_diag`` to fp tolerance.
    """
    area = grid.area                                     # (n_lat, n_lon)
    inv_area = 1.0 / area

    H_u_E = H_u[:, 1:]   # east face of cell j: u-face (j, i+1)
    H_u_W = H_u[:, :-1]  # west face of cell j: u-face (j, i)
    H_v_N = H_v[1:, :]   # north face of cell j: v-face (j+1, i)
    H_v_S = H_v[:-1, :]  # south face of cell j: v-face (j, i)

    if is_tripolar(grid):
        # Tripolar: full per-face 2D metrics (longitude variation in the
        # bipolar cap; column-0-only metrics made PCG diverge).
        dy_u_E = grid.dy_u[:, 1:]    # (n_lat, n_lon)
        dy_u_W = grid.dy_u[:, :-1]
        dx_u_E = grid.dx_u[:, 1:]
        dx_u_W = grid.dx_u[:, :-1]
        dx_v_N = grid.dx_v[1:, :]    # (n_lat, n_lon)
        dx_v_S = grid.dx_v[:-1, :]
        dy_v_N = grid.dy_v[1:, :]
        dy_v_S = grid.dy_v[:-1, :]

        zonal_E = H_u_E * dy_u_E / jnp.maximum(dx_u_E, 1.0e-30) * inv_area
        zonal_W = H_u_W * dy_u_W / jnp.maximum(dx_u_W, 1.0e-30) * inv_area
        merid_sum = (
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

        # v-face cell-centre-to-cell-centre distance: row-pair dependent.
        dy_v_int = 0.5 * (dy_h[1:] + dy_h[:-1])          # (n_lat-1,)
        dy_v_face = jnp.pad(dy_v_int, (1, 1), mode='edge')  # (n_lat+1,)

        # v-face zonal extent dx_v = R*cos(lat_v)*dlon at all n_lat+1 faces,
        # computed BACKEND-AWARE exactly as divergence_cgrid (cell-pad-first):
        # pad the CELL latitudes via pad_with_pole_bc_lat (at an interior MPI
        # band cut the ghost row is the NEIGHBOUR rank's edge-cell latitude via
        # AD-safe sendrecv) then midpoint, so a band-edge v-face carries its
        # REAL latitude instead of the wall ±pi/2.  zero_polar_lat_ends zeros
        # cos at the PHYSICAL poles only.  Serial: bit-identical to the old
        # [-pi/2, .., pi/2] reconstruction at interior faces; the pole faces
        # (cos->0) are multiplied by the walled H_v=0 in merid_sum, so this is
        # bit-exact serially while fixing the BANDED zonal-line smoother, which
        # otherwise set dx_v=0 at rank cuts and dropped its meridional diagonal
        # there (codex MED, job 8487913), eroding the M-cut at scale.
        if reads_stored_vface_metric(grid):
            # Rich geometry (beta-plane): READ the stored v-face metric (#514).
            # dx_v is lon-uniform here (tripolar took the 2D branch above), so
            # column 0 is the full 1D metric the regular path expects; the
            # Cartesian beta-plane gives the uniform dx_m the recompute of a
            # nonzero pseudo-lat would get ~1.8% wrong.  Pole rows keep their
            # stored 0 (multiplied by the walled H_v=0 in merid_sum).
            dx_v = grid.dx_v[:, 0]                        # (n_lat+1,)
        else:
            from legoesm.grids.halo_latlon import (
                pad_with_pole_bc_lat, zero_polar_lat_ends,
            )
            lat_pad = pad_with_pole_bc_lat(
                grid.lat, halo=1, south_value=0.0, north_value=0.0)
            lat_v = 0.5 * (lat_pad[:-1] + lat_pad[1:])      # (n_lat+1,)
            cos_lat_v = zero_polar_lat_ends(jnp.cos(lat_v))
            dx_v = R * cos_lat_v * dlon                      # (n_lat+1,)

        zonal_E = H_u_E * dy_u[:, None] / dx_u[:, None] * inv_area
        zonal_W = H_u_W * dy_u[:, None] / dx_u[:, None] * inv_area
        merid_sum = (
            H_v_N * dx_v[1:, None] / dy_v_face[1:, None]
            + H_v_S * dx_v[:-1, None] / dy_v_face[:-1, None]
        ) * inv_area
    return zonal_E, zonal_W, merid_sum


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

        # #516: single-source v-face zonal cos(lat_v) — interior
        # cos(0.5·(lat[j]+lat[j+1])), poles exactly 0.  Routes through the
        # shared helper (backend-aware pad_with_pole_bc_lat + zero_polar_lat_ends)
        # so an MPI band cut keeps the neighbour-rank metric instead of the
        # old synthetic ±π/2 endpoints (which dropped the meridional diagonal
        # at interior cuts — #515).  Bit-identical on serial.
        cos_lat_v = vface_zonal_cos_lat(grid)            # (n_lat+1,)
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


def _make_zonal_line_preconditioner(
    H_u: jnp.ndarray,
    H_v: jnp.ndarray,
    coeff: jnp.ndarray,
    grid: LatLonGrid,
    mask: jnp.ndarray,
):
    """Zonal-line preconditioner: exact periodic-tridiagonal row solves.

    ``M = I + coeff·(zonal part of -∇·(H·∇), EXACT) + coeff·diag(merid)``
    — the meridional couplings stay on the diagonal (classic line
    relaxation).  Each latitude row is an independent PERIODIC
    tridiagonal system in longitude, solved exactly by
    :func:`cyclic_thomas_batched` (Sherman–Morrison over two batched
    Thomas sweeps).  Properties that matter here:

    * **Communication-free under band MPI**: every rank owns its full
      longitude rows, so applying M⁻¹ is rank-local — unlike a global
      preconditioner it adds ZERO collectives to the PCG iteration
      (the EVP-block-preconditioner principle, CESM POP GMD 9:4209:
      trade cheap local FLOPs for fewer latency-bound iterations).
    * **Targets the stiff direction**: on a lat-lon grid the zonal
      spacing collapses near the poles (dx ∝ cos φ), so the polar rows
      dominate the Helmholtz condition number — exactly the couplings
      this M inverts exactly.
    * **W-self-adjoint**: built from the SAME FV stencil pieces as
      ``A`` (area-weighted), so M is self-adjoint in the
      ``<x,y> = Σ x·y·area`` inner product like A itself — the
      requirement the single_reduce (Chronopoulos–Gear) recurrences
      impose on the preconditioner (the CG-CG inner-product lesson).
      Verified numerically in ``test_zonal_line_preconditioner.py``.

    Land cells: M⁻¹ rows are decoupled by zeroing the couplings across
    masked faces and forcing identity on land (output additionally
    masked, matching the Jacobi convention M⁻¹=0 on land).
    """
    from legoesm.timestepping.tridiagonal import cyclic_thomas_batched

    if mask.shape[-1] < 3:
        raise ValueError(
            "zonal_line preconditioner needs n_lon >= 3 (periodic "
            f"tridiagonal rows); got n_lon={mask.shape[-1]}.  Use "
            "preconditioner='jacobi' on degenerate-longitude grids."
        )
    zonal_E, zonal_W, merid_sum = _helmholtz_coupling_pieces(H_u, H_v, grid)
    wet = mask > 0.5
    # Couplings across a land face must vanish: H on land faces is
    # normally zero already (depth), but the neighbour-cell mask makes
    # this robust to nonzero-H walls (same wet-coupling convention as
    # A_op's u_mask/v_mask).
    wet_E = jnp.roll(wet, shift=-1, axis=1)   # east neighbour wet
    wet_W = jnp.roll(wet, shift=1, axis=1)    # west neighbour wet
    zE = jnp.where(wet & wet_E, zonal_E, 0.0)
    zW = jnp.where(wet & wet_W, zonal_W, 0.0)

    diag = 1.0 + coeff * (zE + zW + merid_sum)
    diag = jnp.where(wet, diag, 1.0)          # identity rows on land
    off_E = jnp.where(wet, -coeff * zE, 0.0)  # couples x[i+1 mod n]
    off_W = jnp.where(wet, -coeff * zW, 0.0)  # couples x[i-1 mod n]

    def M_inv(r: jnp.ndarray) -> jnp.ndarray:
        dt = r.dtype
        x = cyclic_thomas_batched(
            off_W.astype(dt), diag.astype(dt), off_E.astype(dt),
            (r * mask).astype(dt),
        )
        return x * mask.astype(dt)

    return M_inv


def _make_chebyshev_preconditioner(A_op, inv_diag, mask, degree: int):
    """Degree-``degree`` Chebyshev-polynomial preconditioner:
    ``M⁻¹r ≈ p(A)·r`` via ``degree`` applications of the Helmholtz matvec
    ``A_op`` with scalar Chebyshev coefficients on ``[λ_min, λ_max]``.

    Why this (scaling review 2026-06-13, ocean lever #1): it cuts the OUTER
    PCG iteration count using ONLY matvecs + scalar coefficients — NO extra
    global reductions per iteration (one stop_gradient'd global_max at
    setup for λ_max).  On the latency-bound cross-node fabric, trading a
    reduction (all-ranks barrier) for a matvec (nearest-neighbour sendrecv
    halo) is the right direction.  Cheaper than multigrid (~80 LOC, no
    restriction/prolongation/coarse-grid-under-MPI), and it is the standard
    smoother MPAS-O/CESM use inside multigrid anyway.

    Eigenvalue window:
      * ``λ_min = 1`` is ANALYTIC — ``A = I - coeff·∇·(H·∇)`` and
        ``-coeff·∇·(H·∇)`` is positive-semidefinite (coeff,H ≥ 0), so every
        eigenvalue of A is ≥ 1 (the free-surface identity floor; equality at
        the k=0 mode).
      * ``λ_max`` from the Gershgorin row-sum bound: for A's FV stencil the
        off-diagonals sum to ``diag - 1``, so ``λ_max ≤ max(2·diag - 1)``.
        ``diag = 1/inv_diag`` on wet cells.

    W-self-adjoint: a polynomial in the area-weighted-self-adjoint A is
    itself W-self-adjoint, so M⁻¹ composes with the single_reduce
    (Chronopoulos–Gear) CG (the dot-weight requirement) — verified in
    ``test_chebyshev_preconditioner.py``.  AD-safe: pure matvec + scalar
    coeffs, no new custom_vjp (A_op's halo ``_sendrecv_vjp`` carries the
    gradient; the eigenvalue bound is stop_gradient'd, as a preconditioner
    parameter needs no gradient — the converged solution is unchanged).
    """
    import jax

    from legoesm.parallel.reductions import global_max_mpi, is_multi_process

    if degree < 1:
        raise ValueError(
            f"chebyshev preconditioner needs degree >= 1, got {degree}.")

    wet = mask > 0.5
    diag = jnp.where(wet, 1.0 / jnp.maximum(inv_diag, 1.0e-30), 0.0)
    gersh = jnp.where(wet, 2.0 * diag - 1.0, -jnp.inf)
    lmax_local = jnp.max(gersh)
    # Global spectral-radius bound: under the single-controller lat-band shard_map
    # lmax_local is this band's PARTIAL max -> take the max across "lat" with
    # jax.lax.pmax (the SPMD analogue of global_max_mpi; mpi4jax not even required
    # on the route-B multi-GPU path).  Checked FIRST (is_multi_process() is False
    # under one process), and -- like eta_floor._global_sum_pair -- gated on a
    # "lat" mesh axis so a coupled cube-atm SPMD mesh (NON-lat axes) falls through
    # to MPI/local instead of crashing (codex).  pmax/global_max are MAX, so NON-
    # differentiable -- fine because lmax is stop_gradient'd just below (a
    # preconditioner tuning parameter, not part of the converged answer).
    from legoesm.grids.halo import (
        get_halo_backend as _get_halo_backend, get_spmd_mesh as _get_spmd_mesh,
    )
    lmax = None
    if _get_halo_backend() == "spmd":
        _mesh = _get_spmd_mesh()
        if _mesh is None:
            raise RuntimeError(
                "chebyshev preconditioner: halo backend is 'spmd' but no SPMD "
                "mesh is set; arm it via activate_latlon_spmd_halo(mesh).")
        if "lat" in tuple(_mesh.axis_names):
            lmax = jax.lax.pmax(lmax_local, "lat")
    if lmax is None:
        lmax = global_max_mpi(lmax_local) if is_multi_process() else lmax_local
    # stop_gradient: the preconditioner's spectral window is a tuning
    # parameter (only affects convergence speed, not the converged answer),
    # so the non-differentiable allreduce(MAX) is safe here.
    lmax = jax.lax.stop_gradient(jnp.maximum(lmax, 1.0 + 1.0e-12))
    lmin = 1.0
    c = (lmax - lmin) / 2.0   # half-width of the spectral interval
    d = (lmax + lmin) / 2.0   # centre

    def M_inv(r: jnp.ndarray) -> jnp.ndarray:
        dt = r.dtype
        # Cast the spectral-window scalars to the carry dtype: lmax/d/c are
        # computed in f64 (from inv_diag under x64), so alpha/beta derived from
        # them are f64 — and ``x = x + alpha*p`` would UPCAST an f32 PCG carry
        # to f64, tripping the fori_loop carry-dtype check in the full ocean
        # (f32 barotropic carry; job 8489358).  d_/c_ keep all coeffs in dt, so
        # M_inv preserves the input dtype.  f64 lanes are unchanged (d_==d).
        d_ = jnp.asarray(d, dt)
        c_ = jnp.asarray(c, dt)
        b = (r * mask).astype(dt)
        x = jnp.zeros_like(b)
        resid = b               # b - A·0
        p = jnp.zeros_like(b)
        alpha = (1.0 / d_)
        # Chebyshev iteration (Templates / Barrett et al.): a fixed-degree
        # polynomial acceleration of Richardson on A; one A_op matvec per
        # step, zero reductions.
        for i in range(degree):
            if i == 0:
                p = resid
                alpha = 1.0 / d_
            elif i == 1:
                beta = 0.5 * (c_ * alpha) ** 2
                alpha = 1.0 / (d_ - beta / alpha)
                p = resid + beta * p
            else:
                beta = (c_ * alpha / 2.0) ** 2
                alpha = 1.0 / (d_ - beta / alpha)
                p = resid + beta * p
            ap = A_op(p).astype(dt)
            x = x + alpha * p
            resid = resid - alpha * ap
        return x * mask.astype(dt)

    return M_inv


# ---------------------------------------------------------------------------
# Geometric multigrid preconditioner (anisotropic: zonal-line smoother)
# ---------------------------------------------------------------------------
# POC 8487762 (docs/performance/scaling/scaling_levers_audit_2026-06-14.md): a geometric
# V-cycle with a ZONAL-LINE smoother cuts the barotropic-PCG outer iteration
# count M60 -> M2-4 on the polar-anisotropic lat-lon Helmholtz (textbook
# O(log n)), where the pointwise-Jacobi smoother gives only ~3x.  The line
# smoother handles the strong near-pole zonal coupling (dx -> 0) that the
# coarse-grid correction alone cannot; together they smash the
# reduction-latency wall (~120 -> ~8 global allreduces/step under MPI).  The
# transfer pair is Galerkin-symmetric (R = 0.25 P^T) even with land masks, so
# the V-cycle stays ~self-adjoint (CG-friendly).


def _mg_prolong(xc: jnp.ndarray, mask_f: jnp.ndarray) -> jnp.ndarray:
    """Coarse->fine injection prolongation (each fine child = its coarse
    parent), masked to the fine wet domain.  P; its W=area-free adjoint is
    :func:`_mg_restrict` up to the 0.25 factor."""
    xf = jnp.repeat(jnp.repeat(xc, 2, axis=0), 2, axis=1)
    return xf * mask_f


def _mg_restrict(rf: jnp.ndarray, mask_f: jnp.ndarray,
                 nlc: int, nloc: int) -> jnp.ndarray:
    """Full-weighting fine->coarse restriction = 0.25 * P^T: sum the (wet)
    fine 2x2 children.  Exactly 0.25*P^T (P injects + masks), so R and P form
    a Galerkin-symmetric pair on the masked grid."""
    rfm = rf * mask_f
    blk = rfm.reshape(nlc, 2, nloc, 2)
    return 0.25 * (blk[:, 0, :, 0] + blk[:, 1, :, 0]
                   + blk[:, 0, :, 1] + blk[:, 1, :, 1])


def _faces_from_cell_depth(H_cell, mask, n_lat, n_lon):
    """(H_u, H_v, u_mask, v_mask) from a cell depth + mask via the production
    min-rule (periodic lon, pole v-faces zeroed) — the same construction the
    fidelity tests use, so each MG level is a faithful coarse Helmholtz."""
    Hu_inner = jnp.minimum(jnp.roll(H_cell, 1, axis=1), H_cell)
    H_u = jnp.concatenate([Hu_inner, Hu_inner[:, 0:1]], axis=1)
    Hv_inner = jnp.minimum(H_cell[:-1], H_cell[1:])
    H_v = jnp.concatenate(
        [jnp.zeros((1, n_lon)), Hv_inner, jnp.zeros((1, n_lon))], axis=0)
    wet = mask > 0.5
    u_inner = (wet & jnp.roll(wet, 1, axis=1)).astype(H_cell.dtype)
    u_mask = jnp.concatenate([u_inner, u_inner[:, 0:1]], axis=1)
    v_inner = (wet[:-1] & wet[1:]).astype(H_cell.dtype)
    v_mask = jnp.concatenate(
        [jnp.zeros((1, n_lon)), v_inner, jnp.zeros((1, n_lon))], axis=0)
    return H_u * u_mask, H_v * v_mask, u_mask, v_mask


def _faces_from_cell_depth_banded(H_cell, mask, layout):
    """Band-aware ``(H_u, H_v, u_mask, v_mask)`` for the distributed MG.

    Same production min-rule as :func:`_faces_from_cell_depth`, but the
    band-EDGE v-faces use the NEIGHBOUR band's edge row (a 1-row lat halo via
    the backend-dispatched :func:`pad_halo_latlon` — MPI sendrecv / SPMD
    ppermute / local pole-fold) instead of being zeroed, so the meridional
    Helmholtz coupling spans a rank cut.  A v-face is zeroed (wall BC) ONLY at a
    TRUE pole — the south edge when ``layout.south_rank is None`` and the north
    edge when ``layout.north_rank is None``.

    The ``min``-rule is symmetric, so two neighbouring ranks compute the SAME
    value on their shared face (each uses the other's haloed row) — the
    operator is consistent across the cut.  On a global single band
    (``south_rank == north_rank == None``) this reduces EXACTLY to
    :func:`_faces_from_cell_depth` (both edges zeroed; interior identical),
    the property the serial-equivalence test pins.

    u-faces are periodic in longitude and need no lat halo (purely local)."""
    from legoesm.grids.halo_latlon import pad_halo_latlon
    n_lat_local, n_lon = H_cell.shape
    # 1-row lat halo (strip the lon halo pad_halo_latlon also adds).
    Hc_h = pad_halo_latlon(H_cell, halo=1)[:, 1:-1]   # (n_lat_local+2, n_lon)
    m_h = pad_halo_latlon(mask, halo=1)[:, 1:-1]
    # u-faces: periodic lon, local (identical to the serial helper).
    Hu_inner = jnp.minimum(jnp.roll(H_cell, 1, axis=1), H_cell)
    H_u = jnp.concatenate([Hu_inner, Hu_inner[:, 0:1]], axis=1)
    wet = mask > 0.5
    u_inner = (wet & jnp.roll(wet, 1, axis=1)).astype(H_cell.dtype)
    u_mask = jnp.concatenate([u_inner, u_inner[:, 0:1]], axis=1)
    # v-faces: min-rule over the HALOED column → (n_lat_local+1, n_lon).  Face k
    # (k=0..n_lat_local) lies between haloed rows k and k+1.
    H_v = jnp.minimum(Hc_h[:-1], Hc_h[1:])
    wet_h = m_h > 0.5
    v_mask = (wet_h[:-1] & wet_h[1:]).astype(H_cell.dtype)
    # Wall BC at TRUE poles only (no neighbour band there).
    if layout.south_rank is None:
        H_v = H_v.at[0].set(0.0)
        v_mask = v_mask.at[0].set(0.0)
    if layout.north_rank is None:
        H_v = H_v.at[-1].set(0.0)
        v_mask = v_mask.at[-1].set(0.0)
    return H_u * u_mask, H_v * v_mask, u_mask, v_mask


def _coarse_band_layout(layout):
    """2x-coarsened ``LatLonBandLayout`` for the banded multigrid, or ``None``
    if the band is not EVEN-ALIGNED (``lat_start``/``n_lat_local``/global dims
    odd).

    Even-aligned bands are the simplifying requirement for distributed MG: each
    coarse cell's two fine lat rows lie WITHIN one band, so the 2x2 restriction
    stays band-LOCAL (no cross-rank halo / offset).  An odd band would straddle
    a cut -> return ``None`` so the caller falls back / refuses rather than
    build a wrong coarse problem.  Rank topology (rank/n_ranks/neighbours) is
    preserved; ``fold`` is dropped (MG refuses tripolar upstream)."""
    if (layout.lat_start % 2 or layout.n_lat_local % 2
            or layout.n_lat_global % 2 or layout.n_lon_global % 2):
        return None
    return layout._replace(
        n_lat_global=layout.n_lat_global // 2,
        n_lon_global=layout.n_lon_global // 2,
        n_lat_local=layout.n_lat_local // 2,
        lat_start=layout.lat_start // 2,
        lat_end=layout.lat_end // 2,
        fold=None,
    )


def _coarse_band_hierarchy(layout, *, min_coarse_rows: int = 8):
    """Fine->coarsest list of EVEN-ALIGNED band layouts for the banded V-cycle.

    Starts at ``layout`` (L0) and 2x-coarsens (:func:`_coarse_band_layout`)
    while the GLOBAL row count stays ``>= 2*min_coarse_rows`` AND every rank's
    band stays even-aligned.  Two stopping criteria:

    * GLOBAL size — coarsen until the global coarsest has ``< 2*min_coarse_rows``
      rows (so the coarsest GLOBAL problem is ``[min, 2*min)`` rows, cheap for
      the smoother-as-solver).  Using the GLOBAL (not local) count makes the
      depth match the serial hierarchy at ``n_ranks==1`` and keeps the V-cycle
      deep enough to clear the smooth modes regardless of rank count.
    * EVEN-ALIGNMENT — :func:`_coarse_band_layout` returns ``None`` once a band
      would become odd (the 2x2 restriction would straddle a rank cut).  With P
      ranks the band hits an odd size after ``log2(n_lat/P)`` levels, so MANY
      ranks cap the depth below what the global-size rule wants (the known
      1-D-decomposition MG limit; coarsest-grid AGGLOMERATION is the documented
      follow-up).

    Returns ``[L0, L1, ...]``, ``len >= 1`` (L0 always present).  ``len == 1``
    => no admissible coarsening, so the banded factory builds a single-level
    (zonal-line-smoother-only) preconditioner from L0 alone.  For EQUAL bands
    (``n_lat_global % n_ranks == 0``) every rank derives the SAME depth from the
    global dims, so the V-cycle runs in lock-step with no runtime depth
    reduction (the banded factory enforces equal bands)."""
    levels = [layout]
    cur = layout
    # Coarsen while the GLOBAL coarsest would still have >= min_coarse_rows rows.
    while cur.n_lat_global >= 2 * min_coarse_rows:
        nxt = _coarse_band_layout(cur)
        # Stop on the even-alignment limit (None) OR a degenerate coarse
        # longitude: the periodic-tridiagonal zonal-line smoother needs
        # n_lon >= 3 (mirrors the serial hierarchy's nlo//2 < 3 stop).
        if nxt is None or nxt.n_lon_global < 3:
            break
        levels.append(nxt)
        cur = nxt
    return levels


def _make_multigrid_preconditioner(
    H_cell: jnp.ndarray,
    coeff: jnp.ndarray,
    grid: LatLonGrid,
    mask: jnp.ndarray,
    *,
    pre: int = 2,
    post: int = 2,
    coarse_sweeps: int = 10,
    omega: float = 0.8,
    min_coarse_rows: int = 8,
):
    """Anisotropic geometric-multigrid V-cycle preconditioner (POC 8487762).

    Builds a 2x-coarsening level hierarchy ONCE (host-side ``create_latlon_grid``
    per level + cell-depth 2x2 restriction + the production min-rule faces);
    each level carries its Helmholtz ``A_op`` and a ZONAL-LINE smoother
    (:func:`_make_zonal_line_preconditioner`).  The returned ``M_inv`` runs one
    recursive V-cycle (pre-smooth -> restrict residual -> recurse -> prolong
    correction -> post-smooth; coarsest level = ``coarse_sweeps`` line sweeps).
    No global reductions inside the V-cycle (smoothing is comm-free per row;
    transfers are local/halo), so it slashes the OUTER PCG reduction count.

    ``H_cell`` is the cell-centred water-column depth the faces derive from
    (the hierarchy must coarsen the cell field, not the staggered faces).

    SCOPE (codex review of 38fec66b): currently valid for a SINGLE-RANK,
    GLOBAL, regular/Mercator lat-lon grid only.  The coarse levels rebuild
    geometry with ``create_latlon_grid`` (uniform) and the transfers are
    rank-local 2x2 — so it FAILS LOUD on (a) distributed band-MPI/SPMD (a
    coarse 2-row aggregate straddles a rank cut => the local restriction
    builds a different coarse problem and the M-cut claim is void) and (b)
    tripolar grids (the north-fold connectivity is dropped by the
    coarse-grid construction).  Halo-aware + tripolar-fold coarsening is the
    documented follow-up (task #26).  As a PRECONDITIONER a geometry
    mismatch (e.g. Mercator on a uniform coarse grid) only slows
    convergence — the outer PCG's exact ``A_op`` + residual keep the answer
    correct — so Mercator is allowed (degraded, not wrong).
    """
    from legoesm.grids.latlon import create_latlon_grid, ensure_geometry
    from legoesm.core.operators import is_distributed
    from legoesm.grids.halo import get_halo_backend

    # Refuse on BOTH distributed paths: route-A MPI (is_distributed) AND
    # route-B single-process lat-lon SPMD (halo backend == "spmd",
    # is_distributed()==False) — both shard latitude, so the rank/shard-local
    # 2x2 restriction straddles band cuts and builds a wrong coarse problem.
    if is_distributed() or get_halo_backend() == "spmd":
        raise ValueError(
            "multigrid preconditioner: the geometric MG transfers are not yet "
            "halo-aware under band MPI / lat-lon SPMD (a rank/shard-local 2x2 "
            "restriction straddles band cuts), so the coarse problem and the "
            "M-cut are invalid distributed.  Use 'zonal_line' (comm-free) or "
            "'chebyshev' under MPI; 'multigrid' is single-rank only for now.")
    if is_tripolar(grid):
        raise ValueError(
            "multigrid preconditioner: coarse-grid construction "
            "(create_latlon_grid + periodic-lon/zero-pole faces) does not "
            "handle the tripolar north-fold connectivity.  Use 'zonal_line' "
            "or 'chebyshev' on tripolar grids.")

    n_lat, n_lon = mask.shape
    levels = []   # (A_op, mask, smoother, n_lat, n_lon)
    Hc, m = H_cell, mask
    g = grid
    nl, nlo = n_lat, n_lon
    while True:
        H_u, H_v, u_mask, v_mask = _faces_from_cell_depth(Hc, m, nl, nlo)
        A_op = _make_helmholtz(H_u, H_v, coeff, g, m, u_mask, v_mask)
        smoother = _make_zonal_line_preconditioner(H_u, H_v, coeff, g, m)
        levels.append((A_op, m, smoother, nl, nlo))
        if nl // 2 < min_coarse_rows or nl % 2 or nlo % 2 or nlo // 2 < 3:
            break
        nlc, nloc = nl // 2, nlo // 2
        # cell-depth 2x2 MEAN over wet children (_mg_restrict = 0.25*sum;
        # = exact mean on all-wet blocks, the common interior; biased low on
        # partial coastal blocks, acceptable for a PRECONDITIONER).
        Hc = _mg_restrict(Hc, m, nlc, nloc)
        # coarse cell wet if ANY fine child wet: 0.25*sum(m*4)=count_wet>0.5.
        m = (_mg_restrict(m * 4.0, jnp.ones_like(m), nlc, nloc) > 0.5
             ).astype(mask.dtype)
        g = ensure_geometry(create_latlon_grid(n_lat=nlc, n_lon=nloc))
        nl, nlo = nlc, nloc
    return _run_vcycle_preconditioner(
        levels, mask, pre=pre, post=post, coarse_sweeps=coarse_sweeps,
        omega=omega)


def _run_vcycle_preconditioner(levels, mask, *, pre, post, coarse_sweeps,
                               omega):
    """Shared V-cycle engine for the SERIAL and BANDED MG factories.

    ``levels`` is the fine->coarsest list of ``(A_op, mask, smoother,
    n_lat_local, n_lon_local)`` tuples each factory builds (serial: global
    rebuilt grids; banded: per-band sliced grids + halo-aware faces).  The
    recursion is IDENTICAL for both because the 2x2 transfers are LOCAL on an
    even-aligned band (each coarse cell's two fine lat rows are owned by the
    same rank), and every ``A_op`` halo-exchanges internally
    (:func:`gradient_y_cgrid`).  The V-cycle therefore has NO global reduction
    — only neighbour halos — which is the whole reduction-latency win."""
    n_levels = len(levels)

    def _smooth(lvl, b, x, sweeps):
        A_op, m_l, sm, _, _ = levels[lvl]
        for _ in range(sweeps):
            x = x + omega * sm(b - A_op(x))
        return x * m_l

    def _vcycle(lvl, b, x):
        A_op, m_l, _sm, nl_l, nlo_l = levels[lvl]
        if lvl == n_levels - 1:
            return _smooth(lvl, b, x, coarse_sweeps)
        x = _smooth(lvl, b, x, pre)
        r = (b - A_op(x)) * m_l
        nlc, nloc = nl_l // 2, nlo_l // 2
        rc = _mg_restrict(r, m_l, nlc, nloc) * levels[lvl + 1][1]
        ec = _vcycle(lvl + 1, rc, jnp.zeros_like(rc))
        x = x + _mg_prolong(ec, m_l)
        x = _smooth(lvl, b, x, post)
        return x * m_l

    def M_inv(r: jnp.ndarray) -> jnp.ndarray:
        return _vcycle(0, r * mask.astype(r.dtype),
                       jnp.zeros_like(r)).astype(r.dtype)

    return M_inv


def _make_multigrid_preconditioner_banded(
    H_cell: jnp.ndarray,
    coeff: jnp.ndarray,
    grid: LatLonGrid,
    mask: jnp.ndarray,
    layout,
    *,
    pre: int = 2,
    post: int = 2,
    coarse_sweeps: int = 10,
    omega: float = 0.8,
    min_coarse_rows: int = 8,
):
    """BANDED distributed anisotropic geometric-MG V-cycle (task #26).

    The halo-aware sibling of :func:`_make_multigrid_preconditioner`: builds the
    SAME level hierarchy on each rank's LATITUDE BAND so the V-cycle slashes the
    barotropic-PCG outer count (M~60->4) WITHOUT the per-iteration global
    allreduce — the multinode weak-scaling reduction-latency wall.

    Per level: a 2x-coarsened band layout (:func:`_coarse_band_hierarchy`); the
    coarse geometry is the GLOBAL coarse ``create_latlon_grid`` SLICED to the
    band (:func:`slice_latlon_grid_to_band`); band-aware faces
    (:func:`_faces_from_cell_depth_banded`, edge v-faces span the rank cut, only
    true poles walled); the production Helmholtz ``A_op`` (halo-exchanges
    internally) + a comm-free zonal-line smoother.  The 2x2 cell-depth/mask
    restriction is band-LOCAL (the even-aligned requirement).

    LOCK-STEP: requires EQUAL even bands (``n_lat_global % n_ranks == 0`` and
    each band even-aligned) so every rank derives the SAME depth from global
    dims — no runtime depth reduction.  Refuses (caller falls back to the
    comm-free 'zonal_line') when the decomposition is unequal/odd or tripolar.
    A geometry mismatch (uniform coarse grid vs Mercator) only slows
    convergence — the outer PCG's exact ``A_op``/residual keep the answer
    correct — so it is allowed (degraded, not wrong), as in the serial case.
    """
    from legoesm.grids.latlon import create_latlon_grid, ensure_geometry
    from legoesm.parallel.latlon_mpi import slice_latlon_grid_to_band

    if is_tripolar(grid):
        raise ValueError(
            "multigrid_banded: coarse-grid construction does not handle the "
            "tripolar north-fold; use 'zonal_line' or 'chebyshev'.")
    n_ranks = layout.n_ranks
    if n_ranks > 1:
        # Unequal bands => ranks would coarsen to different depths (some bands
        # odd/unaligned) and the V-cycle would deadlock on mismatched halo
        # schedules. Divisibility alone no longer proves equal bands now that
        # make_latlon_band_layout accepts EXPLICIT boundaries (wet-cell
        # balancing, codex): check the ACTUAL band this rank owns is the
        # uniform even split. Non-uniform boundaries make at least one rank
        # fail this and raise BEFORE any collective — a loud abort, never a
        # mismatched schedule.
        base, rem = divmod(layout.n_lat_global, n_ranks)
        uniform = (rem == 0
                   and layout.n_lat_local == base
                   and layout.lat_start == layout.rank * base)
        if not uniform:
            raise ValueError(
                "multigrid_banded: requires EQUAL bands (the uniform even "
                f"split) for lock-step coarsening (n_lat_global="
                f"{layout.n_lat_global}, n_ranks={n_ranks}, this rank owns "
                f"[{layout.lat_start}, {layout.lat_end}) = "
                f"{layout.n_lat_local} rows); wet-cell-balanced / explicit "
                "band boundaries are not supported here — use 'zonal_line' "
                "(comm-free) under this decomposition.")

    band_layouts = _coarse_band_hierarchy(layout, min_coarse_rows=min_coarse_rows)
    n_levels = len(band_layouts)

    levels = []   # (A_op, mask, smoother, n_lat_local, n_lon_local)
    Hc, m = H_cell, mask
    for lvl, blay in enumerate(band_layouts):
        if lvl == 0:
            g = grid
        else:
            # Global coarse grid sliced to THIS band, THEN ensure_geometry (the
            # production order; slice_latlon_grid_to_band needs the LatLonGrid's
            # lat_v, which ensure_geometry drops, and is regular-grid-safe —
            # unlike slice_cgrid_geometry_to_band, which assumes a tripolar
            # fold).  Uniform dlat => the band-edge metrics are consistent
            # across ranks.
            g = ensure_geometry(slice_latlon_grid_to_band(
                create_latlon_grid(
                    n_lat=blay.n_lat_global, n_lon=blay.n_lon_global),
                blay, skip_total_area_reduce=True))
        H_u, H_v, u_mask, v_mask = _faces_from_cell_depth_banded(Hc, m, blay)
        A_op = _make_helmholtz(H_u, H_v, coeff, g, m, u_mask, v_mask)
        smoother = _make_zonal_line_preconditioner(H_u, H_v, coeff, g, m)
        levels.append((A_op, m, smoother, blay.n_lat_local, blay.n_lon_global))
        if lvl + 1 < n_levels:
            nlc, nloc = blay.n_lat_local // 2, blay.n_lon_global // 2
            Hc = _mg_restrict(Hc, m, nlc, nloc)        # band-local 2x2 mean
            m = (_mg_restrict(m * 4.0, jnp.ones_like(m), nlc, nloc) > 0.5
                 ).astype(mask.dtype)

    return _run_vcycle_preconditioner(
        levels, mask, pre=pre, post=post, coarse_sweeps=coarse_sweeps,
        omega=omega)


def _select_preconditioner(
    name: str,
    inv_diag: jnp.ndarray,
    H_u: jnp.ndarray,
    H_v: jnp.ndarray,
    coeff: jnp.ndarray,
    grid: LatLonGrid,
    mask: jnp.ndarray,
    A_op=None,
    cheby_degree: int = 4,
    H_cell=None,
    layout=None,
):
    """Dispatch the implicit-CN PCG preconditioner by config name.

    "jacobi" — inverse diagonal (legacy default; reuses the caller's
    ``inv_diag`` so the Jacobi path stays bit-identical).
    "zonal_line" — exact periodic-tridiagonal row solves (comm-free
    under band MPI; POP EVP-class iteration cutter — see
    :func:`_make_zonal_line_preconditioner`).
    "chebyshev" — degree-``cheby_degree`` Chebyshev polynomial of the
    Helmholtz matvec ``A_op`` (iteration-count cut with NO per-iteration
    reduction; see :func:`_make_chebyshev_preconditioner`).
    "multigrid" — anisotropic geometric V-cycle with a zonal-line smoother
    (cuts the outer M ~M60->M4, textbook O(log n); needs ``H_cell``, the
    cell-centred water-column depth, to coarsen).  Auto-selects the SINGLE-RANK
    :func:`_make_multigrid_preconditioner` serially and the BANDED
    :func:`_make_multigrid_preconditioner_banded` under MPI (``layout`` = the
    :class:`LatLonBandLayout`); the banded V-cycle keeps the M-cut WITHOUT the
    per-iteration global allreduce — the multinode reduction-latency win.
    Unknown names refuse loudly (dispatch-hardening convention).
    """
    if name == "jacobi":
        def M_inv(r: jnp.ndarray) -> jnp.ndarray:
            return r * inv_diag.astype(r.dtype)
        return M_inv
    if name == "zonal_line":
        return _make_zonal_line_preconditioner(
            H_u, H_v, coeff, grid, mask,
        )
    if name == "chebyshev":
        if A_op is None:
            raise ValueError(
                "barotropic_implicit_latlon_cgrid: chebyshev preconditioner "
                "requires the Helmholtz A_op (pass A_op=...).")
        return _make_chebyshev_preconditioner(
            A_op, inv_diag, mask, int(cheby_degree),
        )
    if name == "multigrid":
        if H_cell is None:
            raise ValueError(
                "barotropic_implicit_latlon_cgrid: multigrid preconditioner "
                "requires H_cell (the cell water-column depth; pass H_cell=...).")
        from legoesm.core.operators import is_distributed
        if is_distributed():
            # Band MPI: the halo-aware banded V-cycle (no global reduction).
            from legoesm.parallel.latlon_mpi import LatLonBandLayout
            if not isinstance(layout, LatLonBandLayout):
                raise ValueError(
                    "barotropic_implicit_latlon_cgrid: 'multigrid' under MPI "
                    "needs the LatLonBandLayout (pass layout=get_mpi_topology()); "
                    "the rank-local serial V-cycle is not halo-aware.  Use "
                    "'zonal_line' if no band layout is available.")
            return _make_multigrid_preconditioner_banded(
                H_cell, coeff, grid, mask, layout)
        return _make_multigrid_preconditioner(H_cell, coeff, grid, mask)
    raise ValueError(
        "barotropic_implicit_latlon_cgrid: unknown "
        f"barotropic_implicit_preconditioner {name!r}; expected "
        "'jacobi', 'zonal_line', 'chebyshev', or 'multigrid'."
    )


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

    # ----- Step 3: Coriolis face values (#517: shared fold-safe helper) --
    from legoesm.ocean.dynamics.barotropic_common import coriolis_at_faces
    f_u, f_v = coriolis_at_faces(grid, eta_dtype)

    # ----- Step 4: predictor (FB Coriolis, OLD eta gradient) -----------
    grad_x_eta_old = gradient_x_cgrid(eta_old, grid).astype(eta_dtype)
    grad_y_eta_old = gradient_y_cgrid(eta_old, grid).astype(eta_dtype)

    if getattr(config, "coriolis_energy_conserving", False):
        # Sadourny energy-conserving Coriolis with the SHARED VERTEX f, kept in
        # forward-backward order: cor_u from V_old (forward), cor_v from U_pred
        # (backward).  cor_v already carries the -f sign.  This removes the
        # β-plane energy leak of the face-f form (f_u != f_v).
        cor_u, _ = coriolis_cgrid_energy_conserving(
            U_old, V_old, grid, u_mask=u_mask, v_mask=v_mask,
        )
        U_pred = (
            U_old + dt_t * (-g * grad_x_eta_old + cor_u.astype(eta_dtype) + F_slow_u)
        ) * u_mask
        _, cor_v = coriolis_cgrid_energy_conserving(
            U_pred, V_old, grid, u_mask=u_mask, v_mask=v_mask,
        )
        V_pred = (
            V_old + dt_t * (-g * grad_y_eta_old + cor_v.astype(eta_dtype) + F_slow_v)
        ) * v_mask
    else:
        # Forward-backward face-f Coriolis on the barotropic mode — UNLESS
        # coriolis_scheme="explicit_ab2", where the planetary Coriolis already
        # entered the 3-D du_dt (AB2-extrapolated) and its depth-mean is carried
        # in F_slow_u/v; the solver must then NOT add its own f×U_bt again (no
        # double count).  This mirrors the rigid-lid solver's
        # ``add_barotropic_coriolis`` gate (ocean_model_latlon_cgrid.py) and the
        # outer Matsuno-Coriolis skip.  _cori_fac=1.0 for every other scheme
        # (Matsuno split / default) keeps the original FB Coriolis bit-identical.
        _cori_fac = 0.0 if getattr(config, "coriolis_scheme", "matsuno_split") == "explicit_ab2" else 1.0
        # V at u-points (4-pt average over surrounding v-faces)
        V_west = jnp.roll(V_old, 1, axis=1)
        V_at_u = 0.25 * (V_old[:-1] + V_old[1:] + V_west[:-1] + V_west[1:])
        V_at_u = jnp.concatenate([V_at_u, V_at_u[:, 0:1]], axis=1)

        U_pred = (
            U_old + dt_t * (-g * grad_x_eta_old + _cori_fac * f_u * V_at_u + F_slow_u)
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
            V_old + dt_t * (-g * grad_y_eta_old - _cori_fac * f_v * U_pred_at_v + F_slow_v)
        ) * v_mask

    # ----- Step 5: build elliptic RHS and solve ------------------------
    theta_eta = jnp.asarray(config.barotropic.barotropic_implicit_theta_eta, dtype=eta_dtype)
    theta_pgf = jnp.asarray(config.barotropic.barotropic_implicit_theta_pgf, dtype=eta_dtype)
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

    # Preconditioner dispatch (static config; validated at entry so a
    # typo fails loudly, not as a silently-Jacobi run).  Applies to the
    # fixed-M PCG branch below; the single-rank stock-CG branch keeps
    # its internal Jacobi (the custom-VJP solver owns inv_diag for its
    # exact adjoint).
    # Cell-centred water-column depth for the multigrid coarsening (the same
    # sum(h_k) total H_u_old/H_v_old derive from via the min-rule); None-safe
    # for the other preconditioners (they ignore H_cell).
    _H_cell = jnp.maximum(jnp.sum(h_k_old, axis=-1), min_water_col) * mask
    # Band layout for the distributed 'multigrid' path (None serially / for the
    # other preconditioners, which ignore it).  get_mpi_topology() returns the
    # active LatLonBandLayout under band MPI.
    from legoesm.core.operators import is_distributed as _is_dist_pc
    _pc_layout = None
    if _is_dist_pc():
        from legoesm.grids.halo import get_mpi_topology as _get_topo
        _pc_layout = _get_topo()
    M_inv = _select_preconditioner(
        str(getattr(config.barotropic, "barotropic_implicit_preconditioner",
                    "jacobi")),
        inv_diag, H_u_old, H_v_old, coeff, grid, mask,
        A_op=A_op,
        cheby_degree=int(config.barotropic.barotropic_chebyshev_degree),
        H_cell=_H_cell,
        layout=_pc_layout,
    )

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
        precision_aware_rel_tol,
        solve_helmholtz_implicit,
    )
    _area_eta = grid.area.astype(eta_dtype)
    # Floor the relative-residual tolerance to what the working dtype can
    # reach (f64: 1e-10 default unchanged; f32: raised above ~1.2e-4 since a
    # 1e-10 rel-residual is unreachable below f32 machine epsilon).  Keeps the
    # converged diagnostic meaningful and the stock-CG while_loop terminating.
    _residual_tol = precision_aware_rel_tol(
        config.barotropic.barotropic_implicit_pcg_residual_tol, eta_dtype,
    )
    # ``force_pcg`` selects the fixed-M PCG body even single-rank
    # (solver-matched parity references + the faster-single-rank
    # option, job 8458701); its global dots reduce locally when not
    # multi-process, so the flag is safe pre-arming.
    # The lat-band SPMD backend (single-controller shard_map) is not
    # multi-PROCESS, so _is_distributed() is False — but the stock-CG branch
    # (solve_helmholtz_freesurface -> jax.scipy.sparse.linalg.cg) runs its
    # A_op halo ppermute + CG reductions inside a DATA-DEPENDENT while_loop,
    # whose collectives have no static schedule under shard_map (SIGABRT).
    # Route SPMD to the FIXED-iteration distributed PCG (static scan schedule,
    # SPMD-routed reductions — the same path MPI uses), exactly like the
    # explicit_substep barotropic loop that is already SPMD-validated.
    from legoesm.grids.halo import get_halo_backend as _get_halo_backend
    _spmd_armed = _get_halo_backend() == "spmd"
    _use_pcg = (_is_distributed() or _spmd_armed
                or bool(config.barotropic.barotropic_implicit_force_pcg))
    if not _use_pcg:
        # The stock-CG branch solves with its INTERNAL Jacobi (the
        # custom-VJP solver owns inv_diag for its exact adjoint) — a
        # non-default preconditioner cannot take effect here.  Refuse
        # loudly instead of silently running Jacobi (codex review MAJOR,
        # 2026-06-12): the fixed-M PCG honors it — set
        # barotropic_implicit_force_pcg=True for single-rank runs.
        _precond_req = str(getattr(
            config.barotropic, "barotropic_implicit_preconditioner", "jacobi"))
        if _precond_req != "jacobi":
            raise ValueError(
                "barotropic_implicit_latlon_cgrid: "
                f"barotropic_implicit_preconditioner={_precond_req!r} "
                "only applies to the fixed-M PCG path, but this "
                "single-rank run dispatches the stock-CG solver "
                "(internal Jacobi).  Set "
                "barotropic_implicit_force_pcg=True (PCG is also the "
                "faster single-rank solver, job 8458701) or use "
                "preconditioner='jacobi'."
            )
        # f32: a 1e-10 rel-tol is unreachable, so stock CG would run to
        # maxiter every step — floor it to the f32-reachable value (f64
        # unchanged).
        pcg_tol = precision_aware_rel_tol(
            config.barotropic.barotropic_implicit_pcg_tol, eta_dtype,
        )
        eta_new = solve_helmholtz_freesurface(
            rhs, eta_old, H_u_old, H_v_old, coeff, mask, u_mask, v_mask,
            inv_diag, grid, tol=pcg_tol,
            maxiter=int(config.barotropic.barotropic_implicit_pcg_maxiter),
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
            fixed_iters=int(config.barotropic.barotropic_implicit_pcg_fixed_iters),
            residual_tol=_residual_tol,
            stock_cg_tol=config.barotropic.barotropic_implicit_pcg_tol,
            stock_cg_maxiter=int(config.barotropic.barotropic_implicit_pcg_maxiter),
            pcg_variant=str(config.barotropic.barotropic_implicit_pcg_variant),
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
    if bool(config.runtime_checks.enable_runtime_checks):
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
        batch_psum_spmd as _batch_psum_spmd,
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
    # SPMD-aware batched reduction (SAME dispatch as eta_floor._global_sum_pair /
    # barotropic_common._global_dot_batch -- check "spmd" backend AND a "lat" mesh
    # axis, not just the backend): under the single-controller lat-band shard_map
    # these are PARTIAL sums over this device's latitude band -> sum across "lat"
    # with jax.lax.psum (NOT mpi4jax, not even required on the route-B multi-GPU
    # path).  Checked FIRST because is_multi_process() is False under one process;
    # without it each band would keep its PARTIAL (target_mass, actual_mass, area)
    # and the global mass-conservation correction would be per-band WRONG (this
    # fires every implicit_cn step under SPMD).  The "lat"-in-axis_names guard lets
    # a coupled cube-atm SPMD mesh (backend "spmd", NON-lat axes) fall through to
    # MPI/local instead of crashing on a missing "lat" axis (codex).  psum is
    # self-transposing => AD-safe.
    from legoesm.grids.halo import (
        get_halo_backend as _get_halo_backend, get_spmd_mesh as _get_spmd_mesh,
    )
    _reduced = None
    if _get_halo_backend() == "spmd":
        _mesh = _get_spmd_mesh()
        if _mesh is None:
            raise RuntimeError(
                "barotropic_implicit_latlon_cgrid: halo backend is 'spmd' but no "
                "SPMD mesh is set; arm it via activate_latlon_spmd_halo(mesh).")
        if "lat" in tuple(_mesh.axis_names):
            _reduced = _batch_psum_spmd(
                [_ocean_area_l, _target_mass_l, _actual_mass_l], "lat")
    if _reduced is not None:
        _ocean_area, _target_mass, _actual_mass = _reduced
    elif _is_multi_process():
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
    nmask = north_fold_mask(grid)
    if fold_is_local(grid) or nmask is not None:
        north_3d = h_active_3d[-1:] * h_active_3d[-1:, grid.fold.perm_T, :]
        v_active_3d = apply_north_fold(v_active_3d, north_3d, grid, north_mask=nmask)
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


def solve_unsplit_freesurface(
    eta_old: jnp.ndarray,
    u_star: jnp.ndarray,
    v_star: jnp.ndarray,
    h_u: jnp.ndarray,
    h_v: jnp.ndarray,
    dt: float,
    g: float,
    grid: LatLonGrid,
    mask: jnp.ndarray,
    u_mask: jnp.ndarray,
    v_mask: jnp.ndarray,
    *,
    tol: float = 1.0e-10,
    maxiter: int = 200,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """MITgcm-faithful UNSPLIT implicit free-surface solve + uniform correction.

    No barotropic/baroclinic mode split: given the explicit predictor velocity
    ``(u*, v*)`` on the FULL 3-D field, solve the implicit free surface for
    ``eta^{n+1}`` from the depth-integrated divergence of the predictor transport,
    then apply the SAME (vertically-uniform) surface-pressure-gradient correction
    to every level (MITgcm ``correction_step.F``)::

        eta^{n+1} - dt^2 g div(H grad eta^{n+1}) = eta^n - dt div(int u* dz)
        u^{n+1} = u* - dt g d(eta^{n+1})/dx          (all levels)

    Reuses the SAME area-weighted-adjoint Helmholtz solve as ``implicit_cn``
    (:func:`solve_helmholtz_freesurface`), so the discrete operator and its AD are
    identical.  The fully-implicit (theta=1) backward-Euler surface mode matches
    MITgcm's ``implicSurfPress=1``.  Returns ``(eta_new, u_new, v_new)``.

    The elliptic CG runs in float64 for conditioning (cast back to the eta dtype).
    """
    edt = eta_old.dtype

    def f64(a):
        return jnp.asarray(a, dtype=jnp.float64)

    hu_tot = f64(jnp.maximum(jnp.sum(h_u, axis=-1), 1.0e-10) * u_mask)   # depth-integrated face H
    hv_tot = f64(jnp.maximum(jnp.sum(h_v, axis=-1), 1.0e-10) * v_mask)
    flux_u = jnp.sum(h_u * u_star, axis=-1) * u_mask          # depth-integrated transport
    flux_v = jnp.sum(h_v * v_star, axis=-1) * v_mask
    div = divergence_cgrid(flux_u, flux_v, grid, u_mask=u_mask, v_mask=v_mask)
    coeff = jnp.asarray(dt * dt * g, dtype=jnp.float64)
    rhs = f64((eta_old - dt * div) * mask)
    inv_diag = _helmholtz_inv_diag(hu_tot, hv_tot, coeff, grid, f64(mask))
    eta_new = solve_helmholtz_freesurface(
        rhs, f64(eta_old), hu_tot, hv_tot, coeff, f64(mask), f64(u_mask), f64(v_mask),
        inv_diag, grid, tol=tol, maxiter=maxiter).astype(edt) * mask
    grad_x = gradient_x_cgrid(eta_new, grid)[..., jnp.newaxis]
    grad_y = gradient_y_cgrid(eta_new, grid)[..., jnp.newaxis]
    u_new = (u_star - dt * g * grad_x) * u_mask[..., jnp.newaxis]
    v_new = (v_star - dt * g * grad_y) * v_mask[..., jnp.newaxis]
    return eta_new, u_new, v_new
