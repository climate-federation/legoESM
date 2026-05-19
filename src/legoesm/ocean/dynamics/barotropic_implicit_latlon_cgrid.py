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
through implicit-function-theorem custom-VJP.

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
    divergence_cgrid,
    gradient_x_cgrid,
    gradient_y_cgrid,
    is_tripolar,
)
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
    from legoesm.ocean.dynamics.latlon_cgrid_operators import fold_vface_row

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

    h_v_int = jnp.minimum(h_k[:-1], h_k[1:])
    n_lon = h_k.shape[1]
    nlev = h_k.shape[2]
    south_row = jnp.zeros((1, n_lon, nlev), dtype=h_k.dtype)
    # On tripolar grids, the fold face connects cell (fold_j, i) with
    # its fold partner (fold_j, perm_T[i]).  Face thickness is the min
    # of both sides — identical to the interior min-rule.
    h_k_partner = fold_vface_row(h_k, grid)
    north_row = jnp.minimum(h_k[-1:], h_k_partner)
    h_v = jnp.concatenate([south_row, h_v_int, north_row], axis=0)
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
    from legoesm.ocean.dynamics.latlon_cgrid_operators import fold_vface_row

    H_total = jnp.maximum(jnp.sum(h_k, axis=-1), min_water_col) * mask

    H_u_inner = jnp.minimum(jnp.roll(H_total, 1, axis=1), H_total)
    H_u = jnp.concatenate([H_u_inner, H_u_inner[:, 0:1]], axis=1)

    H_v_int = jnp.minimum(H_total[:-1], H_total[1:])
    n_lon = H_total.shape[1]
    south_row = jnp.zeros((1, n_lon), dtype=H_total.dtype)
    H_total_partner = fold_vface_row(H_total, grid)
    north_row = jnp.minimum(H_total[-1:], H_total_partner)
    H_v = jnp.concatenate([south_row, H_v_int, north_row], axis=0)

    return H_u, H_v


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
    positive-definite (for ``coeff ≥ 0`` and ``H ≥ 0``).
    """
    H_u_face = H_u * u_mask
    H_v_face = H_v * v_mask

    def A_op(eta_in: jnp.ndarray) -> jnp.ndarray:
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

    return A_op


def _make_diag_preconditioner(
    H_u: jnp.ndarray,
    H_v: jnp.ndarray,
    coeff: jnp.ndarray,
    grid: LatLonGrid,
    mask: jnp.ndarray,
):
    """Return ``M(r) = diag⁻¹ · r`` Jacobi preconditioner for the Helmholtz.

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
    inv_diag = jnp.where(mask > 0.5, 1.0 / jnp.maximum(diag, 1.0e-30), 0.0)

    def M_inv(r: jnp.ndarray) -> jnp.ndarray:
        return r * inv_diag

    return M_inv


def barotropic_implicit_latlon_cgrid(
    state: LatLonCGridOceanState,
    dt: float,
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    config: LatLonCGridOceanConfig,
    F_slow_eta: jnp.ndarray | None = None,
    F_slow_u: jnp.ndarray | None = None,
    F_slow_v: jnp.ndarray | None = None,
) -> tuple[LatLonCGridOceanState, tuple[jnp.ndarray, jnp.ndarray]]:
    """Single-step implicit free-surface solver (lat-lon C-grid).

    Returns
    -------
    (state_new, (Hu_avg, Hv_avg))
        Same return signature as ``barotropic_substeps_latlon_cgrid``.

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

    # U_pred at v-points (4-pt average) for FB Coriolis on V
    from legoesm.ocean.dynamics.latlon_cgrid_operators import pad_ns_vector_u
    U_pred_at_v_int = 0.25 * (
        U_pred[:-1, :-1] + U_pred[:-1, 1:]
        + U_pred[1:, :-1] + U_pred[1:, 1:]
    )
    U_pred_at_v = pad_ns_vector_u(U_pred_at_v_int, grid)

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

    # Helmholtz operator + Jacobi preconditioner
    A_op = _make_helmholtz(
        H_u_old, H_v_old, coeff, grid, mask, u_mask, v_mask,
    )
    M_inv = _make_diag_preconditioner(
        H_u_old, H_v_old, coeff, grid, mask,
    )

    # Solve.  Tolerance is on relative residual; 1e-10 is plenty for our
    # mass-conservation needs (||A·η - rhs|| < 1e-10 implies mass error
    # per step is well below floating-point precision of the integral).
    pcg_tol = jnp.asarray(config.barotropic_implicit_pcg_tol, dtype=eta_dtype)
    pcg_maxiter = int(config.barotropic_implicit_pcg_maxiter)
    eta_new, _info = jax.scipy.sparse.linalg.cg(
        A_op, rhs, x0=eta_old, tol=pcg_tol, maxiter=pcg_maxiter, M=M_inv,
    )
    eta_new = eta_new * mask

    # Global mass conservation correction.  The CG solve minimizes the
    # L2 residual but does not guarantee that sum(r * area) = 0 — the
    # area-weighted integral of the residual can have a small nonzero
    # bias that accumulates over 10⁵-10⁶ steps.  Project out the global
    # mean drift so that sum(eta_new * area) = sum(rhs * area) exactly.
    # This is standard practice (MOM6, NEMO, MITgcm).
    _area_eta = grid.area.astype(eta_dtype)
    _ocean_area = jnp.sum(_area_eta * mask)
    _target_mass = jnp.sum(rhs * _area_eta)
    _actual_mass = jnp.sum(eta_new * _area_eta)
    _correction = (_target_mass - _actual_mass) / jnp.maximum(_ocean_area, 1e-30)
    eta_new = (eta_new + _correction * mask) * mask

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
    v_active_3d_int = h_active_3d[:-1] * h_active_3d[1:]
    fold = getattr(grid, "fold", None)
    if fold is not None and fold.is_active:
        south_3d = jnp.zeros_like(v_active_3d_int[:1])
        north_3d = h_active_3d[-1:] * h_active_3d[-1:, fold.perm_T, :]
        v_active_3d = jnp.concatenate(
            [south_3d, v_active_3d_int, north_3d], axis=0,
        )
    else:
        n_lon_grid = h_active_3d.shape[1]
        nlev_g = h_active_3d.shape[2]
        zero_row_3d = jnp.zeros(
            (1, n_lon_grid, nlev_g), dtype=u_3d.dtype,
        )
        v_active_3d = jnp.concatenate(
            [zero_row_3d, v_active_3d_int, zero_row_3d], axis=0,
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
    return state_new, (Hu_avg, Hv_avg)
