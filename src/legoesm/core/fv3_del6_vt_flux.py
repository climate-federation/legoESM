"""Port of Fortran `del6_vt_flux` (sw_core.F90:2008-2121).

Del-nord damping for the relative vorticity, flux-form, metric-aware.
Applied after d_sw6 wind assembly in the FV3 Fortran forward-backward
chain.  The returned edge fluxes are added to u, v as circulation
increments: `u(i,j) += vt(i,j); v(i,j) -= ut(i,j)` where vt, ut are
derived from the fx, fy outputs.

Iter-752 delivers the core standalone algorithm.
Iter-753 adds the high-level `fv3_del6_vorticity_damping` helper
that combines circulation computation + vorticity + del-n flux + the
Fortran-faithful application rule `u += fy2; v -= fx2`.

Wiring into `fv3_sw_tendencies` comes iter-754+.
"""
from __future__ import annotations

import jax.numpy as jnp
import jax

from legoesm.grids.halo import pad_halo


def compute_del6_metrics(cdgrid):
    """Compute Fortran `del6_u, del6_v` metric coefficients.

    Per `fv_grid_utils.F90:713, 725`:
        del6_u(i,j) = sina_v(i,j) * dx(i,j)  / dyc(i,j)
        del6_v(i,j) = sina_u(i,j) * dy(i,j)  / dxc(i,j)

    These are the non-USE_SG form.  USE_SG uses `sin_sg` edge
    averages, which the duo-grid path would use — not needed for the
    initial port.

    Parameters
    ----------
    cdgrid : CubedSphereCDGrid
        Must expose `cosa_u, cosa_v` (cell-interface non-orthogonality
        metrics) and `base.dx, base.dy`.

    Returns
    -------
    del6_u, del6_v : jax.Array
        del6_u shape (6, n+1, n)  at v-edge positions
        del6_v shape (6, n, n+1)  at u-edge positions
    """
    _EPS = 1e-20
    # sina_u = sqrt(1 - cosa_u**2); cosa_u is at u-positions (y-edges).
    cosa_u = cdgrid.cosa_u  # (6, n+1, n)
    cosa_v = cdgrid.cosa_v  # (6, n, n+1)
    sina_u = jnp.sqrt(jnp.maximum(1.0 - cosa_u**2, _EPS))
    sina_v = jnp.sqrt(jnp.maximum(1.0 - cosa_v**2, _EPS))

    # Fortran's `dx(i,j)` and `dy(i,j)` are at EDGE positions, not
    # cell centres.  In our cdgrid convention these are stored as
    # `dx_edge_y` (length of horizontal edge at v-interface,
    # shape (6, n, n+1)) and `dy_edge_x` (length of vertical edge at
    # u-interface, shape (6, n+1, n)) — matching Fortran's
    # (isd:ied, jsd:jed+1) and (isd:ied+1, jsd:jed) shapes.
    dx_v = cdgrid.dx_edge_y   # (6, n, n+1)   at v-interface
    dy_u = cdgrid.dy_edge_x   # (6, n+1, n)   at u-interface

    # `dxc, dyc` are Fortran's cell-centre-to-cell-centre distances.
    # cdgrid ALREADY stores these as first-class fields, computed from
    # the FV3 supergrid per `fv_grid_tools.F90:883`:
    #   dxc(i,j) = 2 * great_circle_dist(edge_midpoint, cell_center)
    # Shapes match Fortran's (isd:ied+1, jsd:jed) and
    # (isd:ied, jsd:jed+1) conventions:
    #   cdgrid.dxc : shape (6, n+1, n)  at u-interface
    #   cdgrid.dyc : shape (6, n, n+1)  at v-interface
    # Using the locked cdgrid fields keeps metric consistency with the
    # rest of the CDGrid operators (discrete Stokes theorem etc).
    dxc_at_u = cdgrid.dxc     # (6, n+1, n)
    dyc_at_v = cdgrid.dyc     # (6, n, n+1)

    # Fortran formula (fv_grid_utils.F90:713, 725):
    #   del6_u(i,j) = sina_v(i,j) * dx(i,j) / dyc(i,j)    at v-interface
    #   del6_v(i,j) = sina_u(i,j) * dy(i,j) / dxc(i,j)    at u-interface
    del6_u = sina_v * dx_v / dyc_at_v   # (6, n, n+1)
    del6_v = sina_u * dy_u / dxc_at_u   # (6, n+1, n)

    return del6_u, del6_v


def _del6_vt_flux(q, damp, nord, del6_u, del6_v, rarea, cdgrid):
    """Fortran `del6_vt_flux` port (sw_core.F90:2008-2121).

    Del-nord damping for the relative vorticity.  Returns edge
    diffusive fluxes `fx2, fy2` that caller adds to u, v as
    circulation increments.

    Parameters
    ----------
    q : jax.Array, shape (6, n, n)
        Cell-mean relative vorticity.  Should be halo-exchanged by
        caller if nord > 0.
    damp : float
        Damping coefficient.  In Fortran: `damp = (damp_v * da_min_c)^(nord+1)`.
    nord : int
        0 = del-2, 1 = del-4, 2 = del-6.
    del6_u : jax.Array, shape (6, n, n+1)
        Metric coefficient at v-edge positions.
    del6_v : jax.Array, shape (6, n+1, n)
        Metric coefficient at u-edge positions.
    rarea : jax.Array, shape (6, n, n)
        1/cell_area (Fortran `gridstruct%rarea`).
    cdgrid : CubedSphereCDGrid
        For halo exchange of intermediate d2.

    Returns
    -------
    fx2 : jax.Array, shape (6, n+1, n)  — diffusive flux in x-direction
    fy2 : jax.Array, shape (6, n, n+1)  — diffusive flux in y-direction

    Notes
    -----
    In Fortran the output shapes are (isd:ied+1, jsd:jed) and
    (isd:ied, jsd:jed+1); the extra +1 is the "east-edge" column.
    In our Python convention with u-edges at (n+1, n) and v-edges at
    (n, n+1), the shapes align with del6_v, del6_u respectively.
    """
    if nord not in (0, 1, 2):
        raise ValueError(f"nord must be 0, 1, or 2; got {nord}")
    if q.ndim not in (3, 4):
        raise ValueError(
            f"_del6_vt_flux expects q.ndim ∈ {{3, 4}}; "
            f"got ndim={q.ndim}, shape={tuple(q.shape)}."
        )

    _EPS = 1e-20
    dg = getattr(cdgrid.base, 'duogrid', None)
    offsets = None if dg is not None else cdgrid.base.halo_interp_offsets

    # FV3_3D iter-1045: ndim-aware halo dispatch.  For 4D ``q`` use
    # ``pad_halo_4d`` (one batched MPI sendrecv per call); for 3D use
    # the legacy ``pad_halo``.  3D static metrics ``del6_u``, ``del6_v``,
    # ``rarea`` get a trailing ``[..., None]`` axis for broadcasting
    # against 4D data.  Closes the last documented vmap-around-
    # ``pad_halo`` site in the NH compressible-Euler 3D path
    # (``damp_v`` / ``damp_w`` post-step at compressible_euler_cdgrid.py:
    # 1197 / 1389).
    _is_4d = q.ndim == 4
    if _is_4d:
        from legoesm.grids.halo import pad_halo_4d as _pad_halo_4d

        def pad_scalar(field):
            return _pad_halo_4d(field, interp_offsets=offsets, duogrid=dg)

        del6_u_b = del6_u[..., None]
        del6_v_b = del6_v[..., None]
        rarea_b = rarea[..., None]
    else:
        def pad_scalar(field):
            return pad_halo(field, interp_offsets=offsets, duogrid=dg)

        del6_u_b = del6_u
        del6_v_b = del6_v
        rarea_b = rarea

    # Step 1: initial damping.
    #
    # Iter-937 (sibling fix to iter-934's `_deln_flux` change): factor
    # `damp` out of the iteration and apply at the final flux-output
    # stage instead of multiplying it into `d2` here.  All operations
    # between this point and the final return are LINEAR in d2, so
    # the result is mathematically identical, but the intermediate
    # `d2`/`fx2`/`fy2` arrays no longer carry the huge `damp` factor
    # which scales as `(damp_v * area)^(nord+1)`.  At low resolution
    # the un-factored intermediates overflow float32 (same class as
    # iter-934).  Production at C36 is bit-identical because float64
    # arithmetic is unchanged and float32 doesn't overflow at C36
    # for the typical vorticity scale.
    d2 = q                                           # (6, n, n)

    # First flux computation (n=0 pre-loop in Fortran).  Note sign:
    # Fortran: fx2 = del6_v * (d2(i-1,j) - d2(i,j)) initially,
    #          then in the nord>0 loop: fx2 = del6_v * (d2(i,j) - d2(i-1,j)).
    # We follow the same convention.
    d2_pad = pad_scalar(d2)                          # (6, n+2, n+2)
    # fx2 at u-edge (6, n+1, n): d2[west] - d2[east]
    # In padded coords, d2 cell at (face, i, j) corresponds to
    # d2_pad[face, i+1, j+1].  Left neighbour at (i-1, j) is
    # d2_pad[face, i, j+1].  Right neighbour at (i+1, j) is
    # d2_pad[face, i+2, j+1].
    # u-edge positions are at (i+0.5, j) for i in [0, n], j in [0, n-1].
    # fx2[i, j] = del6_v[i, j] * (d2[i-1, j] - d2[i, j])
    #          = del6_v[i, j] * (d2_pad[i, j+1] - d2_pad[i+1, j+1])  # where i ranges 0..n, so n+1 values
    d2_west = d2_pad[:, :-1, 1:-1]   # (6, n+1, n[, nlev])
    d2_east = d2_pad[:, 1:, 1:-1]
    fx2 = del6_v_b * (d2_west - d2_east)

    # fy2 at v-edge (6, n, n+1): d2[south] - d2[north] convention.
    # fy2[i, j] = del6_u[i, j] * (d2[i, j-1] - d2[i, j])
    d2_south = d2_pad[:, 1:-1, :-1]  # (6, n, n+1[, nlev])
    d2_north = d2_pad[:, 1:-1, 1:]
    fy2 = del6_u_b * (d2_south - d2_north)

    # Iterate the del-n operator.
    # nord=0: return fx2, fy2 as-is (del-2).
    # nord>0: loop nord times.
    for n in range(nord):
        # d2 = rarea * (fx2 - fx2_east + fy2 - fy2_north)
        # fx2 shape (6, n+1, n): fx2[i, j] and fx2[i+1, j] at each cell.
        # For cell (i, j), fx2 contribution is fx2[i, j] - fx2[i+1, j].
        fx2_west = fx2[:, :-1, :]
        fx2_east_contrib = fx2[:, 1:, :]
        fy2_south = fy2[:, :, :-1]
        fy2_north_contrib = fy2[:, :, 1:]
        d2 = rarea_b * (fx2_west - fx2_east_contrib
                        + fy2_south - fy2_north_contrib)

        # Halo exchange d2.
        d2_pad = pad_scalar(d2)

        # Recompute fx2, fy2 with SIGN FLIPPED (note Fortran line 2099
        # has `d2(i,j) - d2(i-1,j)` whereas pre-loop has
        # `d2(i-1,j) - d2(i,j)`).
        d2_west = d2_pad[:, :-1, 1:-1]
        d2_east = d2_pad[:, 1:, 1:-1]
        fx2 = del6_v_b * (d2_east - d2_west)           # sign flipped

        d2_south = d2_pad[:, 1:-1, :-1]
        d2_north = d2_pad[:, 1:-1, 1:]
        fy2 = del6_u_b * (d2_north - d2_south)         # sign flipped

    # Iter-937: apply the deferred `damp` factor at the final output
    # stage.  See Step 1 comment for the float32-overflow rationale.
    fx2 = damp * fx2
    fy2 = damp * fy2

    return fx2, fy2


def fv3_del6_vorticity_damping(u_d, v_d, damp, nord, cdgrid):
    """Fortran-faithful del-n vorticity damping on D-grid winds.

    Combines the full Fortran d_sw6 damping sequence (sw_core.F90:
    1582-1597, 1948-1999) into a single self-contained step:

    1. Circulation:  `vt = u * dx_at_v`, `ut = v * dy_at_u`.
    2. Cell-mean vorticity:
         `wk = rarea * (vt(i,j) - vt(i,j+1) - ut(i,j) + ut(i+1,j))`
    3. Del-n flux: `(fx2, fy2) = _del6_vt_flux(wk, damp, nord, ...)`
    4. Convert circulation-form damping to velocity form:
         `du_d = +fy2 / dx_edge_y`   (Fortran: u += fy2, but Fortran u
                                      is in circulation at that point)
         `dv_d = -fx2 / dy_edge_x`   (Fortran: v -= fx2, circulation)

    Fortran performs `u(i,j) += vt(i,j)` while `u` is in CIRCULATION
    form (u*dx, units [m²/s]) inside d_sw; the final conversion to
    velocity happens via `*rdx` later.  Our Python `u_d, v_d` are in
    VELOCITY form, so step 4 applies the `*rdx = 1/dx_edge_y` and
    `*rdy = 1/dy_edge_x` conversions to return velocity-form updates
    directly.  This matches Fortran's `rdx, rdy` metric convention
    (`fv_grid_utils.F90`) and keeps the helper's output unit contract
    consistent with the velocity-form input.

    Parameters
    ----------
    u_d : jax.Array, shape (6, n, n+1)
        D-grid u-wind at v-edge positions, VELOCITY form [m/s].
    v_d : jax.Array, shape (6, n+1, n)
        D-grid v-wind at u-edge positions, VELOCITY form [m/s].
    damp : float
        Fortran `damp4 = (damp_v * da_min_c)^(nord+1)`.  Units
        `[m^(2*(nord+1))]`.  Caller must supply with correct scaling.
    nord : int
        0 = del-2, 1 = del-4, 2 = del-6.
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    du_d_damping : jax.Array, shape (6, n, n+1)
        Velocity-unit update for u_d [m/s].  Caller:
        `u_d_new = u_d + du_d`.
    dv_d_damping : jax.Array, shape (6, n+1, n)
        Velocity-unit update for v_d [m/s].  Caller:
        `v_d_new = v_d + dv_d` (Fortran sign `v -= fx2` already
        baked in via the `-fx2 / dy_edge_x` return).

    Notes
    -----
    These are PER-TIMESTEP velocity updates (Fortran convention), not
    continuous tendencies.  To use with a tendency-based integrator,
    caller should divide by dt.

    Unit check (for nord=2, del-6):
      damp has units [m^6].
      q = wk has units [1/s].
      After nord+1=3 iterations of del-n with dimensionless del6_u,v
      and rarea [1/m²]: final fx2/fy2 have units [m²/s] (circulation).
      fy2 / dx_edge_y = [m²/s] / [m] = [m/s] — velocity unit.
    """
    if u_d.ndim not in (3, 4) or v_d.ndim != u_d.ndim:
        raise ValueError(
            f"fv3_del6_vorticity_damping: u_d and v_d must share "
            f"ndim ∈ {{3, 4}}; got u_d.ndim={u_d.ndim}, "
            f"v_d.ndim={v_d.ndim}."
        )

    del6_u_m, del6_v_m = compute_del6_metrics(cdgrid)
    rarea = 1.0 / cdgrid.base.area

    # FV3_3D iter-1045: ndim-aware broadcasting of 3D static metrics
    # against 4D dynamic data.  ``dx_edge_y`` and ``dy_edge_x`` are
    # 3D; multiply against per-level (u_d, v_d) via ``[..., None]``.
    _is_4d = u_d.ndim == 4
    if _is_4d:
        dx_edge_y_b = cdgrid.dx_edge_y[..., None]
        dy_edge_x_b = cdgrid.dy_edge_x[..., None]
        rarea_b = rarea[..., None]
    else:
        dx_edge_y_b = cdgrid.dx_edge_y
        dy_edge_x_b = cdgrid.dy_edge_x
        rarea_b = rarea

    # Step 1: circulation.
    vt = u_d * dx_edge_y_b
    ut = v_d * dy_edge_x_b

    # Step 2: cell-mean vorticity wk.
    vt_south = vt[:, :, :-1]
    vt_north = vt[:, :, 1:]
    ut_west  = ut[:, :-1, :]
    ut_east  = ut[:, 1:, :]
    wk = rarea_b * (vt_south - vt_north - ut_west + ut_east)

    # Step 3: del-n flux.  ``_del6_vt_flux`` is iter-1045 ndim-aware.
    fx2, fy2 = _del6_vt_flux(
        wk, damp, nord,
        del6_u=del6_u_m, del6_v=del6_v_m,
        rarea=rarea, cdgrid=cdgrid,
    )

    # Step 4: convert circulation to velocity (Fortran rdx, rdy).
    du_d_damping = fy2 / dx_edge_y_b
    dv_d_damping = -fx2 / dy_edge_x_b

    return du_d_damping, dv_d_damping
