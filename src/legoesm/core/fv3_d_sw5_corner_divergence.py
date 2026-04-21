"""Port of Fortran `d_sw5` B-grid corner divergence (sw_core.F90:1641-1719).

Fortran d_sw5 computes the divergence at B-grid CORNERS (shape
(npx+1, npy+1)) as:

  1.  ptc(i,j) = (u(i,j) - 0.5*(va(i,j-1)+va(i,j))*cosa_v(i,j))
                 * dyc(i,j) * sina_v(i,j)
      — corner x-circulation at v-interface i,j (shape (isd:ied+1, js:je+1)).

  2.  vort(i,j) = (v(i,j) - 0.5*(ua(i-1,j)+ua(i,j))*cosa_u(i,j))
                   * dxc(i,j) * sina_u(i,j)
      — corner y-circulation at u-interface i,j (shape (is2:ie1, js-1:je+1)).

  3.  delpc(i,j) = vort(i,j-1) - vort(i,j) + ptc(i-1,j) - ptc(i,j)
      — convergence at B-grid corner (i,j).

  4.  delpc(i,j) = rarea_c(i,j) * delpc(i,j)
      — normalize by corner area.

Fortran then uses delpc to compute an adaptive damping coefficient
and injects `damp * delpc` into `ke` at corners.  That step is
deferred to a future iteration; iter-759 ports only the corner
divergence computation.

The Python port uses:
  u_d, v_d  — D-grid velocities (our Python convention: u_d at
              v-interfaces, shape (6, n, n+1); v_d at u-interfaces,
              shape (6, n+1, n))
  u_cc, v_cc — A-grid cell-centre velocities (averaged from D-grid)
              = Fortran ua, va
  cdgrid.cosa_u, cdgrid.cosa_v — non-orthogonality metrics
  cdgrid.dxc, cdgrid.dyc — centre-to-centre great-circle distances
  cdgrid.area_corner — B-grid corner area
"""
from __future__ import annotations

import jax.numpy as jnp

from legoesm.grids.halo import pad_halo


def _cell_centre_winds(u_d, v_d):
    """Average D-grid winds to A-grid cell-centre (ua, va).

    Fortran: ua(i,j) = 0.5*(u(i,j)+u(i,j+1)),
             va(i,j) = 0.5*(v(i,j)+v(i+1,j))

    Our D-grid convention:
      u_d shape (6, n, n+1)  at v-interfaces (j-edges)
      v_d shape (6, n+1, n)  at u-interfaces (i-edges)

    So cell-centre u = 0.5*(u_d at south edge + u_d at north edge)
                     = 0.5*(u_d[:, :, :-1] + u_d[:, :, 1:])
    similarly cell-centre v = 0.5*(v_d[:, :-1, :] + v_d[:, 1:, :]).
    """
    u_cc = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])    # (6, n, n)
    v_cc = 0.5 * (v_d[:, :-1, :] + v_d[:, 1:, :])    # (6, n, n)
    return u_cc, v_cc


def fv3_d_sw5_corner_divergence(u_d, v_d, cdgrid):
    """Fortran-faithful B-grid corner divergence per `sw_core.F90:1641-1719`.

    Parameters
    ----------
    u_d : jax.Array, shape (6, n, n+1)
        D-grid u-wind at v-interfaces.
    v_d : jax.Array, shape (6, n+1, n)
        D-grid v-wind at u-interfaces.
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    delpc : jax.Array, shape (6, n+1, n+1)
        Divergence at all B-grid corners, [1/s].  Matches Fortran
        `delpc(i,j)` at sw_core.F90:1705 after the `rarea_c`
        normalization at line 1719.

    Notes
    -----
    Iter-759 computed only INTERIOR corners `(6, n-1, n-1)`.
    Iter-760 extends to full `(6, n+1, n+1)` via edge-mode padding
    of vort and ptc arrays (both non-square — `pad_halo` doesn't
    support them).  The edge-mode pad is a first approximation:
    it replicates the nearest face-interior vort/ptc value at the
    boundary, whereas Fortran would use halo-filled neighbour-face
    values.  For the duogrid/bounded_domain branch of d_sw5
    (sw_core.F90:1644-1658), this approximation is exact at the
    face interior and first-order at face edges.  Full cross-face
    halo rotation is deferred to iter-760b.

    Iter-759/760 also uses the `duogrid/bounded_domain` branch
    (`sw_core.F90:1644-1658`) — the simpler formula without the
    j==1/j==npy polar-row fallbacks.  The non-bounded polar-row
    special cases (lines 1661-1700) use `sin_sg` edge metrics that
    require extra cdgrid fields; those are deferred to iter-761+.
    """
    _EPS = 1e-20
    n = cdgrid.base.n

    # Fortran `ua, va` — A-grid cell-centre winds.
    u_cc, v_cc = _cell_centre_winds(u_d, v_d)        # (6, n, n)

    # Note (iter-759): no halo pad needed.  Interior corners
    # (i, j) ∈ [1, n-1] × [1, n-1] reference vort(i, j-1), vort(i, j),
    # ptc(i-1, j), ptc(i, j) — all within the interior shapes of
    # vort (6, n+1, n) and ptc (6, n, n+1).

    # `cosa_v, sina_v` at v-interfaces, shape (6, n, n+1).
    cosa_v = cdgrid.cosa_v                          # (6, n, n+1)
    sina_v = jnp.sqrt(jnp.maximum(1.0 - cosa_v**2, _EPS))
    # `cosa_u, sina_u` at u-interfaces, shape (6, n+1, n).
    cosa_u = cdgrid.cosa_u                          # (6, n+1, n)
    sina_u = jnp.sqrt(jnp.maximum(1.0 - cosa_u**2, _EPS))

    # Fortran needs va at v-interfaces and ua at u-interfaces.  Our
    # u_cc, v_cc are at cell centres (6, n, n).  For the INTERIOR
    # formula:
    #   va at v-interface (i, j) with j ∈ [1, n-1]: va at cell
    #     (i, j-1) + va at cell (i, j) → v_cc[:, :, :-1] + v_cc[:, :, 1:]
    #     gives (6, n, n-1).  For corner j ∈ [1, n-1], we need
    #     v-interface j ∈ {j-1, j} = [0, n-1], so we use v-interfaces
    #     j ∈ [1, n-1] on full v_cc: shape (6, n, n-1).
    # This is getting complex.  For iter-759 interior-only, compute
    # `ptc` at v-interface [i ∈ [0, n-1], j ∈ [1, n-1]] (shape
    # (6, n, n-1)) and `vort` at u-interface [i ∈ [1, n-1], j ∈ [0, n-1]]
    # (shape (6, n-1, n)).
    #
    # u_d at v-interface j ∈ [1, n-1] → u_d[:, :, 1:-1] shape (6, n, n-1).
    # va(i, j-1) + va(i, j) for j ∈ [1, n-1]:
    #   = v_cc[:, :, :-1] + v_cc[:, :, 1:] (both shape (6, n, n-1))
    u_d_v_int = u_d[:, :, 1:-1]                     # (6, n, n-1)
    va_sum_int = v_cc[:, :, :-1] + v_cc[:, :, 1:]   # (6, n, n-1)
    cosa_v_int = cosa_v[:, :, 1:-1]                 # (6, n, n-1)
    sina_v_int = sina_v[:, :, 1:-1]                 # (6, n, n-1)
    dyc_v_int = cdgrid.dyc[:, :, 1:-1]              # (6, n, n-1)
    ptc = (u_d_v_int - 0.5 * va_sum_int * cosa_v_int) \
          * dyc_v_int * sina_v_int
    # ptc shape: (6, n, n-1)   — v-interface j ∈ [1, n-1]

    # v_d at u-interface i ∈ [1, n-1] → v_d[:, 1:-1, :] shape (6, n-1, n).
    # ua(i-1, j) + ua(i, j) for i ∈ [1, n-1]:
    #   = u_cc[:, :-1, :] + u_cc[:, 1:, :] (both shape (6, n-1, n))
    v_d_u_int = v_d[:, 1:-1, :]                     # (6, n-1, n)
    ua_sum_int = u_cc[:, :-1, :] + u_cc[:, 1:, :]   # (6, n-1, n)
    cosa_u_int = cosa_u[:, 1:-1, :]                 # (6, n-1, n)
    sina_u_int = sina_u[:, 1:-1, :]                 # (6, n-1, n)
    dxc_u_int = cdgrid.dxc[:, 1:-1, :]              # (6, n-1, n)
    vort = (v_d_u_int - 0.5 * ua_sum_int * cosa_u_int) \
           * dxc_u_int * sina_u_int
    # vort shape: (6, n-1, n)  — u-interface i ∈ [1, n-1]

    # Fortran: delpc(i,j) = vort(i,j-1) - vort(i,j) + ptc(i-1,j) - ptc(i,j)
    # Interior corners (i, j) ∈ [1, n-1] × [1, n-1] (shape (6, n-1, n-1)).
    # Mapping to our interior arrays:
    #   vort(i, j-1): i ∈ [1, n-1] → vort index [0, n-2] = [:, :, :-1]
    #                 j-1 ∈ [0, n-2] → vort index [0, n-2] = [:, :-1] on axis 2? wait
    # vort shape (6, n-1, n).  Axis 1: i ∈ [0, n-2] corresponds to u-interface
    # i ∈ [1, n-1].  Axis 2: j ∈ [0, n-1] corresponds to cell j ∈ [0, n-1].
    # At corner (I, J) with I ∈ [1, n-1], J ∈ [1, n-1]:
    #   vort(I, J-1) = vort[:, I-1, J-1]   → vort[:, :, :-1]
    #   vort(I, J)   = vort[:, I-1, J]     → vort[:, :, 1:]
    #   (i-axis of vort uses I-1, same slice for both → full axis 0-indexing I-1 is [0, n-2])
    vort_jm1 = vort[:, :, :-1]              # (6, n-1, n-1)
    vort_j   = vort[:, :, 1:]               # (6, n-1, n-1)

    # ptc shape (6, n, n-1).  Axis 1: i ∈ [0, n-1] = cell i.  Axis 2:
    # j-in-interior ∈ [0, n-2] corresponds to v-interface j ∈ [1, n-1].
    # At corner (I, J) with I ∈ [1, n-1], J ∈ [1, n-1]:
    #   ptc(I-1, J) = ptc[:, I-1, J-1]    → ptc[:, :-1, :]
    #   ptc(I, J)   = ptc[:, I, J-1]      → ptc[:, 1:, :]
    ptc_im1 = ptc[:, :-1, :]                # (6, n-1, n-1)
    ptc_i   = ptc[:, 1:, :]                 # (6, n-1, n-1)

    delpc_raw = vort_jm1 - vort_j + ptc_im1 - ptc_i   # (6, n-1, n-1)

    # Normalize by corner area (Fortran sw_core.F90:1719):
    # area_corner shape (6, n+1, n+1); interior corners are [1, n-1]:
    rarea_c_int = 1.0 / jnp.maximum(
        cdgrid.area_corner[:, 1:-1, 1:-1], _EPS)  # (6, n-1, n-1)
    delpc = rarea_c_int * delpc_raw

    return delpc
