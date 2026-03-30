"""FV3-style D-grid to A-grid to C-grid vector conversion.

4th-order Lagrange interpolation in the interior, falling back to
2nd-order near face boundaries.  Includes contravariant conversion
at cell centres and C-grid interfaces.

Shared by all cubed-sphere dycores (SW, PE, NH, ocean).

References
----------
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core
- Harris et al. (2021): Scientific Description of GFDL FV3
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.grids.cubed_sphere_cdgrid import CubedSphereCDGrid
from legoesm.grids.halo import pad_halo_vector

# 4th-order Lagrange coefficients
_A1 = 9.0 / 16.0    # weight for nearest two points
_A2 = -1.0 / 16.0   # weight for next two points


def d2a2c_vect(u_d, v_d, cdgrid):
    """Edge-midpoint D-grid → A-grid (cell-centre) → C-grid conversion.

    **Interior** (>= 2 cells from face boundary): 4th-order Lagrange
    interpolation from edge midpoints to cell centres.

    **Near boundary** (< 2 cells): 2nd-order average (same as fv3_d2cc).

    **A-grid contravariant** at cell centres uses the non-orthogonality
    correction ``(u - v*cosa) * rsin2``.

    **C-grid** velocities are obtained by haloing the A-grid winds and
    averaging to face positions with non-orthogonality projection.

    Parameters
    ----------
    u_d : (6, n, n+1) — D-grid x-velocity at x-edge midpoints
    v_d : (6, n+1, n) — D-grid y-velocity at y-edge midpoints
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    ua : (6, n, n) — contravariant x-velocity at cell centres
    va : (6, n, n) — contravariant y-velocity at cell centres
    uc : (6, n+1, n) — covariant x-velocity at x-faces (C-grid)
    vc : (6, n, n+1) — covariant y-velocity at y-faces (C-grid)
    """
    n = cdgrid.n
    grid = cdgrid.base

    # ==================================================================
    # Step 1: D-grid → A-grid (cell-centre covariant components)
    # ==================================================================
    # u_d (6, n, n+1): u at midpoint between corners (i,j) and (i+1,j).
    # Cell centre (i, j) is flanked by u_d[:, i, j] and u_d[:, i, j+1].
    # 4th-order needs u_d[:, i, j-1], u_d[:, i, j], u_d[:, i, j+1], u_d[:, i, j+2].

    # For cells far enough from j-boundaries (j >= 1 and j <= n-2),
    # use 4th-order Lagrange.  For j=0 and j=n-1, fall back to 2-point.
    #
    # 2-point average for all cells (baseline)
    utmp = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])      # (6, n, n)

    # 4th-order correction for interior cells (j = 1..n-2)
    if n >= 4:
        utmp_4 = (_A1 * (u_d[:, :, 1:-2] + u_d[:, :, 2:-1])
                  + _A2 * (u_d[:, :, :-3] + u_d[:, :, 3:]))  # (6, n, n-2)
        utmp = utmp.at[:, :, 1:-1].set(utmp_4)

    # v_d (6, n+1, n): v at midpoint between corners (i,j) and (i,j+1).
    # Cell centre is flanked by v_d[:, i, j] and v_d[:, i+1, j].
    vtmp = 0.5 * (v_d[:, :-1, :] + v_d[:, 1:, :])      # (6, n, n)

    if n >= 4:
        vtmp_4 = (_A1 * (v_d[:, 1:-2, :] + v_d[:, 2:-1, :])
                  + _A2 * (v_d[:, :-3, :] + v_d[:, 3:, :]))  # (6, n-2, n)
        vtmp = vtmp.at[:, 1:-1, :].set(vtmp_4)

    # ==================================================================
    # Step 2: Contravariant at cell centres
    # ==================================================================
    cosa_s = cdgrid.cosa_cell    # (6, n, n)
    rsin2 = cdgrid.rsin2_cell    # (6, n, n)

    ua = (utmp - vtmp * cosa_s) * rsin2
    va = (vtmp - utmp * cosa_s) * rsin2

    # ==================================================================
    # Step 3: A-grid → C-grid (covariant at faces)
    # ==================================================================
    # Halo exchange of cell-centre winds (geographic rotation at faces)
    u_pad, v_pad = pad_halo_vector(
        utmp, vtmp,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=grid.halo_interp_offsets,
    )   # each (6, n+2, n+2)

    # C-grid covariant u at x-faces (6, n+1, n)
    # Average in i-direction, interior j
    uc = 0.5 * (u_pad[:, :-1, 1:-1] + u_pad[:, 1:, 1:-1])

    # 4th-order correction in i for interior faces (i = 2..n-2)
    if n >= 5:
        uc_4 = (_A1 * (u_pad[:, 1:-2, 1:-1] + u_pad[:, 2:-1, 1:-1])
                + _A2 * (u_pad[:, :-3, 1:-1] + u_pad[:, 3:, 1:-1]))
        # Interior i-faces: padded indices 2..n → output indices 1..n-1
        uc = uc.at[:, 1:-1, :].set(uc_4)

    # C-grid covariant v at y-faces (6, n, n+1)
    vc = 0.5 * (v_pad[:, 1:-1, :-1] + v_pad[:, 1:-1, 1:])

    if n >= 5:
        vc_4 = (_A1 * (v_pad[:, 1:-1, 1:-2] + v_pad[:, 1:-1, 2:-1])
                + _A2 * (v_pad[:, 1:-1, :-3] + v_pad[:, 1:-1, 3:]))
        vc = vc.at[:, :, 1:-1].set(vc_4)

    return ua, va, uc, vc


def d2a2c_vect_full(u_d, v_d, cdgrid):
    """Same as :func:`d2a2c_vect` but also returns contravariant C-grid
    winds (``ut``, ``vt``) with non-orthogonality projection, and the
    covariant A-grid winds (``utmp``, ``vtmp``).

    Returns
    -------
    ua, va : (6, n, n) — contravariant A-grid
    uc, vc : (6, n+1, n), (6, n, n+1) — covariant C-grid
    ut, vt : (6, n+1, n), (6, n, n+1) — contravariant C-grid
    utmp, vtmp : (6, n, n) — covariant A-grid
    """
    n = cdgrid.n
    ua, va, uc, vc = d2a2c_vect(u_d, v_d, cdgrid)

    # Contravariant C-grid: project out the cross-component
    # ut at x-faces: (uc - avg_vc*cosa_u) * rsin_u²
    # Simple approach: interpolate va to x-face positions
    grid = cdgrid.base
    # Reuse the haloed cell-centre winds
    u_pad, v_pad = pad_halo_vector(
        ua, va,  # contravariant for transport
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=grid.halo_interp_offsets,
    )

    # Contravariant u at x-faces
    v_at_xface = 0.5 * (v_pad[:, :-1, 1:-1] + v_pad[:, 1:, 1:-1])
    cosa_u = cdgrid.cosa_u                     # (6, n+1, n)
    sina_u = jnp.sqrt(jnp.maximum(1.0 - cosa_u**2, 1e-7))
    ut = (uc - v_at_xface * cosa_u) / jnp.maximum(sina_u**2, 1e-7)

    # Contravariant v at y-faces
    u_at_yface = 0.5 * (u_pad[:, 1:-1, :-1] + u_pad[:, 1:-1, 1:])
    cosa_v = cdgrid.cosa_v                     # (6, n, n+1)
    sina_v = jnp.sqrt(jnp.maximum(1.0 - cosa_v**2, 1e-7))
    vt = (vc - u_at_yface * cosa_v) / jnp.maximum(sina_v**2, 1e-7)

    # Covariant A-grid (simple averages of edge winds)
    utmp = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])
    vtmp = 0.5 * (v_d[:, :-1, :] + v_d[:, 1:, :])

    return ua, va, uc, vc, ut, vt, utmp, vtmp
