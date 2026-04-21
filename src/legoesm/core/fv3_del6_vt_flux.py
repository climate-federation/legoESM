"""Port of Fortran `del6_vt_flux` (sw_core.F90:2008-2121).

Del-nord damping for the relative vorticity, flux-form, metric-aware.
Applied after d_sw6 wind assembly in the FV3 Fortran forward-backward
chain.  The returned edge fluxes are added to u, v as circulation
increments: `u(i,j) += vt(i,j); v(i,j) -= ut(i,j)` where vt, ut are
derived from the fx, fy outputs.

Iter-752 delivers the CORE standalone algorithm.  Wiring into
`fv3_sw_tendencies` comes iter-753+.
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

    _EPS = 1e-20
    dg = getattr(cdgrid.base, 'duogrid', None)
    offsets = None if dg is not None else cdgrid.base.halo_interp_offsets

    def pad_scalar(field):
        return pad_halo(field, interp_offsets=offsets, duogrid=dg)

    # Step 1: initial damping.
    d2 = damp * q                                    # (6, n, n)

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
    d2_west = d2_pad[:, :-1, 1:-1]   # (6, n+1, n)
    d2_east = d2_pad[:, 1:, 1:-1]    # (6, n+1, n)
    fx2 = del6_v * (d2_west - d2_east)               # (6, n+1, n)

    # fy2 at v-edge (6, n, n+1): d2[south] - d2[north] convention.
    # fy2[i, j] = del6_u[i, j] * (d2[i, j-1] - d2[i, j])
    d2_south = d2_pad[:, 1:-1, :-1]  # (6, n, n+1)
    d2_north = d2_pad[:, 1:-1, 1:]   # (6, n, n+1)
    fy2 = del6_u * (d2_south - d2_north)             # (6, n, n+1)

    # Iterate the del-n operator.
    # nord=0: return fx2, fy2 as-is (del-2).
    # nord>0: loop nord times.
    for n in range(nord):
        # d2 = rarea * (fx2 - fx2_east + fy2 - fy2_north)
        # fx2 shape (6, n+1, n): fx2[i, j] and fx2[i+1, j] at each cell.
        # For cell (i, j), fx2 contribution is fx2[i, j] - fx2[i+1, j].
        fx2_west = fx2[:, :-1, :]    # (6, n, n)
        fx2_east_contrib = fx2[:, 1:, :]  # (6, n, n)
        fy2_south = fy2[:, :, :-1]   # (6, n, n)
        fy2_north_contrib = fy2[:, :, 1:]  # (6, n, n)
        d2 = rarea * (fx2_west - fx2_east_contrib
                      + fy2_south - fy2_north_contrib)

        # Halo exchange d2.
        d2_pad = pad_scalar(d2)

        # Recompute fx2, fy2 with SIGN FLIPPED (note Fortran line 2099
        # has `d2(i,j) - d2(i-1,j)` whereas pre-loop has
        # `d2(i-1,j) - d2(i,j)`).
        d2_west = d2_pad[:, :-1, 1:-1]
        d2_east = d2_pad[:, 1:, 1:-1]
        fx2 = del6_v * (d2_east - d2_west)           # sign flipped

        d2_south = d2_pad[:, 1:-1, :-1]
        d2_north = d2_pad[:, 1:-1, 1:]
        fy2 = del6_u * (d2_north - d2_south)         # sign flipped

    return fx2, fy2
