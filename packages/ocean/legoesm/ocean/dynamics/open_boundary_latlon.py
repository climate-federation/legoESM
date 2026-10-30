"""Open lateral boundaries for the regional lat-lon ocean (barotropic part).

Layout.  The regional lat-lon grid carries a ring of cells around the
domain.  On an open side the ring cells are made wet (land mask override)
and act as the boundary row of ROMS/CROCO: their values are set by the
boundary condition, never by the model's own update.  The open face is the
one between the ring and the first interior cell (west ``U[:, 1]``, south
``V[1]``, north ``V[-2]``); the faces at the array edge stay closed.

Barotropic condition, applied at the end of every fast substep, as CROCO
does with ``OBC_M2CHARACT`` + ``OBC_VOLCONS`` and no ``Z`` key
(``u2dbc.F``, ``v2dbc.F``, ``zetabc.F``, ``obc_volcons.F``):

* normal depth-mean velocity on the open face from the characteristic
  (Flather type) condition: the outgoing characteristic comes from the
  interior, the incoming one from the data ``ubar_ext``, ``zeta_ext``;
* sea level in the ring cells: zero gradient (copy of the first interior);
* a uniform normal velocity on all open faces that cancels the net volume
  flux through the open boundary.

The baroclinic and tracer conditions (radiation with nudging) and the
relaxation sponge act once per outer step and are applied elsewhere.
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

_CX_CRIT = 0.292893218813452          # 1 - 1/sqrt(2), u2dbc.F threshold of the zx extrapolation
_EPS = 1.0e-20


class OpenBoundaryData(NamedTuple):
    """Boundary data of the open sides; a side with ``None`` stays closed.

    ``*_ubar`` / ``*_vbar``: normal depth-mean velocity on the open face
    (positive toward +x / +y), shape along the side (n_lat for west, n_lon
    for south and north).  ``*_zeta``: sea level of the ring cells, same
    shape.  Values outside the wet part of a side are ignored.
    """
    west_ubar: jnp.ndarray | None = None
    west_zeta: jnp.ndarray | None = None
    south_vbar: jnp.ndarray | None = None
    south_zeta: jnp.ndarray | None = None
    north_vbar: jnp.ndarray | None = None
    north_zeta: jnp.ndarray | None = None
    # Optional absorbing band: Rayleigh relaxation of the depth-mean velocity
    # toward band_U_ref / band_V_ref in every fast substep, rate band_gamma_u
    # (n_lat, n_lon+1) and band_gamma_v (n_lat+1, n_lon) in s-1, zero outside
    # the band.  Velocity only: sea level changes through the fluxes alone, so
    # the transports the tracers see stay consistent with it.
    band_gamma_u: jnp.ndarray | None = None
    band_gamma_v: jnp.ndarray | None = None
    band_U_ref: jnp.ndarray | None = None
    band_V_ref: jnp.ndarray | None = None
    # Any non-None value switches the volume correction off (a structural
    # switch, so it stays static under jit).  For tests of wave transparency:
    # with the correction on, a sea level anomaly's own volume cannot leave.
    volume_correction_off: object | None = None


def _characteristic(ub_old, u1_old, zb_old, z1_old, z1_new, hb, h1, dn, dt_s, g,
                    u_ext, z_ext, sign):
    """CROCO u2dbc.F, OBC_M2CHARACT with M2_FRC_BRY and Z_FRC_BRY; sign +1 for
    west and south, -1 for north (the outgoing characteristic changes sign)."""
    D = 0.5 * (hb + zb_old + h1 + z1_old)
    hx = jnp.sqrt(g / jnp.maximum(D, _EPS))
    cx = dt_s * D * hx / dn
    zx = (0.5 + cx) * z1_old + (0.5 - cx) * zb_old
    corr = (z1_new + cx * zb_old - (1.0 + cx) * z1_old) * (
        1.0 - _CX_CRIT / jnp.maximum(cx, _EPS)) ** 2
    zx = jnp.where(cx > _CX_CRIT, zx + corr, zx)
    return 0.5 * ((1.0 - cx) * ub_old + cx * u1_old + u_ext - sign * hx * (zx - z_ext))


def apply_barotropic_open_boundary(eta_c, U_c, V_c, eta_new, U_new, V_new,
                                   H_bathy, grid, dt_s, g, ob: OpenBoundaryData,
                                   u_mask, v_mask, mask):
    """Set the open-face normal velocity, the ring sea level and the volume
    correction after a fast substep.  Arrays: eta (n_lat, n_lon),
    U (n_lat, n_lon+1), V (n_lat+1, n_lon); returns (U_new, V_new, eta_new)."""
    dx = jnp.asarray(grid.dx)                          # cell widths (n_lat, n_lon)
    dy_cell = 0.5 * jnp.asarray(grid.dy)               # cell height per row (half of grid.dy, as divergence_cgrid)
    faces_u, faces_v = [], []                          # (column or row index, sign, face depth, face length)

    if ob.west_ubar is not None:
        ub = _characteristic(U_c[:, 1], U_c[:, 2], eta_c[:, 0], eta_c[:, 1], eta_new[:, 1],
                             H_bathy[:, 0], H_bathy[:, 1], dx[:, 1], dt_s, g,
                             ob.west_ubar, ob.west_zeta, 1.0)
        U_new = U_new.at[:, 1].set(ub * u_mask[:, 1])
        eta_new = eta_new.at[:, 0].set(eta_new[:, 1] * mask[:, 0])
        faces_u.append((1, 1.0, jnp.minimum(H_bathy[:, 0], H_bathy[:, 1]) * u_mask[:, 1], dy_cell))
    if ob.south_vbar is not None:
        dy_r = dy_cell[1]
        vb = _characteristic(V_c[1], V_c[2], eta_c[0], eta_c[1], eta_new[1],
                             H_bathy[0], H_bathy[1], dy_r, dt_s, g,
                             ob.south_vbar, ob.south_zeta, 1.0)
        V_new = V_new.at[1].set(vb * v_mask[1])
        eta_new = eta_new.at[0].set(eta_new[1] * mask[0])
        faces_v.append((1, 1.0, jnp.minimum(H_bathy[0], H_bathy[1]) * v_mask[1], _v_face_length(grid, 1)))
    if ob.north_vbar is not None:
        dy_r = dy_cell[-2]
        vb = _characteristic(V_c[-2], V_c[-3], eta_c[-1], eta_c[-2], eta_new[-2],
                             H_bathy[-1], H_bathy[-2], dy_r, dt_s, g,
                             ob.north_vbar, ob.north_zeta, -1.0)
        V_new = V_new.at[-2].set(vb * v_mask[-2])
        eta_new = eta_new.at[-1].set(eta_new[-2] * mask[-1])
        nv = V_new.shape[0] - 2
        faces_v.append((nv, -1.0, jnp.minimum(H_bathy[-1], H_bathy[-2]) * v_mask[-2], _v_face_length(grid, nv)))

    if ob.band_gamma_u is not None:
        ru = 0.0 if ob.band_U_ref is None else ob.band_U_ref
        rv = 0.0 if ob.band_V_ref is None else ob.band_V_ref
        U_new = (U_new + dt_s * ob.band_gamma_u * ru) / (1.0 + dt_s * ob.band_gamma_u) * u_mask
        V_new = (V_new + dt_s * ob.band_gamma_v * rv) / (1.0 + dt_s * ob.band_gamma_v) * v_mask

    if (faces_u or faces_v) and ob.volume_correction_off is None:
        # obc_volcons.F: remove the net inflow with one uniform inward velocity
        flux, section = 0.0, 0.0
        for k, s, d, length in faces_u:
            flux = flux + s * jnp.sum(U_new[:, k] * d * length)
            section = section + jnp.sum(d * length)
        for k, s, d, length in faces_v:
            flux = flux + s * jnp.sum(V_new[k] * d * length)
            section = section + jnp.sum(d * length)
        corr = -flux / jnp.maximum(section, _EPS)
        for k, s, d, length in faces_u:
            U_new = U_new.at[:, k].add(s * corr * (d > 0))
        for k, s, d, length in faces_v:
            V_new = V_new.at[k].add(s * corr * (d > 0))
    return U_new, V_new, eta_new


def _v_face_length(grid, row):
    """Length of the v faces of one v row, R cos(lat_v) dlon (as divergence_cgrid away from the poles)."""
    return float(grid.radius) * jnp.cos(jnp.asarray(grid.lat_v)[row]) * float(grid.dlon)
