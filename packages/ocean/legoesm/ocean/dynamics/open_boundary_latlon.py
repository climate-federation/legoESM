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
  flux through the open boundary, or sets it to ``target_net_flux``.

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
    # Net volume flux (m3 s-1, positive into the domain) that the volume
    # correction keeps through all open faces together, for a nested child
    # that must follow its parent's mass changes; a scalar, None means zero
    # (CROCO obc_volcons).  Ignored when the correction is off.
    target_net_flux: jnp.ndarray | None = None
    # Baroclinic and tracer data (CROCO u3dbc/v3dbc/t3dbc, OBC_M3ORLANSKI,
    # OBC_TORLANSKI with FRC_BRY), shape (nlev, along): *_T, *_S in the ring
    # cells, *_u3 the normal velocity on the open face, *_t3 the tangential
    # velocity in the ring row.  With tau_in (s) set, the ring cells get
    # adaptive Orlanski radiation nudged toward these data after every outer
    # step: strong nudging (tau_in) where the phase speed points inward, weak
    # (tau_out) where it points out.  Passive tracers get zero gradient.
    west_T: jnp.ndarray | None = None
    west_S: jnp.ndarray | None = None
    west_u3: jnp.ndarray | None = None
    west_t3: jnp.ndarray | None = None
    south_T: jnp.ndarray | None = None
    south_S: jnp.ndarray | None = None
    south_u3: jnp.ndarray | None = None
    south_t3: jnp.ndarray | None = None
    north_T: jnp.ndarray | None = None
    north_S: jnp.ndarray | None = None
    north_u3: jnp.ndarray | None = None
    north_t3: jnp.ndarray | None = None
    tau_in: jnp.ndarray | None = None
    tau_out: jnp.ndarray | None = None


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
        faces_u.append((1, 1.0, (slice(None), 0), (slice(None), 1), u_mask[:, 1], dy_cell))
    if ob.south_vbar is not None:
        dy_r = dy_cell[1]
        vb = _characteristic(V_c[1], V_c[2], eta_c[0], eta_c[1], eta_new[1],
                             H_bathy[0], H_bathy[1], dy_r, dt_s, g,
                             ob.south_vbar, ob.south_zeta, 1.0)
        V_new = V_new.at[1].set(vb * v_mask[1])
        eta_new = eta_new.at[0].set(eta_new[1] * mask[0])
        faces_v.append((1, 1.0, 0, 1, v_mask[1], _v_face_length(grid, 1)))
    if ob.north_vbar is not None:
        dy_r = dy_cell[-2]
        vb = _characteristic(V_c[-2], V_c[-3], eta_c[-1], eta_c[-2], eta_new[-2],
                             H_bathy[-1], H_bathy[-2], dy_r, dt_s, g,
                             ob.north_vbar, ob.north_zeta, -1.0)
        V_new = V_new.at[-2].set(vb * v_mask[-2])
        eta_new = eta_new.at[-1].set(eta_new[-2] * mask[-1])
        nv = V_new.shape[0] - 2
        faces_v.append((nv, -1.0, -1, -2, v_mask[-2], _v_face_length(grid, nv)))

    if ob.band_gamma_u is not None:
        ru = 0.0 if ob.band_U_ref is None else ob.band_U_ref
        rv = 0.0 if ob.band_V_ref is None else ob.band_V_ref
        U_new = (U_new + dt_s * ob.band_gamma_u * ru) / (1.0 + dt_s * ob.band_gamma_u) * u_mask
        V_new = (V_new + dt_s * ob.band_gamma_v * rv) / (1.0 + dt_s * ob.band_gamma_v) * v_mask

    if (faces_u or faces_v) and ob.volume_correction_off is None:
        # obc_volcons.F: remove the net inflow with one uniform inward velocity
        # (or bring it to target_net_flux).  The face depth is the total depth
        # H + eta by the min rule of the substep's continuity flux, as CROCO's
        # Dstp = h + zeta: with H alone the correction misses the realised net
        # flux by the sum of (normal velocity x eta) along the boundary, which
        # is large when big inflows and outflows nearly cancel.
        D = (H_bathy + eta_new) * mask
        faces = [(U_new, k, s, jnp.minimum(D[a], D[b]) * m, length, True) for k, s, a, b, m, length in faces_u]
        faces += [(V_new, k, s, jnp.minimum(D[a], D[b]) * m, length, False) for k, s, a, b, m, length in faces_v]
        flux, section = 0.0, 0.0
        for F, k, s, d, length, is_u in faces:
            vel = F[:, k] if is_u else F[k]
            flux = flux + s * jnp.sum(vel * d * length)
            section = section + jnp.sum(d * length)
        target = 0.0 if ob.target_net_flux is None else ob.target_net_flux
        corr = (target - flux) / jnp.maximum(section, _EPS)
        for _, k, s, d, length, is_u in faces:
            if is_u:
                U_new = U_new.at[:, k].add(s * corr * (d > 0))
            else:
                V_new = V_new.at[k].add(s * corr * (d > 0))
    return U_new, V_new, eta_new


def _v_face_length(grid, row):
    """Length of the v faces of one v row, R cos(lat_v) dlon (as divergence_cgrid away from the poles)."""
    return float(grid.radius) * jnp.cos(jnp.asarray(grid.lat_v)[row]) * float(grid.dlon)


def _orlanski(fb_old, f1_old, f1_new, f2_new, data, r_in, r_out):
    """CROCO t3dbc.F / u3dbc.F adaptive oblique radiation with nudging, on
    arrays (..., along) ordered boundary, first, second interior; r_in,
    r_out = dt / tau (dimensionless)."""
    def tgrad(f):
        gb = f - jnp.concatenate([f[..., :1], f[..., :-1]], axis=-1)
        return jnp.concatenate([gb, gb[..., -1:]], axis=-1)
    gb, g1 = tgrad(fb_old), tgrad(f1_old)
    dft = f1_old - f1_new
    dfx = f1_new - f2_new
    inflow = dfx * dft < 0.0
    dft = jnp.where(inflow, 0.0, dft)
    tau = jnp.where(inflow, r_in, r_out)
    up = dft * (g1[..., :-1] + g1[..., 1:]) > 0.0
    dfy = jnp.where(up, g1[..., :-1], g1[..., 1:])
    cff = jnp.maximum(dfx * dfx + dfy * dfy, _EPS)
    cx = dft * dfx
    cy = jnp.clip(dft * dfy, -cff, cff)
    fb = (cff * fb_old + cx * f1_new - jnp.maximum(cy, 0.0) * gb[..., :-1]
          - jnp.minimum(cy, 0.0) * gb[..., 1:]) / (cff + cx)
    return (1.0 - tau) * fb + tau * data


def _side(a, side, b0, k):
    """Row k inward of the boundary row b0 of a 3-D field, as (nlev, along).
    b0: index along the normal of the boundary row -- the ring cell (0 or -1)
    for cell fields and tangential velocities, the open face (1 or -2) for the
    normal velocity."""
    if side == 'west':
        return a[:, b0 + k, :].T
    if side == 'south':
        return a[b0 + k].T
    return a[b0 - k].T


def _set_side(a, side, b0, row):
    if side == 'west':
        return a.at[:, b0, :].set(row.T)
    return a.at[b0].set(row.T)


def apply_baroclinic_open_boundary(state_old, state_new, ob: OpenBoundaryData, dt, dz_ref):
    """Ring cells and open faces after an outer step: T, S radiated and nudged
    toward the data; the open-face normal velocity and the ring tangential
    velocity likewise for their baroclinic part, with the depth mean kept from
    the barotropic boundary condition; passive tracers zero gradient.  The
    depth mean is weighted by the reference layer thicknesses dz_ref (partial
    bottom cells not accounted for)."""
    if ob.tau_in is None:
        return state_new
    r_in, r_out = dt / ob.tau_in, dt / ob.tau_out
    T, S, u, v = state_new.T.data, state_new.S.data, state_new.u.data, state_new.v.data
    To, So, uo, vo = state_old.T.data, state_old.S.data, state_old.u.data, state_old.v.data
    P = getattr(state_new, 'passive', None)
    w = jnp.asarray(dz_ref)[:, None] / jnp.sum(jnp.asarray(dz_ref))
    for side in ('west', 'south', 'north'):
        dT = getattr(ob, f'{side}_T')
        if dT is None:
            continue
        ring = 0 if side in ('west', 'south') else -1
        face = 1 if side in ('west', 'south') else -2
        for name, data in (('T', dT), ('S', getattr(ob, f'{side}_S'))):
            new, old = (T, To) if name == 'T' else (S, So)
            row = _orlanski(_side(old, side, ring, 0), _side(old, side, ring, 1), _side(new, side, ring, 1),
                            _side(new, side, ring, 2), data, r_in, r_out)
            if name == 'T':
                T = _set_side(T, side, ring, row)
            else:
                S = _set_side(S, side, ring, row)
        # normal velocity on the open face, tangential velocity in the ring row
        normal_is_u = side == 'west'
        for is_normal, data in ((True, getattr(ob, f'{side}_u3')), (False, getattr(ob, f'{side}_t3'))):
            if data is None:
                continue
            use_u = normal_is_u == is_normal
            new, old = (u, uo) if use_u else (v, vo)
            b0 = face if is_normal else ring
            mean_cur = jnp.sum(w * _side(new, side, b0, 0), axis=0, keepdims=True)    # barotropic part kept
            rad = _orlanski(_side(old, side, b0, 0), _side(old, side, b0, 1), _side(new, side, b0, 1),
                            _side(new, side, b0, 2), data, r_in, r_out)
            row = rad - jnp.sum(w * rad, axis=0, keepdims=True) + mean_cur
            if use_u:
                u = _set_side(u, side, b0, row)
            else:
                v = _set_side(v, side, b0, row)
        if P is not None:
            Pn = P
            if side == 'west':
                Pn = Pn.at[:, 0].set(Pn[:, 1])
            elif side == 'south':
                Pn = Pn.at[0].set(Pn[1])
            else:
                Pn = Pn.at[-1].set(Pn[-2])
            P = Pn
    out = state_new._replace(T=state_new.T.replace(data=T), S=state_new.S.replace(data=S),
                             u=state_new.u.replace(data=u), v=state_new.v.replace(data=v))
    if P is not None:
        out = out._replace(passive=P)
    return out
