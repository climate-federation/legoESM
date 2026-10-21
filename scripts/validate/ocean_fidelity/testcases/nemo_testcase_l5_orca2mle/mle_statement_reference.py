#!/usr/bin/env python3
"""Statement-order NEMO ``tramle`` reference on captured ORCA2 operands."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


def nemo_mle_transport_reference(
    d: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(U, V, zmld, bm)`` in tramle.f90 statement order."""

    rho, um2, vm2, hk = (np.asarray(d[k]) for k in ("in2", "in4", "in5", "in6"))
    wet = np.asarray(d["is_active"], dtype=np.float64)
    stretch = np.asarray(d["stretch_half0"])
    e1u, e2u, e1v, e2v = (np.asarray(d[k]) for k in
                            ("dx_u", "dy_u", "dx_v", "dy_v"))
    ny, nx, nk = hk.shape

    gdepw_1d = np.abs(np.asarray(d["z_half_ref"]))
    rho0, grav, rho_c = 1026.0, 9.80665, 0.01
    r1_rho0 = 1.0 / rho0
    rpi = 3.141592653589793
    rday = 24.0 * 60.0 * 60.0
    rsiyea = 365.25 * rday * 2.0 * rpi / 6.283076
    rsiday = rday / (1.0 + rday / rsiyea)
    omega = 2.0 * rpi / rsiday
    rad = rpi / 180.0
    rc_f = 0.06 / (5.0e3 * 2.0 * omega * math.sin(rad * 20.0))
    r5_21 = 5.0 / 21.0
    nla10, nlb10 = 1, 2

    rhop = rho * wet
    mbkt = np.maximum(wet.sum(-1).astype(int), 1)
    inml = mbkt + 1
    for jk in range(nk, nlb10 - 1, -1):
        hit = rhop[..., jk - 1] > rhop[..., nla10 - 1] + rho_c
        inml = np.where(hit, jk, inml)
    ikmax = min(int(inml.max()), nk)
    zmld = np.zeros((ny, nx), dtype=np.float64)
    zbm = np.zeros_like(zmld)
    for jk in range(1, ikmax + 1):
        member = np.minimum(np.maximum(0, inml - jk), 1).astype(np.float64)
        zc = hk[..., jk - 1] * member
        zmld = zmld + zc
        zbm = zbm + zc * (rho0 - rhop[..., jk - 1]) * r1_rho0
    with np.errstate(invalid="ignore", divide="ignore"):
        zbm = grav * zbm / np.maximum(hk[..., 0], zmld)

    U = np.zeros((ny, nx + 1, nk), dtype=np.float64)
    V = np.zeros((ny + 1, nx, nk), dtype=np.float64)
    iw = (np.arange(nx + 1) - 1) % nx
    ie = np.arange(nx + 1) % nx
    zhu = np.minimum(zmld[:, ie], zmld[:, iw])
    psim = rc_f * zhu * zhu * (e2u / e1u)
    psim = psim * (zbm[:, ie] - zbm[:, iw])
    psim = psim * np.minimum(111.0e3, e1u)
    rzhu = 1.0 / np.maximum(zhu, 0.5 * np.finfo(np.float64).eps)
    w_u = wet[:, iw, :] * wet[:, ie, :]
    psi1 = np.zeros_like(psim)
    for jk in range(1, ikmax + 1):
        zc = 1.0 - (gdepw_1d[jk] * stretch[:, ie]
                    + gdepw_1d[jk] * stretch[:, iw]) * rzhu
        zc = zc * zc
        mu = np.maximum(0.0, (1.0 - zc) * (1.0 + r5_21 * zc))
        wk1 = w_u[..., jk] if jk < nk else 0.0
        psi2 = psim * mu * (wk1 * w_u[..., jk - 1]) * w_u[..., 0]
        U[..., jk - 1] = psi1 - psi2
        psi1 = psi2
    U = U * um2[..., None]

    js, jn = np.arange(ny - 1), np.arange(1, ny)
    zhv = np.minimum(zmld[jn], zmld[js])
    psim = rc_f * zhv * zhv * (e1v[1:ny] / e2v[1:ny])
    psim = psim * (zbm[jn] - zbm[js])
    psim = psim * np.minimum(111.0e3, e2v[1:ny])
    rzhv = 1.0 / np.maximum(zhv, 0.5 * np.finfo(np.float64).eps)
    w_v = wet[js] * wet[jn]
    psi1 = np.zeros_like(psim)
    for jk in range(1, ikmax + 1):
        zc = 1.0 - (gdepw_1d[jk] * stretch[jn]
                    + gdepw_1d[jk] * stretch[js]) * rzhv
        zc = zc * zc
        mu = np.maximum(0.0, (1.0 - zc) * (1.0 + r5_21 * zc))
        wk1 = w_v[..., jk] if jk < nk else 0.0
        psi2 = psim * mu * (wk1 * w_v[..., jk - 1]) * w_v[..., 0]
        V[1:ny, :, jk - 1] = psi1 - psi2
        psi1 = psi2
    V[1:ny] = V[1:ny] * vm2[1:ny, :, None]
    perm = np.asarray(d["perm_v"])
    V[-1] = np.asarray(d["sign_v"]) * V[-2][perm]

    # The acquisition's zero-first observer defines only wet native faces plus
    # the NEMO fold halo.  Canonicalise every undefined/dry word identically.
    uw = np.zeros_like(U, dtype=bool)
    vw = np.zeros_like(V, dtype=bool)
    for face in range(nx + 1):
        uw[:, face] = (wet[:, (face - 1) % nx] * wet[:, face % nx]
                       * um2[:, face, None]) > 0.0
    vw[1:ny] = (wet[:-1] * wet[1:] * vm2[1:ny, :, None]) > 0.0
    vw[-1] = vw[-2][perm]
    U = np.where(uw, U, 0.0)
    V = np.where(vw, V, 0.0)
    wet2 = np.asarray(d["in3"]) > 0.0
    return U, V, np.where(wet2, zmld, 0.0), np.where(wet2, zbm, 0.0)


__all__ = ("nemo_mle_transport_reference",)
