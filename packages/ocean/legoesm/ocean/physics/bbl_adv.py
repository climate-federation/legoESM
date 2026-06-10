"""Advective bottom boundary layer (Campin & Goosse 1999) — NEMO trabbl
``nn_bbl_adv=2``.

At coarse resolution (1 deg), dense shelf/overflow water (Mediterranean at
Gibraltar, Denmark Strait, Antarctic shelves) cannot descend the continental
slope: the staircase topography mixes it horizontally at sill depth and the
overflow stalls.  NEMO ORCA1 solves this with an ADVECTIVE bottom-boundary-
layer scheme (``ln_trabbl``, ``nn_bbl_adv=2``, ``rn_gambbl=20 s``): wherever
the up-slope (shelf) BOTTOM cell is denser than the down-slope (deep) BOTTOM
cell at a common reference depth, a down-slope transport

    tr_bbl = (face width) * e3_bbl * g * gamma * max(0, drho/rho0)

carries the dense water from the shelf bottom to the DEEP column's bottom,
with an upward return flow through the deep column and a horizontal return
at shelf level — a closed, exactly tracer-conserving circulation cell
(NEMO ``tra_bbl_adv``):

    shelf bottom  (iis, ks):  += |tr| * (pt[deep, ks]  - pt[shelf, ks]) / V
    deep interior (iid, k) :  += |tr| * (pt[k+1]       - pt[k])         / V
                                          for ks <= k < kd
    deep bottom   (iid, kd):  += |tr| * (pt[shelf, ks] - pt[deep, kd])  / V

(V = cell area * thickness; the weighted sum telescopes to zero.)

Geometry is STATIC (NEMO tra_bbl_init): per interior u/v face,
``mgrh = sign(bottom-depth difference)`` (0 when flat), shelf/deep bottom
level indices, and ``e3_bbl = min`` of the two columns' bottom-cell
thicknesses.

This module implements the lat-lon C-grid family version (regular lat-lon +
ORCA tripole, array layout ``(n_lat, n_lon, nlev)``) as pure JAX with STATIC
unrolled level loops — jit/grad safe.  East-west periodicity: faces are built
for column pairs ``(i, i+1)`` for ``i in 0..n_lon-2`` (interior faces only;
on the tripole the cyclic halo columns are slaved by ``ew_cyclic_overlap``
so the seam exchange is represented through the overlap, and on a regular
grid the wrap face is omitted — one face of ~360 at 1 deg, negligible and
safe).  References: Beckmann & Doscher (1997) JPO; Campin & Goosse (1999)
Tellus; NEMO 5.0.1 ``TRA/trabbl.F90``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants


__physics_contract__ = {
    "name": "bbl_adv",
    "units": {
        "T": "degC", "S": "PSU", "h_k": "m", "area": "m^2",
        "utr_bbl": "m^3/s", "vtr_bbl": "m^3/s",
        "dT_dt": "degC/s", "dS_dt": "PSU/s",
        "gamma_s": "s",
    },
    "signs": {
        "transport": "positive toward the DEEPER column (down-slope); "
                     "active only when the shelf bottom cell is denser at "
                     "a common reference depth",
        "tendency": "tracer tendency added to dT_dt/dS_dt (PSU degC per s)",
    },
    "conserves": [
        "tracer mass: the 3-leg circulation cell telescopes to zero under "
        "the volume weights area*h_k (verified by unit test to fp tol)",
    ],
    "differentiable": True,
    "reference": "Campin & Goosse (1999) Tellus 51A 412-430; Beckmann & "
                 "Doscher (1997) JPO 27 581-591; NEMO 5.0.1 TRA/trabbl.F90 "
                 "(nn_bbl_adv=2, rn_gambbl=20 s, ORCA1 RUN_REF namelist)",
    "idealized_test": "tests/ocean/unit/test_bbl_adv.py — analytic 2-column "
                      "dense-shelf overflow (transport formula exact, "
                      "down-slope sign, conservation, gate-off bit-exact)",
}


class BBLGeometry(NamedTuple):
    """Static BBL face geometry (NEMO tra_bbl_init).

    All arrays are FACE-indexed: i-faces ``(n_lat, n_lon-1)`` between columns
    ``(j, i)`` and ``(j, i+1)``; j-faces ``(n_lat-1, n_lon)`` between
    ``(j, i)`` and ``(j+1, i)``.
    """
    mgrhu: jnp.ndarray      # i-face slope sign: +1 deeper at i+1, -1 deeper at i, 0 flat
    mgrhv: jnp.ndarray      # j-face slope sign
    ku_s: jnp.ndarray       # i-face SHELF (shallow) bottom level index
    ku_d: jnp.ndarray       # i-face DEEP bottom level index
    kv_s: jnp.ndarray       # j-face shelf bottom level
    kv_d: jnp.ndarray       # j-face deep bottom level
    e3u_bbl: jnp.ndarray    # i-face BBL thickness = min(bottom e3 of the 2 columns)
    e3v_bbl: jnp.ndarray    # j-face BBL thickness
    dep_u: jnp.ndarray      # i-face common reference depth (mean bottom depth) [m]
    dep_v: jnp.ndarray      # j-face common reference depth [m]
    u_active: jnp.ndarray   # i-face both-columns-wet AND sloped (float 0/1)
    v_active: jnp.ndarray   # j-face mask
    bot_k: jnp.ndarray      # per-CELL bottom level index (n_lat, n_lon)


def bbl_static_geometry(h_ref: jnp.ndarray, land_mask: jnp.ndarray
                        ) -> BBLGeometry:
    """Build the static BBL geometry from per-cell REFERENCE layer
    thicknesses ``h_ref`` (n_lat, n_lon, nlev; 0 below the seafloor —
    partial-cell aware) and the 2-D ocean ``land_mask``.

    NEMO equivalents: ``mbkt`` (bottom level), ``gdept_0`` at the bottom
    (here: cumulative mid-cell depth), ``mgrhu/v = sign(d_bot(i+1)-d_bot(i))``,
    ``mbku_d = max(mbkt, mbkt_neighbour)``, ``e3u_bbl_0 = min`` of the two
    bottom thicknesses.
    """
    h = jnp.asarray(h_ref, dtype=jnp.float64)
    mask = jnp.asarray(land_mask, dtype=jnp.float64)
    wet3 = h > 1.0e-3
    n_active = jnp.sum(wet3.astype(jnp.int32), axis=-1)          # (ny, nx)
    bot_k = jnp.maximum(n_active - 1, 0)                          # bottom index
    # mid-cell depths + bottom-cell depth/thickness per column
    cum = jnp.cumsum(h, axis=-1)
    z_mid = cum - 0.5 * h                                         # (ny,nx,nl)
    dep_bot = jnp.take_along_axis(z_mid, bot_k[..., None], axis=-1)[..., 0]
    e3_bot = jnp.take_along_axis(h, bot_k[..., None], axis=-1)[..., 0]

    def _faces(a_l, a_r):
        return a_l, a_r

    # --- i-faces: columns (j, i) vs (j, i+1) -------------------------------
    dL, dR = dep_bot[:, :-1], dep_bot[:, 1:]
    mgrhu = jnp.sign(dR - dL)
    ku_s = jnp.where(mgrhu >= 0, bot_k[:, :-1], bot_k[:, 1:])     # shallow col
    ku_d = jnp.maximum(bot_k[:, :-1], bot_k[:, 1:])               # NEMO mbku_d
    e3u_bbl = jnp.minimum(e3_bot[:, :-1], e3_bot[:, 1:])
    dep_u = 0.5 * (dL + dR)
    u_active = ((mask[:, :-1] > 0.5) & (mask[:, 1:] > 0.5)
                & (mgrhu != 0)).astype(jnp.float64)

    # --- j-faces: columns (j, i) vs (j+1, i) -------------------------------
    dS_, dN = dep_bot[:-1, :], dep_bot[1:, :]
    mgrhv = jnp.sign(dN - dS_)
    kv_s = jnp.where(mgrhv >= 0, bot_k[:-1, :], bot_k[1:, :])
    kv_d = jnp.maximum(bot_k[:-1, :], bot_k[1:, :])
    e3v_bbl = jnp.minimum(e3_bot[:-1, :], e3_bot[1:, :])
    dep_v = 0.5 * (dS_ + dN)
    v_active = ((mask[:-1, :] > 0.5) & (mask[1:, :] > 0.5)
                & (mgrhv != 0)).astype(jnp.float64)

    return BBLGeometry(
        mgrhu=mgrhu, mgrhv=mgrhv, ku_s=ku_s.astype(jnp.int32),
        ku_d=ku_d.astype(jnp.int32), kv_s=kv_s.astype(jnp.int32),
        kv_d=kv_d.astype(jnp.int32), e3u_bbl=e3u_bbl, e3v_bbl=e3v_bbl,
        dep_u=dep_u, dep_v=dep_v, u_active=u_active, v_active=v_active,
        bot_k=bot_k.astype(jnp.int32),
    )


def _bottom_ts(T, S, bot_k):
    """Gather bottom-cell T, S per column."""
    Tb = jnp.take_along_axis(T, bot_k[..., None], axis=-1)[..., 0]
    Sb = jnp.take_along_axis(S, bot_k[..., None], axis=-1)[..., 0]
    return Tb, Sb


def bbl_transports(T: jnp.ndarray, S: jnp.ndarray, geom: BBLGeometry,
                   dy_u: jnp.ndarray, dx_v: jnp.ndarray, *,
                   gamma_s: float, rho_0: float, eos_fn=None):
    """Campin-Goosse down-slope transports per face [m^3/s].

        tr = facewidth * e3_bbl * (g*gamma) * max(0, (rho_shelf-rho_deep)/rho0)

    The density difference is evaluated with the canonical EOS at the FACE's
    common reference pressure (NEMO uses the alpha/beta linearisation at the
    common bottom depth — identical to linear order; the direct Delta-rho
    avoids re-deriving eos_rab).  Returns ``(utr, vtr)`` SIGNED down-slope
    (multiplied by mgrh like NEMO, so + means toward larger index).
    """
    from legoesm.ocean.eos import wright_eos
    eos = wright_eos if eos_fn is None else eos_fn
    g_gamma = constants.g * gamma_s

    Tb, Sb = _bottom_ts(T, S, geom.bot_k)

    def _face_tr(axis):
        if axis == 0:   # j-faces
            T1, T2 = Tb[:-1, :], Tb[1:, :]
            S1, S2 = Sb[:-1, :], Sb[1:, :]
            mgrh, dep = geom.mgrhv, geom.dep_v
            e3, act, width = geom.e3v_bbl, geom.v_active, dx_v
        else:           # i-faces
            T1, T2 = Tb[:, :-1], Tb[:, 1:]
            S1, S2 = Sb[:, :-1], Sb[:, 1:]
            mgrh, dep = geom.mgrhu, geom.dep_u
            e3, act, width = geom.e3u_bbl, geom.u_active, dy_u
        # shelf = up-slope column, deep = down-slope column (by mgrh)
        T_sh = jnp.where(mgrh >= 0, T1, T2)
        T_dp = jnp.where(mgrh >= 0, T2, T1)
        S_sh = jnp.where(mgrh >= 0, S1, S2)
        S_dp = jnp.where(mgrh >= 0, S2, S1)
        p = rho_0 * constants.g * dep
        drho = eos(T_sh, S_sh, p) - eos(T_dp, S_dp, p)
        zgdrho = jnp.maximum(drho / rho_0, 0.0)       # only shelf-denser
        return width * e3 * g_gamma * zgdrho * mgrh * act

    return _face_tr(1), _face_tr(0)


def apply_bbl_adv_tendency(dT_dt, dS_dt, T, S, h_k, area, geom: BBLGeometry,
                           utr, vtr, *, nlev: int):
    """Add the NEMO ``tra_bbl_adv`` 3-leg exchange to the tracer tendencies.

    ``nlev`` must be a static Python int (the per-level loop is unrolled).
    All scatter updates use ``.at[].add`` — pure JAX, jit/grad safe.
    """
    a3 = area[..., None] * jnp.maximum(h_k, 1.0e-3)   # cell volumes (..., nl)
    inv_v = 1.0 / a3

    def _apply(dpt, pt, tr, axis):
        if axis == 1:   # i-faces between (j,i) and (j,i+1)
            mgrh, ks, kd, act = geom.mgrhu, geom.ku_s, geom.ku_d, geom.u_active
            sl_L = (slice(None), slice(0, -1))
            sl_R = (slice(None), slice(1, None))
        else:           # j-faces
            mgrh, ks, kd, act = geom.mgrhv, geom.kv_s, geom.kv_d, geom.v_active
            sl_L = (slice(0, -1), slice(None))
            sl_R = (slice(1, None), slice(None))
        zu = jnp.abs(tr) * act
        up = mgrh >= 0   # True: LEFT column is shelf, RIGHT is deep

        # gathered shelf/deep columns (face-shaped, full nlev)
        pt_L, pt_R = pt[sl_L], pt[sl_R]
        iv_L, iv_R = inv_v[sl_L], inv_v[sl_R]
        pt_sh = jnp.where(up[..., None], pt_L, pt_R)
        pt_dp = jnp.where(up[..., None], pt_R, pt_L)
        iv_sh = jnp.where(up[..., None], iv_L, iv_R)
        iv_dp = jnp.where(up[..., None], iv_R, iv_L)

        ks_ = ks.astype(jnp.int32)
        kd_ = kd.astype(jnp.int32)
        pt_sh_b = jnp.take_along_axis(pt_sh, ks_[..., None], axis=-1)[..., 0]
        pt_dp_at_ks = jnp.take_along_axis(pt_dp, ks_[..., None], axis=-1)[..., 0]
        pt_dp_b = jnp.take_along_axis(pt_dp, kd_[..., None], axis=-1)[..., 0]

        # face-shaped tendency contributions for shelf + deep columns
        d_sh = jnp.zeros_like(pt_sh)
        d_dp = jnp.zeros_like(pt_dp)
        # (1) shelf bottom: exchange with deep column at shelf level
        ztra_sh = zu * (pt_dp_at_ks - pt_sh_b) * jnp.take_along_axis(
            iv_sh, ks_[..., None], axis=-1)[..., 0]
        d_sh = d_sh.at[
            jnp.arange(d_sh.shape[0])[:, None], jnp.arange(d_sh.shape[1])[None, :],
            ks_].add(ztra_sh)
        # (2) deep interior return flow ks <= k < kd  (static unrolled loop)
        for k in range(nlev - 1):
            in_rng = ((k >= ks_) & (k < kd_)).astype(pt.dtype)
            ztra = zu * in_rng * (pt_dp[..., k + 1] - pt_dp[..., k]) \
                * iv_dp[..., k]
            d_dp = d_dp.at[..., k].add(ztra)
        # (3) deep bottom: receives the shelf bottom water
        ztra_dp = zu * (pt_sh_b - pt_dp_b) * jnp.take_along_axis(
            iv_dp, kd_[..., None], axis=-1)[..., 0]
        d_dp = d_dp.at[
            jnp.arange(d_dp.shape[0])[:, None], jnp.arange(d_dp.shape[1])[None, :],
            kd_].add(ztra_dp)

        # un-gather face contributions back onto LEFT/RIGHT cell columns
        d_L = jnp.where(up[..., None], d_sh, d_dp)
        d_R = jnp.where(up[..., None], d_dp, d_sh)
        dpt = dpt.at[sl_L].add(d_L.astype(dpt.dtype))
        dpt = dpt.at[sl_R].add(d_R.astype(dpt.dtype))
        return dpt

    for axis in (1, 0):
        dT_dt = _apply(dT_dt, T, utr if axis == 1 else vtr, axis)
        dS_dt = _apply(dS_dt, S, utr if axis == 1 else vtr, axis)
    return dT_dt, dS_dt


__all__ = [
    "BBLGeometry",
    "apply_bbl_adv_tendency",
    "bbl_static_geometry",
    "bbl_transports",
]
