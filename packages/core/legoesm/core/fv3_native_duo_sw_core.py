"""FV3-native DUO-GRID c_sw pieces — loop-faithful port, phase 4c.

The duo-grid FV3 (Mouallem/Xi-Chen, Zenodo 8327578, the authoritative
``atmos_cubed_sphere-symmetryclean`` tree) takes DIFFERENT c_sw branches
from the plain FV3 certified in phase 4a (``fv3_native_sw_core``): with
``flagstruct%duogrid`` it calls ``divergence_corner_duo`` instead of
``divergence_corner``, skips the ``fill2/fill_4corners`` corner fills
(the duo halos carry real cross-face data), and takes the
``bounded_domain .or. grid_type>=3 .or. duogrid`` KE/vorticity branch (no
``sin_sg`` panel-edge special-case).  legoESM's PRODUCTION solver runs
these duo branches, so full duo-grid FV3 fidelity certifies THESE, not the
plain ones.

This module ports the duo-specific routines loop-faithfully (index-exact,
plain numpy, Fortran statement order).  ``divergence_corner_duo`` is a
verified-faithful TRANSLATION of the authoritative Fortran — bit-exact
(uint64 words) on the ONE stored C12, ``grid_type=0`` cubed-sphere fixture
(``grid_type>3`` is not ported), with the fixture's ``input_sha256``
ENFORCED against its stored arrays (``test_fv3_native_duo_phase4c``), and
its extraction pinned byte-identical to sw_core.F90:2345-2447 (SHA-256).

SCOPE: the duo c_sw feeds ``divergence_corner_duo`` (and the KE/vorticity
path) with ua/va from ``d2a2c_vect``'s dg-initialized cross-face branch,
which differs from the plain ``c_sw`` ua/va at panel edges/corners.  That
duo ``d2a2c_vect`` is ported + certified here as ``d2a2c_vect_duo``
(bit-exact vs the authoritative Fortran DUO branch,
``test_fv3_native_d2a2c_duo``), and its ua/va are now CHAINED into
``divergence_corner_duo`` and certified bit-exact end-to-end
(``test_fv3_native_duo_chain``) — so the divergence sub-pipeline is closed
on the real duo ua/va.  The remaining phase-4c work is assembling these
certified duo leaves into a full no-corner-fill / no-``sin_sg``-edge duo
c_sw body (+ the simple upwind KE/vorticity) for full-c_sw certification.

Conventions match ``fv3_native_sw_core`` / ``fv3_native_d_sw`` (``fort``
views, ``Bounds``).
"""

from __future__ import annotations

import numpy as np
from legoesm.core.fv3_native_sw_core import Bounds, _fa
from legoesm.grids.fv3_native_gridstruct import fort

# --- d2a2c 4th-order interpolation coefficients (sw_core.F90:53-54) ---
_A1_D2A2C = 0.5625     # 9/16   coeff-ok: FV3 published d2a2c/PPM interpolant
_A2_D2A2C = -0.0625    # -1/16  coeff-ok: FV3 published d2a2c/PPM interpolant
_BIG_NUMBER = 1.0e30   # sw_core.F90:43 corner-init sentinel (guard)


def divergence_corner_duo(u: np.ndarray, v: np.ndarray,
                          ua: np.ndarray, va: np.ndarray,
                          gs: dict, bd: Bounds, npx: int, npy: int,
                          grid_type: int = 0) -> np.ndarray:
    """sw_core.F90 divergence_corner_duo (verbatim; symmetryclean tree).

    Returns divg_d (B-node array, isd:ied+1, jsd:jed+1).  Like
    divergence_corner_nest's interior formula (no is2/ie1 clamp) PLUS the
    duo panel-edge zeroing/quartering that removes the cube-seam
    divergence the duo halos would otherwise double-count.

    u (isd:ied, jsd:jed+1); v (isd:ied+1, jsd:jed); ua/va (isd:ied,
    jsd:jed).  FV3 cos_sg/sin_sg positions 1..4 map to python 0..3
    (W,S,E,N): cos_sg(.,.,4)->[3] N, (.,.,2)->[1] S, (.,.,3)->[2] E,
    (.,.,1)->[0] W.
    """
    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, ied, jsd, jed = bd.isd, bd.ied, bd.jsd, bd.jed

    U = fort(u, isd, jsd)
    V = fort(v, isd, jsd)
    UA = fort(ua, isd, jsd)
    VA = fort(va, isd, jsd)
    SIN_SG = fort(gs["sin_sg"], isd, jsd)
    COS_SG = fort(gs["cos_sg"], isd, jsd)
    DXC = fort(gs["dxc"], isd, jsd)
    DYC = fort(gs["dyc"], isd, jsd)
    RAREA_C = fort(gs["rarea_c"], isd, jsd)

    DIVG = _fa(bd, 1, 1, 1.0e25)   # divg_d = 1.e25 initial

    if grid_type > 3:  # pragma: no cover - cubed-sphere oracle
        raise NotImplementedError("grid_type > 3 not ported")

    uf = fort(np.full((ied - isd + 1, jed + 1 - jsd + 1), np.nan), isd, jsd)
    vf = fort(np.full((ied + 1 - isd + 1, jed - jsd + 1), np.nan), isd, jsd)

    for j in range(jsd + 1, jed + 1):
        for i in range(isd, ied + 1):
            uf[i, j] = (U[i, j] - 0.25 * (VA[i, j - 1] + VA[i, j])
                        * (COS_SG[i, j - 1, 4 - 1] + COS_SG[i, j, 2 - 1])) \
                * DYC[i, j] * 0.5 \
                * (SIN_SG[i, j - 1, 4 - 1] + SIN_SG[i, j, 2 - 1])

    for j in range(jsd, jed + 1):
        for i in range(isd + 1, ied + 1):
            vf[i, j] = (V[i, j] - 0.25 * (UA[i - 1, j] + UA[i, j])
                        * (COS_SG[i - 1, j, 3 - 1] + COS_SG[i, j, 1 - 1])) \
                * DXC[i, j] * 0.5 \
                * (SIN_SG[i - 1, j, 3 - 1] + SIN_SG[i, j, 1 - 1])

    for j in range(jsd + 1, jed + 1):
        for i in range(isd + 1, ied + 1):
            DIVG[i, j] = (vf[i, j - 1] - vf[i, j]
                          + uf[i - 1, j] - uf[i, j]) * RAREA_C[i, j]
            # duo panel-edge zeroing (the seam B-nodes)
            if is_ == 1 and i == is_:
                DIVG[i, j] = 0.0
            if (ie + 1) == npx and i == ie + 1:
                DIVG[i, j] = 0.0
            if js == 1 and j == 1:
                DIVG[i, j] = 0.0
            if je + 1 == npx and j == je + 1:
                DIVG[i, j] = 0.0
            # next-to-seam quartering (verbatim; note the upstream j-tests
            # use npx not npy — harmless on the square single tile)
            if is_ == 1 and i == is_ + 1:
                DIVG[i, j] = 0.25 * DIVG[i, j]
            if (ie + 1) == npx and i == ie:
                DIVG[i, j] = 0.25 * DIVG[i, j]
            if js == 1 and j == 1 + 1:
                DIVG[i, j] = 0.25 * DIVG[i, j]
            if je + 1 == npx and j == je:
                DIVG[i, j] = 0.25 * DIVG[i, j]

    return DIVG.a


def d2a2c_vect_duo(u: np.ndarray, v: np.ndarray, gs: dict, bd: Bounds,
                   npx: int, npy: int, dord4: bool = True,
                   grid_type: int = 0) -> dict:
    """sw_core.F90 d2a2c_vect, DUO branch (``gridstruct%dg%is_initialized``).

    D->A->C vector reconstruction: D-grid (u, v) -> A-grid (ua, va) ->
    C-grid (uc, vc) + contravariant (ut, vt).  The duo/bounded branch runs
    the INTERIOR 4th-order formulas over the full data domain and SKIPS
    every panel-edge / cube-corner special-case (the duo halos carry real
    cross-face winds) — exactly like ``divergence_corner_duo``.

    This is the ua/va source the duo c_sw actually feeds
    ``divergence_corner_duo`` (the plain ``c_sw`` used by the phase-4a
    reference takes d2a2c_vect's NON-duo branch, so its ua/va differ at
    panel edges/corners).  This LEAF is certified bit-exact here, and its
    ua/va -> ``divergence_corner_duo`` chain is certified end-to-end in
    ``test_fv3_native_duo_chain``; the full duo c_sw pipeline (KE/vorticity
    + the delp/pt/u/v update) remains.

    Returns ``{ua, va, uc, vc, ut, vt}`` (numpy).  u (isd:ied, jsd:jed+1);
    v (isd:ied+1, jsd:jed).  ``dord4``/``id`` is UNUSED on the duo branch
    (its interior covers the full domain id-independently; the ``id``
    stencil widening only applies to the non-duo else-branch); kept for
    signature parity.  FV3 cosa/rsin fields are read from ``gs``.
    """
    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, ied, jsd, jed = bd.isd, bd.ied, bd.jsd, bd.jed
    a1, a2 = _A1_D2A2C, _A2_D2A2C

    if grid_type >= 3:  # pragma: no cover - cubed-sphere oracle
        raise NotImplementedError("grid_type >= 3 not ported")

    U = fort(u, isd, jsd)
    V = fort(v, isd, jsd)
    COSA_S = fort(gs["cosa_s"], isd, jsd)
    RSIN2 = fort(gs["rsin2"], isd, jsd)
    COSA_U = fort(gs["cosa_u"], isd, jsd)
    RSIN_U = fort(gs["rsin_u"], isd, jsd)
    COSA_V = fort(gs["cosa_v"], isd, jsd)
    RSIN_V = fort(gs["rsin_v"], isd, jsd)

    utmp = _fa(bd, 0, 0, _BIG_NUMBER)   # (isd:ied, jsd:jed)
    vtmp = _fa(bd, 0, 0, _BIG_NUMBER)
    UA = _fa(bd, 0, 0, _BIG_NUMBER)     # (isd:ied, jsd:jed)
    VA = _fa(bd, 0, 0, _BIG_NUMBER)
    UC = _fa(bd, 1, 0, _BIG_NUMBER)     # (isd:ied+1, jsd:jed)
    VC = _fa(bd, 0, 1, _BIG_NUMBER)     # (isd:ied,   jsd:jed+1)
    UT = _fa(bd, 0, 0, _BIG_NUMBER)     # (isd:ied, jsd:jed)
    VT = _fa(bd, 0, 0, _BIG_NUMBER)

    # ---- D -> A (duo interior; sw_core.F90:3421-3454) ----
    for i in range(isd, ied + 1):
        for j in range(jsd + 1, jed):          # do j=jsd+1,jed-1
            utmp[i, j] = a2 * (U[i, j - 1] + U[i, j + 2]) \
                + a1 * (U[i, j] + U[i, j + 1])
    for i in range(isd, ied + 1):
        utmp[i, jsd] = U[i, jsd + 1]           # 0.5*(u(j+1)+u(j+1))
        utmp[i, jed] = U[i, jed + 1]
    for j in range(jsd, jed + 1):
        for i in range(isd + 1, ied):          # do i=isd+1,ied-1
            vtmp[i, j] = a2 * (V[i - 1, j] + V[i + 2, j]) \
                + a1 * (V[i, j] + V[i + 1, j])
        vtmp[isd, j] = V[isd + 1, j]           # 0.5*(v(i+1)+v(i+1))
        vtmp[ied, j] = V[ied + 1, j]
    for i in range(isd, ied + 1):
        for j in range(jsd, jed + 1):
            UA[i, j] = (utmp[i, j] - vtmp[i, j] * COSA_S[i, j]) * RSIN2[i, j]
            VA[i, j] = (vtmp[i, j] - utmp[i, j] * COSA_S[i, j]) * RSIN2[i, j]

    # ---- A -> C  X-dir (duo ifirst=is-1, ilast=ie+2; 3558-3563) ----
    for j in range(js - 1, je + 2):            # do j=js-1,je+1
        for i in range(is_ - 1, ie + 3):       # do i=is-1,ie+2
            UC[i, j] = a2 * (utmp[i - 2, j] + utmp[i + 1, j]) \
                + a1 * (utmp[i - 1, j] + utmp[i, j])
            UT[i, j] = (UC[i, j] - V[i, j] * COSA_U[i, j]) * RSIN_U[i, j]

    # ---- A -> C  Y-dir (duo interior for every j; 3690-3693) ----
    for j in range(js - 1, je + 3):            # do j=js-1,je+2
        for i in range(is_ - 1, ie + 2):       # do i=is-1,ie+1
            VC[i, j] = a2 * (vtmp[i, j - 2] + vtmp[i, j + 1]) \
                + a1 * (vtmp[i, j - 1] + vtmp[i, j])
            VT[i, j] = (VC[i, j] - U[i, j] * COSA_V[i, j]) * RSIN_V[i, j]

    return {"ua": UA.a, "va": VA.a, "uc": UC.a, "vc": VC.a,
            "ut": UT.a, "vt": VT.a}


def d_sw1_duo(delp, pt, w, uc, vc, xflux, yflux, cx, cy, gs: dict,
              bd: Bounds, npx: int, npy: int, *, dt: float,
              hord_tr: int = 8, hord_vt: int = 6, hord_tm: int = 6,
              hord_dp: int = 6, nord_v: int = 1, nord_t: int = 0,
              damp_v: float = 0.2, damp_t: float = 0.0,
              hydrostatic: bool = True, inline_q: bool = False,
              lim_fac: float = 1.0, duogrid: bool = True,
              workspace_sentinel: float = 1.0e30) -> dict:
    """sw_core.F90 d_sw1 (symmetryclean 500-998), DUO branch — the D-grid
    TRANSPORT stage: ut/vt, crx/cry/xfx/yfx, ra_x/ra_y, the delp/pt flux
    computation (fv_tp_2d), the allflux pack consumed by dyn_core's
    inter-panel flux averaging, and the cx/cy/xflux/yflux capacitors.
    d_sw1 computes FLUXES only — the delp/pt UPDATES happen in d_sw2 AFTER
    the averaging.

    DUO semantics: the interior ut/vt formula runs over the full ranges
    (auth 622-634); the panel-edge + corner 2x2 blocks (656-813) fire
    UNCONDITIONALLY on the global cube (their ``.not.(bounded .and. duo)``
    guard is always true) and READ ut/vt workspace cells the duo interior
    never writes — in dyn_core those are UNINITIALISED stack memory
    (upstream declares the dummies intent(out), so even a pre-call init
    would be undefined on entry).  The oracle therefore certifies a
    DEFINED workspace contract: the extract carries a documented
    intent(out)->intent(inout) shim (its ONLY deviation, drift-guarded
    by the pinned extract SHA) and both sides initialise ut/vt to
    ``workspace_sentinel`` — the sentinel-dependent cells are then
    deterministic, standard-defined, and bit-comparable (a TRANSLATION
    certification; the integrated 6-face pipeline supplies real
    exchanged halos there).  The transport uses the
    duo-gated tp_core chain (copy_corners early-return, xppm/yppm edge
    reconstructions skipped).  VERBATIM notes: the pt transport uses
    ``nord=nord_v, damp_c=damp_v`` (auth 959-961 — the plain monolithic
    d_sw used nord_t/damp_t there); hydrostatic skips the w transport;
    USE_COND / SW_DYNAMICS undefined, matching the phase-4b extraction
    convention.

    Returns a dict with crx_adv/cry_adv/xfx_adv/yfx_adv, ra_x/ra_y,
    ut/vt, allflux_x/allflux_y (k=1 slab, 4+nq=5 slots; slots 2/3/5
    untouched NaN on the hydrostatic no-tracer lane), the (corner-ghost
    mutated on plain; untouched on duo) delp/pt/w, and the accumulated
    cx/cy/xflux/yflux.
    """
    from legoesm.core.fv3_native_d_sw import _fl, fv_tp_2d

    if not duogrid:
        raise NotImplementedError(
            "d_sw1_duo is the DUO-stage port; the plain path is the "
            "certified monolithic d_sw (phase-4b)")
    if not hydrostatic or inline_q:
        raise NotImplementedError(
            "d_sw1_duo: hydrostatic, inline_q=False lane only (matches "
            "the oracle driver; w/q_con/tracer transports not exercised)")

    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, ied, jsd, jed = bd.isd, bd.ied, bd.jsd, bd.jed

    delp = fort(np.array(delp, dtype=np.float64, copy=True), isd, jsd)
    pt = fort(np.array(pt, dtype=np.float64, copy=True), isd, jsd)
    w = fort(np.array(w, dtype=np.float64, copy=True), isd, jsd)
    uc = fort(np.array(uc, dtype=np.float64, copy=True), isd, jsd)
    vc = fort(np.array(vc, dtype=np.float64, copy=True), isd, jsd)
    xflux = fort(np.array(xflux, dtype=np.float64, copy=True), is_, js)
    yflux = fort(np.array(yflux, dtype=np.float64, copy=True), is_, js)
    cx = fort(np.array(cx, dtype=np.float64, copy=True), is_, jsd)
    cy = fort(np.array(cy, dtype=np.float64, copy=True), isd, js)

    AREA = fort(gs["area"], isd, jsd)
    SIN_SG = fort(gs["sin_sg"], isd, jsd)
    COSA_U = fort(gs["cosa_u"], isd, jsd)
    COSA_V = fort(gs["cosa_v"], isd, jsd)
    # (sina_u/sina_v unused on the duo lane — only the skipped non-duo
    # edge ut/vt formulas consume them)
    RSIN_U = fort(gs["rsin_u"], isd, jsd)
    RSIN_V = fort(gs["rsin_v"], isd, jsd)
    DX = fort(gs["dx"], isd, jsd)
    DY = fort(gs["dy"], isd, jsd)
    RDXA = fort(gs["rdxa"], isd, jsd)
    RDYA = fort(gs["rdya"], isd, jsd)
    sw_corner = bool(gs.get("sw_corner", True))
    se_corner = bool(gs.get("se_corner", True))
    ne_corner = bool(gs.get("ne_corner", True))
    nw_corner = bool(gs.get("nw_corner", True))

    # OUT workspaces at the d_sw1 declared bounds, sentinel-initialised
    # (see docstring): ut (isd:ied+1, jsd:jed), vt (isd:ied, jsd:jed+1)
    ut = _fl(isd, ied + 1, jsd, jed, workspace_sentinel)
    vt = _fl(isd, ied, jsd, jed + 1, workspace_sentinel)
    crx_adv = _fl(is_, ie + 1, jsd, jed)
    xfx_adv = _fl(is_, ie + 1, jsd, jed)
    cry_adv = _fl(isd, ied, js, je + 1)
    yfx_adv = _fl(isd, ied, js, je + 1)
    ra_x = _fl(is_, ie, jsd, jed)
    ra_y = _fl(isd, ied, js, je)
    fx = _fl(is_, ie + 1, js, je)
    fy = _fl(is_, ie, js, je + 1)
    gx = _fl(is_, ie + 1, js, je)
    gy = _fl(is_, ie, js, je + 1)
    nq = 1
    allflux_x = np.full((ie + 1 - is_ + 1, je - js + 1, 4 + nq), np.nan)
    allflux_y = np.full((ie - is_ + 1, je + 1 - js + 1, 4 + nq), np.nan)

    # ---- ut/vt DUO interior (auth 622-634) ----
    for j in range(jsd, jed + 1):
        for i in range(is_, ie + 1 + 1):
            ut[i, j] = (uc[i, j] - 0.25 * COSA_U[i, j] * (
                vc[i - 1, j] + vc[i, j]
                + vc[i - 1, j + 1] + vc[i, j + 1])) * RSIN_U[i, j]
    for j in range(js, je + 1 + 1):
        for i in range(isd, ied + 1):
            vt[i, j] = (vc[i, j] - 0.25 * COSA_V[i, j] * (
                uc[i, j - 1] + uc[i + 1, j - 1]
                + uc[i, j] + uc[i + 1, j])) * RSIN_V[i, j]

    # ---- panel edges (auth 656-726; fire always on the global cube) ----
    if is_ == 1:                                     # West edge
        for j in range(jsd, jed + 1):
            if uc[1, j] * dt > 0.0:
                ut[1, j] = uc[1, j] / SIN_SG[0, j, 3 - 1]
            else:
                ut[1, j] = uc[1, j] / SIN_SG[1, j, 1 - 1]
        for j in range(max(3, js), min(npy - 2, je + 1) + 1):
            vt[0, j] = vc[0, j] - 0.25 * COSA_V[0, j] * (
                ut[0, j - 1] + ut[1, j - 1] + ut[0, j] + ut[1, j])
            vt[1, j] = vc[1, j] - 0.25 * COSA_V[1, j] * (
                ut[1, j - 1] + ut[2, j - 1] + ut[1, j] + ut[2, j])
    if (ie + 1) == npx:                              # East edge
        for j in range(jsd, jed + 1):
            if uc[npx, j] * dt > 0.0:
                ut[npx, j] = uc[npx, j] / SIN_SG[npx - 1, j, 3 - 1]
            else:
                ut[npx, j] = uc[npx, j] / SIN_SG[npx, j, 1 - 1]
        for j in range(max(3, js), min(npy - 2, je + 1) + 1):
            vt[npx - 1, j] = vc[npx - 1, j] - 0.25 * COSA_V[npx - 1, j] * (
                ut[npx - 1, j - 1] + ut[npx, j - 1]
                + ut[npx - 1, j] + ut[npx, j])
            vt[npx, j] = vc[npx, j] - 0.25 * COSA_V[npx, j] * (
                ut[npx, j - 1] + ut[npx + 1, j - 1]
                + ut[npx, j] + ut[npx + 1, j])
    if js == 1:                                      # South edge
        for i in range(isd, ied + 1):
            if vc[i, 1] * dt > 0.0:
                vt[i, 1] = vc[i, 1] / SIN_SG[i, 0, 4 - 1]
            else:
                vt[i, 1] = vc[i, 1] / SIN_SG[i, 1, 2 - 1]
        for i in range(max(3, is_), min(npx - 2, ie + 1) + 1):
            ut[i, 0] = uc[i, 0] - 0.25 * COSA_U[i, 0] * (
                vt[i - 1, 0] + vt[i, 0] + vt[i - 1, 1] + vt[i, 1])
            ut[i, 1] = uc[i, 1] - 0.25 * COSA_U[i, 1] * (
                vt[i - 1, 1] + vt[i, 1] + vt[i - 1, 2] + vt[i, 2])
    if (je + 1) == npy:                              # North edge
        for i in range(isd, ied + 1):
            if vc[i, npy] * dt > 0.0:
                vt[i, npy] = vc[i, npy] / SIN_SG[i, npy - 1, 4 - 1]
            else:
                vt[i, npy] = vc[i, npy] / SIN_SG[i, npy, 2 - 1]
        for i in range(max(3, is_), min(npx - 2, ie + 1) + 1):
            ut[i, npy - 1] = uc[i, npy - 1] - 0.25 * COSA_U[i, npy - 1] * (
                vt[i - 1, npy - 1] + vt[i, npy - 1]
                + vt[i - 1, npy] + vt[i, npy])
            ut[i, npy] = uc[i, npy] - 0.25 * COSA_U[i, npy] * (
                vt[i - 1, npy] + vt[i, npy]
                + vt[i - 1, npy + 1] + vt[i, npy + 1])

    # ---- corner 2x2 systems (auth 739-811) ----
    if sw_corner:
        damp = 1.0 / (1.0 - 0.0625 * COSA_U[2, 0] * COSA_V[1, 0])
        ut[2, 0] = (uc[2, 0] - 0.25 * COSA_U[2, 0] * (
            vt[1, 1] + vt[2, 1] + vt[2, 0] + vc[1, 0]
            - 0.25 * COSA_V[1, 0] * (ut[1, 0] + ut[1, -1] + ut[2, -1]))) \
            * damp
        damp = 1.0 / (1.0 - 0.0625 * COSA_U[0, 1] * COSA_V[0, 2])
        vt[0, 2] = (vc[0, 2] - 0.25 * COSA_V[0, 2] * (
            ut[1, 1] + ut[1, 2] + ut[0, 2] + uc[0, 1]
            - 0.25 * COSA_U[0, 1] * (vt[0, 1] + vt[-1, 1] + vt[-1, 2]))) \
            * damp
        damp = 1.0 / (1.0 - 0.0625 * COSA_U[2, 1] * COSA_V[1, 2])
        ut[2, 1] = (uc[2, 1] - 0.25 * COSA_U[2, 1] * (
            vt[1, 1] + vt[2, 1] + vt[2, 2] + vc[1, 2]
            - 0.25 * COSA_V[1, 2] * (ut[1, 1] + ut[1, 2] + ut[2, 2]))) \
            * damp
        vt[1, 2] = (vc[1, 2] - 0.25 * COSA_V[1, 2] * (
            ut[1, 1] + ut[1, 2] + ut[2, 2] + uc[2, 1]
            - 0.25 * COSA_U[2, 1] * (vt[1, 1] + vt[2, 1] + vt[2, 2]))) \
            * damp
    if se_corner:
        damp = 1.0 / (1.0 - 0.0625 * COSA_U[npx - 1, 0] * COSA_V[npx - 1, 0])
        ut[npx - 1, 0] = (uc[npx - 1, 0] - 0.25 * COSA_U[npx - 1, 0] * (
            vt[npx - 1, 1] + vt[npx - 2, 1] + vt[npx - 2, 0] + vc[npx - 1, 0]
            - 0.25 * COSA_V[npx - 1, 0] * (
                ut[npx, 0] + ut[npx, -1] + ut[npx - 1, -1]))) * damp
        damp = 1.0 / (1.0 - 0.0625 * COSA_U[npx + 1, 1] * COSA_V[npx, 2])
        vt[npx, 2] = (vc[npx, 2] - 0.25 * COSA_V[npx, 2] * (
            ut[npx, 1] + ut[npx, 2] + ut[npx + 1, 2] + uc[npx + 1, 1]
            - 0.25 * COSA_U[npx + 1, 1] * (
                vt[npx, 1] + vt[npx + 1, 1] + vt[npx + 1, 2]))) * damp
        damp = 1.0 / (1.0 - 0.0625 * COSA_U[npx - 1, 1] * COSA_V[npx - 1, 2])
        ut[npx - 1, 1] = (uc[npx - 1, 1] - 0.25 * COSA_U[npx - 1, 1] * (
            vt[npx - 1, 1] + vt[npx - 2, 1] + vt[npx - 2, 2] + vc[npx - 1, 2]
            - 0.25 * COSA_V[npx - 1, 2] * (
                ut[npx, 1] + ut[npx, 2] + ut[npx - 1, 2]))) * damp
        vt[npx - 1, 2] = (vc[npx - 1, 2] - 0.25 * COSA_V[npx - 1, 2] * (
            ut[npx, 1] + ut[npx, 2] + ut[npx - 1, 2] + uc[npx - 1, 1]
            - 0.25 * COSA_U[npx - 1, 1] * (
                vt[npx - 1, 1] + vt[npx - 2, 1] + vt[npx - 2, 2]))) * damp
    if ne_corner:
        damp = 1.0 / (1.0 - 0.0625 * COSA_U[npx - 1, npy]
                      * COSA_V[npx - 1, npy + 1])
        ut[npx - 1, npy] = (uc[npx - 1, npy]
                            - 0.25 * COSA_U[npx - 1, npy] * (
            vt[npx - 1, npy] + vt[npx - 2, npy] + vt[npx - 2, npy + 1]
            + vc[npx - 1, npy + 1]
            - 0.25 * COSA_V[npx - 1, npy + 1] * (
                ut[npx, npy] + ut[npx, npy + 1] + ut[npx - 1, npy + 1]))) \
            * damp
        damp = 1.0 / (1.0 - 0.0625 * COSA_U[npx + 1, npy - 1]
                      * COSA_V[npx, npy - 1])
        vt[npx, npy - 1] = (vc[npx, npy - 1]
                            - 0.25 * COSA_V[npx, npy - 1] * (
            ut[npx, npy - 1] + ut[npx, npy - 2] + ut[npx + 1, npy - 2]
            + uc[npx + 1, npy - 1]
            - 0.25 * COSA_U[npx + 1, npy - 1] * (
                vt[npx, npy] + vt[npx + 1, npy] + vt[npx + 1, npy - 1]))) \
            * damp
        damp = 1.0 / (1.0 - 0.0625 * COSA_U[npx - 1, npy - 1]
                      * COSA_V[npx - 1, npy - 1])
        ut[npx - 1, npy - 1] = (uc[npx - 1, npy - 1]
                                - 0.25 * COSA_U[npx - 1, npy - 1] * (
            vt[npx - 1, npy] + vt[npx - 2, npy] + vt[npx - 2, npy - 1]
            + vc[npx - 1, npy - 1]
            - 0.25 * COSA_V[npx - 1, npy - 1] * (
                ut[npx, npy - 1] + ut[npx, npy - 2] + ut[npx - 1, npy - 2]))) \
            * damp
        vt[npx - 1, npy - 1] = (vc[npx - 1, npy - 1]
                                - 0.25 * COSA_V[npx - 1, npy - 1] * (
            ut[npx, npy - 1] + ut[npx, npy - 2] + ut[npx - 1, npy - 2]
            + uc[npx - 1, npy - 1]
            - 0.25 * COSA_U[npx - 1, npy - 1] * (
                vt[npx - 1, npy] + vt[npx - 2, npy] + vt[npx - 2, npy - 1]))) \
            * damp
    if nw_corner:
        damp = 1.0 / (1.0 - 0.0625 * COSA_U[2, npy] * COSA_V[1, npy + 1])
        ut[2, npy] = (uc[2, npy] - 0.25 * COSA_U[2, npy] * (
            vt[1, npy] + vt[2, npy] + vt[2, npy + 1] + vc[1, npy + 1]
            - 0.25 * COSA_V[1, npy + 1] * (
                ut[1, npy] + ut[1, npy + 1] + ut[2, npy + 1]))) * damp
        damp = 1.0 / (1.0 - 0.0625 * COSA_U[0, npy - 1] * COSA_V[0, npy - 1])
        vt[0, npy - 1] = (vc[0, npy - 1] - 0.25 * COSA_V[0, npy - 1] * (
            ut[1, npy - 1] + ut[1, npy - 2] + ut[0, npy - 2] + uc[0, npy - 1]
            - 0.25 * COSA_U[0, npy - 1] * (
                vt[0, npy] + vt[-1, npy] + vt[-1, npy - 1]))) * damp
        damp = 1.0 / (1.0 - 0.0625 * COSA_U[2, npy - 1] * COSA_V[1, npy - 1])
        ut[2, npy - 1] = (uc[2, npy - 1] - 0.25 * COSA_U[2, npy - 1] * (
            vt[1, npy] + vt[2, npy] + vt[2, npy - 1] + vc[1, npy - 1]
            - 0.25 * COSA_V[1, npy - 1] * (
                ut[1, npy - 1] + ut[1, npy - 2] + ut[2, npy - 2]))) * damp
        vt[1, npy - 1] = (vc[1, npy - 1] - 0.25 * COSA_V[1, npy - 1] * (
            ut[1, npy - 1] + ut[1, npy - 2] + ut[2, npy - 2] + uc[2, npy - 1]
            - 0.25 * COSA_U[2, npy - 1] * (
                vt[1, npy] + vt[2, npy] + vt[2, npy - 1]))) * damp

    # ---- xfx/crx, yfx/cry (auth 830-869) ----
    for j in range(jsd, jed + 1):
        for i in range(is_, ie + 1 + 1):
            xfx_adv[i, j] = dt * ut[i, j]
    for j in range(js, je + 1 + 1):
        for i in range(isd, ied + 1):
            yfx_adv[i, j] = dt * vt[i, j]
    for j in range(jsd, jed + 1):
        for i in range(is_, ie + 1 + 1):
            if xfx_adv[i, j] > 0.0:
                crx_adv[i, j] = xfx_adv[i, j] * RDXA[i - 1, j]
                xfx_adv[i, j] = DY[i, j] * xfx_adv[i, j] * SIN_SG[i - 1, j, 3 - 1]
            else:
                crx_adv[i, j] = xfx_adv[i, j] * RDXA[i, j]
                xfx_adv[i, j] = DY[i, j] * xfx_adv[i, j] * SIN_SG[i, j, 1 - 1]
    for j in range(js, je + 1 + 1):
        for i in range(isd, ied + 1):
            if yfx_adv[i, j] > 0.0:
                cry_adv[i, j] = yfx_adv[i, j] * RDYA[i, j - 1]
                yfx_adv[i, j] = DX[i, j] * yfx_adv[i, j] * SIN_SG[i, j - 1, 4 - 1]
            else:
                cry_adv[i, j] = yfx_adv[i, j] * RDYA[i, j]
                yfx_adv[i, j] = DX[i, j] * yfx_adv[i, j] * SIN_SG[i, j, 2 - 1]

    # ---- ra_x/ra_y (auth 875-884) ----
    for j in range(jsd, jed + 1):
        for i in range(is_, ie + 1):
            ra_x[i, j] = AREA[i, j] + xfx_adv[i, j] - xfx_adv[i + 1, j]
    for j in range(js, je + 1):
        for i in range(isd, ied + 1):
            ra_y[i, j] = AREA[i, j] + yfx_adv[i, j] - yfx_adv[i, j + 1]

    # ---- delp fluxes (auth 886-887) + allflux slot 1 + capacitors ----
    # fort-wrapped gridstruct dict for the leaf callees (fv_tp_2d -> xppm/
    # yppm/deln_flux), mirroring the monolithic d_sw's gsf: the callees
    # dereference these as Fortran-index views — handing them RAW numpy
    # arrays silently wraps negative indices (isd-origin reads become
    # from-the-end reads) and corrupts the deln_flux damping increment.
    gsf = {
        "dxa": fort(gs["dxa"], isd, jsd),
        "dya": fort(gs["dya"], isd, jsd),
        "area": AREA,
        "rarea": fort(gs["rarea"], isd, jsd),
        "del6_v": fort(gs["del6_v"], isd, jsd),
        "del6_u": fort(gs["del6_u"], isd, jsd),
        "da_min": float(gs["da_min"]),
        "bounded_domain": bool(gs.get("bounded_domain", False)),
        "grid_type": int(gs.get("grid_type", 0)),
        "sw_corner": sw_corner, "se_corner": se_corner,
        "nw_corner": nw_corner, "ne_corner": ne_corner,
    }
    fv_tp_2d(delp, crx_adv, cry_adv, npx, npy, hord_dp, fx, fy,
             xfx_adv, yfx_adv, gsf, bd, ra_x, ra_y, lim_fac,
             nord=nord_v, damp_c=damp_v, duogrid=duogrid)
    for j in range(js, je + 1):
        for i in range(is_, ie + 1 + 1):
            allflux_x[i - is_, j - js, 1 - 1] = fx[i, j]
    for j in range(js, je + 1 + 1):
        for i in range(is_, ie + 1):
            allflux_y[i - is_, j - js, 1 - 1] = fy[i, j]
    for j in range(jsd, jed + 1):
        for i in range(is_, ie + 1 + 1):
            cx[i, j] = cx[i, j] + crx_adv[i, j]
    for j in range(js, je + 1):
        for i in range(is_, ie + 1 + 1):
            xflux[i, j] = xflux[i, j] + fx[i, j]
    for j in range(js, je + 1 + 1):
        for i in range(isd, ied + 1):
            cy[i, j] = cy[i, j] + cry_adv[i, j]
        for i in range(is_, ie + 1):
            yflux[i, j] = yflux[i, j] + fy[i, j]

    # ---- pt fluxes (auth 959-961: nord=nord_v, damp_c=damp_v) ----
    fv_tp_2d(pt, crx_adv, cry_adv, npx, npy, hord_tm, gx, gy,
             xfx_adv, yfx_adv, gsf, bd, ra_x, ra_y, lim_fac,
             mfx=fx, mfy=fy, mass=delp, nord=nord_v, damp_c=damp_v,
             duogrid=duogrid)
    for j in range(js, je + 1):
        for i in range(is_, ie + 1 + 1):
            allflux_x[i - is_, j - js, 4 - 1] = gx[i, j]
    for j in range(js, je + 1 + 1):
        for i in range(is_, ie + 1):
            allflux_y[i - is_, j - js, 4 - 1] = gy[i, j]

    return {"crx_adv": crx_adv.a, "cry_adv": cry_adv.a,
            "xfx_adv": xfx_adv.a, "yfx_adv": yfx_adv.a,
            "ra_x": ra_x.a, "ra_y": ra_y.a, "ut": ut.a, "vt": vt.a,
            "allflux_x": allflux_x, "allflux_y": allflux_y,
            "delp": delp.a, "pt": pt.a, "w": w.a,
            "cx": cx.a, "cy": cy.a, "xflux": xflux.a, "yflux": yflux.a}


def d_sw2_duo(delp, pt, allflux_x, allflux_y, gs: dict, bd: Bounds, *,
              hydrostatic: bool = True, inline_q: bool = False,
              workspace_sentinel: float = 1.0e30) -> dict:
    """sw_core.F90 d_sw2 (symmetryclean 1000-1199) — the post-averaging
    delp/pt UPDATE stage, on the oracle lane (hydrostatic, inline_q=F,
    no SW_DYNAMICS/USE_COND): heat_source zeroed over the compute
    domain, then the else-arm update from allflux slots 1 (delp fx/fy)
    and 4 (pt gx/gy), auth 1181-1196:

        pt   = pt*delp + (gx(i,j)-gx(i+1,j)+gy(i,j)-gy(i,j+1))*rarea
        delp = delp    + (fx(i,j)-fx(i+1,j)+fy(i,j)-fy(i,j+1))*rarea
        pt   = pt / delp

    d_sw2 has NO duo/edge branches — a pure cell update given the
    fluxes.  ptc and dw (upstream intent(OUT); shimmed to inout in the
    extract so the sentinel round-trip is standard-defined — see the
    extract header) are never written on this lane; the port returns
    them filled with ``workspace_sentinel`` mirroring the oracle
    driver's 1e30 init (documented scratch semantics, bit-comparable).

    ``allflux_x``/``allflux_y`` are the d_sw1_duo output stacks
    ((res+1, res, 5) / (res, res+1, 5), slot axis last).  ``delp``/
    ``pt`` are data-domain arrays (d_sw1 duo leaves them unmutated).
    Returns dict(delp, pt, heat_source, ptc, dw).
    """
    if not hydrostatic or inline_q:
        raise NotImplementedError(
            "d_sw2_duo: hydrostatic, inline_q=False lane only (matches "
            "the oracle driver; w/q_con/tracer updates not exercised)")

    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, jsd = bd.isd, bd.jsd

    delp = fort(np.array(delp, dtype=np.float64, copy=True), isd, jsd)
    pt = fort(np.array(pt, dtype=np.float64, copy=True), isd, jsd)
    RAREA = fort(gs["rarea"], isd, jsd)

    # allflux compute rings -> fort views at (is, js) like the Fortran
    # dummy bounds allflux_x(is:ie+1, js:je, k, slot)
    fx = fort(np.array(allflux_x[:, :, 0], dtype=np.float64), is_, js)
    gx = fort(np.array(allflux_x[:, :, 3], dtype=np.float64), is_, js)
    fy = fort(np.array(allflux_y[:, :, 0], dtype=np.float64), is_, js)
    gy = fort(np.array(allflux_y[:, :, 3], dtype=np.float64), is_, js)

    # heat_source zeroing (auth 1074-1078, #ifndef SW_DYNAMICS)
    heat_source = np.zeros((ie - is_ + 1, je - js + 1))

    # else-arm update (auth 1181-1196; inline_q=F)
    for j in range(js, je + 1):
        for i in range(is_, ie + 1):
            pt[i, j] = pt[i, j] * delp[i, j] + (
                gx[i, j] - gx[i + 1, j] + gy[i, j] - gy[i, j + 1]
            ) * RAREA[i, j]
            delp[i, j] = delp[i, j] + (
                fx[i, j] - fx[i + 1, j] + fy[i, j] - fy[i, j + 1]
            ) * RAREA[i, j]
            pt[i, j] = pt[i, j] / delp[i, j]

    ptc = np.full_like(delp.a, workspace_sentinel)
    dw = np.full((ie - is_ + 1, je - js + 1), workspace_sentinel)
    return {"delp": delp.a, "pt": pt.a, "heat_source": heat_source,
            "ptc": ptc, "dw": dw}
