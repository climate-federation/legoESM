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


def d_sw3_duo(u, v, uc, vc, gs: dict, bd: Bounds, npx: int, npy: int, *,
              dt: float, hord_mt: int = 6, duogrid: bool = True) -> dict:
    """sw_core.F90 d_sw3 (symmetryclean 1201-1388), DUO branch — the
    KE-flux stage: B-grid contravariant vb/ub over the FULL unclamped
    ranges (auth 1260-1266: ``bounded .or. duogrid`` -> is2=is, ie1=ie+1,
    js2=js, je1=je+1) with the INTERIOR formula everywhere (auth
    1271-1275 / 1330-1338; the vt/ut-based edge + corner extrapolations
    live in the SKIPPED non-duo else, so ut/vt are never read on this
    lane — no workspace sentinel needed), then the ytp_v/xtp_u advective
    fluxes and the four outputs dyn_core carries to d_sw5's KE assembly:
    ubbtemp/vbbtemp (post-ytp_v ub + pre-xtp_u vb, auth 1321-1326) and
    ubb/vbb (final, auth 1381-1386).

    VERBATIM quirks preserved: d_sw3 passes ``bounded_domain=.false.``
    to ytp_v/xtp_u as a LITERAL (auth 1318/1378 — the commented-out
    originals passed the real flag); the duo behaviour inside them comes
    from the symmetryclean ``gridstruct%dg%is_initialized`` gates, which
    the port carries as ``duogrid=True`` on the certified plain
    ytp_v/xtp_u (range gate + the WMP smt5/smt6 edge-fix blocks that the
    symmetryclean tree REMOVED).

    u/v are read-only here (only ytp_v/xtp_u consume them).  grid_type=0
    lane only.  Returns dict(ubbtemp, vbbtemp, ubb, vbb) on the B-grid
    compute ring (is:ie+1, js:je+1).
    """
    from legoesm.core.fv3_native_d_sw import _fl, xtp_u, ytp_v

    if not duogrid:
        raise NotImplementedError(
            "d_sw3_duo is the DUO-stage port; the plain path is the "
            "certified monolithic d_sw (phase-4b)")

    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, ied, jsd, jed = bd.isd, bd.ied, bd.jsd, bd.jed

    u = fort(np.array(u, dtype=np.float64, copy=True), isd, jsd)
    v = fort(np.array(v, dtype=np.float64, copy=True), isd, jsd)
    uc = fort(np.array(uc, dtype=np.float64, copy=True), isd, jsd)
    vc = fort(np.array(vc, dtype=np.float64, copy=True), isd, jsd)

    COSA = fort(gs["cosa"], isd, jsd)
    RSINA = fort(gs["rsina"], isd, jsd)
    DX = fort(gs["dx"], isd, jsd)
    RDX = fort(gs["rdx"], isd, jsd)
    DY = fort(gs["dy"], isd, jsd)
    RDY = fort(gs["rdy"], isd, jsd)

    dt5 = 0.5 * dt

    # duo ranges (auth 1260-1263)
    is2, ie1 = is_, ie + 1
    js2, je1 = js, je + 1

    ub = _fl(is_, ie + 1, js, je + 1)
    vb = _fl(is_, ie + 1, js, je + 1)

    # vb: duo interior formula everywhere (auth 1271-1275)
    for j in range(js2, je1 + 1):
        for i in range(is2, ie1 + 1):
            vb[i, j] = dt5 * (vc[i - 1, j] + vc[i, j]
                              - (uc[i, j - 1] + uc[i, j]) * COSA[i, j]) \
                * RSINA[i, j]

    # auth 1318: bounded_domain literal .false.; duo via dg gates
    ytp_v(is_, ie, js, je, isd, ied, jsd, jed, vb, u, v, ub, hord_mt,
          DY, RDY, npx, npy, 0, False, 1.0, duogrid=True)

    ubbtemp = _fl(is_, ie + 1, js, je + 1)
    vbbtemp = _fl(is_, ie + 1, js, je + 1)
    for j in range(js, je + 1 + 1):
        for i in range(is_, ie + 1 + 1):
            ubbtemp[i, j] = ub[i, j]
            vbbtemp[i, j] = vb[i, j]

    # ub: duo interior formula everywhere (auth 1330-1338)
    for j in range(js, je + 1 + 1):
        for i in range(is2, ie1 + 1):
            ub[i, j] = dt5 * (uc[i, j - 1] + uc[i, j]
                              - (vc[i - 1, j] + vc[i, j]) * COSA[i, j]) \
                * RSINA[i, j]

    # auth 1378: bounded_domain literal .false.
    xtp_u(is_, ie, js, je, isd, ied, jsd, jed, ub, u, v, vb, hord_mt,
          DX, RDX, npx, npy, 0, False, 1.0, duogrid=True)

    ubb = _fl(is_, ie + 1, js, je + 1)
    vbb = _fl(is_, ie + 1, js, je + 1)
    for j in range(js, je + 1 + 1):
        for i in range(is_, ie + 1 + 1):
            ubb[i, j] = ub[i, j]
            vbb[i, j] = vb[i, j]

    return {"ubbtemp": ubbtemp.a, "vbbtemp": vbbtemp.a,
            "ubb": ubb.a, "vbb": vbb.a}


def d_sw4_duo(u, v, ut, vt, ke, gs: dict, bd: Bounds, npx: int, npy: int,
              *, dt: float) -> dict:
    """sw_core.F90 d_sw4 (symmetryclean 1390-1472) — the 4-corner KE
    fix.  Its guard ``.not.bounded .or. .not.duogrid`` (auth 1440) is
    ALWAYS TRUE on the global cube (bounded=F), so the corner formulas
    fire on the duo lane too, reading u/v and the d_sw1 ut/vt workspace
    at cells the duo interior DOES write (ut over is:ie+1 x jsd:jed, vt
    over isd:ied x js:je+1 cover every corner read) — all inputs fully
    defined, no sentinel dependence.

    ke is INTENT(INOUT) upstream (in dyn_core it arrives as the inline
    KE assembly kee = 0.5*(ubbtemp*vbbtemp + ubb*vbb)); the oracle
    initialises it to 1e30 on both sides and the stage writes ONLY the
    four corner B-nodes — the full-domain dump proves exactly that.
    Returns dict(ke) (data-domain B-array, corners updated).
    """
    isd, jsd = bd.isd, bd.jsd  # d_sw4 uses only corner indices (1,npx,npy)

    u = fort(np.array(u, dtype=np.float64, copy=True), isd, jsd)
    v = fort(np.array(v, dtype=np.float64, copy=True), isd, jsd)
    ut = fort(np.array(ut, dtype=np.float64, copy=True), isd, jsd)
    vt = fort(np.array(vt, dtype=np.float64, copy=True), isd, jsd)
    ke = fort(np.array(ke, dtype=np.float64, copy=True), isd, jsd)

    sw_corner = bool(gs.get("sw_corner", True))
    se_corner = bool(gs.get("se_corner", True))
    ne_corner = bool(gs.get("ne_corner", True))
    nw_corner = bool(gs.get("nw_corner", True))

    # auth 1440-1466 (guard always true at bounded=F)
    dt6 = dt / 6.0
    if sw_corner:
        ke[1, 1] = dt6 * ((ut[1, 1] + ut[1, 0]) * u[1, 1]
                          + (vt[1, 1] + vt[0, 1]) * v[1, 1]
                          + (ut[1, 1] + vt[1, 1]) * u[0, 1])
    if se_corner:
        i = npx
        ke[i, 1] = dt6 * ((ut[i, 1] + ut[i, 0]) * u[i - 1, 1]
                          + (vt[i, 1] + vt[i - 1, 1]) * v[i, 1]
                          + (ut[i, 1] - vt[i - 1, 1]) * u[i, 1])
    if ne_corner:
        i, j = npx, npy
        ke[i, j] = dt6 * ((ut[i, j] + ut[i, j - 1]) * u[i - 1, j]
                          + (vt[i, j] + vt[i - 1, j]) * v[i, j - 1]
                          + (ut[i, j - 1] + vt[i - 1, j]) * u[i, j])
    if nw_corner:
        j = npy
        ke[1, j] = dt6 * ((ut[1, j] + ut[1, j - 1]) * u[1, j]
                          + (vt[1, j] + vt[0, j]) * v[1, j - 1]
                          + (ut[1, j - 1] - vt[1, j]) * u[0, j])

    return {"ke": ke.a}


def d_sw5_duo(delp, u, v, uc, vc, ua, va, divg_d, crx_adv, cry_adv,
              xfx_adv, yfx_adv, ra_x, ra_y, ke, gs: dict, bd: Bounds,
              npx: int, npy: int, *, dt: float, hord_vt: int = 6,
              nord: int = 1, dddmp: float = 0.2, d2_bg: float = 0.0,
              d4_bg: float = 0.12, d_con: float = 0.0,
              hydrostatic: bool = True, lim_fac: float = 1.0,
              workspace_sentinel: float = 1.0e30) -> dict:
    """sw_core.F90 d_sw5 (symmetryclean 1474-1869), DUO branch, on the
    oracle lane (nord=1, hydrostatic, d_con=0, grid_type=0, not
    stretched): vorticity prep (vt=u*dx, ut=v*dy over the FULL data
    domain — d_sw5 OVERWRITES the d_sw1 ut/vt workspace, so the port
    builds fresh arrays and takes no ut/vt inputs), wk = volume-mean
    relative vorticity, the nord=1 higher-order divergence damping
    n-loop (delpc = saved divg_d; uc/vc CLOBBERED as gradient
    workspaces — auth semantics, returned as outputs; duo SKIPS the
    corner-term removal; fill_c is false at nt=0 and duo-excluded
    anyway), a2b_ord4(duo)+Smagorinsky vort, ke += damping increment
    over the B compute ring, and the vorticity-flux transport
    fv_tp_2d(wk+f0) -> vortfluxx/vortfluxy.

    crx/cry/xfx/yfx/ra_x/ra_y are the d_sw1 outputs (upstream declares
    the first four intent(OUT) yet only reads them — the extract shims
    them to inout; see the extract header).  ptc is unwritten on the
    nord>0 branch and ub/vb are untouched at d_con=0 — all three
    returned as ``workspace_sentinel`` fills mirroring the driver.
    Returns dict(delpc, divg_d, wk, ke, vortfluxx, vortfluxy, uc, vc,
    ptc, ub, vb).
    """
    from legoesm.core.fv3_native_d_sw import _fl, a2b_ord4, fv_tp_2d

    if not hydrostatic or d_con > 1.0e-5 or nord != 1:
        raise NotImplementedError(
            "d_sw5_duo: oracle lane only (hydrostatic, d_con=0, nord=1)")

    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, ied, jsd, jed = bd.isd, bd.ied, bd.jsd, bd.jed
    ng = bd.ng

    u = fort(np.array(u, dtype=np.float64, copy=True), isd, jsd)
    v = fort(np.array(v, dtype=np.float64, copy=True), isd, jsd)
    uc = fort(np.array(uc, dtype=np.float64, copy=True), isd, jsd)
    vc = fort(np.array(vc, dtype=np.float64, copy=True), isd, jsd)
    ua = fort(np.array(ua, dtype=np.float64, copy=True), isd, jsd)
    va = fort(np.array(va, dtype=np.float64, copy=True), isd, jsd)
    del delp, ua, va  # read only on the nord=0 / non-hydro branches
    divg_d = fort(np.array(divg_d, dtype=np.float64, copy=True), isd, jsd)
    ke = fort(np.array(ke, dtype=np.float64, copy=True), isd, jsd)
    crx = fort(np.array(crx_adv, dtype=np.float64), is_, jsd)
    xfx = fort(np.array(xfx_adv, dtype=np.float64), is_, jsd)
    cry = fort(np.array(cry_adv, dtype=np.float64), isd, js)
    yfx = fort(np.array(yfx_adv, dtype=np.float64), isd, js)
    ra_x = fort(np.array(ra_x, dtype=np.float64), is_, jsd)
    ra_y = fort(np.array(ra_y, dtype=np.float64), isd, js)

    RAREA = fort(gs["rarea"], isd, jsd)
    RAREA_C = fort(gs["rarea_c"], isd, jsd)
    DIVG_U = fort(gs["divg_u"], isd, jsd)
    DIVG_V = fort(gs["divg_v"], isd, jsd)
    DX = fort(gs["dx"], isd, jsd)
    DY = fort(gs["dy"], isd, jsd)
    F0 = fort(gs["f0"], isd, jsd)
    da_min_c = float(gs["da_min_c"])

    # fort-wrapped gridstruct for the a2b_ord4 + fv_tp_2d callees
    # (raw arrays silently wrap negative indices — d_sw1 lesson)
    gsf = {
        "dxa": fort(gs["dxa"], isd, jsd),
        "dya": fort(gs["dya"], isd, jsd),
        "area": fort(gs["area"], isd, jsd),
        "rarea": RAREA,
        "del6_v": fort(gs["del6_v"], isd, jsd),
        "del6_u": fort(gs["del6_u"], isd, jsd),
        "grid_lon": fort(gs["grid_lon"], isd, jsd),
        "grid_lat": fort(gs["grid_lat"], isd, jsd),
        "agrid_lon": fort(gs["agrid_lon"], isd, jsd),
        "agrid_lat": fort(gs["agrid_lat"], isd, jsd),
        "edge_w": gs["edge_w"], "edge_e": gs["edge_e"],
        "edge_s": gs["edge_s"], "edge_n": gs["edge_n"],
        "da_min": float(gs["da_min"]), "da_min_c": da_min_c,
        "bounded_domain": bool(gs.get("bounded_domain", False)),
        "grid_type": 0,
        "sw_corner": bool(gs.get("sw_corner", True)),
        "se_corner": bool(gs.get("se_corner", True)),
        "nw_corner": bool(gs.get("nw_corner", True)),
        "ne_corner": bool(gs.get("ne_corner", True)),
    }

    # ---- vorticity prep (auth 1582-1598): fresh ut/vt workspaces ----
    ut = _fl(isd, ied + 1, jsd, jed)
    vt = _fl(isd, ied, jsd, jed + 1)
    for j in range(jsd, jed + 1 + 1):
        for i in range(isd, ied + 1):
            vt[i, j] = u[i, j] * DX[i, j]
    for j in range(jsd, jed + 1):
        for i in range(isd, ied + 1 + 1):
            ut[i, j] = v[i, j] * DY[i, j]
    wk = _fl(isd, ied, jsd, jed)
    for j in range(jsd, jed + 1):
        for i in range(isd, ied + 1):
            wk[i, j] = RAREA[i, j] * (vt[i, j] - vt[i, j + 1]
                                      - ut[i, j] + ut[i + 1, j])

    # ---- nord=1 higher-order divergence damping (auth 1731-1824) ----
    delpc = _fl(isd, ied, jsd, jed)
    delpc.a.fill(workspace_sentinel)  # only the B compute ring written
    for j in range(js, je + 1 + 1):
        for i in range(is_, ie + 1 + 1):
            delpc[i, j] = divg_d[i, j]

    nt = 0  # n-loop: n=1..nord with nord=1; fill_c false (duo-excluded)
    for j in range(js - nt, je + 1 + nt + 1):
        for i in range(is_ - 1 - nt, ie + 1 + nt + 1):
            vc[i, j] = (divg_d[i + 1, j] - divg_d[i, j]) * DIVG_U[i, j]
    for j in range(js - 1 - nt, je + 1 + nt + 1):
        for i in range(is_ - nt, ie + 1 + nt + 1):
            uc[i, j] = (divg_d[i, j + 1] - divg_d[i, j]) * DIVG_V[i, j]
    for j in range(js - nt, je + 1 + nt + 1):
        for i in range(is_ - nt, ie + 1 + nt + 1):
            divg_d[i, j] = uc[i, j - 1] - uc[i, j] + vc[i - 1, j] - vc[i, j]
    # duo: corner-term removal SKIPPED (auth 1771 guard .not.duogrid)
    for j in range(js - nt, je + 1 + nt + 1):
        for i in range(is_ - nt, ie + 1 + nt + 1):
            divg_d[i, j] = divg_d[i, j] * RAREA_C[i, j]

    # ---- Smagorinsky vort (auth 1790-1806; dddmp >= 1e-5) ----
    vort = _fl(isd, ied, jsd, jed)
    if dddmp < 1.0e-5:
        vort.a.fill(0.0)
    else:
        a2b_ord4(wk, vort, gsf, npx, npy, is_, ie, js, je, ng,
                 replace=False, duogrid=True)
        for j in range(js, je + 1 + 1):
            for i in range(is_, ie + 1 + 1):
                vort[i, j] = abs(dt) * np.sqrt(delpc[i, j] ** 2
                                               + vort[i, j] ** 2)

    dd8 = (da_min_c * d4_bg) ** (nord + 1)
    for j in range(js, je + 1 + 1):
        for i in range(is_, ie + 1 + 1):
            damp2 = da_min_c * max(d2_bg, min(0.20, dddmp * vort[i, j]))
            vort[i, j] = damp2 * delpc[i, j] + dd8 * divg_d[i, j]
            ke[i, j] = ke[i, j] + vort[i, j]

    # d_con=0: ub/vb dissipation strips untouched (sentinel round-trip)

    # ---- vorticity transport (auth 1838-1862, hydrostatic) ----
    for j in range(jsd, jed + 1):
        for i in range(isd, ied + 1):
            vort[i, j] = wk[i, j] + F0[i, j]

    vortfluxx = _fl(is_, ie + 1, js, je)
    vortfluxy = _fl(is_, ie, js, je + 1)
    fv_tp_2d(vort, crx, cry, npx, npy, hord_vt, vortfluxx, vortfluxy,
             xfx, yfx, gsf, bd, ra_x, ra_y, lim_fac, duogrid=True)

    ptc = np.full((ied - isd + 1, jed - jsd + 1), workspace_sentinel)
    ub = np.full((ie + 1 - is_ + 1, je + 1 - js + 1), workspace_sentinel)
    vb = np.full((ie + 1 - is_ + 1, je + 1 - js + 1), workspace_sentinel)
    return {"delpc": delpc.a, "divg_d": divg_d.a, "wk": wk.a, "ke": ke.a,
            "vortfluxx": vortfluxx.a, "vortfluxy": vortfluxy.a,
            "uc": uc.a, "vc": vc.a, "ut": ut.a, "vt": vt.a,
            "ptc": ptc, "ub": ub, "vb": vb}


def d_sw6_duo(u, v, ut, vt, ke, wk, vortfluxx, vortfluxy, gs: dict,
              bd: Bounds, npx: int, npy: int, *, nord_v: int = 1,
              damp_v: float = 0.2, d_con: float = 0.0,
              duogrid: bool = True,
              workspace_sentinel: float = 1.0e30) -> dict:
    """sw_core.F90 d_sw6 (symmetryclean 1871-2006) — the final
    circulation-form wind update on the oracle lane (damp_v=0.2,
    d_con=0):

        u = vt + ke - ke(i+1,j) + fy      (fy = vortfluxy)
        v = ut + ke - ke(i,j+1) - fx      (fx = vortfluxx)

    then the damp_v del6 vorticity damping (del6_vt_flux CLOBBERS
    ut/vt as its flux outputs — the certified plain port with the duo
    copy_corners gating) and the diffusive-flux add u += vt, v -= ut.
    d_con=0 skips the heating block, so ub/vb/heat_source are
    untouched (sentinel round-trips; heat_source stays at the driver's
    1e30 because d_sw2 — which zeroes it in the real pipeline — is
    not part of this chain).

    Inputs come from the certified d_sw5 stage: ut/vt are its v*dy /
    u*dx overwrites, ke its damped-KE output, wk its relative
    vorticity, vortfluxx/vortfluxy its transport fluxes.  u/v are the
    committed D-grid winds; only the compute ranges are rewritten
    (halo strips keep the inputs — dumped and compared full-domain).
    Returns dict(u, v, ut, vt, ub, vb, heat_source).
    """
    from legoesm.core.fv3_native_d_sw import _fl, del6_vt_flux

    if d_con > 1.0e-5:
        raise NotImplementedError("d_sw6_duo: d_con=0 oracle lane only")
    if not duogrid:
        raise NotImplementedError(
            "d_sw6_duo is the DUO-stage port; the plain path is the "
            "certified monolithic d_sw (phase-4b)")

    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, ied, jsd, jed = bd.isd, bd.ied, bd.jsd, bd.jed

    u = fort(np.array(u, dtype=np.float64, copy=True), isd, jsd)
    v = fort(np.array(v, dtype=np.float64, copy=True), isd, jsd)
    ut = fort(np.array(ut, dtype=np.float64, copy=True), isd, jsd)
    vt = fort(np.array(vt, dtype=np.float64, copy=True), isd, jsd)
    ke = fort(np.array(ke, dtype=np.float64, copy=True), isd, jsd)
    wk = fort(np.array(wk, dtype=np.float64, copy=True), isd, jsd)
    fx = fort(np.array(vortfluxx, dtype=np.float64, copy=True), is_, js)
    fy = fort(np.array(vortfluxy, dtype=np.float64, copy=True), is_, js)

    da_min_c = float(gs["da_min_c"])
    gsf = {
        "del6_v": fort(gs["del6_v"], isd, jsd),
        "del6_u": fort(gs["del6_u"], isd, jsd),
        "rarea": fort(gs["rarea"], isd, jsd),
        "bounded_domain": bool(gs.get("bounded_domain", False)),
        "grid_type": 0,
        "sw_corner": bool(gs.get("sw_corner", True)),
        "se_corner": bool(gs.get("se_corner", True)),
        "nw_corner": bool(gs.get("nw_corner", True)),
        "ne_corner": bool(gs.get("ne_corner", True)),
    }

    # ---- final circulation-form wind update (auth 1934-1944) ----
    for j in range(js, je + 1 + 1):
        for i in range(is_, ie + 1):
            u[i, j] = vt[i, j] + ke[i, j] - ke[i + 1, j] + fy[i, j]
    for j in range(js, je + 1):
        for i in range(is_, ie + 1 + 1):
            v[i, j] = ut[i, j] + ke[i, j] - ke[i, j + 1] - fx[i, j]

    # ---- damp_v del6 vorticity damping (auth 1948-1951) ----
    vort = _fl(isd, ied, jsd, jed)
    if damp_v > 1.0e-5:
        damp4 = (damp_v * da_min_c) ** (nord_v + 1)
        del6_vt_flux(nord_v, npx, npy, damp4, wk, vort, ut, vt, gsf, bd,
                     duogrid=True)

    # d_con=0: heating block skipped (auth 1953-1986)

    # ---- add diffusive fluxes to the momentum equation (auth 1989-2000) --
    if damp_v > 1.0e-5:
        for j in range(js, je + 1 + 1):
            for i in range(is_, ie + 1):
                u[i, j] = u[i, j] + vt[i, j]
        for j in range(js, je + 1):
            for i in range(is_, ie + 1 + 1):
                v[i, j] = v[i, j] - ut[i, j]

    ub = np.full((ie + 1 - is_ + 1, je + 1 - js + 1), workspace_sentinel)
    vb = np.full((ie + 1 - is_ + 1, je + 1 - js + 1), workspace_sentinel)
    heat_source = np.full((ie - is_ + 1, je - js + 1), workspace_sentinel)
    return {"u": u.a, "v": v.a, "ut": ut.a, "vt": vt.a,
            "ub": ub, "vb": vb, "heat_source": heat_source}
