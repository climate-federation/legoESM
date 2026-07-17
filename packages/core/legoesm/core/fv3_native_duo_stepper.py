"""Six-face integrated duo SW stepper — phase 4c production assembly.

Wires the certified duo stage ports (c_sw duo, d_sw1..d_sw6 duo) and
the six-face exchange/averaging analogs into the dyn_core duo sequence
on all six cube faces.  This is the production caller the per-stage
TRANSLATION certificates deliberately excluded (codex avg-r1 P0): the
two inter-panel averaging sites (dyn_core.F90:853-900 C-ring,
968-1020 BGRID_NE) and the c_sw output exchanges run on REAL neighbor
data here instead of the single-face sentinel contracts.

Assembly ladder (each sub-brick gated before the next):
  SB1  build_six_face_duo_context + csw_step_sixface   (this module)
  SB2  PG-C update + d_sw1 + allflux averaging + d_sw2
  SB3  d_sw3 + BGRID averaging + kee + d_sw4/5/6 = one acoustic step
  SB4  time loop + Williamson-2 run -> the duo-target gate
       (scripts/validate/fv3_native/w2_duo_oracle_gate.py)

Workspace semantics: dyn_core's utt/vtt are UNINITIALIZED stack
upstream (the always-fire d_sw1 edge blocks read them; benign there
because the panel-edge cosa metrics are near-zero).  The stepper makes
the DEFINED choice ut/vt = 0 at entry each step — documented, and the
Williamson-2 duo-target gate is the arbiter.
"""

from __future__ import annotations

import numpy as np
from legoesm.grids.fv3_native_gridstruct import (
    analytic_swcore_state,
    build_fv3_native_gridstruct,
    exchange_bgrid_scalar_halos,
    exchange_cgrid_vector_halos,
)


def build_six_face_duo_context(n: int, ng: int = 3) -> dict:
    """Gridstructs + Bounds for all six faces (certified builders)."""
    from legoesm.core.fv3_native_sw_core import Bounds

    gs6 = [build_fv3_native_gridstruct(n, ng, tile=t) for t in range(1, 7)]
    for gs in gs6:
        gs.setdefault("bounded_domain", False)
        gs.setdefault("grid_type", 0)
        gs.setdefault("sw_corner", True)
        gs.setdefault("se_corner", True)
        gs.setdefault("ne_corner", True)
        gs.setdefault("nw_corner", True)
    return {"n": n, "ng": ng, "gs6": gs6,
            "bd": Bounds.single_tile(n, ng)}


def analytic_six_face_state(ctx: dict, **kw) -> list:
    """Per-face analytic solid-body SW state (Williamson-2-like)."""
    return [analytic_swcore_state(gs, **kw) for gs in ctx["gs6"]]


def csw_step_sixface(ctx: dict, states: list, dt2: float,
                     duogrid: bool = True) -> list:
    """SB1: certified duo c_sw on every face + the two post-c_sw
    exchanges dyn_core performs before d_sw (divgd CORNER-scalar,
    uc/vc CGRID_NE vector) on the six-face neighbor machinery.

    Returns the per-face c_sw output dicts with exchanged halos.
    """
    from legoesm.core.fv3_native_sw_core import c_sw

    n, ng = ctx["n"], ctx["ng"]
    bd = ctx["bd"]
    npx = n + 1
    outs = []
    for t in range(1, 7):
        st = states[t - 1]
        outs.append(c_sw(st["delp"], st["pt"], st.get("w", st["pt"] * 0.0),
                         st["u"], st["v"], ctx["gs6"][t - 1], bd,
                         npx, npx, dt2, duogrid=duogrid))

    divgd6 = [o["divg_d"] for o in outs]
    uc6 = [o["uc"] for o in outs]
    vc6 = [o["vc"] for o in outs]
    for t in range(1, 7):
        exchange_bgrid_scalar_halos(divgd6, t, n, ng)
        exchange_cgrid_vector_halos(uc6, vc6, t, n, ng)
    return outs


def geopk_sw_1lev(delpc: np.ndarray, hs: np.ndarray, bd) -> tuple:
    """dyn_core.F90 geopk (2660-2790), SW_DYNAMICS branch, km=1, CG=T.

    SW convention: akap=1, ptop=0, pt≡1 — pk(1)=ptop**akap=0,
    pk(2)=exp(akap*log(pe))=delp, gz(2)=hs, gz(1)=gz(2)+pt*(pk(2)-pk(1))
    (the SW_DYNAMICS increment, NO cp_air).  CG=.true. ranges:
    ifirst=is-1..ie+1 (c_sw's delpc compute ring covers exactly this).
    Returns (pkc, gz) with a trailing 2-level axis on the data domain.
    """
    is_, ie = bd.is_, bd.ie
    m = delpc.shape[0]
    pkc = np.zeros((m, m, 2))
    gz = np.zeros((m, m, 2))
    lo = 1 - bd.ng
    sl = slice(is_ - 1 - lo, ie + 1 - lo + 1)
    pkc[sl, sl, 0] = 0.0
    pkc[sl, sl, 1] = np.exp(1.0 * np.log(delpc[sl, sl]))
    gz[sl, sl, 1] = hs[sl, sl]
    gz[sl, sl, 0] = gz[sl, sl, 1] + 1.0 * (pkc[sl, sl, 1] - pkc[sl, sl, 0])
    return pkc, gz


def p_grad_c_1lev(dt2: float, delpc, pkc, gz, uc, vc, gs: dict, bd):
    """dyn_core.F90 p_grad_c (2073-2132), km=1, hydrostatic.

    wk = pkc(:,:,2) - pkc(:,:,1); the classic cross-term PG updates
    uc over (is:ie+1, js:je) and vc over (is:ie, js:je+1).  Mutates
    uc/vc (numpy data-domain arrays) in place; delpc unused on the
    hydrostatic branch (kept for signature fidelity).
    """
    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    lo = 1 - bd.ng
    rdxc = gs["rdxc"]
    rdyc = gs["rdyc"]
    wk = pkc[:, :, 1] - pkc[:, :, 0]

    def wk_at(i, j):
        return wk[i - lo, j - lo]

    def gz_at(i, j, k):
        return gz[i - lo, j - lo, k - 1]

    def pk_at(i, j, k):
        return pkc[i - lo, j - lo, k - 1]

    for j in range(js, je + 1):
        for i in range(is_, ie + 1 + 1):
            uc[i - lo, j - lo] += dt2 * rdxc[i - lo, j - lo] / (
                wk_at(i - 1, j) + wk_at(i, j)) * (
                (gz_at(i - 1, j, 2) - gz_at(i, j, 1))
                * (pk_at(i, j, 2) - pk_at(i - 1, j, 1))
                + (gz_at(i - 1, j, 1) - gz_at(i, j, 2))
                * (pk_at(i - 1, j, 2) - pk_at(i, j, 1)))
    for j in range(js, je + 1 + 1):
        for i in range(is_, ie + 1):
            vc[i - lo, j - lo] += dt2 * rdyc[i - lo, j - lo] / (
                wk_at(i, j - 1) + wk_at(i, j)) * (
                (gz_at(i, j - 1, 2) - gz_at(i, j, 1))
                * (pk_at(i, j, 2) - pk_at(i, j - 1, 1))
                + (gz_at(i, j - 1, 1) - gz_at(i, j, 2))
                * (pk_at(i, j - 1, 2) - pk_at(i, j, 1)))


def dsw12_step_sixface(ctx: dict, states: list, csw_outs: list,
                       dt: float) -> list:
    """SB2: geopk(SW,1-lev) + p_grad_c per face, the post-PG duo
    exchanges, d_sw1 per face, the C-ring inter-panel flux averaging
    (dyn_core.F90:853-900 — the first excluded mpp site, now live),
    then d_sw2 per face on the AVERAGED slots.

    Workspace choice: d_sw1's ut/vt workspaces enter as ZEROS
    (workspace_sentinel=0.0) — the defined analog of upstream's
    uninitialized stack (benign via near-zero panel-edge cosa).
    INTERIM (documented): the post-PG uc/vc + divgd exchanges use the
    mpp-analog index-copy helpers; the authoritative duo lane uses the
    ext_scalar/ext_vector k2e machinery (dyn_core 652-655) — swap
    staged with SB3 before the W2 gate.

    Returns per-face dicts: d_sw1 outputs + averaged allflux + the
    d_sw2-updated delp/pt.
    """
    from legoesm.core.fv3_native_duo_sw_core import d_sw1_duo, d_sw2_duo
    from legoesm.grids.fv3_native_gridstruct import (
        average_allflux_shared_edges,
    )

    n, ng = ctx["n"], ctx["ng"]
    bd = ctx["bd"]
    npx = n + 1
    m_a = n + 2 * ng

    uc6 = [np.array(o["uc"], copy=True) for o in csw_outs]
    vc6 = [np.array(o["vc"], copy=True) for o in csw_outs]
    for t in range(1, 7):
        gs = ctx["gs6"][t - 1]
        hs = np.zeros_like(states[t - 1]["delp"])
        pkc, gz = geopk_sw_1lev(csw_outs[t - 1]["delpc"], hs, bd)
        p_grad_c_1lev(0.5 * dt, csw_outs[t - 1]["delpc"], pkc, gz,
                      uc6[t - 1], vc6[t - 1], gs, bd)
    divgd6 = [o["divg_d"] for o in csw_outs]
    for t in range(1, 7):
        exchange_bgrid_scalar_halos(divgd6, t, n, ng)
        exchange_cgrid_vector_halos(uc6, vc6, t, n, ng)

    s1 = []
    for t in range(1, 7):
        st = states[t - 1]
        s1.append(d_sw1_duo(
            st["delp"], st["pt"], st.get("w", st["pt"] * 0.0),
            uc6[t - 1], vc6[t - 1],
            np.zeros((npx, n)), np.zeros((n, npx)),
            np.zeros((npx, m_a)), np.zeros((m_a, npx)),
            ctx["gs6"][t - 1], bd, npx, npx, dt=dt,
            hord_tr=8, hord_vt=6, hord_tm=6, hord_dp=6,
            nord_v=1, nord_t=0, damp_v=0.2, damp_t=0.0,
            workspace_sentinel=0.0))

    afx6 = [o["allflux_x"] for o in s1]
    afy6 = [o["allflux_y"] for o in s1]
    average_allflux_shared_edges(afx6, afy6, 1, n, ng)

    outs = []
    for t in range(1, 7):
        s2 = d_sw2_duo(s1[t - 1]["delp"], s1[t - 1]["pt"],
                       afx6[t - 1], afy6[t - 1],
                       ctx["gs6"][t - 1], bd)
        outs.append({**s1[t - 1], "allflux_x": afx6[t - 1],
                     "allflux_y": afy6[t - 1],
                     "delp": s2["delp"], "pt": s2["pt"],
                     "uc": uc6[t - 1], "vc": vc6[t - 1],
                     "divg_d": divgd6[t - 1]})
    return outs


def acoustic_step_sixface(ctx: dict, states: list, dt: float) -> list:
    """SB3: ONE full duo acoustic step on all six faces —
    c_sw -> geopk/PG-C -> d_sw1 -> C-ring averaging -> d_sw2 -> d_sw3
    -> BGRID averaging of (ubb, vbbtemp) (dyn_core.F90:968-1020, the
    second excluded mpp site, now LIVE) -> kee -> d_sw4 -> d_sw5 ->
    d_sw6 final winds.

    Returns per-face new states {delp, pt, u, v} plus diagnostics
    (ke, wk, divg_d).  Same interim exchange semantics as SB2
    (documented there).
    """
    from legoesm.core.fv3_native_duo_sw_core import (
        d_sw3_duo,
        d_sw4_duo,
        d_sw5_duo,
        d_sw6_duo,
    )
    from legoesm.grids.fv3_native_gridstruct import (
        average_shared_edge_bgrid,
    )

    n, ng = ctx["n"], ctx["ng"]
    bd = ctx["bd"]
    npx = n + 1
    m_a = n + 2 * ng

    csw = csw_step_sixface(ctx, states, dt2=0.5 * dt)
    s12 = dsw12_step_sixface(ctx, states, csw, dt=dt)

    s3 = [d_sw3_duo(states[t - 1]["u"], states[t - 1]["v"],
                    s12[t - 1]["uc"], s12[t - 1]["vc"],
                    ctx["gs6"][t - 1], bd, npx, npx, dt=dt, hord_mt=6)
          for t in range(1, 7)]

    ubb6 = [np.array(o["ubb"], copy=True) for o in s3]
    vbbtemp6 = [np.array(o["vbbtemp"], copy=True) for o in s3]
    average_shared_edge_bgrid(ubb6, vbbtemp6, n, ng)

    outs = []
    for t in range(1, 7):
        ke = np.full((m_a + 1, m_a + 1), 0.0)
        ring = slice(ng, ng + npx)
        kee = s3[t - 1]["ubbtemp"] * vbbtemp6[t - 1]
        ke[ring, ring] = 0.5 * (kee + ubb6[t - 1] * s3[t - 1]["vbb"])
        s4 = d_sw4_duo(states[t - 1]["u"], states[t - 1]["v"],
                       s12[t - 1]["ut"], s12[t - 1]["vt"], ke,
                       ctx["gs6"][t - 1], bd, npx, npx, dt=dt)
        s5 = d_sw5_duo(s12[t - 1]["delp"], states[t - 1]["u"],
                       states[t - 1]["v"], s12[t - 1]["uc"],
                       s12[t - 1]["vc"], csw[t - 1]["ua"],
                       csw[t - 1]["va"], s12[t - 1]["divg_d"],
                       s12[t - 1]["crx_adv"], s12[t - 1]["cry_adv"],
                       s12[t - 1]["xfx_adv"], s12[t - 1]["yfx_adv"],
                       s12[t - 1]["ra_x"], s12[t - 1]["ra_y"],
                       s4["ke"], ctx["gs6"][t - 1], bd, npx, npx,
                       dt=dt, hord_vt=6, nord=1, dddmp=0.2,
                       d2_bg=0.0, d4_bg=0.12, d_con=0.0)
        s6 = d_sw6_duo(states[t - 1]["u"], states[t - 1]["v"],
                       s5["ut"], s5["vt"], s5["ke"], s5["wk"],
                       s5["vortfluxx"], s5["vortfluxy"],
                       ctx["gs6"][t - 1], bd, npx, npx,
                       nord_v=1, damp_v=0.2, d_con=0.0)
        outs.append({"delp": s12[t - 1]["delp"], "pt": s12[t - 1]["pt"],
                     "u": s6["u"], "v": s6["v"],
                     "ke": s5["ke"], "wk": s5["wk"],
                     "divg_d": s5["divg_d"]})
    return outs
