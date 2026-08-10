"""D-grid tail: d_sw3, BARRIER 2, d_sw4/5/6, D geopk, one_grad_p.

STAGE 1 unit 5. Continues where ``fv3_native_dsw_phase_3d`` (unit 4)
stops, completing one acoustic sub-step of the duo cadence.

ORACLE CADENCE (published, checksum-pinned tree):

    dyn_core.F90:961   call d_sw3
                :984   mpp_get_boundary BGRID_NE   <-- BARRIER 2
                :1102  call d_sw4
                :1107  call d_sw5
                :1256  call d_sw6
                :1401  call geopk        (D-grid)
                :1531  call one_grad_p   (hydrostatic, beta <= 0)

BARRIER 2 EXTENT DIFFERS FROM BARRIER 1. Barrier 1 blends C-ring fluxes
over ``i = is..ie`` / ``j = js..je``; barrier 2 blends the B-grid corner
ingredients over ``i = is..ie+1``, ``j = js..je+1`` -- tile corners
included. Two adjacent routines with different extents is a classic
silent-defect site, so the two are kept in separate units with separate
tests rather than folded into one "apply the barriers" helper.

WHAT BARRIER 2 ACTS ON. Not the final ``u``/``v``: it blends ``ubb`` and
``vbbtemp``, the B-grid corner velocity INGREDIENTS, before the kinetic
energy is formed. The KE assembly that follows is
``ke = 0.5*(ubbtemp*vbbtemp + ubb*vbb)`` (``dyn_core.F90:1080-1085``),
so blending after the KE product would be a different operator.

``d_con = 0.0`` throughout: the shipped duo decks set ``d_con = 0.``, so
the KE-to-heat pathway and its ``heat_source`` allocation
(``dyn_core.F90:322-325``, gated on ``d_con > 1.0E-5``) are inactive.
Passing a non-zero value here would activate a path no fixture certifies.
"""
from __future__ import annotations

import numpy as np

from legoesm.core.fv3_native_state_3d import level_slice, require_no_remap_needed

# Shipped duo deck (resolved namelist, not the fv_arrays initialiser).
DUO_TAIL_CFG = {
    "hord_mt": 6, "hord_vt": 6,
    "nord": 2, "nord_v": 2,
    "dddmp": 0.0, "d2_bg": 0.0, "d4_bg": 0.12, "damp_v": 0.12,
}


def dsw_tail_phase_3d(ctx: dict, state: list, csw_outs: list,
                      dsw_outs: list, dt: float, km: int, *,
                      cfg: dict | None = None,
                      remap_follows: bool = False) -> list:
    """``d_sw3`` -> BARRIER 2 -> ``d_sw4/5/6``, per level, all six faces.

    Returns per-face dicts with the updated D winds ``u``/``v`` at every
    level plus the ``d_sw5`` diagnostics later stages read.
    """
    require_no_remap_needed(km, remap_follows=remap_follows)
    from legoesm.core.fv3_native_duo_sw_core import (
        d_sw3_duo, d_sw4_duo, d_sw5_duo, d_sw6_duo,
    )
    from legoesm.grids.fv3_native_gridstruct import average_shared_edge_bgrid

    c = dict(DUO_TAIL_CFG)
    c.update(cfg or {})
    n, ng, bd = ctx["n"], ctx["ng"], ctx["bd"]
    npx = n + 1
    m_a = n + 2 * ng

    # --- d_sw3 at every level, all faces (needed before barrier 2) --------
    s3 = [[None] * km for _ in range(6)]
    for t in range(6):
        for k in range(km):
            lev = level_slice(state[t], k, km)
            # uc/vc are c_sw outputs, updated IN PLACE by p_grad_c; d_sw1
            # takes them as inputs and does not return them.
            s3[t][k] = d_sw3_duo(lev["u"], lev["v"],
                                 csw_outs[t]["uc"][:, :, k],
                                 csw_outs[t]["vc"][:, :, k],
                                 ctx["gs6"][t], bd, npx, npx, dt=dt,
                                 hord_mt=c["hord_mt"])

    # --- BARRIER 2: B-grid corner ingredients, one level at a time --------
    # dyn_core.F90:984. All six faces must be present for a level before it
    # is blended, or a face is averaged against a stale neighbour.
    ubb6_k, vbb6_k = [], []
    for k in range(km):
        ubb6 = [np.array(s3[t][k]["ubb"], copy=True) for t in range(6)]
        vbbtemp6 = [np.array(s3[t][k]["vbbtemp"], copy=True) for t in range(6)]
        average_shared_edge_bgrid(ubb6, vbbtemp6, n, ng)
        ubb6_k.append(ubb6)
        vbb6_k.append(vbbtemp6)

    # --- d_sw4 / d_sw5 / d_sw6 on the blended ingredients -----------------
    outs = []
    for t in range(6):
        # Allocate diagnostics from the SHAPE THE STAGE ACTUALLY RETURNS.
        # Guessing (m_a, m_a) is wrong for the B-grid members: divg_d and
        # the corner KE are (m_a+1, m_a+1). Hard-coding a shape here would
        # either raise (as it did) or, worse, broadcast silently.
        acc: dict = {}
        acc["u"] = np.zeros((m_a, m_a + 1, km), dtype=np.float64)
        acc["v"] = np.zeros((m_a + 1, m_a, km), dtype=np.float64)
        for k in range(km):
            lev = level_slice(state[t], k, km)
            s1 = dsw_outs[t]["levels"][k]
            # KE at corners: dyn_core.F90:1080-1085 (Lin-Rood).
            ke = np.zeros((m_a + 1, m_a + 1), dtype=np.float64)
            ring = slice(ng, ng + npx)
            kee = s3[t][k]["ubbtemp"] * vbb6_k[k][t]
            ke[ring, ring] = 0.5 * (kee + ubb6_k[k][t] * s3[t][k]["vbb"])

            s4 = d_sw4_duo(lev["u"], lev["v"], s1["ut"], s1["vt"], ke,
                           ctx["gs6"][t], bd, npx, npx, dt=dt)
            s5 = d_sw5_duo(dsw_outs[t]["delp"][:, :, k], lev["u"], lev["v"],
                           csw_outs[t]["uc"][:, :, k],
                           csw_outs[t]["vc"][:, :, k],
                           csw_outs[t]["ua"][:, :, k],
                           csw_outs[t]["va"][:, :, k],
                           csw_outs[t]["divg_d"][:, :, k],
                           s1["crx_adv"], s1["cry_adv"],
                           s1["xfx_adv"], s1["yfx_adv"],
                           s1["ra_x"], s1["ra_y"], s4["ke"],
                           ctx["gs6"][t], bd, npx, npx, dt=dt,
                           hord_vt=c["hord_vt"], nord=c["nord"],
                           dddmp=c["dddmp"], d2_bg=c["d2_bg"],
                           d4_bg=c["d4_bg"], d_con=0.0)
            s6 = d_sw6_duo(lev["u"], lev["v"], s5["ut"], s5["vt"],
                           s5["ke"], s5["wk"], s5["vortfluxx"],
                           s5["vortfluxy"], ctx["gs6"][t], bd, npx, npx,
                           nord_v=c["nord_v"], damp_v=c["damp_v"],
                           d_con=0.0)
            acc["u"][:, :, k] = s6["u"]
            acc["v"][:, :, k] = s6["v"]
            for nm in ("ke", "wk", "divg_d", "delpc"):
                if nm not in s5:
                    continue
                arr = np.asarray(s5[nm])
                if nm not in acc:
                    acc[nm] = np.zeros((*arr.shape, km), dtype=np.float64)
                if arr.shape != acc[nm].shape[:2]:
                    raise ValueError(
                        f"face {t + 1} level {k} {nm!r}: d_sw5 gave "
                        f"{arr.shape}, level 0 gave {acc[nm].shape[:2]} -- "
                        f"a stage must not change shape between levels")
                acc[nm][:, :, k] = arr
        outs.append(acc)
    return outs


def dgrid_pressure_phase_3d(ctx: dict, dsw_outs: list, tail_outs: list,
                            km: int, *, dt: float, ptop: float, akap: float,
                            cp_air: float, a2b_ord: int = 4,
                            d_ext: float = 0.0,
                            remap_step: bool = False,
                            remap_follows: bool = False) -> list:
    """D-grid ``geopk`` (``:1401``) then ``one_grad_p`` (``:1531``).

    ``one_grad_p`` mutates ``u``, ``v``, AND ``pk``/``gz`` in place --
    ``a2b_ord4`` is called with ``replace=.true.``, so on return ``pk`` and
    ``gz`` hold B-grid corner values on ``[is,ie+1] x [js,je+1]``.

    ``d_ext = 0.0`` matches every shipped duo deck's RESOLVED namelist
    (``D_EXT = 0.000000000000000E+000``). At ``d_ext = 0`` the oracle sets
    ``divg2(:,:) = 0.`` and skips the ``a2b_ord2``/mass-weighted branch
    entirely, so passing zero is faithful, not a simplification.

    ``remap_step`` is ``dyn_core.F90:344-348``'s flag (true on
    ``it == n_split``).  At ``:1511-1519`` the oracle copies
    ``pk(i,j,k) = pkc(i,j,k)`` over the COMPUTE WINDOW **before**
    ``one_grad_p`` at ``:1531`` overwrites ``pkc`` with B-grid corner
    values (``a2b_ord4`` is called with ``replace=.true.``).  That
    saved ``pk`` is the array ``Lagrangian_to_Eulerian`` then reads, so
    this snapshot is not a convenience: taking ``pk`` after
    ``one_grad_p`` would feed the remap a corner-staggered field.  It is
    returned as ``pk_remap`` and is present ONLY when ``remap_step`` is
    true, so a caller cannot silently pick up a stale one.
    """
    require_no_remap_needed(km, remap_follows=remap_follows)
    from legoesm.core.fv3_native_pgrad import geopk, one_grad_p

    bd = ctx["bd"]
    n, ng = ctx["n"], ctx["ng"]
    hs6 = ctx.get("hs6")
    press = []
    for t in range(6):
        hs = (np.asarray(hs6[t], dtype=np.float64) if hs6 is not None
              else np.zeros((n + 2 * ng, n + 2 * ng), dtype=np.float64))
        got = geopk(dsw_outs[t]["delp"], dsw_outs[t]["pt"], hs, bd,
                    km=km, ptop=ptop, akap=akap, cp_air=cp_air,
                    cg=False, duogrid=True, computehalo=False,
                    npx=bd.ie + 1, npy=bd.je + 1, a2b_ord=a2b_ord,
                    bounded_domain=False, sw_dynamics=False)
        if remap_step:
            # dyn_core.F90:1511-1519, taken BEFORE :1531 one_grad_p.
            got["pk_remap"] = np.array(got["pk"], copy=True)
        divg2 = np.zeros((n + 2 * ng + 1, n + 2 * ng + 1), dtype=np.float64)
        one_grad_p(tail_outs[t]["u"], tail_outs[t]["v"],
                   got["pk"], got["gz"], divg2, dsw_outs[t]["delp"],
                   ctx["gs6"][t], bd, npx=bd.ie + 1, npy=bd.je + 1,
                   npz=km, dt=dt, ptop=ptop, akap=akap,
                   hydrostatic=True, a2b_ord=a2b_ord, d_ext=d_ext,
                   ng=ng, duogrid=True)
        press.append(got)
    return press
