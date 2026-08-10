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
# hord_tm rides here too: update_dz_d's transport order is
# flagstruct%hord_tm (dyn_core.F90:1406).
DUO_TAIL_CFG = {
    "hord_mt": 6, "hord_vt": 6, "hord_tm": 6,
    "nord": 2, "nord_v": 2,
    "dddmp": 0.0, "d2_bg": 0.0, "d4_bg": 0.12, "damp_v": 0.12,
}


def dsw_tail_phase_3d(ctx: dict, state: list, csw_outs: list,
                      dsw_outs: list, dt: float, km: int, *,
                      cfg: dict | None = None,
                      hydrostatic: bool = True,
                      remap_follows: bool = False) -> list:
    """``d_sw3`` -> BARRIER 2 -> ``d_sw4/5/6``, per level, all six faces.

    Returns per-face dicts with the updated D winds ``u``/``v`` at every
    level plus the ``d_sw5`` diagnostics later stages read.

    ``hydrostatic=False`` threads the NH w finalisation through d_sw5
    (``w`` back to velocity against the UPDATED delp, plus the d_sw2
    ``dw`` increment); the final ``w`` is stacked into the returned
    per-face dicts.
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
        if not hydrostatic:
            acc["w"] = np.zeros((m_a, m_a, km), dtype=np.float64)
        nh_damp_w = (float(c.get("damp_w", c["damp_v"]))
                     if not hydrostatic else 0.0)
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
                           d4_bg=c["d4_bg"], d_con=0.0,
                           hydrostatic=hydrostatic,
                           w=(None if hydrostatic
                              else dsw_outs[t]["w"][:, :, k]),
                           dw=(None if hydrostatic else s1.get("dw")),
                           damp_w=nh_damp_w)
            s6 = d_sw6_duo(lev["u"], lev["v"], s5["ut"], s5["vt"],
                           s5["ke"], s5["wk"], s5["vortfluxx"],
                           s5["vortfluxy"], ctx["gs6"][t], bd, npx, npx,
                           nord_v=c["nord_v"], damp_v=c["damp_v"],
                           d_con=0.0)
            acc["u"][:, :, k] = s6["u"]
            acc["v"][:, :, k] = s6["v"]
            if not hydrostatic:
                acc["w"][:, :, k] = s5["w"]
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


def _ext_scalar_planes_6(ctx: dict, planes6: list) -> None:
    """Duo A-grid scalar exchange for one 2-D plane per face, fail-closed.

    The NH D-stage sites (``dyn_core.F90:1482-1483``: ``ext_scalar(zh)``,
    ``ext_scalar(pkc)``) are the same operator as the delp/pt sites, so
    this mirrors the acoustic module's ``_pad_scalars_6`` contract: an
    ext bundle is REQUIRED unless the caller declared the substitution.
    """
    use_ext = bool(ctx.get("use_ext_bundle")) and ctx.get("ectx") is not None
    excluded = "ascalar" in tuple(ctx.get("ext_exclude", ()))
    if use_ext and not excluded:
        from legoesm.grids.fv3_native_ext_vector import ext_scalar_sixface
        ext_scalar_sixface(planes6, "A", ctx["ectx"])
        return
    if not excluded:
        raise ValueError(
            "_ext_scalar_planes_6: dyn_core.F90:1482-1483 exchanges "
            "zh/pkc with ext_scalar on the duo lane, and this context "
            "has no ext bundle. Build it with use_ext_bundle=True, or "
            "declare the substitution with ext_exclude=('ascalar',).")
    from legoesm.grids.fv3_native_gridstruct import (
        exchange_agrid_scalar_halos,
    )
    for t in range(1, 7):
        exchange_agrid_scalar_halos(planes6, t, ctx["n"], ctx["ng"])


def nh_exchanged_area6(ctx: dict) -> list:
    """Sentinel-free per-face ``area`` with REAL corner-diagonal halos.

    The single-tile gridstruct leaves BIG_NUMBER sentinels in the
    corner-diagonal halo cells of ``area``; ``fv_tp_2d``'s inner
    updates inside ``update_dz_d`` read them (measured: a constant
    field's flux then deviates from ``xfx*C`` by 35% near corners; with
    real areas the deviation is pure rounding).  The ORACLE's
    ``gridstruct%area`` halos are exchange-filled at grid init, so the
    six-face NH integration owns supplying the analog: one ext_scalar
    exchange of the area planes, cached on the ctx (the metric is
    time-invariant).  PLAUSIBLE, to be adjudicated by the full-step
    oracle gate: the Lagrange corner-region fill of an exchanged FIELD
    may differ from the oracle's own corner-area construction in the
    last bits; any disagreement will localise to corner-adjacent
    stencils in the parity map.
    """
    if "nh_area6" in ctx:
        return ctx["nh_area6"]
    a6 = [np.array(np.asarray(ctx["gs6"][t]["area"]), dtype=np.float64,
                   copy=True) for t in range(6)]
    _ext_scalar_planes_6(ctx, a6)
    ctx["nh_area6"] = a6
    ctx["nh_rarea6"] = [1.0 / a for a in a6]
    return a6


def dgrid_nh_pressure_phase_3d(ctx: dict, csw_press: list, dsw_outs: list,
                               tail_outs: list, nh: dict, km: int, *,
                               dt: float, ptop: float, akap: float,
                               cp_air: float, p_fac: float, a_imp: float,
                               dp0: np.ndarray, delz6: list,
                               remap_step: bool = False,
                               use_logp: bool = False,
                               square_domain: bool = True,
                               cfg: dict | None = None,
                               remap_follows: bool = False) -> list:
    """The NH D-grid tail (``dyn_core.F90:1403-1543``): ``update_dz_d``
    -> ``Riem_Solver3`` -> ``pe_halo``/``pk3_halo`` -> zh/pkc duo
    exchanges -> ``gz = zh*grav`` -> ``nh_p_grad``.

    Replaces the hydrostatic ``geopk`` + ``one_grad_p`` chain.  Consumes
    the post-d_sw2 ``delp``/``pt`` (whose halos the caller has already
    refreshed -- the :1336-1337 exchange precedes this block) and the
    d_sw5-final ``w`` in ``tail_outs``; mutates ``w`` (Riem_Solver3),
    ``delz6``, the persistent ``nh`` carry (``zh6``, ``gz6``, ``pkc``
    from the C stage as ``ppe``, ``pk3_6``, ``pe6``, ``pk6``,
    ``peln6``, ``ws6``) and the D winds in ``tail_outs`` (nh_p_grad).

    ``nh`` keys (built by the acoustic driver, persistent across
    substeps): ``zh6``, ``gz6``, ``zs6``, ``pk3_6``, ``pe6``, ``pk6``,
    ``peln6``, ``ws6``.

    trap #6 is honoured: ``csw_press[t]['pkc']`` -- FULL pressure out
    of the C stage -- is the SAME storage Riem_Solver3 overwrites with
    the D-stage PERTURBATION and nh_p_grad then B-grid-scratches.

    ``square_domain=True`` adds the second pkc exchange
    (``dyn_core.F90:1496-1502``).  Between the first exchange and it,
    only ``gz`` changes, so the second is idempotent -- included for
    statement-order fidelity, and harmless if the fresh run resolves
    square_domain false.
    """
    require_no_remap_needed(km, remap_follows=remap_follows)
    from legoesm.core.fv3_native_nh_core import riem_solver3, update_dz_d
    from legoesm.core.fv3_native_pgrad import nh_p_grad, pe_halo, pk3_halo
    from legoesm.grids.fv3_native_gridstruct import FV3_GRAV

    c = dict(DUO_TAIL_CFG)
    c.update(cfg or {})
    bd = ctx["bd"]
    n, ng = ctx["n"], ctx["ng"]
    npx = n + 1
    nord_v = int(c.get("nord_w", c["nord_v"]))
    damp_vt = float(c.get("damp_w", c["damp_v"]))

    area6 = nh_exchanged_area6(ctx)
    rarea6 = ctx["nh_rarea6"]

    rdt = 1.0 / dt
    for t in range(6):
        s1_levels = dsw_outs[t]["levels"]
        crx = np.stack([s1_levels[k]["crx_adv"] for k in range(km)], axis=2)
        cry = np.stack([s1_levels[k]["cry_adv"] for k in range(km)], axis=2)
        xfx = np.stack([s1_levels[k]["xfx_adv"] for k in range(km)], axis=2)
        yfx = np.stack([s1_levels[k]["yfx_adv"] for k in range(km)], axis=2)
        # nord_v/damp_vt are level-invariant on this deck (n_sponge=-1);
        # slot km is free for update_dz_d's :231-232 mutation.
        ndif = np.full(km + 1, float(nord_v))
        damp = np.full(km + 1, damp_vt)
        gs_nh = dict(ctx["gs6"][t])
        gs_nh["area"] = area6[t]
        gs_nh["rarea"] = rarea6[t]
        update_dz_d(ndif, damp, int(c["hord_tm"]), bd, km, npx, npx,
                    area6[t], rarea6[t], dp0, nh["zs6"][t], nh["zh6"][t],
                    crx, cry, xfx, yfx, nh["ws6"][t], rdt, gs_nh,
                    lim_fac=1.0)

        riem_solver3(0, dt, bd, km, akap, cp_air, ptop, nh["zs6"][t],
                     tail_outs[t]["w"], delz6[t], dsw_outs[t]["pt"],
                     dsw_outs[t]["delp"], nh["zh6"][t], nh["pe6"][t],
                     csw_press[t]["pkc"], nh["pk3_6"][t], nh["pk6"][t],
                     nh["peln6"][t], nh["ws6"][t], p_fac, a_imp,
                     use_logp=use_logp, last_call=remap_step,
                     fp_out=False)

        if remap_step:
            pe_halo(nh["pe6"][t], dsw_outs[t]["delp"], bd,
                    npz=km, ptop=ptop)
        pk3_halo(nh["pk3_6"][t], dsw_outs[t]["delp"], bd,
                 npz=km, ptop=ptop, akap=akap)

    # zh + pkc duo exchanges (:1482-1483), per interface level.
    for k in range(km + 1):
        zh_k = [nh["zh6"][t][:, :, k] for t in range(6)]
        _ext_scalar_planes_6(ctx, zh_k)
        for t in range(6):
            nh["zh6"][t][:, :, k] = zh_k[t]
        pkc_k = [csw_press[t]["pkc"][:, :, k] for t in range(6)]
        _ext_scalar_planes_6(ctx, pkc_k)
        for t in range(6):
            csw_press[t]["pkc"][:, :, k] = pkc_k[t]

    # gz = zh*grav over the two-cell halo box (:1487-1494).
    i0 = bd.is_ - bd.isd
    j0 = bd.js - bd.jsd
    sl_i = slice(i0 - 2, i0 + n + 2)
    sl_j = slice(j0 - 2, j0 + n + 2)
    for t in range(6):
        nh["gz6"][t][sl_i, sl_j, :] = nh["zh6"][t][sl_i, sl_j, :] * FV3_GRAV

    # Second pkc exchange, square_domain only (:1496-1502).
    if square_domain:
        for k in range(km + 1):
            pkc_k = [csw_press[t]["pkc"][:, :, k] for t in range(6)]
            _ext_scalar_planes_6(ctx, pkc_k)
            for t in range(6):
                csw_press[t]["pkc"][:, :, k] = pkc_k[t]

    # nh_p_grad (:1534-1543): beta = 0 on the deck.
    press = []
    for t in range(6):
        nh_p_grad(tail_outs[t]["u"], tail_outs[t]["v"],
                  csw_press[t]["pkc"], nh["gz6"][t],
                  dsw_outs[t]["delp"], nh["pk3_6"][t],
                  ctx["gs6"][t], bd, npx=npx, npy=npx, npz=km, dt=dt,
                  ptop=ptop, akap=akap, use_logp=use_logp,
                  ng=ng, duogrid=True)
        press.append({"pe": nh["pe6"][t], "pk": nh["pk6"][t],
                      "peln": nh["peln6"][t], "ws": nh["ws6"][t]})
    return press
