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

import os

import numpy as np
from legoesm.grids.fv3_native_gridstruct import (
    analytic_swcore_state,
    build_fv3_native_gridstruct,
    exchange_bgrid_scalar_halos,
    exchange_cgrid_vector_halos,
)

# Vertex-instability diagnostic mode (codex vertex-kill C3), frozen at
# import — no per-call env reads, no mid-run env mutation (codex
# screens-r1 F6).  Default OFF = faithful.
_PG_BVERTEX_MEAN2 = os.environ.get("LEGOESM_DUO_PG_BVERTEX", "") == "mean2"

# Entry A-scalar exchange gating (marginal-stability probe).  Upstream
# runs the ENTRY ext_scalar(delp,pt) only on the first acoustic step of
# each dt_atmos block (dyn_core.F90:432-439 `if (it==1)`), while the
# TAIL refresh (:1336-1337) runs every step; our stepper has no n_split
# structure and so runs BOTH every step -- two A-scalar wedge
# applications per step where upstream averages ~1.3.  The exchange is
# idempotent on an unchanged field, so this should be a no-op; with a
# faithful ~300x corner-wedge amplifier sitting in a near-cancellation
# against del-6 (measured per-step excess gain only ~1.0033), "should
# be" is worth measuring.  "off" drops the entry refresh, keeping the
# tail one.
_ENTRY_ASCALAR_OFF = (
    os.environ.get("LEGOESM_DUO_ENTRY_ASCALAR", "") == "off")


def build_six_face_duo_context(n: int, ng: int = 3,
                               use_ext_bundle: bool = False,
                               vector_corner: str = "lagrange",
                               ext_exclude: tuple = (),
                               use_ext_metrics: bool = False,
                               oracle_conventions: bool = False,
                               omega: float | None = None,
                               k2e_nord: int = 2,
                               topo_fn=None) -> dict:
    """Gridstructs + Bounds for all six faces (certified builders).

    ``oracle_conventions=True`` = the BOUNDED-conventions lane the
    Zenodo duo runs actually execute (proven by the C48 fms.out
    da_min_c: duo took the ``bounded_domain`` grid-init arms).  Metrics
    come from ``build_fv3_native_gridstruct_bounded`` (extended
    own-face lattice, real geometry everywhere incl the regular
    120-degree vertex kink cosa=-1/2/sina=sqrt(3)/2/rsina=4/3), and
    every stage guard sees ``bounded_domain=True`` with the four
    corner flags FALSE (upstream sets them only at
    ``.not.bounded``) — so the d_sw4 corner-KE fix and every
    plain-lane corner/edge special-case switch off exactly as in the
    duo runs.

    ``oracle_conventions=False`` keeps the plain-conventions stack
    (kinked mpp-state metrics + the SB4-era angle/area overrides that
    approximate the bounded values at edges).
    """
    from legoesm.core.fv3_native_sw_core import Bounds
    from legoesm.grids.fv3_native_halos import ed_supergrid_lonlat_ref
    from legoesm.grids.fv3_native_metrics import compute_fv3_native_angles
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_OMEGA,
        FV3_RADIUS_M,
        build_fv3_native_gridstruct_bounded,
    )

    # radius/omega: the W2 balanced state, the duo-target gate and the
    # Zenodo reference all use the FMS constants printed by the duo run
    # log ("Radius is 6371200.0, omega is 7.2921e-5") — the builder's
    # constants.R_earth/Omega defaults put a broad scale error on every
    # metric and Coriolis term (codex vertex-diff P2).
    # omega override: the colliding-modon case runs a NON-ROTATING
    # planet (FV3 case 8; omega=0); default = the FMS value
    if omega is None:
        omega = FV3_OMEGA
    if oracle_conventions:
        gs6 = [build_fv3_native_gridstruct_bounded(n, ng, tile=t,
                                                   omega=omega)
               for t in range(1, 7)]
    else:
        gs6 = [build_fv3_native_gridstruct(n, ng, tile=t,
                                           radius=FV3_RADIUS_M,
                                           omega=omega)
               for t in range(1, 7)]

    # DUO angle override: the plain-mpp gridstruct poisons the panel-edge
    # B-node sina/rsina (plain FV3 never reads them — its non-duo d_sw3
    # edge branches extrapolate instead), but the duo interior-everywhere
    # contravariant formulas DIVIDE by rsina at edge B-nodes.  The duo
    # grid has REAL cross-face angles there — exactly the certified
    # phase-2B compute_fv3_native_angles construction.  The four
    # cube-vertex B-nodes stay poisoned (upstream convention: d_sw4's
    # corner-KE fix replaces exactly those values).
    lon6s, lat6s = ed_supergrid_lonlat_ref(n)
    npx = n + 1
    lon6c = np.stack([lon6s[t][::2, ::2] for t in range(6)])
    lat6c = np.stack([lat6s[t][::2, ::2] for t in range(6)])
    ang = compute_fv3_native_angles(lon6c, lat6c)
    for t in range(6) if not oracle_conventions else ():
        gs = gs6[t]
        blk = (slice(ng, ng + npx), slice(ng, ng + npx))
        cb = np.array(ang["cosa_b"][t])
        sb = np.array(ang["sina_b"][t])
        good = np.isfinite(cb) & np.isfinite(sb)
        cosa = gs["cosa"][blk]
        sina = gs["sina"][blk]
        rsina = gs["rsina"][blk]
        cosa[good] = cb[good]
        sina[good] = sb[good]
        # fv_grid_utils.F90:540: rsina = 1/max(tiny_number, sina**2)
        rs = 1.0 / np.maximum(1.0e-8, sb * sb)
        rsina[good] = rs[good]
        gs["cosa"][blk] = cosa
        gs["sina"][blk] = sina
        gs["rsina"][blk] = rsina

    # DUO metric corner fill: the plain convention poisons the
    # corner-diagonal cell areas (area = -big -> rarea = -1e-8); the
    # duo extended grid has REAL metrics there, and d_sw5's full-domain
    # relative vorticity divides by area at those cells — the del6
    # vorticity damping then amplifies the fringe jump by
    # damp4=(damp_v*da_min_c)^2.  Interim: FV3 AGRID corner index-fill
    # on area (real side values; the true wedge areas arrive with the
    # ext-machinery swap) + rarea recomputed there.
    from legoesm.grids.fv3_native_gridstruct import (
        fill_corners_agrid_x as _fill_corners_agrid_x,
    )
    from legoesm.grids.fv3_native_gridstruct import (
        fort as _fort,
    )

    lo = 1 - ng
    for gs in (gs6 if not oracle_conventions else ()):
        area = np.array(gs["area"], copy=True)
        _fill_corners_agrid_x(_fort(area, lo, lo), npx, ng)
        rarea = np.array(gs["rarea"], copy=True)
        corner = gs["area"] != area          # exactly the filled cells
        rarea[corner] = 1.0 / area[corner]
        gs["area"] = area
        gs["rarea"] = rarea

    # k2e duo ext machinery (SB5b): the certified phase-3 DuoGridData
    # drives the A-scalar halo exchanges (delp/pt) — the authoritative
    # ext_scalar(…,0,0) analog.  Vector/staggered ext = follow-up.
    from legoesm.grids.fv3_native_halos import (
        create_fv3_native_duogrid_data,
    )

    # k2e_nord=2 = the AUTHORITATIVE live default (fv_arrays.F90:150 +
    # global_grid_data.F90:57, no nml override; the dg/gg FATAL only
    # enforces equality of the two 2-defaults).  The historic 4 came
    # from the luanfs mirror monolith and is the vertex amplifier
    # (4-pt corner-adjacent ring Lagrange has oscillating extrapolation
    # lobes; measured differential gain 1.56x/block, gamma=0.054/blk).
    dg = create_fv3_native_duogrid_data(n, ng=min(ng, 3),
                                        k2e_nord=k2e_nord)

    for gs in gs6:
        # bounded lane: guards see bounded_domain=True + corner flags
        # FALSE (upstream sets sw..ne_corner only at .not.bounded) —
        # d_sw4's corner-KE fix and every plain corner special switch
        # off exactly as in the duo runs
        gs["bounded_domain"] = bool(oracle_conventions)
        gs["grid_type"] = 0
        for k in ("sw_corner", "se_corner", "ne_corner", "nw_corner"):
            gs[k] = not oracle_conventions
    ectx = None
    if use_ext_bundle:
        # FULL ext consistency bundle: the faithful ext_scalar /
        # ext_vector machinery (fv3_native_ext_vector — c2l latlon
        # intermediary + ng=4 geographic lattice + a2stag ext bases +
        # Lagrange corner regions) over extended-lattice halo metrics
        # (extend_gridstruct — interior restored bitwise).  The ext
        # context snapshots the KINKED mpp-state metrics BEFORE the
        # gridstructs are extended (upstream c2l reads the model
        # gridstruct, not the ext lattice).
        from legoesm.grids.fv3_native_ext_vector import build_ext_context
        from legoesm.grids.fv3_native_gridstruct import extend_gridstruct

        bad = set(ext_exclude) - {"divgd", "cvec", "metrics", "dvec", "ascalar"}
        if bad:
            raise ValueError(f"ext_exclude: unknown families {sorted(bad)}")
        ectx = build_ext_context(n, ng, gs6,
                                 vector_corner=vector_corner,
                                 k2e_nord=k2e_nord)
        # ORACLE-FAITHFUL DEFAULT (Zenodo grep): upstream duo d_sw
        # consumes the model gridstruct metrics — the model tree never
        # reads the dg ext metrics; extend_gridstruct was OUR coherence
        # intuition with no upstream counterpart, so ext halo METRICS
        # are a NON-FAITHFUL measurement opt-in.  (The 2026-07-18
        # attribution scores once cited here were old-diagnostic-lens
        # numbers — void, see docs/dycore/fv3_native_p4c_oracle.md
        # RE-BASELINE — but the grep-based faithfulness argument
        # stands on its own.)
        if use_ext_metrics and "metrics" not in ext_exclude:
            gs6 = [extend_gridstruct(gs6[t], n, ng, tile=t + 1)
                   for t in range(6)]

    from legoesm.grids.fv3_native_gridstruct import (
        build_extended_corner_lonlat,
        build_kinked_corner_lonlat,
    )

    kk6 = [build_kinked_corner_lonlat(n, ng, tile=t) for t in range(1, 7)]
    ee6 = [build_extended_corner_lonlat(n, ng, tile=t)
           for t in range(1, 7)]

    # surface geopotential (phis, m^2/s^2) per face on the data domain:
    # topo_fn(lon, lat) -> phis; geopk consumes it as gz(km+1) exactly
    # like upstream (test_cases case-5 phis feeds geopk's hs argument).
    # Upstream applies ONE static ext_scalar(phis) before stepping
    # (test_cases.F90:1567-1582) so the halo/wedge slots carry the k2e
    # values, not the raw analytic evaluation (codex r8 P1).
    hs6 = None
    if topo_fn is not None:
        hs6 = [np.asarray(topo_fn(np.asarray(gs["agrid_lon"]),
                                  np.asarray(gs["agrid_lat"])))
               for gs in gs6]
        if use_ext_bundle:
            from legoesm.grids.fv3_native_ext_vector import (
                ext_scalar_sixface,
            )

            ext_scalar_sixface(hs6, "A", ectx)

    return {"n": n, "ng": ng, "gs6": gs6, "dg": dg,
            "use_ext_bundle": use_ext_bundle, "ectx": ectx,
            "ext_exclude": tuple(ext_exclude),
            "oracle_conventions": bool(oracle_conventions),
            "kk6": kk6, "ee6": ee6, "hs6": hs6,
            "bd": Bounds.single_tile(n, ng)}


def analytic_six_face_state(ctx: dict, **kw) -> list:
    """Per-face analytic solid-body SW state (Williamson-2-like)."""
    return [analytic_swcore_state(gs, **kw) for gs in ctx["gs6"]]


def _check_exchange_flags(ctx: dict):
    """use_k2e_scalars (legacy jax duo pad measurement flag) and
    use_ext_bundle are mutually exclusive: the legacy pad would shadow
    the required ext_scalar refreshes (dyn_core.F90:437-438,
    1336-1337; codex ext r1 P1-2)."""
    if ctx.get("use_k2e_scalars") and ctx.get("use_ext_bundle"):
        raise ValueError(
            "use_k2e_scalars and use_ext_bundle are mutually exclusive")


def csw_step_sixface(ctx: dict, states: list, dt2: float,
                     duogrid: bool = True,
                     exchange: bool = False) -> list:
    """Certified duo c_sw on every face.  dyn_core's divgd + uc/vc
    exchanges happen POST-p_grad_c (dyn_core.F90:629-655) — the
    stepper does them there (dsw12_step_sixface); ``exchange=True``
    applies them here instead for standalone halo tests.
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

    if exchange:
        divgd6 = [o["divg_d"] for o in outs]
        uc6 = [o["uc"] for o in outs]
        vc6 = [o["vc"] for o in outs]
        for t in range(1, 7):
            exchange_bgrid_scalar_halos(divgd6, t, n, ng)
            exchange_cgrid_vector_halos(uc6, vc6, t, n, ng)
    return outs


def duo_pad_scalars(f6: list, ctx: dict) -> None:
    """SB5b: A-scalar duo ext — the k2e Lagrange halo remap
    (legoesm.grids.halo.pad_halo with the certified DuoGridData),
    replacing the interim index-copy exchange for delp/pt.  The padded
    (6, n+2ng, n+2ng) result IS the data domain — full in-place
    replacement (interior bits unchanged by construction)."""
    import jax.numpy as jnp
    from legoesm.grids.halo import pad_halo

    n, ng = ctx["n"], ctx["ng"]
    comp = jnp.stack([jnp.asarray(f[ng:ng + n, ng:ng + n])
                      for f in f6])
    padded = np.asarray(pad_halo(comp, halo=ng, duogrid=ctx["dg"]))
    for t in range(6):
        f6[t][:, :] = padded[t]


def _geopk_sw_adapter(delp2d: np.ndarray, hs: np.ndarray, bd,
                      pt: np.ndarray | None, *, cg: bool) -> tuple:
    """Shared body of the two km=1 SW geopk adapters.

    Folds the ``-DSW_DYNAMICS`` convention (akap=1, ptop=0, pt defaults
    to 1, NO cp_air, no peln/pkz) and the 2-D<->3-D staging onto the ONE
    km-general kernel in :mod:`legoesm.core.fv3_native_pgrad`.  Carries
    no numerics of its own.

    ``unwritten_fill=0.0`` (not the oracle's 1e30 sentinel) keeps the
    returned halos byte-identical to the pre-refactor km=1 code, which
    allocated ``np.zeros`` and wrote only the compute box — see
    UNCERTAIN U10.  ``cp_air=1.0`` is inert: the SW branch (:2770) drops
    the factor entirely.
    """
    from legoesm.core.fv3_native_pgrad import geopk as _geopk_km

    delp3 = np.asarray(delp2d, dtype=np.float64)[:, :, None]
    pt3 = (np.ones_like(delp3) if pt is None
           else np.asarray(pt, dtype=np.float64)[:, :, None])
    out = _geopk_km(delp3, pt3, hs, bd, km=1, ptop=0.0, akap=1.0,
                    cp_air=1.0, cg=cg, duogrid=True, computehalo=False,
                    npx=bd.ie + 1, npy=bd.je + 1, a2b_ord=4,
                    bounded_domain=False, sw_dynamics=True,
                    unwritten_fill=0.0)
    return out["pk"], out["gz"]


def geopk_sw_1lev(delpc: np.ndarray, hs: np.ndarray, bd,
                  pt: np.ndarray | None = None) -> tuple:
    """dyn_core.F90 geopk (2660-2790), SW_DYNAMICS branch, km=1, CG=T.

    SW convention: akap=1, ptop=0, pt≡1 — pk(1)=ptop**akap=0,
    pk(2)=exp(akap*log(pe))=delp, gz(2)=hs, gz(1)=gz(2)+pt*(pk(2)-pk(1))
    (the SW_DYNAMICS increment, NO cp_air).  CG=.true. ranges:
    ifirst=is-1..ie+1 (c_sw's delpc compute ring covers exactly this).
    Returns (pkc, gz) with a trailing 2-level axis on the data domain.

    ADAPTER over ``fv3_native_pgrad.geopk`` — no duplicated numerics.
    """
    return _geopk_sw_adapter(delpc, hs, bd, pt, cg=True)


def p_grad_c_1lev(dt2: float, delpc, pkc, gz, uc, vc, gs: dict, bd):
    """dyn_core.F90 p_grad_c (2073-2132), km=1, hydrostatic.

    wk = pkc(:,:,2) - pkc(:,:,1); the classic cross-term PG updates
    uc over (is:ie+1, js:je) and vc over (is:ie, js:je+1).  Mutates
    uc/vc (numpy data-domain arrays) in place; delpc unused on the
    hydrostatic branch (kept for signature fidelity).

    ADAPTER over ``fv3_native_pgrad.p_grad_c`` — the 2-D uc/vc are
    passed as ``[:, :, None]`` VIEWS so the in-place update writes
    through to the caller's arrays.
    """
    from legoesm.core.fv3_native_pgrad import p_grad_c as _p_grad_c_km

    _p_grad_c_km(dt2, np.asarray(delpc, dtype=np.float64)[:, :, None],
                 pkc, gz, uc[:, :, np.newaxis], vc[:, :, np.newaxis],
                 gs, bd, npz=1, hydrostatic=True)


def dsw12_step_sixface(ctx: dict, states: list, csw_outs: list,
                       dt: float, sw_cfg: dict | None = None) -> list:
    """SB2: geopk(SW,1-lev) + p_grad_c per face, the post-PG duo
    exchanges, d_sw1 per face, the C-ring inter-panel flux averaging
    (dyn_core.F90:853-900 — the first excluded mpp site, now live),
    then d_sw2 per face on the AVERAGED slots.

    ``sw_cfg`` overrides the W2-tuned damping/reconstruction defaults
    (``_SW_CFG_DEFAULT``) — e.g. the Zenodo case-8 configuration turns
    vorticity damping OFF and runs hord=8 everywhere.

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
    hs6 = ctx.get("hs6")
    for t in range(1, 7):
        gs = ctx["gs6"][t - 1]
        hs = (np.asarray(hs6[t - 1]) if hs6 is not None
              else np.zeros_like(states[t - 1]["delp"]))
        pkc, gz = geopk_sw_1lev(csw_outs[t - 1]["delpc"], hs, bd,
                                pt=csw_outs[t - 1]["ptc"])
        p_grad_c_1lev(0.5 * dt, csw_outs[t - 1]["delpc"], pkc, gz,
                      uc6[t - 1], vc6[t - 1], gs, bd)
    sd = ctx.get("step_dump")
    if sd:
        for t in range(6):
            sd(203, t, "uc", uc6[t])
            sd(203, t, "vc", vc6[t])
    divgd6 = [o["divg_d"] for o in csw_outs]
    if ctx.get("use_ext_bundle"):
        # authoritative post-p_grad_c duo exchanges (dyn_core.F90:652-655):
        # ext_scalar(divgd, 1,1) + ext_vector(uc, vc, 1,0,0,1)
        from legoesm.grids.fv3_native_ext_vector import (
            ext_scalar_sixface,
            ext_vector_cgrid_sixface,
        )

        if "divgd" in ctx.get("ext_exclude", ()):
            for t in range(1, 7):
                exchange_bgrid_scalar_halos(divgd6, t, n, ng)
        else:
            ext_scalar_sixface(divgd6, "B", ctx["ectx"])
        if "cvec" in ctx.get("ext_exclude", ()):
            for t in range(1, 7):
                exchange_cgrid_vector_halos(uc6, vc6, t, n, ng)
        else:
            ext_vector_cgrid_sixface(uc6, vc6, ctx["ectx"])
    else:
        for t in range(1, 7):
            exchange_bgrid_scalar_halos(divgd6, t, n, ng)
            exchange_cgrid_vector_halos(uc6, vc6, t, n, ng)
    if sd:
        for t in range(6):
            sd(204, t, "uc", uc6[t])
            sd(204, t, "vc", vc6[t])
            sd(204, t, "divgd", divgd6[t])

    cfg = dict(_SW_CFG_DEFAULT)
    cfg.update(sw_cfg or {})
    s1 = []
    for t in range(1, 7):
        st = states[t - 1]
        s1.append(d_sw1_duo(
            st["delp"], st["pt"], st.get("w", st["pt"] * 0.0),
            uc6[t - 1], vc6[t - 1],
            np.zeros((npx, n)), np.zeros((n, npx)),
            np.zeros((npx, m_a)), np.zeros((m_a, npx)),
            ctx["gs6"][t - 1], bd, npx, npx, dt=dt,
            hord_tr=cfg["hord_tr"], hord_vt=cfg["hord_vt"],
            hord_tm=cfg["hord_tm"], hord_dp=cfg["hord_dp"],
            nord_v=cfg["nord_v"], nord_t=0,
            damp_v=cfg["damp_v"], damp_t=0.0,
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


# W2-tuned stage configuration (the historical hardcoded values —
# byte-identical default).  The Zenodo case-8 run uses: hords all 8,
# damp_v=0 (do_vort_damp=.false.), dddmp=0, d2_bg=0, d4_bg=0.12 with
# nord=2 (del-6 — ported and certified bit-exact 6/6, job 9108229;
# SW_CFG_CASE8 below carries it).
_SW_CFG_DEFAULT = {
    "hord_tr": 8, "hord_vt": 6, "hord_tm": 6, "hord_dp": 6,
    "hord_mt": 6, "nord_v": 1, "damp_v": 0.2,
    "dddmp": 0.2, "d2_bg": 0.0, "d4_bg": 0.12, "nord": 1,
}

SW_CFG_CASE8 = {
    # Zenodo C48.sw.case8 fms.out damping block + fv_core_nml:
    # del-6 (nord=2) bg 0.12, vort damping OFF, dddmp 0, hords all 8
    "hord_tr": 8, "hord_vt": 8, "hord_tm": 8, "hord_dp": 8,
    "hord_mt": 8, "nord_v": 1, "damp_v": 0.0,
    "dddmp": 0.0, "d2_bg": 0.0, "d4_bg": 0.12, "nord": 2,
}


def acoustic_step_sixface(ctx: dict, states: list, dt: float,
                          sw_cfg: dict | None = None,
                          entry_ascalar: bool = True) -> list:
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
        exchange_agrid_scalar_halos,
        exchange_dgrid_vector_halos,
    )

    n, ng = ctx["n"], ctx["ng"]
    bd = ctx["bd"]
    npx = n + 1
    m_a = n + 2 * ng

    # pre-c_sw entry exchanges (dyn_core duo 437-471: ext_scalar(delp),
    # ext_scalar(pt), ext_vector(u,v)) — interim mpp-analog versions;
    # ALSO fills the corner-diagonal ghosts (BIG-poisoned in the
    # analytic state) that d_sw5's full-domain vorticity prep and the
    # del6 damping otherwise amplify by damp4 ~ 1e21.
    delp6 = [np.array(st["delp"], copy=True) for st in states]
    pt6 = [np.array(st["pt"], copy=True) for st in states]
    u6 = [np.array(st["u"], copy=True) for st in states]
    v6 = [np.array(st["v"], copy=True) for st in states]
    _check_exchange_flags(ctx)
    if ctx.get("use_k2e_scalars"):
        # MEASURED WORSE in isolation (2026-07-17 ablation: du 14->110,
        # ddelp 2%->54% at 8h): k2e scalars live at EXTENDED-grid
        # positions while the vector halos + halo metrics remain
        # kinked-lattice — mixing the two breaks the discrete geometry.
        # The ext swap must be the FULL consistency bundle (all fields
        # + extended halo metrics together); until then the coherent
        # index-copy interim stays.
        if entry_ascalar and not _ENTRY_ASCALAR_OFF:
            duo_pad_scalars(delp6, ctx)
            duo_pad_scalars(pt6, ctx)
    elif not ctx.get("use_ext_bundle"):
        # same it==1 cadence gate as the ext lane (codex r1 P2-8: the
        # interim/measurement lanes must not run the entry A-scalar
        # exchange every inner step when the caller supplies the
        # upstream n_split schedule)
        if entry_ascalar and not _ENTRY_ASCALAR_OFF:
            for t in range(1, 7):
                exchange_agrid_scalar_halos(delp6, t, n, ng)
                exchange_agrid_scalar_halos(pt6, t, n, ng)
    if ctx.get("use_ext_bundle"):
        # authoritative duo entry exchanges (dyn_core.F90:437-471):
        # ext_scalar(delp/pt, 0,0) + ext_vector(u, v, 0,1,1,0)
        from legoesm.grids.fv3_native_ext_vector import (
            ext_scalar_sixface,
            ext_vector_dgrid_sixface,
        )

        if _ENTRY_ASCALAR_OFF or not entry_ascalar:
            pass          # upstream gates the entry exchange on it==1
        elif "ascalar" in ctx.get("ext_exclude", ()):
            for t in range(1, 7):
                exchange_agrid_scalar_halos(delp6, t, n, ng)
                exchange_agrid_scalar_halos(pt6, t, n, ng)
        else:
            ext_scalar_sixface(delp6, "A", ctx["ectx"])
            ext_scalar_sixface(pt6, "A", ctx["ectx"])
        if "dvec" in ctx.get("ext_exclude", ()):
            for t in range(1, 7):
                exchange_dgrid_vector_halos(u6, v6, t, n, ng)
        else:
            ext_vector_dgrid_sixface(u6, v6, ctx["ectx"])
    else:
        for t in range(1, 7):
            exchange_dgrid_vector_halos(u6, v6, t, n, ng)
    states = [{**states[t], "delp": delp6[t], "pt": pt6[t],
               "u": u6[t], "v": v6[t]} for t in range(6)]

    # optional step-1 stage twin hook: callable(block, tile0, name, arr)
    # at the same dyn_core points as the instrumented oracle (201 =
    # post-entry exchanges, 202 = post-c_sw, 203/204 in dsw12, 205 =
    # post-d_sw2).  None (default) = byte-identical behavior.
    sd = ctx.get("step_dump")
    if sd:
        for t in range(6):
            sd(201, t, "u", states[t]["u"])
            sd(201, t, "v", states[t]["v"])
            sd(201, t, "delp", states[t]["delp"])
            sd(201, t, "pt", states[t]["pt"])

    cfg = dict(_SW_CFG_DEFAULT)
    cfg.update(sw_cfg or {})
    csw = csw_step_sixface(ctx, states, dt2=0.5 * dt)
    if sd:
        for t in range(6):
            sd(202, t, "uc", csw[t]["uc"])
            sd(202, t, "vc", csw[t]["vc"])
            sd(202, t, "delpc", csw[t]["delpc"])
            sd(202, t, "ua", csw[t]["ua"])
            sd(202, t, "va", csw[t]["va"])
            sd(202, t, "divgd", csw[t]["divg_d"])
    s12 = dsw12_step_sixface(ctx, states, csw, dt=dt, sw_cfg=sw_cfg)
    if sd:
        for t in range(6):
            sd(205, t, "delp", s12[t]["delp"])
            sd(205, t, "pt", s12[t]["pt"])

    s3 = [d_sw3_duo(states[t - 1]["u"], states[t - 1]["v"],
                    s12[t - 1]["uc"], s12[t - 1]["vc"],
                    ctx["gs6"][t - 1], bd, npx, npx, dt=dt,
                    hord_mt=cfg["hord_mt"])
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
                       dt=dt, hord_vt=cfg["hord_vt"], nord=cfg["nord"],
                       dddmp=cfg["dddmp"],
                       d2_bg=cfg["d2_bg"], d4_bg=cfg["d4_bg"],
                       d_con=0.0)
        s6 = d_sw6_duo(states[t - 1]["u"], states[t - 1]["v"],
                       s5["ut"], s5["vt"], s5["ke"], s5["wk"],
                       s5["vortfluxx"], s5["vortfluxy"],
                       ctx["gs6"][t - 1], bd, npx, npx,
                       nord_v=cfg["nord_v"], damp_v=cfg["damp_v"],
                       d_con=0.0)
        outs.append({"delp": s12[t - 1]["delp"], "pt": s12[t - 1]["pt"],
                     "u": s6["u"], "v": s6["v"],
                     "ke": s5["ke"], "wk": s5["wk"],
                     "divg_d": s5["divg_d"], "delpc": s5["delpc"]})
    return outs


def geopk_sw_1lev_d(delp: np.ndarray, hs: np.ndarray, bd,
                    pt: np.ndarray | None = None) -> tuple:
    """geopk D-grid call (CG=.false., a2b_ord=4, duo): ranges widen to
    is-2..ie+2 (dyn_core geopk range guard).  Same SW km=1 formulas as
    the C version; delp halos must be freshly exchanged (dyn_core does
    ext_scalar(delp/pt) right before this call).

    ADAPTER over ``fv3_native_pgrad.geopk`` — no duplicated numerics.

    OPEN ITEM (pre-existing, NOT introduced by the adapter refactor):
    dyn_core.F90:1402 passes ``computehalo=.true.`` at this site, which
    on the duo lane extends the write box to the FULL data domain
    (isd..ied, jsd..jed) whenever ``is==1`` and ``ie==npx-1``.  This
    km=1 path has always used the un-extended is-2..ie+2 box and is kept
    that way here so the six-face stepper does not move; it is benign
    for the stepper (a2b needs only is-2..ie+2 and the halos are
    exchange-filled) but it IS a divergence from upstream.  The
    ``computehalo`` extension itself is certified by the km>1 oracle
    (``PKC_D``/``GZ_D`` carry no sentinel), not by this adapter.
    """
    return _geopk_sw_adapter(delp, hs, bd, pt, cg=False)


def one_grad_p_1lev(u, v, pkc, gz, divg2, gs: dict, bd, npx: int,
                    npy: int, *, dt: float, d_ext: float = 0.02):
    """dyn_core.F90 one_grad_p (2347-2480), km=1 hydrostatic SW:
    pk B-node top = ptk = 0; a2b_ord4(replace=True, duo) moves pk(2),
    gz(1), gz(2) to B-nodes; wk2/wk1 from divg2 differences; the final
    D-grid PG update converts the circulation-form d_sw6 winds back to
    covariant: u = rdx*(wk2 + u + dt/(wk+wk(i+1))*(cross-terms)).
    Mutates u/v in place (data-domain numpy arrays).

    ADAPTER over ``fv3_native_pgrad.one_grad_p`` — no duplicated
    numerics.  ``u``/``v`` are passed as ``[:, :, None]`` VIEWS so the
    in-place update writes through.

    ALIASING CONTRACT (codex r21 blocker A — a REGRESSION this adapter
    briefly shipped).  The pre-refactor km=1 body worked on COPIES of the
    pkc/gz planes, so the caller's ``pkc``/``gz`` survived the a2b
    ``replace=True``.  The shared kernel is faithful to dyn_core and
    mutates them IN PLACE; passing the caller's arrays straight through
    silently mutated 169 ``pkc`` words and 338 ``gz`` words where the
    legacy body mutated ZERO.  The earlier justification ("both call
    sites discard them") was wrong on principle: the frozen legacy
    reference is the contract and the burden is on the port, not on
    every present and future caller.  This adapter therefore restores
    COPY-ON-ENTRY.  ``test_sw_adapter_mutation_footprints`` asserts the
    full footprint — which arrays change and by how many words — for all
    four adapters, not just the returned/updated winds.
    """
    from legoesm.core.fv3_native_pgrad import one_grad_p as _one_grad_p_km

    pk_work = np.array(pkc, dtype=np.float64, copy=True)
    gz_work = np.array(gz, dtype=np.float64, copy=True)
    _one_grad_p_km(u[:, :, np.newaxis], v[:, :, np.newaxis], pk_work,
                   gz_work, divg2, None, gs, bd, npx=npx, npy=npy, npz=1,
                   dt=dt, ptop=0.0, akap=1.0, hydrostatic=True,
                   a2b_ord=4, d_ext=d_ext, ng=bd.ng, duogrid=True,
                   bvertex_mean2=_PG_BVERTEX_MEAN2)


def full_acoustic_step_sixface(ctx: dict, states: list, dt: float,
                               d_ext: float = 0.02,
                               sw_cfg: dict | None = None,
                               entry_ascalar: bool = True) -> list:
    """SB4: complete acoustic step INCLUDING the D-grid tail — the
    stage chain (acoustic_step_sixface), delp/pt halo refresh, the
    D geopk, the external-mode divg2 filter (d_ext*da_min_c*saved
    divergence at km=1), and one_grad_p back to covariant winds.

    Returned-state D-wind halos: on the interim path they are
    exchange-refreshed here; on the ext-bundle path they are
    STALE-BY-ONE exactly like upstream (dyn_core.F90:1332-1338
    refreshes only delp/pt post-step; ext_vector runs at the NEXT
    step's entry, :468-472)."""
    from legoesm.grids.fv3_native_gridstruct import (
        exchange_agrid_scalar_halos,
        exchange_dgrid_vector_halos,
    )

    n, ng = ctx["n"], ctx["ng"]
    bd = ctx["bd"]
    npx = n + 1

    stage = acoustic_step_sixface(ctx, states, dt, sw_cfg=sw_cfg,
                                  entry_ascalar=entry_ascalar)

    delp6 = [np.array(o["delp"], copy=True) for o in stage]
    pt6 = [np.array(o["pt"], copy=True) for o in stage]
    _check_exchange_flags(ctx)
    if ctx.get("use_k2e_scalars"):
        duo_pad_scalars(delp6, ctx)
        duo_pad_scalars(pt6, ctx)
    elif ctx.get("use_ext_bundle"):
        # post-step delp/pt refresh (dyn_core.F90:1336-1337)
        from legoesm.grids.fv3_native_ext_vector import ext_scalar_sixface

        if "ascalar" in ctx.get("ext_exclude", ()):
            for t in range(1, 7):
                exchange_agrid_scalar_halos(delp6, t, n, ng)
                exchange_agrid_scalar_halos(pt6, t, n, ng)
        else:
            ext_scalar_sixface(delp6, "A", ctx["ectx"])
            ext_scalar_sixface(pt6, "A", ctx["ectx"])
    else:
        for t in range(1, 7):
            exchange_agrid_scalar_halos(delp6, t, n, ng)
            exchange_agrid_scalar_halos(pt6, t, n, ng)

    u6 = [np.array(o["u"], copy=True) for o in stage]
    v6 = [np.array(o["v"], copy=True) for o in stage]
    hs6 = ctx.get("hs6")
    for t in range(1, 7):
        gs = ctx["gs6"][t - 1]
        hs = (np.asarray(hs6[t - 1]) if hs6 is not None
              else np.zeros_like(delp6[t - 1]))
        pkc, gz = geopk_sw_1lev_d(delp6[t - 1], hs, bd,
                                  pt=pt6[t - 1])
        # divg2 = d_ext*da_min_c*saved divergence (km=1; dyn_core
        # 1310-1325 with the mass weight cancelling at one level)
        divg2 = np.zeros((npx, npx))
        sl = slice(ng, ng + npx)
        divg2[:, :] = (d_ext * float(gs["da_min_c"])
                       * stage[t - 1]["delpc"][sl, sl]
                       if "delpc" in stage[t - 1] else 0.0)
        one_grad_p_1lev(u6[t - 1], v6[t - 1], pkc, gz, divg2, gs, bd,
                        npx, npx, dt=dt, d_ext=d_ext)
    if not ctx.get("use_ext_bundle"):
        for t in range(1, 7):
            exchange_dgrid_vector_halos(u6, v6, t, n, ng)
    # ext path: NO post-step vector refresh — dyn_core.F90:1332-1338
    # refreshes only delp/pt after the step; the D-vector ext_vector
    # runs at the NEXT step's entry (:468-472).  The returned state's
    # D halos are therefore stale-by-one exactly like upstream's
    # (codex ext r1 P2-5); consumers needing fresh halos re-enter the
    # step or call ext_vector_dgrid_sixface themselves.

    return [{"delp": delp6[t], "pt": pt6[t], "u": u6[t], "v": v6[t]}
            for t in range(6)]


def advance_duo_outer_step(ctx: dict, states: list, dt_atmos: float,
                           n_split: int, d_ext: float = 0.02,
                           sw_cfg: dict | None = None) -> list:
    """One dt_atmos block = ``n_split`` acoustic steps at
    dt = dt_atmos/n_split with the UPSTREAM exchange schedule: the
    entry A-scalar ext_scalar(delp, pt) fires only on the FIRST inner
    step of the block (dyn_core.F90:432-439 gates it on ``it == 1``),
    while the D-vector entry (:468-472), the post-PG B/C exchanges
    (:651-655) and the tail A-scalar refresh (:1335-1337) fire every
    inner step.  Per 1200 s at n_split=7 this is the oracle's 8
    A-scalar applications per field (1 entry + 7 tail); the flat
    back-to-back stepper (entry every step) applies 14.
    """
    if n_split < 1:
        raise ValueError(f"n_split must be >= 1, got {n_split}")
    dt = dt_atmos / n_split
    for it in range(n_split):
        states = full_acoustic_step_sixface(ctx, states, dt,
                                            d_ext=d_ext, sw_cfg=sw_cfg,
                                            entry_ascalar=(it == 0))
    return states


def w2_six_face_state(ctx: dict, alpha: float = 0.0,
                      u0: float | None = None,
                      gh0: float = 2.94e4) -> list:
    """Williamson case-2 BALANCED six-face state on the SW-via-
    production convention (pt≡1, delp = g·h):

        h = (gh0 - (a·Omega·u0 + u0^2/2) · S^2) / g,
        S = -cos(lon)·cos(lat)·sin(alpha) + sin(lat)·cos(alpha)

    Winds are the analytic solid-body projection (reuses
    analytic_swcore_state's certified D/C construction with ddelp=0),
    then delp/pt are overwritten with the balanced fields at the
    kinked-lattice cell centres (halos included; corner-diagonals are
    handled by the step-entry exchanges).
    """
    from legoesm import constants
    from legoesm.grids.fv3_native_gridstruct import FV3_OMEGA, FV3_RADIUS_M

    a_r = FV3_RADIUS_M
    omega = FV3_OMEGA
    g = constants.g                     # cancels: delp = g*h = gh0 - coef*S^2
    if u0 is None:
        # upstream test_cases case 2: Ubar = 2*pi*radius / (12 days)
        u0 = 2.0 * np.pi * a_r / (12.0 * 86400.0)
    coef = (a_r * omega * u0 + 0.5 * u0 * u0)

    states = []
    for gs in ctx["gs6"]:
        st = analytic_swcore_state(gs, u0=u0, alpha=alpha)
        lon = gs["agrid_lon"]
        lat = gs["agrid_lat"]
        s = (-np.cos(lon) * np.cos(lat) * np.sin(alpha)
             + np.sin(lat) * np.cos(alpha))
        h = (gh0 - coef * s * s) / g
        st = dict(st)
        st["delp"] = g * h
        st["pt"] = np.ones_like(st["delp"])
        st["w"] = np.zeros_like(st["delp"])
        states.append(st)
    return states


def run_duo_sw(ctx: dict, states: list, dt: float, nsteps: int,
               d_ext: float = 0.02) -> list:
    """SB5 time loop: nsteps full acoustic steps."""
    for _ in range(nsteps):
        states = full_acoustic_step_sixface(ctx, states, dt, d_ext=d_ext)
    return states
