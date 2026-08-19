#!/usr/bin/env python
"""Localise the NH one-step w gap (6.6116e-04) in TIME and OPERATOR.

The NH full-step gate reports the port's standing gap against the pinned
Fortran as w-specific: |d|max ~1.3e-7 m/s, boundary-ring cells, k=0
argmax, 6.6e-4 of the oracle's own one-step w tendency, while u/v/pt/
delp/delz all sit at their ~5e-7-of-tendency floor.  --trace-substeps is
hydro-only, so this probe threads the NH carry through the acoustic loop
one sub-step at a time, mirroring fv_dynamics_step VERBATIM (theta_v
conversion, flux capacitors, tracer_2d, the NH remap call), and prints:

  * per sub-step: the w increment from the d_sw arms (state-entry ->
    pre-Riem_Solver3) and from Riem_Solver3 (pre -> post), each split
    (k=0 vs k>0) x (boundary ring <3 cells vs interior);
  * the remap's own w increment, same split, plus a w_limiter=False
    twin of the remap on copies (predicts bit-identical: the oracle's
    limiter printed nothing in run_nh_1step_gfs);
  * the final residual vs the oracle under the SAME face map as the
    gate, per-k and ring/interior -- must reproduce the 6.6116e-04.

The within-substep split is observed by WRAPPING
dgrid_nh_pressure_phase_3d (function-scope import in
acoustic_substep_3d resolves the module attribute at call time), never
by rebuilding the chain by hand.  INSTRUMENT CONTROL: after the traced
run, fv_dynamics_step is run on a fresh IC with the wrapper removed and
every prognostic field must be BIT-IDENTICAL to the traced run's final
state; the probe refuses to report otherwise.

No verdict is printed.  Pre-registered readings:
  * d_sw-arm increment dominating at the residual's site class (k=0,
    boundary ring) points at the w advection/damping arms;
  * Riem_Solver3 dominating there points at the implicit solver /
    its boundary-region operands;
  * a remap increment comparable to 1.3e-7 at those sites keeps the
    kord_wz remap on the suspect list; orders smaller retires it as
    the birth place (it can still reshape).

usage:  python scripts/validate/fv3_native/nh_substep_w_trace.py
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))

# import the gate runner as a module: its loaders, IC builder, face map
# and rel() ARE the instrument; a re-implementation could differ for its
# own reasons.
_spec = importlib.util.spec_from_file_location(
    "full_step_oracle_parity",
    os.path.join(_HERE, "full_step_oracle_parity.py"))
fsp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fsp)

N, NG, KM = fsp.N, fsp.NG, fsp.KM
N_SPLIT = fsp.N_SPLIT
DT = fsp.DT_ATMOS
RING = 3  # the gate's own "within 3 of a panel boundary" class


def _sha() -> str:
    try:
        return subprocess.run(["git", "-C", _REPO, "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:
        return "unknown"


def _win(arr):
    """Padded (m_a, m_a, ...) -> compute window (n, n, ...)."""
    return np.asarray(arr)[NG:NG + N, NG:NG + N, ...]


_i, _j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
_RINGDIST = np.minimum(np.minimum(_i, _j),
                       np.minimum(N - 1 - _i, N - 1 - _j))
_BND = _RINGDIST < RING  # (n, n) bool


def _split(d6):
    """d6: list of six (n, n, km*) arrays -> the 2x2 max table."""
    out = {}
    for tag, kmask in (("k=0", 0), ("k>0", slice(1, None))):
        top = [d[:, :, kmask] for d in d6]
        out[(tag, "bnd")] = max(
            float(np.abs(np.compress(_BND.ravel(),
                                     t.reshape(N * N, -1), axis=0)).max())
            for t in top)
        out[(tag, "int")] = max(
            float(np.abs(np.compress(~_BND.ravel(),
                                     t.reshape(N * N, -1), axis=0)).max())
            for t in top)
    return out


def _row(name, d6):
    s = _split(d6)
    print(f"    {name:28s} k=0: bnd {s[('k=0', 'bnd')]:.3e} "
          f"int {s[('k=0', 'int')]:.3e}   k>0: bnd {s[('k>0', 'bnd')]:.3e} "
          f"int {s[('k>0', 'int')]:.3e}")


def main() -> int:
    print(f"REPO_SHA={_sha()}  n={N} ng={NG} km={KM} dt={DT} "
          f"n_split={N_SPLIT}  oracle={fsp.ORACLE_ROOT}/run_nh_*_gfs")

    from legoesm.core.fv3_native_acoustic_3d import (
        acoustic_substep_3d,
        build_nh_carry,
    )
    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context,
    )
    from legoesm.core.fv3_native_dynamics import (
        fv_dynamics_step,
        p_var_nonhydrostatic,
        pt_to_theta_v,
    )
    from legoesm.core.fv3_native_eta import set_eta_analytic
    from legoesm.core.fv3_native_mapz import lagrangian_to_eulerian
    from legoesm.core.fv3_native_state_3d import field_shape
    from legoesm.core.fv3_native_tracer2d import (
        alloc_flux_capacitors,
        tracer_2d_1l_sixface,
    )
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_CP_AIR,
        FV3_GRAV,
        FV3_KAPPA,
        FV3_RDGAS,
    )
    import legoesm.core.fv3_native_dsw_tail_3d as tailmod

    ctx = build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                     use_ext_metrics=False,
                                     oracle_conventions=True)
    n, ng = ctx["n"], ctx["ng"]
    ak, bk, ptop, _ks = set_eta_analytic(KM)
    ptop = float(ptop)

    orc_ic = fsp.load_oracle(f"{fsp.ORACLE_ROOT}/run_nh_zerostep_gfs",
                             nh=True)
    orc_1 = fsp.load_oracle(f"{fsp.ORACLE_ROOT}/run_nh_1step_gfs", nh=True)
    fsp.require_flat_orography(orc_ic)
    fsp.require_flat_orography(orc_1)

    def fresh():
        state = fsp.build_port_ic(ctx, ak, bk, nh=True)
        press = [p_var_nonhydrostatic(f["delp"], f["delz"], f["pt"],
                                      ptop=ptop, akap=FV3_KAPPA,
                                      n=n, ng=ng, km=KM) for f in state]
        q = [[np.zeros(field_shape("delp", n, ng, KM), dtype=np.float64)
              for _ in range(fsp.NR_TRACERS)] for _ in range(6)]
        return state, press, q

    state, press, q = fresh()
    p_ic = fsp.port_window(state, ctx)

    # instrument control 1: the face map, derived exactly as the gate
    # derives it, must land at the quad floor.
    _cost, meta, perm, worst, _pf, _wo = fsp.derive_face_map(p_ic, orc_ic)
    print(f"IC control: worst rel {worst:.3e} "
          f"(map {[int(p) + 1 for p in perm]})")
    if worst > fsp.IC_CONTROL_MAX_REL:
        raise SystemExit("IC control failed; refusing to trace")

    m_a = n + 2 * ng
    ctx["hs6"] = [np.zeros((m_a, m_a), dtype=np.float64) for _ in range(6)]

    # ---- the traced step: fv_dynamics_step's NH lane, unrolled -------
    dp0 = ((np.asarray(ak)[1:] - np.asarray(ak)[:-1])
           + (np.asarray(bk)[1:] - np.asarray(bk)[:-1]) * 1.0e5)
    for t in range(6):
        pt_to_theta_v(state[t]["pt"], press[t]["pkz"], n=n, ng=ng)
    w_ic6 = [np.array(_win(state[t]["w"])) for t in range(6)]

    nh = build_nh_carry(ctx, KM, ctx["hs6"])
    dp1_6 = [np.array(state[t]["delp"], copy=True) for t in range(6)]
    flux_cap = alloc_flux_capacitors(n, ng, KM)

    rec = {"pre": None, "post": None}
    orig_dgrid_nh = tailmod.dgrid_nh_pressure_phase_3d

    def wrapped(wctx, csw_press, dsw_outs, tail_outs, wnh, wkm, **kw):
        rec["pre"] = [np.array(t["w"]) for t in tail_outs]
        out = orig_dgrid_nh(wctx, csw_press, dsw_outs, tail_outs,
                            wnh, wkm, **kw)
        rec["post"] = [np.array(t["w"]) for t in tail_outs]
        return out

    tailmod.dgrid_nh_pressure_phase_3d = wrapped
    dt_sub = DT / float(N_SPLIT)
    press_out: list = []
    print("\nPER-SUB-STEP w increments, max|dw| over faces, split "
          "(k-level) x (boundary ring <3 / interior):")
    try:
        for it in range(1, N_SPLIT + 1):
            w_entry6 = [np.array(_win(state[t]["w"])) for t in range(6)]
            acoustic_substep_3d(
                ctx, state, dt_sub, KM, first_substep=(it == 1),
                ptop=ptop, akap=FV3_KAPPA, cp_air=FV3_CP_AIR,
                remap_step=(it == N_SPLIT), remap_follows=True,
                hydrostatic=False, nh=nh, p_fac=0.05, a_imp=1.0,
                dp0=dp0, use_logp=False,
                press_out=(press_out if it == N_SPLIT else None),
                flux_cap=flux_cap)
            if rec["pre"] is None or rec["post"] is None:
                raise SystemExit("wrapper never fired -- the function-"
                                 "scope import no longer resolves the "
                                 "module attribute; probe is blind")
            dw_dsw = [_win(rec["pre"][t]) - w_entry6[t] for t in range(6)]
            dw_riem = [_win(rec["post"][t]) - _win(rec["pre"][t])
                       for t in range(6)]
            cum = [_win(state[t]["w"]) - w_ic6[t] for t in range(6)]
            print(f"  it={it}/{N_SPLIT}:")
            _row("d_sw arms (entry->preRiem)", dw_dsw)
            _row("Riem_Solver3 (pre->post)", dw_riem)
            _row("cumulative |w - IC|", cum)
            rec["pre"] = rec["post"] = None
    finally:
        tailmod.dgrid_nh_pressure_phase_3d = orig_dgrid_nh

    # ---- tracer_2d + the NH remap, verbatim from fv_dynamics_step ----
    tracer_2d_1l_sixface(ctx, q, dp1_6, flux_cap, km=KM,
                         nq=fsp.NR_TRACERS, hord_tr=6, dt=DT,
                         q_split=0, nord_tr=0, trdm=0.0, lim_fac=1.0,
                         z_tracer=True, inline_q=False)
    w_pre_remap6 = [np.array(_win(state[t]["w"])) for t in range(6)]

    # w_limiter=False twin FIRST, on deep copies of everything the remap
    # mutates, so the real call below stays byte-identical to the driver.
    twin_w = []
    for t in range(6):
        g = press_out[t]
        cp_press = {k: np.array(press[t][k]) for k in press[t]}
        cp_press["pe"][:] = g["pe"]
        cp_press["peln"][:] = g["peln"]
        cp_press["pk"][ng:ng + n, ng:ng + n, :] = g["pk"]
        cp_state = {k: np.array(state[t][k]) for k in state[t]}
        cp_q = [np.array(a) for a in q[t]]
        lagrangian_to_eulerian(
            pe=cp_press["pe"], peln=cp_press["peln"], pk=cp_press["pk"],
            pkz=cp_press["pkz"], delp=cp_state["delp"],
            pt=cp_state["pt"], u=cp_state["u"], v=cp_state["v"],
            ps=cp_press["ps"], ak=ak, bk=bk, ptop=ptop, akap=FV3_KAPPA,
            cp=FV3_CP_AIR, r_vir=0.0, km=KM, n=n, ng=ng,
            kord_mt=fsp.KORD_MT, kord_tm=fsp.KORD_TM,
            kord_tr=fsp.KORD_TR, q=cp_q,
            omga=np.zeros(field_shape("delp", n, ng, KM),
                          dtype=np.float64),
            last_step=True, hydrostatic=False, adiabatic=True,
            w=cp_state["w"], delz=cp_state["delz"],
            ws=np.array(g["ws"]), kord_wz=9, w_limiter=False,
            rdgas=FV3_RDGAS, grav=FV3_GRAV,
            consv=0.0, fill=False, do_sat_adj=False,
            do_inline_mp=False, do_adiabatic_init=False)
        twin_w.append(np.array(_win(cp_state["w"])))

    for t in range(6):
        g = press_out[t]
        press[t]["pe"][:] = g["pe"]
        press[t]["peln"][:] = g["peln"]
        press[t]["pk"][ng:ng + n, ng:ng + n, :] = g["pk"]
        lagrangian_to_eulerian(
            pe=press[t]["pe"], peln=press[t]["peln"], pk=press[t]["pk"],
            pkz=press[t]["pkz"], delp=state[t]["delp"],
            pt=state[t]["pt"], u=state[t]["u"], v=state[t]["v"],
            ps=press[t]["ps"], ak=ak, bk=bk, ptop=ptop, akap=FV3_KAPPA,
            cp=FV3_CP_AIR, r_vir=0.0, km=KM, n=n, ng=ng,
            kord_mt=fsp.KORD_MT, kord_tm=fsp.KORD_TM,
            kord_tr=fsp.KORD_TR, q=q[t],
            omga=np.zeros(field_shape("delp", n, ng, KM),
                          dtype=np.float64),
            last_step=True, hydrostatic=False, adiabatic=True,
            w=state[t]["w"], delz=state[t]["delz"],
            ws=press_out[t]["ws"], kord_wz=9, w_limiter=True,
            rdgas=FV3_RDGAS, grav=FV3_GRAV,
            consv=0.0, fill=False, do_sat_adj=False,
            do_inline_mp=False, do_adiabatic_init=False)

    dw_remap = [_win(state[t]["w"]) - w_pre_remap6[t] for t in range(6)]
    print("\nREMAP alone (w, post-acoustic -> final):")
    _row("remap dw", dw_remap)
    lim_diff = max(float(np.abs(_win(state[t]["w"]) - twin_w[t]).max())
                   for t in range(6))
    print(f"  w_limiter True-vs-False twin: max|dw| = {lim_diff:.3e} "
          f"(0.0 == limiter is a NO-OP on this regime, matching the "
          f"oracle's zero W_LIMITER prints)")

    # ---- final residual vs the oracle, gate arithmetic ---------------
    p_1 = fsp.port_window(state, ctx)
    print("\nFINAL w residual vs oracle (same map as the gate):")
    worst_w = 0.0
    for pf in range(6):
        ot = perm[pf]
        pairs, _ws = fsp.apply_map(p_1[pf], orc_1[ot], meta[pf][ot])
        a, b = pairs["w"]
        d = a - b
        r = fsp.rel(a, b)
        worst_w = max(worst_w, r)
        per_k = "  ".join(f"k{k}:{np.abs(d[:, :, k]).max():.2e}"
                          for k in range(KM))
        bnd = float(np.abs(d[_BND, :]).max())
        itr = float(np.abs(d[~_BND, :]).max())
        print(f"  face {pf + 1}->tile {ot + 1}: rel={r:.4e} "
              f"|d|max={np.abs(d).max():.4e}  bnd={bnd:.2e} "
              f"int={itr:.2e}  [{per_k}]")
    print(f"WORST w rel: {worst_w:.4e}  (gate's standing number: "
          f"6.6116e-04 -- a probe that does not reproduce it traced a "
          f"different step)")

    # ---- instrument control 2: bit-identity vs the real driver -------
    state2, press2, q2 = fresh()
    fv_dynamics_step(ctx, state2, press2, bdt=DT, km=KM, k_split=1,
                     n_split=N_SPLIT, ptop=ptop, ak=ak, bk=bk,
                     akap=FV3_KAPPA, cp_air=FV3_CP_AIR,
                     kord_mt=fsp.KORD_MT, kord_tm=fsp.KORD_TM,
                     kord_tr=fsp.KORD_TR, q=q2, hydrostatic=False,
                     w_limiter=True)
    bad = []
    for t in range(6):
        for f in ("u", "v", "pt", "delp", "w", "delz"):
            if not np.array_equal(state[t][f], state2[t][f]):
                bad.append((t + 1, f,
                            float(np.abs(np.asarray(state[t][f])
                                         - np.asarray(state2[t][f])).max())))
    if bad:
        print(f"\nINSTRUMENT CONTROL FAILED: traced step != "
              f"fv_dynamics_step on {bad}; every number above is "
              f"UNTRUSTED (the unrolled loop drifted from the driver).")
        return 1
    print("\nINSTRUMENT CONTROL PASSED: traced step is bit-identical to "
          "fv_dynamics_step on all six faces, all prognostics.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
