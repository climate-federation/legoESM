#!/usr/bin/env python
"""Is the NH gap already there when dyn_core returns, or does the remap make it?

THE OPEN QUESTION since 2026-08-19. The NH dry arm scores 6.6116e-04
against the oracle, which is 1.3003e-07 m/s of ``w`` at k=0 boundary
cells. Metrics (including a bitwise transplant), constants, the
w_limiter and the delz IC have all been ELIMINATED; the cause is
unknown. One measurement splits the remaining space in half: compare
the port's ``w`` BEFORE the remap against the oracle's ``w`` at the
same point.

    gap already present pre-remap -> it is the ACOUSTIC loop
    gap absent pre-remap          -> it is the REMAP (kord_wz)

The oracle side comes from an instrumented build
(``scripts/cluster/fv3_native/build_nh_wdump_oracle.sbatch``) that
dumps w/pt/delp/delz between ``dyn_core`` and
``Lagrangian_to_Eulerian``; the port side from
``fv_dynamics_step(..., return_pre_remap=True)``.

THE INSTRUMENT IS CONTROLLED BEFORE ITS NUMBER IS USED, three ways:
the IC face map is re-derived and must hit its ~1e-14 floor; the
oracle's own pre-remap-vs-restart difference is printed, so a dump that
is secretly the post-remap state is visible; and the port's full-step
residual is recomputed here and must reproduce the established
6.6116e-04, or this script is not running the configuration whose gap
is under investigation.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from full_step_oracle_parity import (          # noqa: E402
    DIHEDRAL,
    IC_CONTROL_MAX_REL,
    apply_map,
    KM,
    N,
    NG,
    ORACLE_ROOT,
    build_port_ic,
    derive_face_map,
    load_oracle,
    oracle_ij,
    port_window,
    rel,
)

WDUMP_ROOT = "/burg-archive/glab/users/pg2328/fv3_wdump/run_nh_1step"


def read_dump(run_dir: str, blk: int, name: str, tile: int) -> np.ndarray:
    """One ``wdump_b<blk>_<name>_t<tile>.dat`` as a (ni, nj) array.

    The writer emits ``ilo ihi jlo jhi`` then ``i j value`` rows, with
    the Fortran halo offsets in the header -- so the array is placed by
    its OWN stated bounds rather than by an assumed shape.
    """
    path = os.path.join(run_dir, f"wdump_b{blk}_{name}_t{tile}.dat")
    if not os.path.exists(path):
        raise SystemExit(f"missing dump {path}")
    with open(path) as fh:
        ilo, ihi, jlo, jhi = (int(x) for x in fh.readline().split())
        out = np.full((ihi - ilo + 1, jhi - jlo + 1), np.nan,
                      dtype=np.float64)
        for line in fh:
            i, j, v = line.split()
            out[int(i) - ilo, int(j) - jlo] = float(v)
    if np.isnan(out).any():
        raise SystemExit(f"{path}: {int(np.isnan(out).sum())} cells never "
                         f"written -- a truncated dump would read as a "
                         f"difference")
    # STORE IT THE WAY load_oracle DOES, (k, j, i) with k of length 1,
    # so oracle_ij can be reused verbatim rather than its logic being
    # re-derived here for a 2-D array. The first version compared
    # without the face map's dihedral at all and control 2 caught it
    # (full-step w read 1.0071e+00 against an established 6.6116e-04).
    return out.T[None, :, :]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wdump", default=WDUMP_ROOT)
    ap.add_argument("--ic-run", default=f"{ORACLE_ROOT}/run_nh_zerostep_gfs")
    ap.add_argument("--step-run", default=f"{ORACLE_ROOT}/run_nh_1step_gfs")
    ap.add_argument("--n-split", type=int, default=8)
    ap.add_argument("--dt", type=float, default=1920.0)
    args = ap.parse_args(argv)

    from legoesm.core.fv3_native_duo_stepper import build_six_face_duo_context
    from legoesm.core.fv3_native_dynamics import (
        fv_dynamics_step, p_var_nonhydrostatic,
    )
    from legoesm.core.fv3_native_eta import set_eta_analytic
    from legoesm.grids.fv3_native_gridstruct import FV3_CP_AIR, FV3_KAPPA

    ctx = build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                     oracle_conventions=True)
    m_a = N + 2 * NG
    ctx["hs6"] = [np.zeros((m_a, m_a), dtype=np.float64) for _ in range(6)]
    ak, bk, ptop, _ks = set_eta_analytic(KM)

    orc_ic = load_oracle(args.ic_run, nh=True)
    orc_1 = load_oracle(args.step_run, nh=True)
    state = build_port_ic(ctx, ak, bk, nh=True)[0]
    p_ic = port_window(state, ctx)

    # CONTROL 1: the face map, re-derived, at its own floor.
    # READ, not inferred: it returns
    # (cost, meta, best_perm, best_worst, per_field, wind_only)
    # -- full_step_oracle_parity.py:661.
    _cost, meta, perm, worst, _pf, _wo = derive_face_map(p_ic, orc_ic)
    print(f"IC face-map control: worst rel {worst:.3e} "
          f"(floor {IC_CONTROL_MAX_REL:.0e})")
    if worst > IC_CONTROL_MAX_REL:
        raise SystemExit("INSTRUMENT CONTROL FAILED: the IC does not "
                         "reproduce the oracle's; nothing below means "
                         "anything.")

    press = [p_var_nonhydrostatic(f["delp"], f["delz"], f["pt"],
                                 ptop=ptop, akap=FV3_KAPPA,
                                 n=N, ng=NG, km=KM) for f in state]
    q = [[np.zeros((m_a, m_a, KM), dtype=np.float64)] for _ in range(6)]
    out = fv_dynamics_step(
        ctx, state, press, bdt=args.dt, km=KM, k_split=1,
        n_split=args.n_split, ptop=ptop, ak=ak, bk=bk, akap=FV3_KAPPA,
        cp_air=FV3_CP_AIR, kord_mt=9, kord_tm=-9, kord_tr=9, q=q,
        hydrostatic=False, w_limiter=True, return_pre_remap=True)
    pre = out["pre_remap"]
    p_1 = port_window(state, ctx)

    # CONTROL 2: this script must reproduce the gap under investigation.
    worst_step = 0.0
    for pf in range(6):
        ot = perm[pf]
        pairs, _ws = apply_map(p_1[pf], orc_1[ot], meta[pf][ot])
        worst_step = max(worst_step, rel(*pairs["w"]))
    print(f"full-step w residual (this script): {worst_step:.4e}   "
          f"[established 6.6116e-04]")
    if not 3.0e-04 <= worst_step <= 1.5e-03:
        raise SystemExit(
            f"INSTRUMENT CONTROL FAILED: this script scores the full "
            f"step at {worst_step:.4e}, not the established 6.6116e-04, "
            f"so it is NOT running the configuration whose gap is under "
            f"investigation. Refusing to report a pre-remap number from "
            f"it. (The first version read 1.0071e+00 -- it had dropped "
            f"the face map's dihedral.)")

    # CONTROL 3: the dump must not secretly be the post-remap state.
    print("\noracle: how far the REMAP moves w (pre-remap dump vs "
          "restart), per tile:")
    row = []
    for t in range(1, 7):
        d = read_dump(args.wdump, 1, "w", t)          # (1, j, i) padded
        rst = orc_1[t - 1]["w"]                        # (k, j, i) compute
        nj = rst.shape[1]
        off = (d.shape[1] - nj) // 2
        dw = d[0, off:off + nj, off:off + nj]
        row.append(float(np.abs(dw - rst[0]).max()))
    print("  " + "  ".join(f"{x:10.4g}" for x in row))
    if max(row) == 0.0:
        raise SystemExit(
            "the pre-remap dump is IDENTICAL to the restart on every "
            "tile -- the dump is not where it claims to be, and the "
            "comparison below would be the full-step one wearing a "
            "different name.")

    # THE MEASUREMENT.
    print("\nPORT vs ORACLE, w BEFORE the remap:")
    worst_pre, worst_face = 0.0, None
    cs = slice(NG, NG + N)
    for pf in range(6):
        ot = perm[pf]
        transposed, nm, _su, _sv = meta[pf][ot]
        d = read_dump(args.wdump, 1, "w", ot + 1)      # (1, j, i) padded
        nj = N
        off = (d.shape[1] - nj) // 2
        ow = oracle_ij(d[:, off:off + nj, off:off + nj], transposed)
        # the PORT side carries the dihedral too -- w is cell-centred,
        # so factor +1, exactly as apply_map treats pt/delp.
        pw = DIHEDRAL[nm](pre[pf]["w"][cs, cs, :1])
        r = rel(pw, ow)
        if r > worst_pre:
            worst_pre, worst_face = r, f"face{pf + 1}->tile{ot + 1}"
        print(f"  face {pf + 1} -> tile {ot + 1}:  rel={r:.4e}")
    print(f"\nWORST pre-remap w rel: {worst_pre:.4e} at {worst_face}")
    print(f"full-step  w rel:      {worst_step:.4e}")
    if worst_step > 0:
        print(f"ratio pre/full:        {worst_pre / worst_step:.3f}")
    print("\nVERDICT: " + (
        "the gap is ALREADY THERE before the remap -> ACOUSTIC LOOP"
        if worst_pre > 0.5 * worst_step else
        "the gap is NOT there before the remap -> THE REMAP (kord_wz)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
