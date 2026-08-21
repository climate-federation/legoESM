#!/usr/bin/env python
"""Is the NH ``w`` error born in ONE acoustic sub-step, or grown over eight?

WHERE THIS PICKS UP. The NH dry arm's 6.6116e-04 against the oracle is
1.3003e-07 m/s of ``w``, and job 9448215 localised it twice: it is
already present when ``dyn_core`` returns (the remap moves it by 0.008%
at k=0), and it is w-SPECIFIC (pt 2.3e-13, delp 3.0e-10, both at the
parity floor). So the defect is born inside ``do it=1,n_split``, which
runs EIGHT times per step. A single end-of-loop reading cannot separate

    error already full-size at sub-step 1  ->  ONE STAGE is wrong
    error growing across the eight         ->  a COEFFICIENT is wrong

and those two want completely different searches. This scores the port's
``w`` against the oracle's at the end of EVERY sub-step.

The oracle side is a second instrumented build
(``scripts/cluster/fv3_native/build_nh_wsubstep_oracle.sbatch``) that
adds a per-sub-step dump inside ``dyn_core`` AND keeps the pre-remap dump
of the previous instrument, so both capture points live in ONE binary.
The port side is ``fv_dynamics_step(..., return_substeps=True)``, by
RETURN -- never a probe that re-runs the acoustic chain itself, which
would be a second implementation of the thing under test.

FIVE CONTROLS RUN BEFORE ANY NUMBER IS REPORTED, and the script exits
non-zero on each:

  1. the IC face map is re-derived and must hit its ~1e-14 floor;
  2. the port's full-step residual must reproduce the established
     6.6116e-04, or this is not the configuration under investigation;
  3. the instrumented binary's OWN restart must be bitwise the certified
     one -- a freshly compiled binary is a different program until shown
     otherwise;
  4. the oracle's sub-step-``n_split`` dump must be BITWISE its own
     pre-remap dump. Nothing between the loop's end and ``dyn_core``'s
     return writes ``w`` (dyn_core.F90:1736-1830), so this is the check
     that the new dump sits where it claims; and
  5. consecutive oracle sub-step dumps must DIFFER, or the series is
     eight copies of one reading and every trend below is an artifact.

NO VERDICT IS PRINTED. The series and its growth ratios are the output;
which of the two signatures it matches is an interpretation, and this
campaign has already been burned once by a probe that baked its own
conclusion into its output.
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
from nh_preremap_w_parity import read_dump      # noqa: E402

SUBSTEP_ROOT = "/burg-archive/glab/users/pg2328/fv3_wsubstep/run_nh_1step"

#: Fields the sub-step instrument dumps (level 1 / k=0).
DUMP_FIELDS = ("w", "pt", "delp")


def _oracle_level0(run_dir: str, blk: int, name: str, tile: int,
                   transposed: bool, prefix: str) -> np.ndarray:
    """The oracle's compute-window level-0 field, in port index order."""
    d = read_dump(run_dir, blk, name, tile, prefix=prefix)
    off = (d.shape[1] - N) // 2
    return oracle_ij(d[:, off:off + N, off:off + N], transposed)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sdump", default=SUBSTEP_ROOT)
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
        hydrostatic=False, w_limiter=True,
        return_pre_remap=True, return_substeps=True)
    pre, subs = out["pre_remap"], out["substeps"]
    p_1 = port_window(state, ctx)
    if len(subs) != args.n_split:
        raise SystemExit(f"port returned {len(subs)} sub-steps, expected "
                         f"{args.n_split}")

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
            f"INSTRUMENT CONTROL FAILED: this script scores the full step "
            f"at {worst_step:.4e}, not the established 6.6116e-04, so it "
            f"is NOT running the configuration whose gap is under "
            f"investigation.")

    # CONTROL 3: is the NEW binary the certified program?
    instr_1 = load_oracle(args.sdump, nh=True)
    worst_auth = max(rel(instr_1[t]["w"], orc_1[t]["w"]) for t in range(6))
    print(f"instrumented-vs-certified restart: worst w rel {worst_auth:.4e}")
    if worst_auth > 1.0e-12:
        raise SystemExit(
            f"INSTRUMENT CONTROL FAILED: the sub-step binary's own restart "
            f"differs from the certified one at {worst_auth:.4e}; it is not "
            f"running the certified program.")

    # CONTROL 4: the last sub-step dump IS the pre-remap dump, bitwise.
    # Both come from THIS binary, so a mismatch is a capture-point error
    # and cannot be blamed on build drift.
    worst_meet = 0.0
    for t in range(1, 7):
        for name in DUMP_FIELDS:
            a = read_dump(args.sdump, args.n_split, name, t,
                          prefix="sdump_it")
            b = read_dump(args.sdump, 1, name, t, prefix="wdump_b")
            if a.shape != b.shape:
                raise SystemExit(
                    f"tile {t} {name}: sub-step dump {a.shape} vs pre-remap "
                    f"dump {b.shape} -- different domains, not comparable")
            if name == "w":
                worst_meet = max(worst_meet,
                                 float(np.abs(a - b).max()))
    print(f"oracle sub-step {args.n_split} vs its own pre-remap dump: "
          f"|dw|max {worst_meet:.3e} (must be 0.0)")
    if worst_meet != 0.0:
        raise SystemExit(
            "INSTRUMENT CONTROL FAILED: the oracle's last sub-step dump is "
            "not bitwise its pre-remap dump, but nothing between the loop's "
            "end and dyn_core's return writes w (dyn_core.F90:1736-1830). "
            "One of the two dumps is not where it claims to be.")

    # CONTROL 5: the series is not eight copies of one reading.
    same = []
    for it in range(1, args.n_split):
        d = max(float(np.abs(read_dump(args.sdump, it, "w", t,
                                       prefix="sdump_it")
                             - read_dump(args.sdump, it + 1, "w", t,
                                         prefix="sdump_it")).max())
                for t in range(1, 7))
        if d == 0.0:
            same.append(it)
    if same:
        raise SystemExit(
            f"INSTRUMENT CONTROL FAILED: oracle sub-steps {same} are "
            f"identical to their successors -- the dump is not being "
            f"rewritten per sub-step, and every trend below is an artifact.")

    # THE MEASUREMENT. One FIXED scale per field across all sub-steps:
    # rel() picks max(peaks) PER CALL, so a per-sub-step denominator that
    # itself grows would hide (or invent) growth in the ratio. Absolute
    # |d|max is the primary number; the scale is printed once beside it.
    cs = slice(NG, NG + N)
    series: dict = {}
    for name in DUMP_FIELDS:
        scale = 0.0
        for it in range(1, args.n_split + 1):
            for pf in range(6):
                ot = perm[pf]
                transposed, nm, _su, _sv = meta[pf][ot]
                ow = _oracle_level0(args.sdump, it, name, ot + 1,
                                    transposed, "sdump_it")
                pw = DIHEDRAL[nm](subs[it - 1][pf][name][cs, cs, :1])
                scale = max(scale, float(np.abs(ow).max()),
                            float(np.abs(pw).max()))
        rows = []
        for it in range(1, args.n_split + 1):
            worst_d, worst_face = 0.0, None
            for pf in range(6):
                ot = perm[pf]
                transposed, nm, _su, _sv = meta[pf][ot]
                ow = _oracle_level0(args.sdump, it, name, ot + 1,
                                    transposed, "sdump_it")
                pw = DIHEDRAL[nm](subs[it - 1][pf][name][cs, cs, :1])
                d = float(np.abs(pw - ow).max())
                if d > worst_d:
                    worst_d, worst_face = d, f"face{pf + 1}->tile{ot + 1}"
            rows.append((it, worst_d, worst_face))
        series[name] = (scale, rows)

    for name in DUMP_FIELDS:
        scale, rows = series[name]
        last = rows[-1][1]
        print(f"\n{name} at k=0, per acoustic sub-step "
              f"(scale {scale:.6e}, fixed across sub-steps):")
        print("   it      |d|max        rel        d(it)/d(last)   worst")
        for it, d, face in rows:
            frac = (d / last) if last > 0.0 else float("nan")
            print(f"  {it:3d}   {d:.6e}   {d / scale:.4e}   "
                  f"{frac:12.4f}   {face}")

    # The two signatures, stated as a reader's guide -- NOT decided here.
    w_rows = series["w"][1]
    if w_rows[-1][1] <= 0.0:
        raise SystemExit(
            "the port matches the oracle BITWISE at the last sub-step, "
            "which contradicts control 2's 6.6116e-04 -- the two sides are "
            "not being read at the same point.")
    print(f"\nw growth: sub-step 1 is {w_rows[0][1] / w_rows[-1][1]:.4f} of "
          f"sub-step {args.n_split}.")
    print("READ IT, do not let this script read it for you: a fraction "
          "near 1 means the whole error appears in ONE sub-step (a stage "
          "is wrong); a fraction near 1/n_split, or a smooth ramp, means "
          "it accumulates (a coefficient is wrong). Anything else is "
          "neither, and is the interesting case.")
    print("\nSCOPE, stated: level k=0 only -- the level where the worst "
          "full-step error lives. It says nothing about where in the "
          "column the damage is born.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
