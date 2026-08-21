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

WHAT THE SERIES CAN AND CANNOT SETTLE (codex BLOCKER, job 9450542, and
the reason no verdict is printed). End-of-sub-step readings alone do NOT
identify the cause. A wrong STAGE that runs every sub-step injects a
similar signed error each time and produces the same ramp a wrong
COEFFICIENT would; a wrong coefficient can excite a mode that is nearly
full-size after one sub-step. What the series DOES give is the shape of
the growth and the LOCATION of the damage, and that is what is reported:
the worst cell's face and indices, the SIGNED error there, and -- because
a max over cells is free to move between sub-steps and then describes no
single error's history -- the FINAL worst cell's error traced backwards
through all the sub-steps. Separating stage from coefficient needs
captures BETWEEN the stages of one sub-step, which this instrument does
not have.

The oracle side is a second instrumented build
(``scripts/cluster/fv3_native/build_nh_wsubstep_oracle.sbatch``) that
adds a per-sub-step dump inside ``dyn_core`` AND keeps the pre-remap dump
of the previous instrument, so both capture points live in ONE binary.
The port side is ``fv_dynamics_step(..., return_substeps=True)``, by
RETURN -- never a probe that re-runs the acoustic chain itself, which
would be a second implementation of the thing under test.

SEVEN CONTROLS, and the script exits non-zero on each:

  1. the IC face map is re-derived and must hit its ~1e-14 floor;
  2. the port's full-step residual must reproduce the established
     6.6116e-04, or this is not the configuration under investigation;
  3. the instrumented binary's OWN restart must be bitwise the certified
     one, on EVERY prognostic and with no tolerance -- a freshly compiled
     binary is a different program until shown otherwise. Note what this
     does NOT cover: a perturbation that the instrumentation introduces
     mid-step and that cancels by the restart. The build script's
     no-dump control binary is what addresses that (codex MAJOR, job
     9450542);
  4. the oracle's sub-step-``n_split`` dump must be BITWISE its own
     pre-remap dump. Nothing between the loop's end and ``dyn_core``'s
     return writes ``w`` (dyn_core.F90:1736-1830), so this is the check
     that the new dump sits where it claims; and
  5. consecutive oracle sub-step dumps must DIFFER, or the series is
     eight copies of one reading and every trend below is an artifact;
  6. the PORT's last sub-step capture must equal its own pre-remap
     capture, the mirror of control 4; and
  7. the last sub-step's ``w`` error must reproduce the pre-remap
     instrument's established 1.3003e-07. Controls 1-2 authenticate the
     ``apply_map`` chain; this is the only one that authenticates the
     hand-rolled window/dihedral chain the series is actually read
     through.

THE ``pt`` SERIES CARRIES A CAVEAT the ``w`` series does not. After the
sub-step loop, ``dyn_core`` can still write ``pt`` in its dissipative
heating block (``:1769, :1774, :1796``) when ``d_con > 1e-5``; the pinned
deck sets ``d_con = 0.0`` so it does not here, but the "nothing writes it
after the loop" statement is a ``w`` statement, not a general one.

NO VERDICT IS PRINTED. The series, the worst cell, and the fixed-cell
history are the output; what they mean is an interpretation, and this
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

#: The established port-vs-oracle |d|max of ``w`` at k=0 just before the
#: remap, measured by the pre-remap instrument (job 9448215). The last
#: sub-step must reproduce it: nothing writes ``w`` between the two
#: capture points, so any disagreement is the MEASUREMENT PATH, not
#: physics. Both reviewers asked for this anchor independently (GLM M2,
#: job 9450546) -- it is the only control that exercises the hand-rolled
#: window/dihedral chain the series is read through, rather than the
#: `apply_map` chain the other controls use.
PREREMAP_W_DMAX_K0 = 1.3003e-07
PREREMAP_W_BAND = 2.0


def _oracle_level0(run_dir: str, blk: int, name: str, tile: int,
                   transposed: bool, prefix: str) -> np.ndarray:
    """The oracle's compute-window level-0 field, in port index order."""
    d = read_dump(run_dir, blk, name, tile, prefix=prefix)
    off = (d.shape[1] - N) // 2
    return oracle_ij(d[:, off:off + N, off:off + N], transposed)


def require_deck_agreement(args) -> None:
    """The oracle run's OWN namelist must match what this probe assumes.

    The probe hardcodes ``k_split = 1`` and defaults ``n_split = 8`` /
    ``dt = 1920``. If the deck drifts, the failure today is a missing-file
    exit at sub-step 7 -- late and cryptic -- and under ``k_split > 1`` it
    is not a failure at all: the sub-step dumps carry no ``n_map`` in
    their names, so a second outer iteration would silently OVERWRITE the
    first and the series would be the last iteration's wearing the whole
    step's name (GLM m4/m7, job 9450546). Read the deck instead.
    """
    nml = os.path.join(args.sdump, "input.nml")
    if not os.path.exists(nml):
        raise SystemExit(f"no input.nml under {args.sdump}: cannot confirm "
                         f"the oracle ran the configuration this probe "
                         f"assumes")
    want = {"k_split": 1, "n_split": args.n_split}
    seen: dict = {}
    with open(nml) as fh:
        for line in fh:
            bare = line.split("!")[0]
            if "=" not in bare:
                continue
            key, _, val = bare.partition("=")
            key = key.strip().lower()
            if key in want:
                try:
                    seen[key] = int(val.strip().rstrip(","))
                except ValueError:
                    pass
    missing = [k for k in want if k not in seen]
    if missing:
        raise SystemExit(f"{nml}: {missing} not found; refusing to assume "
                         f"the deck agrees with this probe")
    bad = [f"{k}={seen[k]} (probe assumes {v})"
           for k, v in want.items() if seen[k] != v]
    if bad:
        raise SystemExit(
            "DECK DISAGREES WITH THE PROBE: " + ", ".join(bad) + ". "
            "The sub-step dumps carry no n_map in their names, so k_split "
            "> 1 would silently give the last outer iteration only.")
    print(f"deck agreement: k_split={seen['k_split']} "
          f"n_split={seen['n_split']} (from {nml})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sdump", default=SUBSTEP_ROOT)
    ap.add_argument("--ic-run", default=f"{ORACLE_ROOT}/run_nh_zerostep_gfs")
    ap.add_argument("--step-run", default=f"{ORACLE_ROOT}/run_nh_1step_gfs")
    ap.add_argument("--n-split", type=int, default=8)
    ap.add_argument("--dt", type=float, default=1920.0)
    args = ap.parse_args(argv)

    require_deck_agreement(args)

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

    # CONTROL 3: is the NEW binary the certified program? BITWISE, and on
    # every prognostic -- not a tolerance on w alone. The sibling probe's
    # version accepted rel <= 1e-12 on w while calling itself a bitwise
    # check, which would have passed with pt or delp arbitrarily different
    # (codex MINOR, job 9450542).
    instr_1 = load_oracle(args.sdump, nh=True)
    bad = []
    for t in range(6):
        for name in sorted(set(instr_1[t]) & set(orc_1[t])):
            a, b = instr_1[t][name], orc_1[t][name]
            if getattr(a, "shape", None) != getattr(b, "shape", None):
                bad.append(f"tile{t + 1} {name}: shape {a.shape} vs {b.shape}")
            elif not np.array_equal(a, b):
                bad.append(f"tile{t + 1} {name}: |d|max "
                           f"{float(np.abs(a - b).max()):.3e}")
    print(f"instrumented-vs-certified restart, BITWISE on every field: "
          f"{'IDENTICAL' if not bad else str(len(bad)) + ' differ'}")
    if bad:
        raise SystemExit(
            "INSTRUMENT CONTROL FAILED: the sub-step binary's restart is "
            "not bitwise the certified one, so its dumps cannot be compared "
            "against numbers measured with the certified binary:\n  "
            + "\n  ".join(bad[:12]))

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

    # CONTROL 6: THE PORT SIDE MEETS TOO. `pre` was requested and then
    # never read (GLM M3, job 9450546). Nothing between the acoustic
    # loop's end and the remap touches the prognostics on this lane, so
    # the last sub-step snapshot must BE the pre-remap one.
    for t in range(6):
        for name in ("delp", "pt", "u", "v", "w"):
            if not np.array_equal(subs[-1][t][name], pre[t][name]):
                raise SystemExit(
                    f"INSTRUMENT CONTROL FAILED: port face {t + 1} {name}: "
                    f"the last sub-step capture is not the pre-remap "
                    f"capture, so the two sides are being read at "
                    f"different points.")
    print("port: last sub-step capture == pre-remap capture, all fields")

    # THE MEASUREMENT. One FIXED scale per field across all sub-steps:
    # rel() picks max(peaks) PER CALL, so a per-sub-step denominator that
    # itself grows would hide (or invent) growth in the ratio. Absolute
    # |d|max is the primary number; the scale is printed once beside it.
    #
    # Two things are reported that a bare max over cells cannot give
    # (codex BLOCKER, job 9450542): WHERE the worst cell is and what SIGN
    # the error has there, and the history of ONE fixed cell -- the final
    # sub-step's worst -- across all sub-steps. A max whose argument
    # wanders between sub-steps is not any single error's growth curve.
    cs = slice(NG, NG + N)

    def _pair(name, it, pf):
        """(port, oracle) level-0 windows for one field/sub-step/face."""
        ot = perm[pf]
        transposed, nm, _su, _sv = meta[pf][ot]
        ow = _oracle_level0(args.sdump, it, name, ot + 1, transposed,
                            "sdump_it")
        pw = DIHEDRAL[nm](subs[it - 1][pf][name][cs, cs, :1])
        return pw, ow

    series: dict = {}
    for name in DUMP_FIELDS:
        scale = 0.0
        for it in range(1, args.n_split + 1):
            for pf in range(6):
                pw, ow = _pair(name, it, pf)
                scale = max(scale, float(np.abs(ow).max()),
                            float(np.abs(pw).max()))
        rows = []
        for it in range(1, args.n_split + 1):
            worst = (0.0, None, None, 0.0)
            for pf in range(6):
                pw, ow = _pair(name, it, pf)
                d = pw - ow
                idx = np.unravel_index(int(np.abs(d).argmax()), d.shape)
                mag = float(np.abs(d[idx]))
                if mag > worst[0]:
                    worst = (mag, pf, idx, float(d[idx]))
            rows.append((it, *worst))
        series[name] = (scale, rows)

    for name in DUMP_FIELDS:
        scale, rows = series[name]
        last = rows[-1][1]
        print(f"\n{name} at k=0, per acoustic sub-step "
              f"(scale {scale:.6e}, fixed across sub-steps):")
        print("   it      |d|max      signed d       rel     d/d(last)   "
              "worst cell")
        for it, mag, pf, idx, signed in rows:
            frac = (mag / last) if last > 0.0 else float("nan")
            where = ("(no difference)" if pf is None else
                     f"face{pf + 1}->tile{perm[pf] + 1} "
                     f"i={idx[0]} j={idx[1]}")
            print(f"  {it:3d}   {mag:.4e}  {signed:+.4e}  "
                  f"{mag / scale:.3e}  {frac:9.4f}   {where}")

    # THE FIXED-CELL HISTORY. The final sub-step's worst cell, read at
    # every sub-step, so these numbers are one error's growth rather than
    # a moving maximum's.
    for name in DUMP_FIELDS:
        _scale, rows = series[name]
        _mag, pf, idx, _signed = rows[-1][1:]
        if pf is None:
            continue
        print(f"\n{name}: the cell that is worst at sub-step "
              f"{args.n_split} (face{pf + 1}->tile{perm[pf] + 1} "
              f"i={idx[0]} j={idx[1]}), read at every sub-step:")
        hist = []
        for it in range(1, args.n_split + 1):
            pw, ow = _pair(name, it, pf)
            hist.append(float((pw - ow)[idx]))
        print("   " + "  ".join(f"{it}:{v:+.3e}"
                                for it, v in enumerate(hist, start=1)))

    # CONTROL 7 (last, because it needs the series): the final sub-step
    # must reproduce the ESTABLISHED pre-remap |d|max. Controls 1-2 run
    # through `derive_face_map`/`apply_map`; the series above runs through
    # a hand-rolled window + dihedral chain, and nothing else exercises
    # it. A wrong dihedral on one face, or a halo off-by-one, produces
    # gradient-scale garbage that every earlier control accepts.
    w_last = series["w"][1][-1][1]
    lo = PREREMAP_W_DMAX_K0 / PREREMAP_W_BAND
    hi = PREREMAP_W_DMAX_K0 * PREREMAP_W_BAND
    print(f"\nanchor: sub-step {args.n_split} |dw|max = {w_last:.4e} "
          f"(established pre-remap {PREREMAP_W_DMAX_K0:.4e}, "
          f"band [{lo:.3e}, {hi:.3e}])")
    if not lo <= w_last <= hi:
        raise SystemExit(
            f"INSTRUMENT CONTROL FAILED: the last sub-step scores "
            f"{w_last:.4e} against the oracle, but the same state measured "
            f"through the pre-remap instrument scores "
            f"{PREREMAP_W_DMAX_K0:.4e}. Nothing writes w between those two "
            f"points, so this is the measurement path -- the window, the "
            f"dihedral or the face pairing -- not physics.")

    if series["w"][1][-1][1] <= 0.0:
        raise SystemExit(
            "the port matches the oracle BITWISE at the last sub-step, "
            "which contradicts control 2's 6.6116e-04 -- the two sides are "
            "not being read at the same point.")
    print("\nNO VERDICT HERE. End-of-sub-step readings cannot separate a "
          "wrong stage from a wrong coefficient: a stage that runs every "
          "sub-step also produces a ramp, and a coefficient can excite a "
          "mode that is nearly full-size after one. That separation needs "
          "captures BETWEEN the stages of a single sub-step.")
    print("\nSCOPE, stated: level k=0 only -- the level where the worst "
          "full-step error lives. It says nothing about where in the "
          "column the damage is born.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
