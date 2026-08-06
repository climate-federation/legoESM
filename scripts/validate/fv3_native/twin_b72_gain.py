#!/usr/bin/env python
"""Per-stage tangent-gain probe, OUR side (codex r4 decisive experiment).

Loads the block-72 state from the d2 twin npz, injects the SAME delp
vertex bump as the instrumented oracle (tile 3, cells (1,1) and
(1,n): +eps/+eps symmetric arm, +eps/-eps antisymmetric arm,
eps=1e-4 Pa), runs ONE acoustic step (entry_ascalar=True — block 73 is
an it==1 block start) with the 201-205 stage-dump hook, for baseline /
sym / anti, and writes one npz with keys <arm>_b<stage>_<name>_t<T>.

The gain comparator diffs perturbed-vs-baseline per stage IN THE SAME
FRAME per side — no cross-model mapping anywhere.

Usage: twin_b72_gain.py --twin twin_ours_c48.npz --block 72 --out g.npz
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

EPS = 1.0e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--twin", required=True)
    ap.add_argument("--block", type=int, default=72)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from pathlib import Path
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parents[3]))

    from legoesm.core.fv3_native_duo_stepper import (
        SW_CFG_CASE8,
        build_six_face_duo_context,
        full_acoustic_step_sixface,
    )

    twin = np.load(args.twin, allow_pickle=False)
    n, ng = int(twin["n"]), int(twin["ng"])
    b = args.block
    ctx = build_six_face_duo_context(n, 3, use_ext_bundle=True,
                                     oracle_conventions=True, omega=0.0)

    def base_states():
        sts = []
        for t in range(1, 7):
            st = {k: np.array(twin[f"b{b}_{k}_t{t}"], copy=True)
                  for k in ("u", "v", "delp", "pt")}
            st["w"] = np.zeros_like(st["delp"])
            sts.append(st)
        return sts

    out = {"n": np.array(n), "ng": np.array(ng),
           "block": np.array(b), "eps": np.array(EPS)}
    lo = 1 - ng          # numpy index of Fortran cell 1

    for arm, s2 in (("base", 0.0), ("sym", +EPS), ("anti", -EPS)):
        states = base_states()
        if arm != "base":
            d3 = states[2]["delp"]           # tile 3 (0-based idx 2)
            d3[1 - lo, 1 - lo] += EPS        # Fortran (1,1)
            d3[1 - lo, n - lo] += s2         # Fortran (1,n) My-pair
        got = {}
        ctx["step_dump"] = (
            lambda st, t, nm, a: got.setdefault(
                f"{st}_{nm}_t{t + 1}", np.array(a, copy=True)))
        states = full_acoustic_step_sixface(
            ctx, states, 1200.0 / 7.0, d_ext=0.0,
            sw_cfg=dict(SW_CFG_CASE8), entry_ascalar=True)
        ctx.pop("step_dump", None)
        # complete the dt_atmos block so stage 300 matches the
        # oracle's block-73 dump (7 inner steps, entry only on 1)
        for _ in range(6):
            states = full_acoustic_step_sixface(
                ctx, states, 1200.0 / 7.0, d_ext=0.0,
                sw_cfg=dict(SW_CFG_CASE8), entry_ascalar=False)
        for k, a in got.items():
            out[f"{arm}_{k}"] = a
        for t in range(6):                   # final state = stage 300
            for k in ("u", "v", "delp", "pt"):
                out[f"{arm}_300_{k}_t{t + 1}"] = np.array(
                    states[t][k], copy=True)
        print(f"arm {arm} done", flush=True)

    np.savez_compressed(args.out, **out)
    print(f"saved {args.out} ({len(out)} arrays)")


if __name__ == "__main__":
    main()
