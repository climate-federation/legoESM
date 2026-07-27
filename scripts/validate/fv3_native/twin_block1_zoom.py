#!/usr/bin/env python
"""Per-inner-step zoom of dt_atmos block 1 (twin pseudo-blocks 101-107).

The block-level twin showed IC (b0) agreement at 2.6e-13 but 3 m/s wind
/ 5.6e3 Pa delp divergence after ONE dt_atmos block at the burst core —
this dumps OUR state after EVERY inner acoustic step of block 1 so
compare_state_twin (--expect-blocks 0,101-107) can localize the first
divergent inner step against the oracle's per-it dumps.

Usage: twin_block1_zoom.py --n 48 --out zoom.npz
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--dt-atmos", type=float, default=1200.0)
    ap.add_argument("--n-split", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from pathlib import Path
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parents[3]))
    from tests.test_cases.colliding_modons import _MODON_H0, _modon_winds_geo

    from legoesm.core.fv3_native_duo_stepper import (
        SW_CFG_CASE8,
        build_six_face_duo_context,
        full_acoustic_step_sixface,
    )
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_RADIUS_M,
        analytic_swcore_state,
    )

    ctx = build_six_face_duo_context(args.n, 3, use_ext_bundle=True,
                                     oracle_conventions=True, omega=0.0)
    fv3_grav = 9.80665   # const-ok: oracle pins upstream's gravity

    def wind_fn(ll):
        u_e, v_n = _modon_winds_geo(ll[..., 0], ll[..., 1], FV3_RADIUS_M)
        return np.asarray(u_e), np.asarray(v_n)

    def scalars_fn(ll):
        delp = np.full(ll.shape[:-1], fv3_grav * _MODON_H0)
        return delp, np.ones_like(delp)

    states = []
    for gs in ctx["gs6"]:
        st = dict(analytic_swcore_state(gs, wind_fn=wind_fn,
                                        scalars_fn=scalars_fn))
        st["w"] = np.zeros_like(st["delp"])
        states.append(st)

    out = {}

    def dump(b, states):
        for t in range(6):
            for k in ("u", "v", "delp", "pt"):
                out[f"b{b}_{k}_t{t + 1}"] = np.array(states[t][k],
                                                    copy=True)

    dump(0, states)
    dt = args.dt_atmos / args.n_split
    for it in range(args.n_split):
        states = full_acoustic_step_sixface(ctx, states, dt, d_ext=0.0,
                                            sw_cfg=dict(SW_CFG_CASE8),
                                            entry_ascalar=(it == 0))
        dump(101 + it, states)
        print(f"inner step {it + 1}/{args.n_split} done", flush=True)

    root = here.parents[3]
    sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    np.savez_compressed(
        args.out, **out, n=np.array(args.n), ng=np.array(3),
        dt_atmos=np.array(args.dt_atmos),
        n_split=np.array(args.n_split), git_sha=np.array(sha),
        protocol=np.array(
            "block-1 per-inner-step zoom: b0=IC, b101..b10N = state "
            "after inner step 1..N of the first dt_atmos block "
            "(entry A-scalar on step 1 only)"))
    print(f"saved {args.out} ({len(out)} arrays)", flush=True)


if __name__ == "__main__":
    main()
