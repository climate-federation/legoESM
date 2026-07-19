#!/usr/bin/env python
"""Colliding modons (FV3 case 8, Lin et al. 2017) on the six-face duo
stepper — the bounded-conventions lane's answer to the production cube's
soliton dispersal (2026-07-18 battery: ico5 preserves the dipoles at
day 100, production A-L C36 disperses them into wave debris).

IC: the faithful case-8 port from tests/test_cases/colliding_modons
(two zonal Gaussian bursts, Umax=50, r0=750 km, h0=5000 m, NON-rotating
planet omega=0), projected to D-grid covariant winds with the certified
analytic_swcore_state edge-midpoint recipe (wind_fn/scalars_fn hooks).

Output: u/v geographic frames on the 1-degree grid via the c2l_ord2
lens (run_duo_stepper_w2 protocol) every --frame-days, plus per-frame
soliton metrics: max|u_east| per burst hemisphere (soliton amplitude)
and global max wind.  Preservation criterion (vs the ico5 battery
reference): coherent dipoles with amplitude within ~2x of ico at
matched day, not wave debris.

Usage: run_duo_stepper_modon.py --n 24 --dt 400 --days 60 --out m.npz
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--dt", type=float, default=400.0)
    ap.add_argument("--days", type=float, default=60.0)
    ap.add_argument("--frame-days", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ext-bundle", action="store_true", default=True)
    ap.add_argument("--oracle-conventions", action="store_true",
                    default=True,
                    help="bounded-conventions lane (default ON — this "
                         "runner exists to test that lane)")
    ap.add_argument("--plain-conventions", action="store_true",
                    help="A/B arm: plain-conventions lane")
    args = ap.parse_args()

    from pathlib import Path
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parent))       # sibling runner import
    sys.path.insert(0, str(here.parents[3]))   # repo root: tests.test_cases
    from run_duo_stepper_w2 import build_nearest_map
    from tests.test_cases.colliding_modons import (
        _MODON_H0,
        _MODON_SIZE,
        _MODON_UMAX,
        _modon_winds_geo,
    )

    from legoesm import constants
    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context,
        full_acoustic_step_sixface,
    )
    from legoesm.grids.fv3_native_ext_vector import (
        c2l_ord2_face,
        center_a_matrix,
    )
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_RADIUS_M,
        analytic_swcore_state,
    )

    oc = not args.plain_conventions
    ctx = build_six_face_duo_context(args.n, 3,
                                     use_ext_bundle=True,
                                     oracle_conventions=oc,
                                     omega=0.0)

    def wind_fn(ll):
        u_e, v_n = _modon_winds_geo(ll[..., 0], ll[..., 1], FV3_RADIUS_M)
        return np.asarray(u_e), np.asarray(v_n)

    def scalars_fn(ll):
        delp = np.full(ll.shape[:-1], constants.g * _MODON_H0)
        return delp, np.ones_like(delp)

    states = []
    for gs in ctx["gs6"]:
        st = analytic_swcore_state(gs, wind_fn=wind_fn,
                                   scalars_fn=scalars_fn)
        st = dict(st)
        st["w"] = np.zeros_like(st["delp"])
        states.append(st)

    nmap = build_nearest_map(ctx)
    n, ng = ctx["n"], ctx["ng"]
    sl = slice(ng, ng + n)

    def sample_uv(states):
        """(u_east, v_north) 1-degree frames via the c2l lens."""
        u6, v6 = [], []
        for t in range(6):
            gs = ctx["gs6"][t]
            amat = center_a_matrix(gs)
            ua, va = c2l_ord2_face(np.asarray(states[t]["u"]),
                                   np.asarray(states[t]["v"]),
                                   gs["dx"], gs["dy"], amat, n, ng)
            u6.append(ua[sl, sl])
            v6.append(va[sl, sl])
        au = np.concatenate([u.ravel() for u in u6])
        av = np.concatenate([v.ravel() for v in v6])
        return au[nmap], av[nmap]

    steps_per_frame = int(round(args.frame_days * 86400.0 / args.dt))
    n_frames = int(round(args.days / args.frame_days))

    times = [0.0]
    u0f, v0f = sample_uv(states)
    uf, vf = [u0f], [v0f]
    print(f"day 0: max|u| {np.nanmax(np.abs(u0f)):.3f} "
          f"(IC Umax {_MODON_UMAX}, r0 {_MODON_SIZE/1e3:.0f} km)",
          flush=True)
    for fr in range(1, n_frames + 1):
        for _ in range(steps_per_frame):
            states = full_acoustic_step_sixface(ctx, states, args.dt)
        day = fr * args.frame_days
        u_ll, v_ll = sample_uv(states)
        times.append(day)
        uf.append(u_ll)
        vf.append(v_ll)
        # soliton metric: peak wind in each burst's half (collision
        # swaps partners but peaks stay diagnostic of coherence)
        umax = float(np.nanmax(np.abs(u_ll)))
        wmax = float(np.nanmax(np.hypot(u_ll, v_ll)))
        print(f"day {day:g}: max|u| {umax:.3f} max|V| {wmax:.3f}",
              flush=True)
        if not np.isfinite(umax):
            print("NaN — aborting", flush=True)
            sys.exit(2)
    np.savez_compressed(
        args.out, times_days=np.array(times),
        u=np.stack(uf), v=np.stack(vf),
        lat=np.linspace(-90, 90, 181), lon=np.arange(360, dtype=float),
        oracle_conventions=np.array(oc),
        dt=args.dt, n=args.n,
        protocol="six-face duo stepper, FV3 case-8 IC "
                 "(tests.test_cases.colliding_modons formulas, certified "
                 "edge-midpoint D projection), omega=0; c2l_ord2 lens + "
                 "nearest-cell 1deg sampling")
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
