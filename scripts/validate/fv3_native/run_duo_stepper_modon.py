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
    ap.add_argument("--nord", type=int, default=None,
                    help="override the preset's divergence-damping "
                         "order (codex lattice-r1 rank-2 screen)")
    ap.add_argument("--d-ext", type=float, default=None,
                    help="override the preset's external-mode filter "
                         "coefficient (seam-ringing discriminator)")
    ap.add_argument("--plain-conventions", action="store_true",
                    help="A/B arm: plain-conventions lane (default is "
                         "the bounded lane + ext bundle — the lane "
                         "this runner exists to test)")
    ap.add_argument("--preset", default="case8",
                    choices=("case8", "w2tuned"),
                    help="stage configuration: 'case8' = the Zenodo "
                         "C48.sw.case8 damping block (vort damping OFF, "
                         "dddmp=0, d_ext=0, hords=8, d4_bg=0.12; the "
                         "ported nord=1/del-4 stands in for their "
                         "nord=2/del-6 — disclosed) + FV3 gravity "
                         "9.80665 in delp; 'w2tuned' = the historical "
                         "W2-tuned defaults (damp_v=0.2, dddmp=0.2, "
                         "d_ext=0.02, hord=6)")
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
        SW_CFG_CASE8,
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

    # FV3 constants_mod GRAV (case-8 delp = 5000*Grav); legoESM
    # constants.g differs by 5e-5 rel (codex modon-r1 P2)
    # const-ok: oracle pins upstream's gravity, not legoESM's
    fv3_grav = 9.80665 if args.preset == "case8" else constants.g
    if args.preset == "case8":
        sw_cfg = dict(SW_CFG_CASE8)
        d_ext = 0.0
    else:
        sw_cfg = None
        d_ext = 0.02
    if args.d_ext is not None:
        d_ext = args.d_ext
    if args.nord is not None and sw_cfg is not None:
        sw_cfg["nord"] = args.nord

    def wind_fn(ll):
        u_e, v_n = _modon_winds_geo(ll[..., 0], ll[..., 1], FV3_RADIUS_M)
        return np.asarray(u_e), np.asarray(v_n)

    def scalars_fn(ll):
        delp = np.full(ll.shape[:-1], fv3_grav * _MODON_H0)
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
    w0 = float(np.max(np.hypot(u0f, v0f)))
    for fr in range(1, n_frames + 1):
        for _ in range(steps_per_frame):
            states = full_acoustic_step_sixface(ctx, states, args.dt,
                                                d_ext=d_ext,
                                                sw_cfg=sw_cfg)
        day = fr * args.frame_days
        u_ll, v_ll = sample_uv(states)
        times.append(day)
        uf.append(u_ll)
        vf.append(v_ll)
        # finite gate FIRST (nanmax would hide localized NaNs — codex
        # modon-r1 P1), then coherence metrics: global peak, normalized
        # amplitude A(t)/A(0), and the two hemispheric peaks (lon 0-180
        # / 180-360 halves track the two modons through the collision)
        if not (np.all(np.isfinite(u_ll)) and np.all(np.isfinite(v_ll))):
            print(f"day {day:g}: NaN — aborting", flush=True)
            sys.exit(2)
        w = np.hypot(u_ll, v_ll)
        wmax = float(np.max(w))
        p_w = float(np.max(w[:, :180]))
        p_e = float(np.max(w[:, 180:]))
        print(f"day {day:g}: max|V| {wmax:.3f} A/A0 {wmax / w0:.3f} "
              f"peakW {p_w:.3f} peakE {p_e:.3f}", flush=True)
    np.savez_compressed(
        args.out, times_days=np.array(times),
        u=np.stack(uf), v=np.stack(vf),
        lat=np.linspace(-90, 90, 181), lon=np.arange(360, dtype=float),
        oracle_conventions=np.array(oc),
        preset=np.array(args.preset),
        dt=args.dt, n=args.n,
        protocol="six-face duo stepper, FV3 case-8 IC "
                 "(tests.test_cases.colliding_modons formulas, certified "
                 "edge-midpoint D projection), omega=0; c2l_ord2 lens + "
                 "nearest-cell 1deg sampling; preset=" + args.preset
                 + (" (Zenodo case-8 damping block; ported nord=1/del-4 "
                    "stands in for their nord=2/del-6 — disclosed)"
                    if args.preset == "case8" else " (W2-tuned)"))
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
