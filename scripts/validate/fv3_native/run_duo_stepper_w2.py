#!/usr/bin/env python
"""Run the six-face duo stepper on balanced Williamson-2 and export the
daily geographic v-wind on the 1-degree lat-lon grid for the duo-target
gate (scripts/validate/fv3_native/w2_duo_oracle_gate.py contract:
times_days strictly increasing ending at --days, v (nt, 181, 360)).

Remap protocol (documented approximation): D winds -> geographic
(east,north) via the CERTIFIED upstream c2l_ord2 a-matrix operator
(the runs' own ua/va output used the ord4 sibling; ord2 residual is
O(dx^2)) -> NEAREST-cell sampling onto the 1-degree grid (C-cell sizes
>= 1.6deg at C24/C48, so nearest is a pattern-level protocol; the
Zenodo reference uses fregrid — scores are comparable at the envelope
level only, stated on output).

Usage: run_duo_stepper_w2.py --n 24 --dt 450 --days 5 --out w2_c24.npz
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def geographic_va(ctx, states):
    """Per-face geographic northward wind at cell centres via the
    upstream c2l_ord2 construction (fv_grid_utils:2547-2628): D winds
    -> geographic (u_lon, v_lat) through center_a_matrix, which
    matches the upstream z->a build to ~1e-16 on compute cells (codex
    bounded-r3 (a); the four diagonal halo-corner cells differ —
    upstream zeroes ec there — and are sliced out below).  Same
    operator family as the Zenodo runs' own ua/va output (their
    c2l_ord=4 is the higher-order sibling; ord2's extra residual is
    O(dx^2), stated on output).  A bounded Fortran z/a dump for full
    independent certification is a follow-up.

    The previous central-difference tangent-basis inversion painted a
    +/-15 m/s vertex butterfly on the DAY-0 balanced state (bases
    straddle the corner kink) — the entire 'vertex imprint' at day 5
    was that diagnostic artifact, not model error."""
    from legoesm.grids.fv3_native_ext_vector import (
        c2l_ord2_face,
        center_a_matrix,
    )

    n, ng = ctx["n"], ctx["ng"]
    sl = slice(ng, ng + n)
    v_geo6 = []
    for t in range(6):
        gs = ctx["gs6"][t]
        amat = center_a_matrix(gs)
        _, va = c2l_ord2_face(np.asarray(states[t]["u"]),
                              np.asarray(states[t]["v"]),
                              gs["dx"], gs["dy"], amat, n, ng)
        v_geo6.append(va[sl, sl])
    return v_geo6


def build_nearest_map(ctx):
    n, ng = ctx["n"], ctx["ng"]
    sl = slice(ng, ng + n)
    cx = []
    for t in range(6):
        lon = ctx["gs6"][t]["agrid_lon"][sl, sl].ravel()
        lat = ctx["gs6"][t]["agrid_lat"][sl, sl].ravel()
        cx.append(np.stack([np.cos(lat) * np.cos(lon),
                            np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1))
    centers = np.concatenate(cx)                 # (6n^2, 3)
    lats = np.deg2rad(np.linspace(-90, 90, 181))
    lons = np.deg2rad(np.arange(360, dtype=float))
    llon, llat = np.meshgrid(lons, lats)
    pts = np.stack([np.cos(llat) * np.cos(llon),
                    np.cos(llat) * np.sin(llon), np.sin(llat)], axis=-1)
    flat = pts.reshape(-1, 3)
    idx = np.empty(flat.shape[0], dtype=np.int64)
    for i0 in range(0, flat.shape[0], 4000):
        blk = flat[i0:i0 + 4000] @ centers.T
        idx[i0:i0 + 4000] = blk.argmax(axis=1)
    return idx.reshape(181, 360)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--dt", type=float, default=450.0)
    ap.add_argument("--days", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ext-bundle", action="store_true",
                    help="faithful ext_scalar/ext_vector duo exchanges "
                         "(fv3_native_ext_vector) + ext halo metrics")
    ap.add_argument("--ext-metrics", action="store_true",
                    help="NON-FAITHFUL opt-in: extended-lattice halo "
                         "metrics (upstream duo never consumes ext "
                         "metrics in the model; measured harmful)")
    ap.add_argument("--ext-exclude", default="",
                    help="comma list of ext families to swap back to the "
                         "interim exchanges (attribution probes): "
                         "divgd,cvec,metrics")
    ap.add_argument("--vector-corner", default="lagrange",
                    choices=("lagrange", "a2d"),
                    help="vector wedge treatment (lagrange = upstream-"
                         "faithful re-extrapolation; a2d = keep the "
                         "projected geographic-corner values)")
    ap.add_argument("--oracle-conventions", action="store_true",
                    help="BOUNDED-conventions gridstruct (the lane the "
                         "Zenodo duo runs execute: extended-lattice "
                         "metrics, bounded_domain=True guards, corner "
                         "flags off — d_sw4 corner-KE fix and plain "
                         "corner specials disabled)")
    args = ap.parse_args()

    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context,
        full_acoustic_step_sixface,
        w2_six_face_state,
    )

    excl = tuple(x for x in args.ext_exclude.split(",") if x)
    ctx = build_six_face_duo_context(args.n, 3,
                                     use_ext_bundle=args.ext_bundle,
                                     vector_corner=args.vector_corner,
                                     ext_exclude=excl,
                                     use_ext_metrics=args.ext_metrics,
                                     oracle_conventions=args.
                                     oracle_conventions)
    states = w2_six_face_state(ctx)
    nmap = build_nearest_map(ctx)

    def sample_v(states):
        v6 = geographic_va(ctx, states)
        allv = np.concatenate([v.ravel() for v in v6])
        return allv[nmap]

    steps_per_day = int(round(86400.0 / args.dt))
    times = []
    frames = []
    times.append(1e-6)
    frames.append(sample_v(states))
    total_days = int(round(args.days))
    for day in range(1, total_days + 1):
        for _ in range(steps_per_day):
            states = full_acoustic_step_sixface(ctx, states, args.dt)
        times.append(float(day))
        frames.append(sample_v(states))
        vmax = float(np.nanmax(np.abs(frames[-1])))
        print(f"day {day}: v_ll absmax {vmax:.4f}", flush=True)
        if not np.isfinite(vmax):
            print("NaN — aborting", flush=True)
            sys.exit(2)
    if args.ext_bundle and args.vector_corner == "lagrange":
        mode = "faithful ext bundle (fv3_native_ext_vector)"
    elif args.ext_bundle:
        mode = ("ext bundle, NONFAITHFUL vector_corner="
                f"{args.vector_corner} variant")
    else:
        mode = "interim exchanges"
    np.savez_compressed(
        args.out, times_days=np.array(times),
        v=np.stack(frames), lat=np.linspace(-90, 90, 181),
        lon=np.arange(360, dtype=float),
        ext_bundle=np.array(bool(args.ext_bundle)),
        vector_corner=np.array(args.vector_corner),
        ext_exclude=np.array(args.ext_exclude),
        ext_metrics=np.array(bool(args.ext_metrics)),
        oracle_conventions=np.array(bool(args.oracle_conventions)),
        protocol=f"duo stepper ({mode}); certified c2l_ord2 D->geographic "
        "(upstream operator family; runs' own output used c2l_ord=4, "
        "ord2 residual O(dx^2)); NEAREST-cell 1deg sampling (pattern-"
        "level protocol, envelope-comparable to the fregrid reference)")
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
