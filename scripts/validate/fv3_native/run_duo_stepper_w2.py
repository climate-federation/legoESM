#!/usr/bin/env python
"""Run the six-face duo stepper on balanced Williamson-2 and export the
daily geographic v-wind on the 1-degree lat-lon grid for the duo-target
gate (scripts/validate/fv3_native/w2_duo_oracle_gate.py contract:
times_days strictly increasing ending at --days, v (nt, 181, 360)).

Remap protocol (documented approximation): covariant A-winds from the
certified d2a2c chain -> geographic (east,north) via the exact local
tangent-basis inversion at cell centres -> NEAREST-cell sampling onto
the 1-degree grid (C-cell sizes >= 1.6deg at C24/C48, so nearest is a
pattern-level protocol; the Zenodo reference uses fregrid — scores are
comparable at the envelope level only, stated on output).

Usage: run_duo_stepper_w2.py --n 24 --dt 450 --days 5 --out w2_c24.npz
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def geographic_va(ctx, states):
    """Per-face geographic northward wind at cell centres."""
    from legoesm.core.fv3_native_duo_stepper import csw_step_sixface

    n, ng = ctx["n"], ctx["ng"]
    sl = slice(ng, ng + n)
    outs = csw_step_sixface(ctx, states, dt2=1.0)  # ua/va via certified d2a2c
    v_geo6 = []
    for t in range(6):
        gs = ctx["gs6"][t]
        lon = gs["agrid_lon"][sl, sl]
        lat = gs["agrid_lat"][sl, sl]
        ua = np.asarray(outs[t]["ua"])[sl, sl]
        va = np.asarray(outs[t]["va"])[sl, sl]

        def xyz(lo, la):
            return np.stack([np.cos(la) * np.cos(lo),
                             np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)

        # local i/j unit tangents at cell centres (central differences
        # over the halo-valid agrid)
        lo_f = gs["agrid_lon"]
        la_f = gs["agrid_lat"]
        p = xyz(lo_f, la_f)
        e1 = p[ng + 1:ng + n + 1, sl] - p[ng - 1:ng + n - 1, sl]
        e2 = p[sl, ng + 1:ng + n + 1] - p[sl, ng - 1:ng + n - 1]
        pc = p[sl, sl]
        for e in (e1, e2):
            e -= (e * pc).sum(-1, keepdims=True) * pc
        e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
        e2 /= np.linalg.norm(e2, axis=-1, keepdims=True)
        # covariant components: ua = V.e1, va = V.e2  ->  solve per cell
        g11 = (e1 * e1).sum(-1)
        g12 = (e1 * e2).sum(-1)
        g22 = (e2 * e2).sum(-1)
        det = g11 * g22 - g12 * g12
        c1 = (g22 * ua - g12 * va) / det      # contravariant coefficients
        c2 = (g11 * va - g12 * ua) / det
        vvec = c1[..., None] * e1 + c2[..., None] * e2
        north = np.stack([-np.sin(lat) * np.cos(lon),
                          -np.sin(lat) * np.sin(lon),
                          np.cos(lat)], axis=-1)
        v_geo6.append((vvec * north).sum(-1))
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
    args = ap.parse_args()

    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context,
        full_acoustic_step_sixface,
        w2_six_face_state,
    )

    ctx = build_six_face_duo_context(args.n, 3,
                                     use_ext_bundle=args.ext_bundle)
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
    np.savez_compressed(
        args.out, times_days=np.array(times),
        v=np.stack(frames), lat=np.linspace(-90, 90, 181),
        lon=np.arange(360, dtype=float),
        protocol="duo stepper (interim exchanges); covariant->geographic "
        "exact tangent inversion; NEAREST-cell 1deg sampling (pattern-"
        "level protocol, envelope-comparable to the fregrid reference)")
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
