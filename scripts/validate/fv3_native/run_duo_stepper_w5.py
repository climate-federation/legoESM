#!/usr/bin/env python
"""Williamson-5 (FV3 test_case 5, zonal flow over an isolated mountain)
on the six-face duo stepper — the topography follow-up of the nord-2
root-cause fix (PR #1372).

IC = the VERBATIM Zenodo test_cases.F90 case(5) recipe (:1173-1210):
  Ubar = 20, gh0 = 5960*Grav, mountain at p1 = (pi/2, pi/6) with the
  FLAT-COORDINATE radius r = sqrt(min(r0^2, dlon^2 + dlat^2)), r0 =
  pi/9, phis = 2000*Grav*(1 - r/r0);  delp = gh0 - (a*omega*Ubar +
  Ubar^2/2)*S^2 - phis with the case-2 solid-body winds (alpha = 0).
No analytic solution exists; no case-5 run ships in the Zenodo
archive, so the twin reference is GENERATED with the certified
fv3_solo.exe (test_case=5) and compared through the same lens.

Output: u/v geographic frames (c2l lens + 1-degree nearest sampling,
run_duo_stepper_w2 protocol) + per-frame max|V|; block-state dumps on
the twin schedule for the symmetry/quotient instruments (the mountain
at 90E preserves the Mx meridian reflection — the gauge-free artifact
probe still applies).
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np


def _git_sha() -> str:
    try:
        from pathlib import Path
        root = Path(__file__).resolve().parents[3]
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status",
                                "--porcelain"], capture_output=True,
                               text=True, check=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


# verbatim test_cases.F90 case(5) parameters
# const-ok: oracle pins upstream's gravity/depths, not legoESM's
_W5_UBAR = 20.0
_W5_GH0_M = 5960.0          # * Grav
_W5_HS0_M = 2000.0          # * Grav
_W5_R0 = np.pi / 9.0
_W5_LON_C = np.pi / 2.0
_W5_LAT_C = np.pi / 6.0
_FV3_GRAV = 9.80665


def w5_phis(lon, lat):
    """phis = 2000*Grav*(1 - r/r0), FLAT-coordinate radius (verbatim
    test_cases.F90:1181-1189 — NOT great-circle)."""
    r2 = np.minimum(_W5_R0 ** 2, (lon - _W5_LON_C) ** 2
                    + (lat - _W5_LAT_C) ** 2)
    return _W5_HS0_M * _FV3_GRAV * (1.0 - np.sqrt(r2) / _W5_R0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--dt-atmos", type=float, default=1200.0)
    ap.add_argument("--n-split", type=int, default=7)
    ap.add_argument("--days", type=float, default=15.0)
    ap.add_argument("--frame-days", type=float, default=1.0)
    ap.add_argument("--k2e-nord", type=int, default=2, choices=(2, 4))
    ap.add_argument("--nord", type=int, default=None, choices=(1, 2))
    ap.add_argument("--dump-state-out", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from pathlib import Path
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parent))
    sys.path.insert(0, str(here.parents[3]))
    from run_duo_stepper_w2 import build_nearest_map

    from legoesm.core.fv3_native_duo_stepper import (
        advance_duo_outer_step,
        build_six_face_duo_context,
    )
    from legoesm.grids.fv3_native_ext_vector import (
        c2l_ord2_face,
        center_a_matrix,
    )
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_OMEGA,
        FV3_RADIUS_M,
        analytic_swcore_state,
    )

    ctx = build_six_face_duo_context(args.n, 3, use_ext_bundle=True,
                                     oracle_conventions=True,
                                     k2e_nord=args.k2e_nord,
                                     topo_fn=w5_phis)

    a_r, omega = FV3_RADIUS_M, FV3_OMEGA
    coef = a_r * omega * _W5_UBAR + 0.5 * _W5_UBAR * _W5_UBAR

    def scalars_fn(ll):
        lon, lat = ll[..., 0], ll[..., 1]
        s = np.sin(lat)                       # alpha = 0
        delp = (_W5_GH0_M * _FV3_GRAV - coef * s * s
                - w5_phis(lon, lat))
        return delp, np.ones_like(delp)

    states = []
    for gs in ctx["gs6"]:
        st = dict(analytic_swcore_state(gs, u0=_W5_UBAR, alpha=0.0,
                                        scalars_fn=scalars_fn))
        st["w"] = np.zeros_like(st["delp"])
        states.append(st)

    sw_cfg = {"nord": args.nord} if args.nord is not None else None

    nmap = build_nearest_map(ctx)
    n, ng = ctx["n"], ctx["ng"]
    sl = slice(ng, ng + n)

    def sample_uv(states):
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

    state_dumps = {}
    blk = [0]

    def maybe_dump(states, b):
        if (args.dump_state_out and b <= 400
                and (b <= 14 or b % 18 == 0)):
            for t in range(6):
                for k in ("u", "v", "delp", "pt"):
                    state_dumps[f"b{b}_{k}_t{t + 1}"] = np.array(
                        states[t][k], copy=True)

    maybe_dump(states, 0)
    blocks_f = args.frame_days * 86400.0 / args.dt_atmos
    bpf = int(round(blocks_f))
    if abs(bpf - blocks_f) > 1e-9:
        ap.error("frame-days*86400 must be a multiple of dt-atmos")
    n_frames = int(round(args.days / args.frame_days))

    times, uf, vf = [0.0], [], []
    u0f, v0f = sample_uv(states)
    uf.append(u0f)
    vf.append(v0f)
    if not (np.all(np.isfinite(u0f)) and np.all(np.isfinite(v0f))):
        print("day 0: NaN in IC — aborting", flush=True)
        sys.exit(2)
    print(f"day 0: max|V| {np.max(np.hypot(u0f, v0f)):.3f}", flush=True)
    for fr in range(1, n_frames + 1):
        for _ in range(bpf):
            states = advance_duo_outer_step(ctx, states, args.dt_atmos,
                                            args.n_split, d_ext=0.0,
                                            sw_cfg=sw_cfg)
            blk[0] += 1
            maybe_dump(states, blk[0])
        day = fr * args.frame_days
        u_ll, v_ll = sample_uv(states)
        times.append(day)
        uf.append(u_ll)
        vf.append(v_ll)
        if not (np.all(np.isfinite(u_ll)) and np.all(np.isfinite(v_ll))):
            print(f"day {day:g}: NaN — aborting", flush=True)
            sys.exit(2)
        print(f"day {day:g}: max|V| {np.max(np.hypot(u_ll, v_ll)):.3f}",
              flush=True)

    np.savez_compressed(
        args.out, times_days=np.array(times), u=np.stack(uf),
        v=np.stack(vf), lat=np.linspace(-90, 90, 181),
        lon=np.arange(360, dtype=float),
        dt_atmos=np.array(args.dt_atmos), n_split=np.array(args.n_split),
        k2e_nord=np.array(args.k2e_nord), n=args.n,
        git_sha=np.array(_git_sha()),
        protocol=np.array(
            "W5 duo stepper: verbatim case-5 IC (flat-metric mountain "
            "at (pi/2, pi/6), delp = gh0 - coef*S^2 - phis), hs via "
            "ctx topo_fn -> geopk; c2l lens + 1-deg nearest sampling"))
    print("saved", args.out, flush=True)
    if args.dump_state_out:
        np.savez_compressed(
            args.dump_state_out, **state_dumps, n=np.array(args.n),
            ng=np.array(3), dt_atmos=np.array(args.dt_atmos),
            n_split=np.array(args.n_split),
            k2e_nord=np.array(args.k2e_nord),
            git_sha=np.array(_git_sha()),
            protocol=np.array("W5 block-state twin dumps"))
        print(f"saved {args.dump_state_out} "
              f"({len(state_dumps)} arrays)", flush=True)


if __name__ == "__main__":
    main()
