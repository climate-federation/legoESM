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

EXTERNAL-MODE FILTER (``--d-ext``): the Zenodo duo decks run the
external-mode del-2 divergence filter OFF.  The RESOLVED namelist echo
of every duo deck reads ``D_EXT = 0.000000000000000E+000``
(``C48.sw.case2.alpha0.duo.hord8/rundir/logfile.000000.out:406``, and
likewise case6, case8 and nh.case-13) -- the deck's ``input.nml`` does
not mention ``d_ext`` at all, so this is the resolved value, not the
declared one, and ``fv_arrays.F90`` carries two conflicting
declarations (``:392`` 0.0 and ``:399`` 0.02) which is exactly why the
echo has to be the authority.

This runner used to pass no value and inherit the stepper's 0.02, i.e.
it applied a divergence filter the oracle did not have -- and that
filter acts on precisely the divergent panel-seam mode the W2 imprint
metric measures, so it flattered the score.  Its two sibling runners
were already correct (``run_duo_stepper_w5.py:175`` passes
``d_ext=0.0``; ``run_duo_stepper_modon.py:192`` sets 0.0 under
``--preset case8``); W2 was the straggler.  The default is now the
deck value 0.0.  Expect the measured imprint to get WORSE: removing a
non-oracle filter is a fidelity gain even when the number moves the
wrong way, and any earlier W2 duo figure was taken under the filter and
is not comparable to one taken without it.

Usage: run_duo_stepper_w2.py --n 24 --dt 450 --days 5 --out w2_c24.npz
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np


def _git_sha() -> str:
    """Repo SHA, recorded in the npz so a stored score carries its
    provenance (an artifact without its commit is not comparable to
    anything)."""
    try:
        return subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)),
             "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


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


def build_nearest_map(ctx, lon_deg=None):
    """Nearest-cube-cell index map for a 181x360 lat-lon canvas.

    ``lon_deg`` (default ``arange(360)`` — the historical W2/modon
    canvas) selects the canvas longitudes: the Zenodo ``atmos_daily.nc``
    files store T-CELL CENTRES at 0.5..359.5, so a field-to-field score
    against them must pass ``0.5 + arange(360)`` (run_duo_stepper_case6
    does).  Latitudes are always ``linspace(-90, 90, 181)``.
    """
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
    if lon_deg is None:
        lon_deg = np.arange(360, dtype=float)
    lons = np.deg2rad(np.asarray(lon_deg, dtype=float))
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
    ap.add_argument("--d-ext", type=float, default=0.0,
                    help="external-mode del-2 divergence filter "
                         "coefficient. Default 0.0 = the value every "
                         "Zenodo duo deck RESOLVES to (D_EXT = 0.0 in "
                         "logfile.000000.out). Pass 0.02 to reproduce "
                         "pre-2026-08-06 runs of this script, which "
                         "inherited the stepper default and so applied a "
                         "filter the oracle does not have.")
    ap.add_argument("--k2e-nord", type=int, default=2, choices=(2, 4),
                    help="along-ring k2e order: 2 = authoritative live "
                         "default (2026-07-27 root cause), 4 = "
                         "historical mirror-monolith order")
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
                                     oracle_conventions,
                                     k2e_nord=args.k2e_nord)
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
            states = full_acoustic_step_sixface(ctx, states, args.dt,
                                                d_ext=args.d_ext)
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
        k2e_nord=np.array(args.k2e_nord),
        v=np.stack(frames), lat=np.linspace(-90, 90, 181),
        lon=np.arange(360, dtype=float),
        ext_bundle=np.array(bool(args.ext_bundle)),
        vector_corner=np.array(args.vector_corner),
        ext_exclude=np.array(args.ext_exclude),
        ext_metrics=np.array(bool(args.ext_metrics)),
        oracle_conventions=np.array(bool(args.oracle_conventions)),
        d_ext=np.array(float(args.d_ext)),
        dt=np.array(float(args.dt)),
        n=np.array(int(args.n)),
        git_sha=np.array(_git_sha()),
        protocol=f"duo stepper ({mode}); certified c2l_ord2 D->geographic "
        "(upstream operator family; runs' own output used c2l_ord=4, "
        "ord2 residual O(dx^2)); NEAREST-cell 1deg sampling (pattern-"
        "level protocol, envelope-comparable to the fregrid reference); "
        f"d_ext={args.d_ext} (Zenodo duo decks resolve D_EXT=0.0)")
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
