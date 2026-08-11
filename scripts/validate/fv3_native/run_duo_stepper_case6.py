#!/usr/bin/env python
"""Run the six-face duo stepper on a Zenodo SW deck — Williamson case 6
(Rossby-Haurwitz wave 4, the original scope of this runner) or case 2
(solid-body, ``--case 2``, optionally rotated ``--alpha``) — and export
daily gh/u/v on the Zenodo reference's lat-lon canvas for
``case6_duo_oracle_gate.py``.

IC (case 6): ``case6_six_face_state`` — the literal port of the pinned
oracle's ``tools/test_cases.F90`` case(6) (:1213-1270), analytic fields
from the shared :mod:`legoesm.core.williamson_sw_analytic` on the
oracle's GFS constants.

IC (case 2): ``w2_six_face_state(alpha=...)`` — the case(2) block
(:991-1040) + ``init_winds`` ``defOnGrid=5`` (:477-507), with the
rotated Coriolis (:783-790) threaded through
``build_six_face_duo_context(rotation_alpha=...)``.

ALPHA UNITS (``--alpha``): the RAW ``test_case_nml`` value.  The pinned
Fortran feeds it straight into ``sin``/``cos`` (no deg->rad anywhere;
``fv_control.F90:1105``'s historical ``alpha = alpha*pi`` is commented
out), so the Zenodo ``alpha45`` decks rotate by 45 RADIANS
(== 45 - 14*pi ~ 1.0177 rad ~ 58.31 deg geographically).  VERIFIED
against the reference IC: analytic case-2 gh at alpha=45 rad matches
``C48.sw.case2.alpha45.duo.hord8`` ``ps_ic*Grav`` at rel-L2 1.12e-04
(C48 discretisation + fregrid floor), while the pi/4 interpretation is
off by 1.11e-01.  Pass ``--alpha 45`` to reproduce the alpha45 deck.

CASE-8 NOTE (why there is no ``--case 8`` despite
``C48.sw.case8.alpha45.duo.hord8`` existing): the deck diff against its
alpha0 sibling shows the case-8 "alpha45" run changes ``target_lat``
-90 -> -135 (a 45-degree SCHMIDT GRID ROTATION) plus a ``alpha = 0.75``
namelist echo that case(8) never reads (its IC has no alpha term and it
overrides ``f0 = 0; fC = 0``, test_cases.F90:1375-1381).  Reproducing
it requires a rotated-target cube gridstruct, which this port does not
have — scoring an unrotated-grid run against that reference would be a
grid confound, not a rotation port, so it is refused rather than
approximated.

DECK CONFIGURATION (defaults) — every value below is the RESOLVED
namelist echo of the Zenodo reference deck
``C48.sw.case6.alpha0.duo.hord8/rundir/logfile.000000.out``, not the
declared ``input.nml`` (the w2 d_ext lesson: the echo is the only
authority):

* damping block (:401-417): ``DDDMP=0  D2_BG=0  D4_BG=0  KE_BG=0
  D_EXT=0  NORD=2``, ``do_vort_damp=F`` / ``vtdm4=0``, hords all 8 —
  i.e. ``SW_CFG_CASE8`` with ``d4_bg = 0.0`` (case 8 runs del-6 bg
  0.12; case 6 runs NO explicit divergence damping at all — hord-8
  limiting is the only dissipation).
* cadence: ``DT_ATMOS=1200`` (:184), ``N_SPLIT=7`` (:358), K_SPLIT=1.
* grid: C48 duo, ``DO_SCHMIDT=T`` with ``STRETCH_FAC=1`` and target
  (lon 0, lat -90) — read against ``fv_grid_utils.F90:859-917``
  (``direct_transform``): with c=1 and sin(lat_p)=-1 this is a PURE
  lon -> lon + pi relabelling of the source cube (plus the skipped
  ``shift_fac`` -10 deg of the non-Schmidt branch,
  ``fv_grid_tools.F90:662``).  It moves where the native cells sit and
  nothing else; the analytic IC and the daily output are GEOGRAPHIC
  fields, so a canvas comparison is unaffected and this runner keeps
  the port's standard cube orientation.

CANVAS: the reference ``atmos_daily.nc`` stores T-cell centres at
lon 0.5..359.5, lat -90..90 (181x360, poles included) — NOT the
historical W2/modon 0..359 canvas.  Frames here are sampled on the
reference's own longitudes (``build_nearest_map(ctx, lon_deg=...)``)
so the gate compares like against like.

REMAP PROTOCOL (documented approximation, same family as the W2/modon
runners): D winds -> geographic via the certified c2l_ord2 lens (the
reference's own ucomp/vcomp used the ord4 sibling + fregrid; ord2
residual is O(dx^2)); gh = delp sampled NEAREST-cell (the reference is
fregrid).  Scores against the reference are therefore envelope/
pattern-level, floored by the remap-protocol difference — stated on
output and in the gate.

Usage: run_duo_stepper_case6.py --n 48 --days 5 --out case6_c48.npz
       (--days 0 exports the IC frame only — the pure IC-port score.)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

# Resolved Zenodo case-6 deck damping block (logfile.000000.out:401-417):
# SW_CFG_CASE8 minus its del-6 background (D4_BG 0.12 -> 0.0).  Built
# from the case-8 block at import time in main() to stay a single
# source; this literal names only the one differing key.
CASE6_D4_BG = 0.0
CASE6_DT_ATMOS_S = 1200.0     # logfile.000000.out:184  DT_ATMOS
CASE6_N_SPLIT = 7             # logfile.000000.out:358  N_SPLIT

# Per-case resolved deck defaults, each key straight from the named
# case's C48 duo hord8 logfile.000000.out echo (never input.nml):
#   case 6: DT_ATMOS=1200 (:184), N_SPLIT=7 (:358), D4_BG=0 (:403)
#   case 2: DT_ATMOS=3600 (:184), N_SPLIT=7 (:358), D4_BG=0.12 (:403)
#           — i.e. the FULL SW_CFG_CASE8 damping block, unmodified.
# gh_band: per-case sentinel plausibility band (see GH_PLAUSIBLE note
# below).  Case-2 steady state spans gh in [gh0 - coef, gh0]
# ~ [1.07e4, 2.94e4] m^2/s^2, so its lower sentinel bound must sit
# below 1e4 (the case-6 band would flag the healthy state).
CASE_DECKS = {
    6: {"dt_atmos": CASE6_DT_ATMOS_S, "n_split": CASE6_N_SPLIT,
        "d4_bg": CASE6_D4_BG, "gh_band": (1.0e4, 5.0e5)},
    2: {"dt_atmos": 3600.0, "n_split": 7,
        "d4_bg": 0.12, "gh_band": (5.0e3, 5.0e5)},
}


# Diagnostic environment modes consumed (at import time) on this
# runner's use_ext_bundle=True path — each changes physics without any
# CLI flag, so each MUST enter the npz manifest or a diagnostic variant
# could pass enforcement at the same git_sha (codex a45 r2 #1):
#   LEGOESM_DUO_PG_BVERTEX      pressure-gradient B-vertex update
#   LEGOESM_DUO_ENTRY_ASCALAR   scalar-exchange cadence
#   LEGOESM_DUO_AVG_B_ENDPOINTS B-grid edge averaging
#   LEGOESM_DUO_CORNER_MODE     ext-vector corner remap
_DIAG_ENV_KNOBS = ("LEGOESM_DUO_AVG_B_ENDPOINTS",
                   "LEGOESM_DUO_CORNER_MODE",
                   "LEGOESM_DUO_ENTRY_ASCALAR",
                   "LEGOESM_DUO_PG_BVERTEX")


def diag_env_record() -> str:
    """Comma list of the SET diagnostic env knobs ('' = deck-faithful).
    Any set-but-unrecognised value still records (and so still fails
    the deck check) — conservative by construction."""
    return ",".join(f"{k}={os.environ[k]}" for k in _DIAG_ENV_KNOBS
                    if os.environ.get(k))


def reference_canvas() -> tuple[np.ndarray, np.ndarray]:
    """(lat_deg, lon_deg) of the Zenodo atmos_daily.nc T-cell canvas."""
    return (np.linspace(-90.0, 90.0, 181),
            0.5 + np.arange(360, dtype=float))


def _git_sha() -> str:
    """Repo SHA recorded in the npz (artifact without commit is not
    comparable to anything)."""
    try:
        return subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)),
             "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# Plausibility bands, mirrored by the gate (sentinel detectors, not
# physics gates): RH4 lives in gh ~ [7.8e4, 1.04e5] m^2/s^2, |V| <~ 100
# m/s; the known failure mode (BIG_NUMBER = 1e8 corner sentinel leaking
# through a sample window) sits orders of magnitude outside.
GH_PLAUSIBLE = (1.0e4, 5.0e5)
WIND_PLAUSIBLE_MAX = 500.0


def frames_plausible(gh, u, v, gh_band=GH_PLAUSIBLE) -> str | None:
    """None if within the sentinel bands, else a description."""
    if not all(np.isfinite(f).all() for f in (gh, u, v)):
        return "non-finite values"
    if gh.min() < gh_band[0] or gh.max() > gh_band[1]:
        return (f"gh [{gh.min():g}, {gh.max():g}] outside "
                f"{tuple(gh_band)} — sentinel leak?")
    m = max(float(np.abs(u).max()), float(np.abs(v).max()))
    if m > WIND_PLAUSIBLE_MAX:
        return f"|wind| max {m:g} > {WIND_PLAUSIBLE_MAX:g} m/s"
    return None


def sample_fields(ctx, states, nmap):
    """(gh, u_east, v_north) 181x360 frames.

    gh: compute-window delp (== g*h on the SW convention) nearest-cell
    sampled.  Winds: the c2l_ord2 lens (run_duo_stepper_w2 protocol).
    """
    from legoesm.grids.fv3_native_ext_vector import (
        c2l_ord2_face,
        center_a_matrix,
    )

    n, ng = ctx["n"], ctx["ng"]
    sl = slice(ng, ng + n)
    gh6, u6, v6 = [], [], []
    for t in range(6):
        gs = ctx["gs6"][t]
        amat = center_a_matrix(gs)
        ua, va = c2l_ord2_face(np.asarray(states[t]["u"]),
                               np.asarray(states[t]["v"]),
                               gs["dx"], gs["dy"], amat, n, ng)
        gh6.append(np.asarray(states[t]["delp"])[sl, sl])
        u6.append(ua[sl, sl])
        v6.append(va[sl, sl])
    gh = np.concatenate([f.ravel() for f in gh6])[nmap]
    u = np.concatenate([f.ravel() for f in u6])[nmap]
    v = np.concatenate([f.ravel() for f in v6])[nmap]
    return gh, u, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=6, choices=sorted(CASE_DECKS),
                    help="Zenodo SW deck: 6 = Rossby-Haurwitz wave 4 "
                         "(default), 2 = solid-body (supports --alpha)")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="test_case_nml alpha, RAW namelist units == "
                         "RADIANS (the alpha45 decks mean 45 radians; see "
                         "module docstring).  Case 2 only.")
    ap.add_argument("--n", type=int, default=48,
                    help="cube resolution (reference decks are C48)")
    ap.add_argument("--dt-atmos", type=float, default=None,
                    help="outer block [s]; default = the deck echo "
                         "(case 6: 1200, case 2: 3600)")
    ap.add_argument("--n-split", type=int, default=None,
                    help="inner acoustic steps per block; deck resolves 7")
    ap.add_argument("--days", type=float, default=5.0,
                    help="whole days to run; 0 = IC frame only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--d-ext", type=float, default=0.0,
                    help="external-mode filter; deck resolves D_EXT=0.0")
    ap.add_argument("--d4-bg", type=float, default=None,
                    help="del-6 divergence-damping bg; default = the deck "
                         "echo (case 6: 0.0, case 2: 0.12); any other "
                         "value is diagnostic, NON-FAITHFUL to the deck")
    ap.add_argument("--k2e-nord", type=int, default=2, choices=(2, 4),
                    help="along-ring k2e order (2 = authoritative live "
                         "default)")
    ap.add_argument("--ext-exclude", default="",
                    help="comma list of ext families to swap to interim "
                         "exchanges (attribution probes)")
    ap.add_argument("--plain-conventions", action="store_true",
                    help="A/B arm: plain-conventions lane (default = the "
                         "bounded/oracle lane the Zenodo duo runs execute)")
    args = ap.parse_args()
    deck = CASE_DECKS[args.case]
    if args.dt_atmos is None:
        args.dt_atmos = deck["dt_atmos"]
    if args.n_split is None:
        args.n_split = deck["n_split"]
    if args.d4_bg is None:
        args.d4_bg = deck["d4_bg"]
    if args.case != 2 and args.alpha != 0.0:
        ap.error("--alpha is only meaningful for --case 2 (case 6 has "
                 "no rotated reference and its IC port has no alpha "
                 "hook)")
    if not np.isfinite(args.alpha):
        ap.error(f"--alpha must be finite, got {args.alpha}")
    if args.n_split < 1:
        ap.error(f"--n-split must be >= 1, got {args.n_split}")
    blocks_per_day_f = 86400.0 / args.dt_atmos
    blocks_per_day = int(round(blocks_per_day_f))
    if abs(blocks_per_day - blocks_per_day_f) > 1e-9:
        ap.error("86400 must be an integer multiple of --dt-atmos "
                 f"(got {blocks_per_day_f} blocks/day)")
    total_days = int(round(args.days))
    if abs(total_days - args.days) > 1e-9 or total_days < 0:
        ap.error(f"--days must be a whole number >= 0, got {args.days}")

    from pathlib import Path
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parent))       # sibling runner import
    from run_duo_stepper_w2 import build_nearest_map

    from legoesm.core.fv3_native_duo_stepper import (
        SW_CFG_CASE8,
        advance_duo_outer_step,
        build_six_face_duo_context,
        case6_six_face_state,
        w2_six_face_state,
    )

    sw_cfg = {**SW_CFG_CASE8, "d4_bg": args.d4_bg}
    oc = not args.plain_conventions
    excl = tuple(x for x in args.ext_exclude.split(",") if x)
    ctx = build_six_face_duo_context(args.n, 3,
                                     use_ext_bundle=True,
                                     oracle_conventions=oc,
                                     ext_exclude=excl,
                                     k2e_nord=args.k2e_nord,
                                     rotation_alpha=args.alpha)
    if args.case == 6:
        states = case6_six_face_state(ctx)
    else:                                     # case 2 (choices-gated)
        states = w2_six_face_state(ctx, alpha=args.alpha)
    lat_deg, lon_deg = reference_canvas()
    nmap = build_nearest_map(ctx, lon_deg=lon_deg)

    gh_band = deck["gh_band"]
    times = [0.0]
    gh0f, u0f, v0f = sample_fields(ctx, states, nmap)
    ghf, uf, vf = [gh0f], [u0f], [v0f]
    bad = frames_plausible(gh0f, u0f, v0f, gh_band)
    if bad:
        print(f"day 0: {bad} — aborting", flush=True)
        sys.exit(2)
    print(f"day 0: gh [{gh0f.min():.1f}, {gh0f.max():.1f}] "
          f"max|u| {np.abs(u0f).max():.3f} max|v| {np.abs(v0f).max():.3f}",
          flush=True)

    ic_desc = ("case6_six_face_state IC (test_cases.F90:1213-1270 via "
               "williamson_sw_analytic, GFS constants)" if args.case == 6
               else "w2_six_face_state IC (test_cases.F90:991-1040 + "
                    "init_winds defOnGrid=5, rotated Coriolis via "
                    f"rotation_alpha={args.alpha} RADIANS — raw namelist "
                    "units)")

    def _save():
        np.savez_compressed(
            args.out, times_days=np.array(times),
            gh=np.stack(ghf), u=np.stack(uf), v=np.stack(vf),
            lat=lat_deg, lon=lon_deg,
            case=np.array(int(args.case)),
            alpha=np.array(float(args.alpha)),
            n=np.array(int(args.n)),
            dt_atmos=np.array(float(args.dt_atmos)),
            n_split=np.array(int(args.n_split)),
            requested_days=np.array(int(total_days)),
            d_ext=np.array(float(args.d_ext)),
            d4_bg=np.array(float(args.d4_bg)),
            k2e_nord=np.array(int(args.k2e_nord)),
            ext_exclude=np.array(args.ext_exclude),
            oracle_conventions=np.array(bool(oc)),
            diag_env=np.array(diag_env_record()),
            git_sha=np.array(_git_sha()),
            protocol=(
                f"six-face duo stepper, {ic_desc}; resolved Zenodo "
                f"case-{args.case} deck config (SW_CFG_CASE8 with d4_bg="
                f"{args.d4_bg}, d_ext={args.d_ext}, dt_atmos="
                f"{args.dt_atmos}, n_split={args.n_split}); c2l_ord2 lens "
                "(reference used c2l_ord=4 + fregrid; ord2 residual "
                "O(dx^2)) + NEAREST-cell sampling on the reference T-cell "
                "canvas (lon 0.5..359.5) — envelope/pattern-level protocol"))
        print("saved", args.out, flush=True)

    _save()
    for day in range(1, total_days + 1):
        for _ in range(blocks_per_day):
            states = advance_duo_outer_step(ctx, states, args.dt_atmos,
                                            args.n_split, d_ext=args.d_ext,
                                            sw_cfg=sw_cfg)
        gh_d, u_d, v_d = sample_fields(ctx, states, nmap)
        bad = frames_plausible(gh_d, u_d, v_d, gh_band)
        if bad:
            print(f"day {day}: {bad} — aborting (partial npz kept)",
                  flush=True)
            sys.exit(2)
        times.append(float(day))
        ghf.append(gh_d)
        uf.append(u_d)
        vf.append(v_d)
        print(f"day {day}: gh [{gh_d.min():.1f}, {gh_d.max():.1f}] "
              f"max|u| {np.abs(u_d).max():.3f} "
              f"max|v| {np.abs(v_d).max():.3f}", flush=True)
        _save()      # incremental: a timeout still leaves day-k frames


if __name__ == "__main__":
    main()
