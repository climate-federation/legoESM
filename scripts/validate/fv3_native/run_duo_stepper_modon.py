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
import subprocess
import sys

import numpy as np


def _git_sha() -> str:
    """Worktree HEAD sha (+ '-dirty') for artifact provenance."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--dt", type=float, default=None,
                    help="flat inner acoustic step [s] (legacy cadence: "
                         "entry A-scalar exchange every step). Mutually "
                         "exclusive with --dt-atmos.")
    ap.add_argument("--dt-atmos", type=float, default=None,
                    help="outer dt_atmos block [s]; runs n_split inner "
                         "steps of dt_atmos/n_split with the upstream "
                         "exchange schedule (entry A-scalar only on the "
                         "first inner step, dyn_core.F90:432-439). "
                         "Zenodo C48 case-8: --dt-atmos 1200 --n-split 7.")
    ap.add_argument("--n-split", type=int, default=7,
                    help="inner acoustic steps per dt_atmos block "
                         "(only with --dt-atmos)")
    ap.add_argument("--days", type=float, default=60.0)
    ap.add_argument("--dump-state-out", default=None,
                    help="npz path for block-state twin dumps (requires "
                         "--dt-atmos): full u/v/delp/pt lattices per tile "
                         "at block 0 (IC) and after outer blocks b<=14 or "
                         "b%%18==0 (cap 200) — the same schedule as the "
                         "instrumented oracle dyn_core")
    ap.add_argument("--frame-days", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--single-vortex", default=None,
                    help="'LON,LAT' (deg): replace the two case-8 "
                         "bursts with ONE burst centred there — an "
                         "eastward Gaussian u-burst (v=0), i.e. exactly "
                         "one case-8 modon half, NOT an azimuthal "
                         "vortex (codex screens-r1 F3).  Vertex-"
                         "locality discriminator; vertex ~ '45,35.26', "
                         "face centre ~ '0,0'")
    ap.add_argument("--nord", type=int, default=None, choices=(1, 2),
                    help="override the divergence-damping order for "
                         "EITHER preset (merged onto the stepper "
                         "defaults; codex lattice-r1 rank-2 screen)")
    ap.add_argument("--d-ext", type=float, default=None,
                    help="override the preset's external-mode filter "
                         "coefficient (seam-ringing discriminator)")
    ap.add_argument("--dump-lattice-days", default=None,
                    help="comma list of frame days at which to savez the "
                         "raw six-face lattice states (delp/pt/u/v full "
                         "halos) next to --out as <out>_lat_dayD.npz — "
                         "feeds the corner-wedge sharp-state "
                         "capture-compare (codex deficiency-r1 endgame)")
    ap.add_argument("--ext-exclude", default=None,
                    help="comma list of ext-bundle exchanges to swap to "
                         "the interim mpp-analog (diagnostic): "
                         "'cvec' isolates the post-PG C-vector vertex "
                         "extension (codex vsrc rank-1 source), "
                         "also ascalar/dvec/divgd/metrics")
    ap.add_argument("--d4-bg", type=float, default=None,
                    help="del-6 divergence-damping strength override "
                         "(NON-FAITHFUL diagnostic; Zenodo case-8 = "
                         "0.12): boost to test whether the day-5 vertex "
                         "spike is a del-6 dissipation deficit")
    ap.add_argument("--ic-perturb", type=float, default=0.0,
                    help="relative IC wind kick (chaos discriminator): "
                         "if day-5 max|V| moves wildly vs the "
                         "unperturbed run, case-8 is Lyapunov-chaotic "
                         "and the vertex escape is not a localizable bug")
    ap.add_argument("--ic-perturb-mode", default="checker",
                    choices=("checker", "uniform", "burst"),
                    help="checker = index-alternating sign (grid-scale; "
                         "del-6 removes it — underestimates the chaos "
                         "floor, 2026-07-25 retraction); uniform = "
                         "smooth (1+eps) wind rescale; burst = scale "
                         "only the westerly burst's Gaussian envelope "
                         "(codex r2: a single smooth bulk direction "
                         "under-samples seam-sensitive growth — use "
                         "BOTH smooth modes for the chaos floor)")
    ap.add_argument("--plain-conventions", action="store_true",
                    help="A/B arm: plain-conventions lane (default is "
                         "the bounded lane + ext bundle — the lane "
                         "this runner exists to test)")
    ap.add_argument("--preset", default="case8",
                    choices=("case8", "w2tuned"),
                    help="stage configuration: 'case8' = the Zenodo "
                         "C48.sw.case8 damping block (vort damping OFF, "
                         "dddmp=0, d_ext=0, hords=8, nord=2/del-6 "
                         "d4_bg=0.12 — certified bit-exact 6/6) + FV3 "
                         "gravity 9.80665 in delp; 'w2tuned' = the historical "
                         "W2-tuned defaults (damp_v=0.2, dddmp=0.2, "
                         "d_ext=0.02, hord=6)")
    args = ap.parse_args()
    if (args.dt is None) == (args.dt_atmos is None):
        ap.error("exactly one of --dt (flat legacy cadence) or "
                 "--dt-atmos [--n-split] (upstream schedule) is required")
    if args.dt_atmos is not None and args.n_split < 1:
        ap.error(f"--n-split must be >= 1, got {args.n_split}")
    if args.dump_state_out and args.dt_atmos is None:
        ap.error("--dump-state-out requires --dt-atmos (block-indexed "
                 "twin dumps are defined on the upstream cadence)")

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
        advance_duo_outer_step,
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
    ext_exclude = tuple(x for x in
                        (args.ext_exclude or "").split(",") if x)
    ctx = build_six_face_duo_context(args.n, 3,
                                     use_ext_bundle=True,
                                     oracle_conventions=oc,
                                     ext_exclude=ext_exclude,
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
    if args.nord is not None:
        # merge onto stepper defaults — works for BOTH presets (the
        # stepper does dict(_SW_CFG_DEFAULT).update(sw_cfg); a bare
        # {"nord": N} overrides just that key).  Was a silent no-op
        # for w2tuned (codex screens-r1 F2).
        sw_cfg = {**(sw_cfg or {}), "nord": args.nord}
    if args.d4_bg is not None:
        # del-6 divergence-damping strength override (NON-FAITHFUL
        # diagnostic): if boosting it above the Zenodo 0.12 suppresses
        # the day-5 vertex spike, the residual is a del-6 dissipation
        # deficit; if not, del-6 strength is not the lever.
        sw_cfg = {**(sw_cfg or {}), "d4_bg": args.d4_bg}
    nord_effective = (sw_cfg or {}).get("nord", 1)
    d4_bg_effective = (sw_cfg or {}).get("d4_bg", None)

    # diagnostic env modes active this run (provenance — screens-r1 F5)
    import os
    diag_env = {k: os.environ[k] for k in
                ("LEGOESM_DUO_CORNER_MODE",
                 "LEGOESM_DUO_AVG_B_ENDPOINTS",
                 "LEGOESM_DUO_PG_BVERTEX") if os.environ.get(k)}

    if args.single_vortex:
        lon0, lat0 = (np.deg2rad(float(x))
                      for x in args.single_vortex.split(","))
        from legoesm.grids.cubed_sphere import great_circle_distance

        def wind_fn(ll):
            r = great_circle_distance(ll[..., 0], ll[..., 1], lon0, lat0,
                                      FV3_RADIUS_M)
            u_e = _MODON_UMAX * np.exp(-(np.asarray(r) / _MODON_SIZE) ** 2)
            return u_e, np.zeros_like(u_e)
    else:
        def wind_fn(ll):
            u_e, v_n = _modon_winds_geo(ll[..., 0], ll[..., 1],
                                        FV3_RADIUS_M)
            return np.asarray(u_e), np.asarray(v_n)

    if args.ic_perturb and args.ic_perturb_mode == "burst":
        # perturb the ANALYTIC field (smooth, seam-agnostic): scale the
        # winds by (1 + eps*g1) with g1 the westerly burst's Gaussian
        # envelope at (90E, 0) — a second chaos-floor direction that is
        # not a bulk rescale (codex r2)
        from legoesm.grids.cubed_sphere import great_circle_distance
        base_wind_fn = wind_fn

        def wind_fn(ll):
            u_e, v_n = base_wind_fn(ll)
            r1 = great_circle_distance(ll[..., 0], ll[..., 1],
                                       np.pi / 2, 0.0, FV3_RADIUS_M)
            f = 1.0 + args.ic_perturb * np.exp(
                -(np.asarray(r1) / _MODON_SIZE) ** 2)
            return u_e * f, v_n * f
        print(f"IC perturbed by rel eps={args.ic_perturb:g} "
              "mode=burst (chaos discriminator)", flush=True)

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

    if args.ic_perturb and args.ic_perturb_mode != "burst":
        # chaos discriminator: a tiny relative kick to the D winds; if
        # a bulk metric (e.g. day-5 max|V|) moves WILDLY vs the
        # unperturbed run, the near-inviscid case-8 is Lyapunov-chaotic
        # and pointwise divergence from the oracle is expected, not a
        # localizable bug.  DETERMINISTIC per (n, seed) via index hash.
        eps = args.ic_perturb
        for t, st in enumerate(states):
            for k in ("u", "v"):
                a = np.asarray(st[k])
                if args.ic_perturb_mode == "uniform":
                    # smooth rescale: survives del-6, so it actually
                    # measures the chaos floor
                    st[k] = a * (1.0 + eps)
                else:
                    # index-varying sign so it is not a uniform rescale
                    ii = np.arange(a.size).reshape(a.shape)
                    st[k] = a * (1.0 + eps * (((ii + t) % 2) * 2 - 1))
        print(f"IC perturbed by rel eps={eps:g} "
              f"mode={args.ic_perturb_mode} (chaos discriminator)",
              flush=True)

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

    state_dumps = {}
    blk_counter = [0]

    def _maybe_dump_state(states, b):
        if (args.dump_state_out and b <= 400
                and (b <= 14 or b % 18 == 0)):
            for t in range(6):
                for k in ("u", "v", "delp", "pt"):
                    state_dumps[f"b{b}_{k}_t{t + 1}"] = np.array(
                        states[t][k], copy=True)

    if args.dt_atmos is not None:
        # upstream cadence: n_split inner steps per dt_atmos block,
        # entry A-scalar only on the first inner step of each block
        blocks_f = args.frame_days * 86400.0 / args.dt_atmos
        steps_per_frame = int(round(blocks_f))
        if abs(steps_per_frame - blocks_f) > 1e-9:
            ap.error("frame-days*86400 must be an integer multiple of "
                     f"dt-atmos (got {blocks_f} blocks/frame)")
        dt_inner = args.dt_atmos / args.n_split

        def step_frame_unit(states):
            states = advance_duo_outer_step(ctx, states, args.dt_atmos,
                                            args.n_split, d_ext=d_ext,
                                            sw_cfg=sw_cfg)
            blk_counter[0] += 1
            _maybe_dump_state(states, blk_counter[0])
            return states
    else:
        steps_per_frame = int(round(args.frame_days * 86400.0 / args.dt))
        dt_inner = args.dt

        def step_frame_unit(states):
            return full_acoustic_step_sixface(ctx, states, args.dt,
                                              d_ext=d_ext, sw_cfg=sw_cfg)
    n_frames = int(round(args.days / args.frame_days))

    if args.dump_state_out:
        _maybe_dump_state(states, 0)      # IC = block 0

    times = [0.0]
    u0f, v0f = sample_uv(states)
    uf, vf = [u0f], [v0f]
    # finite gate at day 0 too — nanmax would hide a localized IC NaN
    # and then poison w0/every A/A0 (codex screens-r1 F7)
    if not (np.all(np.isfinite(u0f)) and np.all(np.isfinite(v0f))):
        print("day 0: NaN in IC — aborting", flush=True)
        sys.exit(2)
    print(f"day 0: max|u| {np.max(np.abs(u0f)):.3f} "
          f"(IC Umax {_MODON_UMAX}, r0 {_MODON_SIZE/1e3:.0f} km)",
          flush=True)
    w0 = float(np.max(np.hypot(u0f, v0f)))
    for fr in range(1, n_frames + 1):
        for _ in range(steps_per_frame):
            states = step_frame_unit(states)
        day = fr * args.frame_days
        if args.dump_lattice_days and any(
                abs(day - float(x)) < 1e-9
                for x in args.dump_lattice_days.split(",")):
            dump = {}
            for t in range(6):
                for k in ("delp", "pt", "u", "v"):
                    dump[f"{k}_t{t + 1}"] = np.asarray(states[t][k])
            dpath = args.out.replace(".npz", f"_lat_day{day:g}.npz")
            np.savez_compressed(dpath, **dump)
            print(f"day {day:g}: lattice dump {dpath}", flush=True)
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
    ic_desc = (f"single case-8 Gaussian u-burst at {args.single_vortex}"
               if args.single_vortex else
               "FV3 case-8 two-burst IC")
    np.savez_compressed(
        args.out, times_days=np.array(times),
        u=np.stack(uf), v=np.stack(vf),
        lat=np.linspace(-90, 90, 181), lon=np.arange(360, dtype=float),
        oracle_conventions=np.array(oc),
        preset=np.array(args.preset),
        single_vortex=np.array(args.single_vortex or ""),
        nord_effective=np.array(nord_effective),
        d_ext_effective=np.array(d_ext),
        d4_bg_effective=np.array(repr(d4_bg_effective)),
        diag_env=np.array(repr(diag_env)),
        dt=dt_inner, n=args.n,
        dt_atmos=np.array(-1.0 if args.dt_atmos is None
                          else args.dt_atmos),
        n_split=np.array(0 if args.dt_atmos is None else args.n_split),
        git_sha=np.array(_git_sha()),
        protocol="six-face duo stepper, " + ic_desc + " "
                 "(tests.test_cases.colliding_modons formulas, certified "
                 "edge-midpoint D projection), omega=0; c2l_ord2 lens + "
                 "nearest-cell 1deg sampling; preset=" + args.preset
                 + f", nord={nord_effective}, d_ext={d_ext}"
                 + (f", diag_env={diag_env}" if diag_env else ""))
    print("saved", args.out, flush=True)
    if args.dump_state_out:
        np.savez_compressed(
            args.dump_state_out, **state_dumps,
            n=np.array(args.n), ng=np.array(3),
            dt_atmos=np.array(args.dt_atmos),
            n_split=np.array(args.n_split),
            git_sha=np.array(_git_sha()),
            ic_perturb=np.array(args.ic_perturb),
            protocol=np.array(
                "block-state twin: full-lattice u/v/delp/pt per tile at "
                "block 0 (IC) + after outer blocks b<=14 or b%18==0 "
                "(cap 200); numpy [p,q] = Fortran (p+1-ng, q+1-ng)"))
        print(f"saved {args.dump_state_out} "
              f"({len(state_dumps)} arrays)", flush=True)


if __name__ == "__main__":
    main()
