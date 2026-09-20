#!/usr/bin/env python3
"""S0 decider: does prognostic CLUBB see Arctic sub-grid cloud that the
diagnostic closure does not?

Pre-registered (2026-09-20) on a protocol-matched pair restarted from one
stripped day-40 checkpoint: control = diagnostic CLUBB, arm = prognostic CLUBB
with the radiative thlp2 source.  Score = area-weighted mean over 75-90N of
the PDF cloud fraction in layers colder than 253 K (supercooled regime), arm
minus control, averaged over pair days 5-10.

  CONFIRM   arm - control >= 0.05
  REFUTE    arm - control <= 0.01
  FAIL      any alarm on the arm's moments: thlp2 > 100 K^2, rtp2 > 1e-4,
            wp2 > 100 m^2/s^2, |wp3| > 1000 m^3/s^3, or any non-finite value

Both arms publish the PDF cloud fraction into the checkpoint's
``physstate_cloud_fraction`` (top-down, troposphere-tapered) so the band reads
the same field on both sides.  The cold-layer mask is taken from the CONTROL's
temperature and applied to both arms, so an arm that warms or cools the cap
cannot move its own scoring mask (GLM review).  PROVENANCE CONTROL, run on the
first scored day: each arm's stored cloud fraction is compared with the
humidity-based cover the run's cloud scheme would diagnose from the same
state; a stored field equal to that cover is the grid-scale cover, not the
PDF's, and the probe aborts instead of scoring PDF against RH.  The diagnostic control does NOT carry its
variances (they are recomputed from mixing length each step and never stored),
so the arm's rtp2/thlp2/wp2 are reported as absolute cold-layer means only; the
"2x control" secondary is not measurable from checkpoints and is said so.

REGISTERED-PAIR GATE (codex review): the two runs' resolved configurations
must differ in ``clubb_prognostic`` (control False, arm True) and the output
directory ONLY, both must restart from the same checkpoint (manifest command
line), share the vertical grid, and the scored days must be exactly the
registered 45-50; anything else is printed as EXPLORATORY, never as the
registered verdict.  Every stored array is shape-checked against the mesh and
vertical grid, temperatures must be finite, cloud fractions in [0, 1], and the
WHOLE packed moment array finite before any unpacking.

Usage:
  s0_clubb_pdf_score.py --control s0_ctl --arm s0_prog --days 45 46 47 48 49 50
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np

_VAL = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("cloud_layers", _VAL / "cloud_layers.py")
cl = importlib.util.module_from_spec(_spec)
sys.modules["cloud_layers"] = cl
_spec.loader.exec_module(cl)
rb = cl.rb

ALARMS = {"thlp2": 100.0, "rtp2": 1.0e-4, "wp2": 100.0, "wp3": 1000.0}
CONFIRM, REFUTE = 0.05, 0.01
REGISTERED_DAYS = list(range(45, 51))
PAIR_MAY_DIFFER = {"clubb_prognostic", "output"}
N_MOMENTS = 15


def registered_pair_check(exp_c, exp_a, run_c, run_a):
    """Abort unless the two resolved configs differ only in the registered
    switch (control diagnostic, arm prognostic) and the output block, and both
    manifests restart from the same checkpoint."""
    diff = sorted(k for k in set(exp_c) | set(exp_a) if exp_c.get(k) != exp_a.get(k))
    extra = [k for k in diff if k not in PAIR_MAY_DIFFER]
    print(f"{'field':>24s} {run_c:>12s} {run_a:>12s}")
    for k in diff:
        if k != "output":
            print(f"{k:>24s} {str(exp_c.get(k)):>12s} {str(exp_a.get(k)):>12s}")
    if extra:
        raise SystemExit(f"FATAL: not the registered pair; configs also differ in {extra}")
    if exp_c.get("clubb_prognostic") is not False or exp_a.get("clubb_prognostic") is not True:
        raise SystemExit("FATAL: control must be diagnostic CLUBB and the arm prognostic")
    if exp_c.get("turbulence") != "clubb":
        raise SystemExit("FATAL: pair does not run CLUBB")
    origin = []
    for run in (run_c, run_a):
        cmd = json.load(open(f"{rb.ROOT}/{run}/run_manifest.json"))["command_line"]
        toks = cmd.split()
        if "--restart-from" not in toks:
            raise SystemExit(f"FATAL: {run} was not restarted from a checkpoint")
        origin.append(toks[toks.index("--restart-from") + 1])
    if origin[0] != origin[1]:
        raise SystemExit(f"FATAL: different restart origins {origin}")
    print(f"restart origin (both): {origin[0]}")


def cold_cap_mean(field, T, lat, area, lat_lo, T_max):
    """Area-weighted mean of ``field`` over cells >= lat_lo and layers T < T_max."""
    m = (T < T_max) & (lat >= lat_lo)[:, None]
    w = np.broadcast_to(area[:, None], T.shape) * m
    if w.sum() == 0.0:
        raise SystemExit("FATAL: empty cold-layer mask")
    return float((field * w).sum() / w.sum()), int(m.sum())


def load_day(run, day, lat_n, require_moments=False):
    z = np.load(f"{rb.ROOT}/{run}/checkpoint_day_{day:04d}.npz", allow_pickle=True)
    order = cl.cell_order(z, lat_n)
    if "physstate_cloud_fraction" not in z.files:
        raise SystemExit(f"FATAL: {run} day {day}: no physstate_cloud_fraction")
    vgrid = np.asarray(z["meta_vgrid"], dtype=np.float64)
    nlev = vgrid.shape[1] - 1
    out = {"T": np.asarray(z["T"], dtype=np.float64)[order],
           "cf": np.asarray(z["physstate_cloud_fraction"], dtype=np.float64)[order],
           "q_v": np.asarray(z["trc_q_v"])[order], "p_s": np.asarray(z["p_s"])[order],
           "vgrid": vgrid}
    for k in ("trc_q_c", "trc_q_i", "trc_N_c", "trc_N_i"):
        out[k] = np.asarray(z[k])[order] if k in z.files else None
    where = f"{run} day {day}"
    for k in ("T", "cf"):
        if out[k].shape != (lat_n, nlev):
            raise SystemExit(f"FATAL: {where}: {k} shape {out[k].shape} != {(lat_n, nlev)}")
        if not np.isfinite(out[k]).all():
            raise SystemExit(f"FATAL: {where}: non-finite {k}")
    if out["cf"].min() < 0.0 or out["cf"].max() > 1.0:
        raise SystemExit(f"FATAL: {where}: cloud fraction outside [0, 1]")
    mom = np.asarray(z["physstate_clubb_moments"]) if "physstate_clubb_moments" in z.files else None
    good = mom is not None and mom.shape == (lat_n, N_MOMENTS, nlev + 1)
    if require_moments and not good:
        raise SystemExit(f"FATAL: {where}: no well-formed physstate_clubb_moments "
                         f"(got {None if mom is None else mom.shape}, want "
                         f"{(lat_n, N_MOMENTS, nlev + 1)}); alarms cannot run")
    out["moments"] = np.asarray(mom, dtype=np.float64)[order] if good else None
    return out


def rh_cover(d, exp):
    """The grid-scale cover the run's cloud scheme diagnoses from the stored
    state (same call as the live radiation entry: raw condensate and number
    tracers handed in), for the provenance control."""
    from legoesm import constants
    from legoesm.atmosphere.physics.clouds.cloud_fraction import compute_cloud_properties
    cfg = cl.resolved_cloud_config(exp)
    vg = d["vgrid"]
    p_half = vg[0][None, :] * constants.p_ref + vg[1][None, :] * d["p_s"][:, None]
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    props = compute_cloud_properties(d["T"], p_full, d["q_v"], np.diff(p_half, axis=1), cfg,
                                     q_cloud=d["trc_q_c"], q_ice=d["trc_q_i"],
                                     n_cloud=d["trc_N_c"], n_ice=d["trc_N_i"])
    return np.asarray(props.cloud_fraction)


def check_provenance(name, d, exp, tol=1.0e-6):
    rh = rh_cover(d, exp)
    same = float(np.mean(np.abs(d["cf"] - rh) <= tol))
    if same > 0.999:
        raise SystemExit(f"FATAL: {name}: stored cloud fraction equals the humidity-based "
                         f"cover on {same:.1%} of points; it is not the PDF cloud fraction")
    return same


def moment_stats(mom, T, lat, area, lat_lo, T_max):
    """Cold-layer means + global extrema of the arm's packed moments.

    zm-level fields (nzm = nlev+1) are averaged onto the nlev zt layers so the
    same T<T_max mask applies; CLUBB stores ascending, so flip to top-down first."""
    from legoesm.atmosphere.physics.turbulence.clubb import unpack_clubb_moments
    import jax.numpy as jnp
    res, alarms = {}, []
    mom = np.asarray(mom, dtype=np.float64)
    if not np.isfinite(mom).all():                     # ALL 15 fields, before unpacking
        alarms.append(f"packed moments non-finite ({int((~np.isfinite(mom)).sum())} entries)")
        mom = np.nan_to_num(mom)
    st = unpack_clubb_moments(jnp.asarray(mom, dtype=jnp.float64))
    for name in ("rtp2", "thlp2", "wp2", "wp3"):
        f = np.asarray(getattr(st, name), dtype=np.float64)
        ext = float(np.abs(f).max())                   # alarm on the RAW stored level
        if f.shape[1] == T.shape[1] + 1:               # zm -> zt
            f = 0.5 * (f[:, 1:] + f[:, :-1])
        f = f[:, ::-1]                                 # ascending -> top-down
        mean, _ = cold_cap_mean(f, T, lat, area, lat_lo, T_max)
        res[name] = (mean, ext)
        if ext > ALARMS[name]:
            alarms.append(f"{name} |max| {ext:.3g} > {ALARMS[name]:g}")
    return res, alarms


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--control", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--days", type=int, nargs="+", required=True)
    ap.add_argument("--lat-lo", type=float, default=75.0)
    ap.add_argument("--t-max", type=float, default=253.0)
    ap.add_argument("--json", default=None, help="write per-day numbers here")
    args = ap.parse_args(argv)
    import jax
    jax.config.update("jax_enable_x64", True)          # stored moments are fp64

    exp = json.load(open(f"{rb.ROOT}/{args.control}/experiment_config.json"))
    exp_a = json.load(open(f"{rb.ROOT}/{args.arm}/experiment_config.json"))
    registered_pair_check(exp, exp_a, args.control, args.arm)
    lat, _lon, area = cl.mesh_coords(exp)
    if not (np.isfinite(area).all() and (area > 0).all()):
        raise SystemExit("FATAL: mesh areas not finite/positive")
    registered = (sorted(set(args.days)) == args.days == REGISTERED_DAYS
                  and args.lat_lo == 75.0 and args.t_max == 253.0)
    label = "REGISTERED" if registered else "EXPLORATORY"
    p_top = exp.get("clubb_trop_cloud_top_press")

    print(f"=== S0 PDF cloud fraction, {args.lat_lo:g}-90N, layers T < {args.t_max:g} K: "
          f"{args.arm} minus {args.control} ===")
    print(f"{'day':>4s} {'ctl cf':>8s} {'arm cf':>8s} {'diff':>8s} {'n_cold':>7s}  "
          f"{'rtp2':>9s} {'thlp2':>9s} {'wp2':>9s} {'|wp3|max':>9s}")
    rows, diffs, all_alarms = [], [], []
    for day in args.days:
        c = load_day(args.control, day, lat.size)
        a = load_day(args.arm, day, lat.size, require_moments=True)
        if not np.array_equal(c["vgrid"], a["vgrid"]):
            raise SystemExit(f"FATAL: day {day}: arms on different vertical grids")
        if day == args.days[0]:
            if p_top:
                from legoesm import constants
                vg = c["vgrid"]
                p_half = vg[0][None, :] * constants.p_ref + vg[1][None, :] * c["p_s"][:, None]
                p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
                above, _ = cold_cap_mean((p_full < p_top).astype(float), c["T"], lat, area,
                                         args.lat_lo, args.t_max)
                print(f"taper: {above:.1%} of the cold-layer weight lies above the CLUBB "
                      f"troposphere-top pressure {p_top:g} Pa (PDF cloud tapered there)")
            for nm, d, e in ((args.control, c, exp), (args.arm, a, exp_a)):
                same = check_provenance(nm, d, e)
                print(f"provenance {nm} day {day}: stored cf equals RH cover on "
                      f"{same:.1%} of points [OK]")
        cf_c, n = cold_cap_mean(c["cf"], c["T"], lat, area, args.lat_lo, args.t_max)
        cf_a, _ = cold_cap_mean(a["cf"], c["T"], lat, area, args.lat_lo, args.t_max)  # CONTROL mask
        d = cf_a - cf_c
        diffs.append(d)
        row = {"day": day, "cf_control": cf_c, "cf_arm": cf_a, "diff": d, "n_cold": n}
        mstr = ""
        if a["moments"] is not None:
            ms, al = moment_stats(a["moments"], c["T"], lat, area, args.lat_lo, args.t_max)
            all_alarms += [f"day {day}: {x}" for x in al]
            row["arm_moments_cold_mean"] = {k: v[0] for k, v in ms.items()}
            row["arm_moments_absmax"] = {k: v[1] for k, v in ms.items()}
            mstr = (f"  {ms['rtp2'][0]:9.2e} {ms['thlp2'][0]:9.2e} {ms['wp2'][0]:9.2e} "
                    f"{ms['wp3'][1]:9.2e}")
        if c["moments"] is not None:
            _, al = moment_stats(c["moments"], c["T"], lat, area, args.lat_lo, args.t_max)
            all_alarms += [f"day {day} (control): {x}" for x in al]
        rows.append(row)
        print(f"{day:4d} {cf_c:8.4f} {cf_a:8.4f} {d:+8.4f} {n:7d}{mstr}")

    mean_d = float(np.mean(diffs))
    if all_alarms:
        verdict = "FAIL"
    elif mean_d >= CONFIRM:
        verdict = "CONFIRM"
    elif mean_d <= REFUTE:
        verdict = "REFUTE"
    else:
        verdict = "INCONCLUSIVE"
    print(f"\nmean arm-control over days {args.days[0]}-{args.days[-1]}: {mean_d:+.4f}  "
          f"(confirm >= {CONFIRM:g}, refute <= {REFUTE:g})  {label} VERDICT: {verdict}")
    for x in all_alarms:
        print("ALARM", x)
    print("control variances: NOT CARRIED in checkpoints (diagnostic closure); "
          "'2x control' secondary not scored")
    if args.json:
        json.dump({"verdict": verdict, "label": label, "mean_diff": mean_d, "rows": rows,
                   "alarms": all_alarms, "lat_lo": args.lat_lo, "t_max": args.t_max},
                  open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
