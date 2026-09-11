#!/usr/bin/env python
"""The DAY-BY-DAY from-rest gap, day x field, in ONE metric on BOTH sides.

WHY THIS EXISTS.  The from-rest 3-D temperature gap was known only at steps 1
and 2 and then at days 10/20/30, where it is already flat:

    step 1  1.176e-07 K    step 2  6.122e-06 K    day 10  7.68e-04 K

Four orders of magnitude are established somewhere in between, and NEMO had no
record there.  ``nemo_dino_earlydays/run.sh`` produced one (restarts at kt
32/64/96/128/160 = days 1..5).  This scores every day the record covers, in a
SINGLE metric, so the day the gap is born can be named instead of bracketed.

THE METRIC, identical on every row and both sides: wet-masked rms difference
over NEMO's own mesh_mask, full frame (no interior crop), fp64.  T and S use
``tmask``, eta uses ``tmask[...,0]``, u uses ``umask`` and v uses ``vmask``.
legoESM's redundant west/south face is dropped exactly as
``kt2_leapfrog_gate.py:310-322`` and ``nemo_dino_step1_gate.py:275-278`` drop
it.  This is NOT ``twin_nemo_ts_maps.py``'s metric -- that one crops the
one-ring interior to (197,50) -- so the numbers here are NOT comparable to the
recorded T3D series and are not presented as its continuation.  Both series
are internally matched; only one of them may be quoted in a sentence.

PROVENANCE HAZARD, stated rather than buried (Rule 7).  Days 1-5 come from
``nemo_earlydays`` and days >=10 from ``cfgs/DINO/RUN_TRAJ``.  Both are the
same 16-rank decomposition (``layout.dat``: jpnij 16, jpimax 30, jpjmax 29),
the same namelist but for ``nn_itend``/``nn_stock``, and the same from-rest
start -- but they were produced by DIFFERENT BUILDS: RUN_TRAJ's
``ocean.output`` is dated 2026-07-17 and ``MY_SRC/stpmlf.F90`` gained the
``stp_dump_ts_krhs`` instrumentation on 2026-08-19.  That instrumentation is a
pure writer (``INTENT(in)`` on every dumped array, gated by
``nn_stpdump_every``), so it cannot change the arithmetic it observes; that is
a SOURCE argument, not a measurement, and no record exists at a shared kt to
turn it into one.  Read the day-5 -> day-10 step with that in mind.

THE FLOOR.  ``--phase0-root`` points at a ``verdict360_fromrest.py --phase0``
ensemble run to the same days.  The floor is that harness's own statistic --
``_pairwise`` imported from it, not re-derived -- so a floor quoted here is
the same object the day-30/90/180/360 floors are.

THE UNLANDED STATEMENT.  ``--size-restoring`` prints, per day, the size of the
surface restoring's TIME LEVEL -- the statement measured at 0.961 of the kt=2
temperature residual and NOT landed, because NEMO evaluates the flux on
``ts(:,:,1,:,Kbb)`` (``usrdef_sbc.f90:388,:436``) and legoESM on its NOW
tracer, which needs the Kbb tracer at the forcing call.  Expression, identical
to ``kt1_surface_gate.py:380``:

    r1_rho0_rcp * rn_trp * ( tn - tb )|surface / e3t(1)

evaluated on NEMO's OWN restart at that day, times ``rDt``, pooled over the
3-D wet count.  Two accumulation bounds are printed next to it and BOTH are
labelled: ``n_steps`` x (fully coherent) and ``sqrt(n_steps)`` x (random
walk).  The truth is between them only if the statement is the sole driver,
which is exactly what is not known -- so these are PREREGISTERED BOUNDS, not
a prediction of the gap.

Usage
-----
    CUDA_VISIBLE_DEVICES=<uuid> JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \\
      python scripts/validate/ocean_fidelity/dino_1226/day_gap_table.py \\
        --run-dino-dir <run with --snapshot-every-days 1> \\
        --phase0-root <verdict360_fromrest --phase0 out-root> --size-restoring
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
from rebuild_nemo_restart import rebuild                        # noqa: E402

DT = 2700.0
STEPS_PER_DAY = 32
EARLY = "/data/abyssal/dbalwada/dino_fromrest_y1/nemo_earlydays"
TRAJ = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ"
#: (day -> which record holds kt = 32*day).  Days 1-5 only exist in the
#: early-days record; days 10/20/30 only in RUN_TRAJ.
DAYS_EARLY = (1, 2, 3, 4, 5)
DAYS_TRAJ = (10, 20, 30)
FIELDS = (("T", "tn"), ("S", "sn"), ("eta", "sshn"), ("u", "un"), ("v", "vn"))
# usrdef_sbc / namelist_cfg:37,:41 and DINOConfig -- the SAME constants
# kt1_surface_gate.py:384-386 uses, not re-derived here.
RN_TRP = -40.0
RHO0, RCP = 1026.0, 3991.86795711963


def _load_pairwise():
    """``verdict360_fromrest._pairwise``, imported so the FLOOR statistic here
    is byte-identical to the one the day-30..360 floors were measured with."""
    spec = importlib.util.spec_from_file_location(
        "_v360fr", os.path.join(_HERE, "verdict360_fromrest.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._pairwise, mod._rms


def _lego(snap, name):
    """legoESM's field on NEMO's own staggering (kt2_leapfrog_gate.py:310)."""
    d = np.asarray(snap[name], dtype=np.float64)
    if name == "u":
        return d[:, 1:, :]
    if name == "v":
        return d[1:, :, :]
    return d


def _snapshot_for_day(run_dir, day):
    meta = os.path.join(run_dir, "run_metadata.json")
    every = None
    if os.path.exists(meta):
        m = json.load(open(meta))
        every = m.get("snapshot_every_days") or m.get("snapshot_interval_days")
    files = sorted(glob.glob(os.path.join(run_dir, "snapshots",
                                          "snapshot_*.npz")))
    if not files:
        raise SystemExit(f"no snapshots under {run_dir}")
    for f in files:
        z = np.load(f)
        if abs(float(z["time_days"]) - day) < 1e-9:
            return f, z
    raise SystemExit(
        f"{run_dir} has no snapshot at day {day} "
        f"(snapshot_every_days={every}; found "
        f"{[float(np.load(x)['time_days']) for x in files]})")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dino-dir", required=True)
    ap.add_argument("--early-run", default=EARLY)
    ap.add_argument("--traj-run", default=TRAJ)
    ap.add_argument("--phase0-root", default=None)
    ap.add_argument("--kt1-run", default="/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_FROMREST_KT1")
    ap.add_argument("--size-restoring", action="store_true")
    ap.add_argument("--days", default=None,
                    help="comma-separated override of the day list")
    ap.add_argument("--plant", action="store_true",
                    help="score every day TWICE -- unplanted, and with ONE wet "
                         "T cell moved by 1 ulp -- and print the DELTA.  The "
                         "first version printed only the planted table and "
                         "asked the reader to eyeball it against another run; "
                         "one ulp on one cell of 342134 is ~1e-19 of an rms "
                         "printed to five digits, so the two tables were "
                         "BYTE-IDENTICAL and the control proved nothing.")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())                          # Rule 1c
    from legoesm.ocean.fidelity import nemo_dino_mesh as ndm

    g = ndm.nemo_dino_mesh()
    masks = {"T": np.asarray(g.tmask > 0.5), "S": np.asarray(g.tmask > 0.5),
             "eta": np.asarray(g.tmask > 0.5)[..., 0],
             "u": np.asarray(g.umask > 0.5), "v": np.asarray(g.vmask > 0.5)}
    for k, m in masks.items():
        print(f"  mask {k:4s} {m.shape} wet {int(m.sum())}")
    print(f"  mesh dtype {np.asarray(g.tmask).dtype}")

    if a.days:
        want = [int(x) for x in a.days.split(",")]
    else:
        want = list(DAYS_EARLY) + list(DAYS_TRAJ)

    # WHICH FIELD LEADS cannot be read off five numbers in five different
    # units.  The second table normalises each gap by the signal NEMO ITSELF
    # has developed from rest by that day -- rms(NEMO(day) - NEMO(rest)) on
    # the same wet mask -- which is dimensionless and comparable across
    # fields.  NEMO's rest state is the kt=1 restart's BEFORE level (tb/sb/
    # sshb/ub/vb), which the step-1 gate certifies IS the initial state.
    rest = rebuild(os.path.join(a.kt1_run, "DINO_00000001_restart_*.nc"),
                   ["tb", "sb", "sshb", "ub", "vb"])
    REST = {"T": "tb", "S": "sb", "eta": "sshb", "u": "ub", "v": "vb"}

    print(f"\nDAY x FIELD GAP -- legoESM {a.run_dino_dir} vs NEMO, wet-masked "
          "rms, full frame, fp64")
    print(f"  {'day':>5s}{'kt':>7s}{'record':>12s}"
          + "".join(f"{n:>14s}" for n, _ in FIELDS))
    rows = {}
    fracs = {}
    prov = {}
    for day in want:
        kt = day * STEPS_PER_DAY
        run = a.early_run if day in DAYS_EARLY else a.traj_run
        pat = os.path.join(run, f"DINO_{kt:08d}_restart*.nc")
        if not glob.glob(pat):
            print(f"  {day:>5d}{kt:>7d}{'MISSING':>12s}   no tiles match {pat}")
            continue
        R = rebuild(pat, [n for _, n in FIELDS])
        spath, snap = _snapshot_for_day(a.run_dino_dir, day)
        prov[day] = {"kt": kt, "nemo_run": run,
                     "nemo_tiles": len(glob.glob(pat)),
                     "lego_snapshot": spath, "lego_sha256": _sha(spath)}
        out, delta, frac = [], [], []
        for lname, nname in FIELDS:
            lego = _lego(snap, lname)
            nemo = (np.nan_to_num(R[nname]) if lname == "eta"
                    else np.nan_to_num(np.moveaxis(R[nname], 0, -1)))
            m = masks[lname]
            if lego.shape != nemo.shape or lego.shape != m.shape:
                raise SystemExit(
                    f"shape mismatch day {day} {lname}: lego {lego.shape} "
                    f"nemo {nemo.shape} mask {m.shape}")
            v = float(np.sqrt(np.mean((lego - nemo)[m] ** 2)))
            out.append(v)
            r0 = (np.nan_to_num(rest[REST[lname]]) if lname == "eta"
                  else np.nan_to_num(np.moveaxis(rest[REST[lname]], 0, -1)))
            sig = float(np.sqrt(np.mean((nemo - r0)[m] ** 2)))
            frac.append(v / sig if sig > 0 else float("nan"))
            if a.plant and lname == "T":
                j, i, k = np.argwhere(m)[0]
                for tag, dv in (("1 ulp", None), ("1e-4 K", 1e-4)):
                    lp = lego.copy()
                    lp[j, i, k] = (np.nextafter(lp[j, i, k], np.inf)
                                   if dv is None else lp[j, i, k] + dv)
                    vp = float(np.sqrt(np.mean((lp - nemo)[m] ** 2)))
                    delta.append((f"{lname}(j{j},i{i},k{k}) {tag}", vp - v))
        rows[day] = out
        fracs[day] = frac
        print(f"  {day:>5d}{kt:>7d}{os.path.basename(run):>12s}"
              + "".join(f"{v:>14.4e}" for v in out))
        for tag, d in delta:
            print(f"        PLANT {tag}: moves the rms by {d:+.4e} K "
                  f"({'RESOLVED' if d != 0.0 else 'BELOW RESOLUTION'})")
    if a.plant:
        print("  PLANT lines are DELTAS computed in this same process, not a "
              "second table to eyeball.  A 1-ulp move on ONE cell of 342134 "
              "is BELOW this table's resolution and says so; that is not a "
              "defect, it is the table's scale -- these rows are a CLIMATE "
              "comparison whose noise floor is the Phase-0 spread above "
              "(~1.7e-10 K), seven orders below the gap they report, and the "
              "1e-4 K plant is the positive control that they move at all.")

    if fracs:
        print("\nTHE SAME GAPS as a FRACTION of the signal NEMO itself has "
              "developed from rest by that day -- rms(NEMO(day) - NEMO(rest)) "
              "on the same mask.  Dimensionless, so the five fields are "
              "comparable and the LEADING one can be named.")
        print(f"  {'day':>5s}" + "".join(f"{n:>14s}" for n, _ in FIELDS)
              + "   leads")
        for day in sorted(fracs):
            f = fracs[day]
            lead = [n for n, _ in FIELDS][int(np.nanargmax(f))]
            print(f"  {day:>5d}" + "".join(f"{v:>14.4e}" for v in f)
                  + f"   {lead}")

    floor = {}
    if a.phase0_root:
        pairwise, _rms = _load_pairwise()
        wet = masks["T"]
        per_day = {}
        for d in sorted(glob.glob(os.path.join(a.phase0_root, "member_*",
                                               "snapshots"))):
            for f in sorted(glob.glob(os.path.join(d, "snapshot_*.npz"))):
                z = np.load(f)
                per_day.setdefault(int(round(float(z["time_days"]))),
                                   []).append(np.asarray(z["T"]))
        print(f"\nPHASE-0 FLOOR (same _pairwise statistic as the day-30..360 "
              f"floors), {a.phase0_root}")
        print(f"  {'day':>5s}{'members':>9s}{'floor rms [K]':>16s}"
              f"{'T gap [K]':>13s}{'gap/2*floor':>13s}")
        for day in sorted(per_day):
            M = per_day[day]
            if len(M) < 2:
                continue
            w, _ = pairwise(M, M[:0], wet)
            f = float(np.sqrt(np.mean(w ** 2)))
            floor[day] = f
            gap = rows.get(day, [float("nan")])[0]
            r = gap / (2 * f) if f > 0 else float("nan")
            print(f"  {day:>5d}{len(M):>9d}{f:>16.4e}{gap:>13.4e}{r:>13.2f}")
        if 0 in floor and floor[0] == 0.0:
            raise SystemExit("two members identical at day 0 -- the floor "
                             "would be manufactured downward")

    sizes = {}
    if a.size_restoring:
        print("\nSIZE of the UNLANDED surface statement (the restoring flux's "
              "TIME LEVEL, usrdef_sbc.f90:388,:436), per day, off NEMO's OWN "
              "restart.  PREREGISTERED BOUNDS, not a prediction.")
        print(f"  {'day':>5s}{'1 step [K]':>14s}{'x sqrt(n)':>13s}"
              f"{'x n':>13s}{'T gap [K]':>13s}{'1step/gap':>12s}"
              f"{'sqrt(n)/gap':>13s}{'n/gap':>11s}")
        wet3 = masks["T"]
        n3 = int(wet3.sum())
        e3t0 = None
        from legoesm.ocean.experiments import dino as dm
        cfg = dm.nemo_faithful_dino_config(
            base=dm.dino_config_for_recipe("nemo_dino_kamm_mlf"))
        grid = dm.dino_lat_lon_grid(cfg)
        e3t0 = float(np.asarray(dm.dino_lat_lon_vertical(grid, cfg).dz_ref)[0])
        for day in want:
            kt = day * STEPS_PER_DAY
            run = a.early_run if day in DAYS_EARLY else a.traj_run
            pat = os.path.join(run, f"DINO_{kt:08d}_restart*.nc")
            if not glob.glob(pat):
                continue
            Rl = rebuild(pat, ["tn", "tb"])
            now = np.nan_to_num(np.asarray(Rl["tn"], dtype=np.float64))
            bef = np.nan_to_num(np.asarray(Rl["tb"], dtype=np.float64))
            d = (1.0 / (RHO0 * RCP)) * (RN_TRP * (now[0] - bef[0])) / e3t0
            m2 = wet3[..., 0]
            one = float(np.sqrt(np.sum((d[m2] * 2.0 * DT) ** 2) / n3))
            n = day * STEPS_PER_DAY
            gap = rows.get(day, [float("nan")])[0]
            sizes[day] = {"one_step_K": one, "n_steps": n,
                          "sqrt_n_K": one * np.sqrt(n), "n_K": one * n,
                          "gap_K": gap}
            print(f"  {day:>5d}{one:>14.4e}{one * np.sqrt(n):>13.4e}"
                  f"{one * n:>13.4e}{gap:>13.4e}{one / gap:>12.4f}"
                  f"{one * np.sqrt(n) / gap:>13.3f}{one * n / gap:>11.2f}")

    if a.json_out:
        json.dump({"rows": {str(k): dict(zip([n for n, _ in FIELDS], v))
                            for k, v in rows.items()},
                   "floor_K": {str(k): v for k, v in floor.items()},
                   "fraction_of_nemo_signal": {
                       str(k): dict(zip([n for n, _ in FIELDS], v))
                       for k, v in fracs.items()},
                   "restoring_time_level": {str(k): v
                                            for k, v in sizes.items()},
                   "provenance": prov,
                   "metric": "wet-masked full-frame rms, fp64"},
                  open(a.json_out, "w"), indent=1)
        print(f"\n  wrote {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
