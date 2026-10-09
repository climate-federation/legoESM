#!/usr/bin/env python3
"""Round 211 / VORTEX_SMT round 1 -- sanity on NEMO's OWN seamount output.

Decision 88 / operator note CC, deliverable (5).  This probe says nothing
about legoESM: it reads only what NEMO wrote, and it answers five questions
the round brief asks.

  1. the resolved geometry: k_bot and the partial e3t at the seamount, read
     from mesh_mask.nc (the deck sets ln_meshmask = .true.);
  2. max |ssh| and max |u| at kt=10 and at day 100;
  3. the bottom-level map, as a histogram and as the i-row through the summit;
  4. no NaN anywhere, at kt=10 or day 100 -- NOT via nanmax, which would hide
     exactly this (campaign rule), but via an explicit isfinite assertion;
  5. whether NEMO's zps run differs from the certified FLAT run after one
     step, and if so WHERE.  Partial steps change e3t near the seamount from
     step 1, long before the vortex arrives, so a difference confined to the
     seamount is the expected answer and a difference under the vortex at
     kt=2 would be the finding.
  6. the vortex still drifts west: the longitude of the ssh extremum per day.

Pre-impl search (RULE 4): grepped scripts/validate/ocean_fidelity for an
existing step-entry reader and found three.  The self-describing one in
nemo_testcase_phase3_trajectory_gate.read_entry (note BD: it parses the
record's own header and predicts no size) is REUSED here rather than a
fourth being written; the first_divergence_gate's copy hard-codes
EXPECTED_DIMS and would refuse a new geometry.  netCDF reading follows
round 210's load_nemo contract (variable names and axis order) exactly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nemo_testcase_phase3_trajectory_gate import read_entry, require  # noqa: E402

HALO = 2


def _interior(a):
    return a[HALO:-HALO, HALO:-HALO] if a.ndim == 2 else a[HALO:-HALO, HALO:-HALO, :]


def geometry(run_dir: Path) -> dict:
    import netCDF4
    with netCDF4.Dataset(run_dir / "mesh_mask.nc") as h:
        # NEMO 5 writes the last-wet-level field as "mbathy" here; a
        # tree that renames it to "bottom_level" is accepted too, and
        # one that has neither is a refusal rather than a guess.
        kname = next((n for n in ("mbathy", "bottom_level")
                      if n in h.variables), None)
        require(kname is not None,
                "mesh_mask.nc carries neither mbathy nor bottom_level")
        kbot = np.asarray(h.variables[kname][0], dtype=np.int64)
        e3t = np.asarray(h.variables["e3t_0"][0], dtype=np.float64)   # (z, y, x)
        glamt = np.asarray(h.variables["glamt"][0], dtype=np.float64)
        tmask = np.asarray(h.variables["tmask"][0], dtype=np.float64)
    require(bool(np.all(np.isfinite(e3t))), "mesh_mask e3t_0 is not finite")
    wet = kbot > 0
    report_kname = kname
    # the thickness of the LAST wet cell of every wet column
    ny, nx = kbot.shape
    bot_e3t = np.full(kbot.shape, np.nan)
    jj, ii = np.nonzero(wet)
    bot_e3t[jj, ii] = e3t[kbot[jj, ii] - 1, jj, ii]
    # bathymetry as the model carries it: the summed wet thickness
    depth = np.zeros(kbot.shape)
    for k in range(e3t.shape[0]):
        depth += np.where(kbot > k, e3t[k], 0.0)
    j_summit, i_summit = np.unravel_index(
        np.argmin(np.where(wet, depth, np.inf)), depth.shape)
    row = []
    for i in range(nx):
        row.append({"i": int(i), "glamt_km": float(glamt[j_summit, i]),
                    "depth_m": float(depth[j_summit, i]),
                    "k_bot": int(kbot[j_summit, i]),
                    "e3t_bot_m": (None if not wet[j_summit, i]
                                  else float(bot_e3t[j_summit, i]))})
    levels, counts = np.unique(kbot[wet], return_counts=True)
    return {
        "bottom_level_variable": report_kname,
        "nx_ny_with_halo": [int(nx), int(ny)],
        "n_wet_columns": int(wet.sum()),
        "k_bot_histogram": {int(a): int(b) for a, b in zip(levels, counts)},
        "k_bot_min_wet": int(kbot[wet].min()), "k_bot_max": int(kbot.max()),
        "depth_min_m": float(depth[wet].min()),
        "depth_max_m": float(depth[wet].max()),
        "bottom_e3t_min_m": float(np.nanmin(bot_e3t)),
        "bottom_e3t_max_m": float(np.nanmax(bot_e3t)),
        "summit": {"j": int(j_summit), "i": int(i_summit),
                   "glamt_km": float(glamt[j_summit, i_summit]),
                   "depth_m": float(depth[j_summit, i_summit]),
                   "k_bot": int(kbot[j_summit, i_summit]),
                   "e3t_bot_m": float(bot_e3t[j_summit, i_summit])},
        "summit_row": row,
        "tmask_wet_cells": int(tmask.sum()),
    }


def entry_extrema(run_dir: Path, kt: int, case: str) -> dict:
    rec = read_entry(run_dir / f"oracle_step_entry_kt{kt:08d}.bin", case)
    out = {"kt": kt}
    for name in ("T", "S", "u", "v", "ssh"):
        a = np.asarray(rec[name], dtype=np.float64)
        require(bool(np.all(np.isfinite(a))), f"kt={kt}: {name} is not finite")
        out[f"max_abs_{name}"] = float(np.max(np.abs(a)))
    return out


def restart_state(run_dir: Path, step: int) -> dict:
    import netCDF4
    matches = sorted(run_dir.glob(f"*_{step:08d}_restart.nc"))
    require(len(matches) == 1,
            f"expected one restart for step {step} in {run_dir}, "
            f"found {len(matches)}")
    with netCDF4.Dataset(matches[0]) as h:
        recorded = int(np.asarray(h.variables["kt"][...]))
        require(recorded == step, f"{matches[0]}: kt={recorded} != {step}")
        fields = {}
        for name, var in (("T", "tn"), ("u", "un"), ("v", "vn")):
            fields[name] = np.asarray(h.variables[var][0], dtype=np.float64)
        fields["ssh"] = np.asarray(h.variables["sshn"][0], dtype=np.float64)
        glamt = np.asarray(h.variables["nav_lon"][...], dtype=np.float64)
    out = {"step": step}
    for name, a in fields.items():
        require(bool(np.all(np.isfinite(a))),
                f"step {step}: NEMO field {name} is not finite")
        out[f"max_abs_{name}"] = float(np.max(np.abs(a)))
    ssh = fields["ssh"]
    j, i = np.unravel_index(np.argmax(np.abs(ssh)), ssh.shape)
    out["ssh_extremum"] = {"j": int(j), "i": int(i),
                           "glamt_km": float(glamt[j, i]),
                           "value_m": float(ssh[j, i])}
    return out


def drift(run_dir: Path, days) -> list:
    track = []
    for day in days:
        state = restart_state(run_dir, day * 30)
        track.append({"day": day, **state["ssh_extremum"]})
    return track


def one_step_vs_flat(smt_dir: Path, flat_dir: Path, case: str) -> dict:
    """Does one step of the zps run already differ from the flat one, and where?

    kt=1's entry record is the INITIAL state, which reads only the 1-D depths
    and must therefore be identical; kt=2's entry is the state after ONE
    step, which is the first thing the partial cells can touch.
    """
    out = {}
    for kt in (1, 2):
        a = read_entry(smt_dir / f"oracle_step_entry_kt{kt:08d}.bin", case)
        b = read_entry(flat_dir / f"oracle_step_entry_kt{kt:08d}.bin", case)
        per = {}
        for name in ("T", "u", "v", "ssh"):
            da = np.abs(np.asarray(a[name]) - np.asarray(b[name]))
            per[name] = float(da.max())
            if name == "ssh" and da.max() > 0.0:
                jj, ii = np.unravel_index(np.argmax(da), da.shape)
                per["ssh_argmax_ji"] = [int(jj), int(ii)]
        out[f"kt{kt}"] = per
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smt-kt-dir", required=True, type=Path)
    ap.add_argument("--smt-day100-dir", type=Path)
    ap.add_argument("--flat-kt-dir", type=Path,
                    help="the certified FLAT 30 km record of the same card")
    ap.add_argument("--case", default="VORTEX_SMT")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    report = {"case": args.case, "smt_kt_dir": str(args.smt_kt_dir)}
    report["geometry"] = geometry(args.smt_kt_dir)
    report["kt10"] = entry_extrema(args.smt_kt_dir, 10, args.case)
    report["kt10_restart"] = restart_state(args.smt_kt_dir, 10)
    if args.flat_kt_dir is not None:
        report["zps_vs_zco"] = one_step_vs_flat(
            args.smt_kt_dir, args.flat_kt_dir, args.case)
    if args.smt_day100_dir is not None:
        report["day100"] = restart_state(args.smt_day100_dir, 3000)
        report["ssh_extremum_track"] = drift(
            args.smt_day100_dir, (1, 10, 30, 60, 100))
        xs = [p["glamt_km"] for p in report["ssh_extremum_track"]]
        report["drifts_west"] = bool(xs[-1] < xs[0])
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
