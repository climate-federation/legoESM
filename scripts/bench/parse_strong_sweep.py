"""Parse the steps/s reported by the strong-scaling sweep and emit a
machine-readable summary at ``results/scaling/strong_sweep.json``.

The sweep wrapper (``scripts/bench/run_strong_scaling_sweep.sh``) writes
``output/baroclinic_wave_diagnostics_<TAG>_<backend>_<grid>_<res>.npz``
plus stdout containing ``Integration complete: <S>s (<X> steps/s)``.
The benchmark itself doesn't store ``steps_per_sec`` in the npz, so
we derive it from the npz's ``nlev`` × ``dt`` and the total wall time
captured from the log if available, else just record the npz metadata.

Usage:
    python scripts/parse_strong_sweep.py [--log /tmp/sweep.log]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def _walk_npz(tag_glob="iter201_strong"):
    out_dir = Path("output")
    rows = []
    for p in sorted(out_dir.glob(f"baroclinic_wave_diagnostics_{tag_glob}_*.npz")):
        m = re.match(
            rf"baroclinic_wave_diagnostics_{tag_glob}_(cpu|gpu)_(spectral|icosahedral|cubed-sphere)_(\d+)\.npz",
            p.name,
        )
        if not m:
            continue
        backend, grid, res = m.group(1), m.group(2), int(m.group(3))
        d = np.load(p, allow_pickle=True)
        nlev = int(d["nlev"]) if "nlev" in d.files else None
        dt = float(d["dt"]) if "dt" in d.files else None
        last_day = float(d["times_days"][-1]) if "times_days" in d.files else None
        ncells = _grid_ncells(grid, res)
        # Prefer the npz-recorded throughput (added by the benchmark
        # itself); fall back to log-scraping further down if the npz
        # is from a pre-iter-202 run that didn't store it.
        sps_npz = (
            float(d["steps_per_sec"]) if "steps_per_sec" in d.files else None
        )
        wall_npz = (
            float(d["wall_time_s"]) if "wall_time_s" in d.files else None
        )
        backend_recorded = (
            str(d["backend"]) if "backend" in d.files else backend
        )
        rows.append({
            "backend": backend, "grid": grid, "resolution": res,
            "ncells": ncells, "nlev": nlev, "dt": dt,
            "days_simulated": last_day,
            "steps_per_sec": sps_npz,
            "wall_time_s": wall_npz,
            "backend_recorded": backend_recorded,
            "npz": str(p),
        })
    return rows


def _grid_ncells(grid: str, res: int) -> int:
    if grid == "spectral":
        # T<n_max>: Gaussian grid (3*n_max + 1) lat × (2*(3*n_max+1)) lon
        n_lat = 3 * res + 1
        return 2 * n_lat * n_lat
    if grid == "icosahedral":
        return 10 * 4**res + 2
    if grid == "cubed-sphere":
        return 6 * res * res
    raise ValueError(grid)


def _augment_with_log(rows, log_path):
    if log_path is None or not Path(log_path).exists():
        return rows
    text = Path(log_path).read_text()
    # Pattern: per cell, the run prints `Integration complete: ...s (X.X steps/s)`
    # following a line containing the tag.
    by_label = {}
    label = None
    for line in text.splitlines():
        m = re.search(r"--tag\s+'?([\w-]+)", line)
        if m:
            label = m.group(1)
            continue
        m2 = re.search(r"Integration complete:\s+\d+s\s+\(([\d.]+)\s+steps/s\)", line)
        if m2 and label:
            by_label[label] = float(m2.group(1))
            label = None
    for r in rows:
        tag = f"iter201_strong_{r['backend']}_{r['grid']}_{r['resolution']}"
        if r.get("steps_per_sec") is None and tag in by_label:
            r["steps_per_sec"] = by_label[tag]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="/tmp/sweep.log")
    parser.add_argument(
        "--out", default="results/scaling/strong_sweep.json",
    )
    args = parser.parse_args()
    rows = _walk_npz()
    rows = _augment_with_log(rows, args.log)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(rows, f, indent=2)
    # also pretty-print to stdout
    print(f"# {len(rows)} cells parsed → {out}")
    hdr = f"{'backend':4s}  {'grid':12s}  {'res':>4s}  {'ncells':>7s}  {'sps':>7s}  {'days':>5s}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        sps = r.get("steps_per_sec")
        sps_str = f"{sps:7.1f}" if sps is not None else "      —"
        ds = r.get("days_simulated") or 0.0
        print(f"{r['backend']:4s}  {r['grid']:12s}  {r['resolution']:>4d}  "
              f"{r['ncells']:>7d}  {sps_str}  {ds:5.2f}")


if __name__ == "__main__":
    main()
