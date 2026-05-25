"""Aggregate 100-day RCEMIP long-run JSON outputs across grids.

Reads ``results/rcemip100/<grid>.json`` for each NH grid and prints
a side-by-side stability + conservation table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GRIDS = ("plane_fd", "plane_spectral", "cubed_sphere", "mpas")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path,
                   default=Path("results/rcemip100"))
    args = p.parse_args()

    rows = []
    for grid in GRIDS:
        path = args.results / f"{grid}.json"
        if not path.exists():
            rows.append((grid, "MISSING", None, None, None, None, None))
            continue
        with path.open() as f:
            data = json.load(f)
        hist = data["history"]
        final = hist[-1]
        mass_drifts = [h["mass_drift"] for h in hist]
        max_w_final = final["max_w"]
        max_th_final = final["max_th"]
        max_drift = max(mass_drifts)
        t_days = final["t_days"]
        wall_min = data["wall_sec"] / 60.0
        blowup = data["blowup"]
        rows.append((
            grid,
            "BLOWUP" if blowup else "OK",
            t_days, max_w_final, max_th_final, max_drift, wall_min,
        ))

    th_label = "max|theta'|"
    print(f"{'grid':<18} {'status':<8} {'t_d':>6} {'max|w|':>10} "
          f"{th_label:>11} {'max_drift':>12} {'wall_min':>9}")
    print("-" * 80)
    for row in rows:
        grid, status, t_d, mw, mt, drift, wall = row
        if t_d is None:
            print(f"{grid:<18} {status:<8} {'-':>6} {'-':>10} "
                  f"{'-':>10} {'-':>12} {'-':>9}")
            continue
        print(f"{grid:<18} {status:<8} {t_d:6.1f} {mw:10.3e} "
              f"{mt:10.3e} {drift:12.3e} {wall:9.2f}")


if __name__ == "__main__":
    main()
