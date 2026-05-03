"""Walk every ``output/baroclinic_wave_diagnostics_*.npz`` file and
emit a tidy CSV + Markdown table of throughput per (tag, grid, res,
backend, dt, days, sps).  The npz format added ``steps_per_sec``,
``wall_time_s``, and ``backend`` in iter-202 so this runs purely from
the npz metadata — no log-scraping required.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/scaling_summary.py [--tag-glob iter204]
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


_NPZ_PATTERN = re.compile(
    r"baroclinic_wave_diagnostics_(.+)\.npz$"
)


def _grid_ncells(grid: str | None, res: int | None) -> int | None:
    if grid is None or res is None:
        return None
    if grid == "spectral":
        n_lat = 3 * res + 1
        return 2 * n_lat * n_lat
    if grid == "icosahedral":
        return 10 * 4 ** res + 2
    if grid == "cubed-sphere":
        return 6 * res * res
    return None


def _guess_grid_from_tag(tag: str) -> str | None:
    for g in ("spectral", "icosahedral", "cubed-sphere", "cubedsphere"):
        if g in tag:
            return g.replace("cubedsphere", "cubed-sphere")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag-glob", default="*",
        help="Filename glob over the npz tag (post-prefix portion).",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Directory holding the diagnostic npz files (default: output/)",
    )
    parser.add_argument(
        "--csv", default="results/scaling/throughput_summary.csv",
    )
    args = parser.parse_args()

    rows = []
    for p in sorted(Path(args.output_dir).glob(
        f"baroclinic_wave_diagnostics_{args.tag_glob}.npz",
    )):
        m = _NPZ_PATTERN.search(p.name)
        if not m:
            continue
        tag = m.group(1)
        d = np.load(p, allow_pickle=True)
        files = d.files
        sps = float(d["steps_per_sec"]) if "steps_per_sec" in files else None
        wall = float(d["wall_time_s"]) if "wall_time_s" in files else None
        n_steps = int(d["n_steps_total"]) if "n_steps_total" in files else None
        backend = str(d["backend"]) if "backend" in files else "?"
        res = int(d["resolution"]) if "resolution" in files else None
        nlev = int(d["nlev"]) if "nlev" in files else None
        dt = float(d["dt"]) if "dt" in files else None
        days = (
            float(d["times_days"][-1])
            if "times_days" in files and d["times_days"].size > 0
            else None
        )
        grid = _guess_grid_from_tag(tag)
        rows.append({
            "tag": tag, "grid": grid, "resolution": res, "nlev": nlev,
            "ncells": _grid_ncells(grid, res), "dt": dt, "days": days,
            "n_steps": n_steps, "wall_time_s": wall, "steps_per_sec": sps,
            "backend": backend,
        })

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"# {len(rows)} npz cells → {csv_path}")

    if not rows:
        return
    hdr = (
        f"{'tag':36s}  {'grid':12s}  {'res':>4s}  {'ncells':>7s}  "
        f"{'dt':>5s}  {'days':>5s}  {'sps':>7s}  {'backend':6s}"
    )
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['tag'][:36]:36s}  {r['grid'] or '?':12s}  "
            f"{r['resolution'] or 0:>4d}  {r['ncells'] or 0:>7d}  "
            f"{r['dt'] or 0:>5.0f}  {r['days'] or 0:>5.2f}  "
            f"{r['steps_per_sec'] or 0:>7.1f}  {r['backend']:6s}"
        )


if __name__ == "__main__":
    main()
