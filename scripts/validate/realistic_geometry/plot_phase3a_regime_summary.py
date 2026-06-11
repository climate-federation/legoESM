#!/usr/bin/env python
"""Aggregate Phase 3a sweep results and produce regime-boundary plot.

Reads ``run.log`` from every
``results/realistic_geometry_validation/phase3a_seamount_*/`` directory
and produces a scatter of final ``max|u|`` vs r-factor max, with the
5 mm/s pass threshold and the B-H literature bound (r=0.2) overlaid.

Usage:
    python scripts/validate/realistic_geometry/plot_phase3a_regime_summary.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_ROOT = Path("results/realistic_geometry_validation")
OUT = RESULTS_ROOT / "phase3a_regime_summary.png"

KV = re.compile(r"^(\w+)\s*=\s*(.+)$")


def _parse_log(p: Path) -> dict:
    out = {}
    for line in p.read_text().splitlines():
        m = KV.match(line.strip())
        if m:
            k, v = m.group(1), m.group(2).strip()
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def main():
    runs = sorted(RESULTS_ROOT.glob("phase3a_seamount_*/run.log"))
    if not runs:
        print(f"No phase3a runs found in {RESULTS_ROOT}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(runs)} runs:")
    rows = []
    for p in runs:
        d = _parse_log(p)
        rows.append({
            "tag": p.parent.name.replace("phase3a_seamount_", ""),
            "smooth": int(d.get("smoothing_passes", 0)),
            "r_max": float(d["r_max"]),
            "u_mm": float(d["final_max_u_mm_per_s"]),
            "status": d["status"],
        })
        print(f"  {p.parent.name:<40s}  r_max={rows[-1]['r_max']:.3f}  "
              f"|u|={rows[-1]['u_mm']:.2f}  {rows[-1]['status']}")

    rows.sort(key=lambda r: r["r_max"])
    r_arr = np.array([r["r_max"] for r in rows])
    u_arr = np.array([r["u_mm"] for r in rows])
    pass_mask = np.array([r["status"] == "PASS" for r in rows])

    fig, ax = plt.subplots(figsize=(9, 6))
    # Connecting line
    ax.plot(r_arr, u_arr, "-", color="grey", lw=1, alpha=0.5, zorder=1)
    # Scatter, coloured by pass/fail
    ax.scatter(r_arr[pass_mask], u_arr[pass_mask], s=100,
                color="C2", edgecolor="black", label="PASS", zorder=2)
    ax.scatter(r_arr[~pass_mask], u_arr[~pass_mask], s=100,
                color="C3", edgecolor="black", label="FAIL", zorder=2)
    # Threshold lines
    ax.axhline(5.0, color="C0", ls="--", lw=1.2,
                label="Pass threshold (5 mm/s)")
    ax.axvline(0.2, color="C1", ls=":", lw=1.2,
                label="B-H literature bound (r=0.2)")
    # Axis
    ax.set_xlabel("r-factor max  $|H_i - H_j| / \\max(H_i, H_j)$")
    ax.set_ylabel("Final max|u| after 30 days (mm/s)")
    ax.set_yscale("log")
    ax.set_title(
        "Phase 3a Beckmann-Haidvogel seamount: regime boundary\n"
        "lat-lon C-grid ocean, 36×72 / 20 levels, implicit-CN, dt=600 s"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {OUT}")

    # Regime summary: largest passing r_max, smallest failing r_max
    if pass_mask.any():
        r_pass_max = r_arr[pass_mask].max()
    else:
        r_pass_max = float("nan")
    if (~pass_mask).any():
        r_fail_min = r_arr[~pass_mask].min()
    else:
        r_fail_min = float("nan")
    print(f"\n--- Regime boundary ---")
    print(f"Largest passing r_max:   {r_pass_max:.3f}")
    print(f"Smallest failing r_max:  {r_fail_min:.3f}")
    if r_pass_max < r_fail_min:
        print(f"=> Acceptable r_max for this model: < {r_pass_max:.2f}")


if __name__ == "__main__":
    main()
