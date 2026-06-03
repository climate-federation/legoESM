"""Compare two BCW diagnostic NPZ files — speedup, drift agreement, and
final-state norm.  Useful for regression-testing benchmark optimisations
(e.g. ``--scan-steps`` tuning) and for cross-checking CPU vs GPU runs.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/compare_scaling_runs.py \\
        output/baroclinic_wave_diagnostics_iter208_default_v2.npz \\
        output/baroclinic_wave_diagnostics_iter208_scan24_v2.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _safe_pct(num: float, den: float) -> str:
    if abs(den) < 1e-30:
        return "—"
    return f"{(num / den - 1.0) * 100:+.2f}%"


def _load(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"missing: {path}")
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--rel-drift-tol", type=float, default=1e-3,
        help="Max relative-drift agreement between the two runs "
             "(absolute fp32 storage tolerance, default 1e-3).",
    )
    args = parser.parse_args()

    a = _load(args.baseline)
    b = _load(args.candidate)

    # Per-row metadata.
    def _meta(d, label):
        sps = float(d.get("steps_per_sec", 0))
        wt  = float(d.get("wall_time_s", 0))
        ns  = int(d.get("n_steps_total", 0)) if "n_steps_total" in d else 0
        be  = str(d.get("backend", "?"))
        nl  = int(d.get("nlev", 0)) if "nlev" in d else 0
        res = int(d.get("resolution", 0)) if "resolution" in d else 0
        dt  = float(d.get("dt", 0))
        ndiag = len(d["times_days"]) if "times_days" in d else 0
        print(f"  {label:14s} backend={be:4s}  res={res:>4d}  nlev={nl:>3d}  "
              f"dt={dt:>5.0f}s  n_steps={ns:>4d}  diag_samples={ndiag:>4d}  "
              f"wall={wt:>6.2f}s  sps={sps:>7.2f}")
        return sps, wt, ns

    print("Comparison:")
    sps_a, wall_a, ns_a = _meta(a, "baseline")
    sps_b, wall_b, ns_b = _meta(b, "candidate")
    if sps_a > 0:
        print(f"\n  speedup = candidate/baseline = {sps_b/sps_a:.3f}× "
              f"({_safe_pct(sps_b, sps_a)})")

    # Conservation series agreement (final point).
    keys = [("dry_mass", 1e-12), ("total_energy", 1e-9),
            ("ps_min", 1.0), ("max_wind", 1.0)]
    print("\n  Final-step diagnostic agreement:")
    print(f"  {'metric':14s}  {'baseline':>15s}  {'candidate':>15s}  {'diff':>15s}  {'rel':>8s}")
    for k, _abs_tol in keys:
        if k not in a or k not in b:
            continue
        va = float(a[k][-1]) if a[k].ndim > 0 else float(a[k])
        vb = float(b[k][-1]) if b[k].ndim > 0 else float(b[k])
        diff = vb - va
        rel = diff / va if abs(va) > 1e-30 else float("nan")
        print(f"  {k:14s}  {va:15.6e}  {vb:15.6e}  {diff:+15.3e}  {rel:+8.2e}")

    # Drift-of-drift check: compare the scalar relative drift of each.
    if "dry_mass" in a and "dry_mass" in b:
        dm_a = a["dry_mass"]; dm_b = b["dry_mass"]
        if dm_a.ndim > 0 and dm_a.size > 1 and dm_b.size > 1:
            rd_a = abs((dm_a[-1] - dm_a[0]) / dm_a[0])
            rd_b = abs((dm_b[-1] - dm_b[0]) / dm_b[0])
            agree = abs(rd_a - rd_b) < args.rel_drift_tol
            print(f"\n  Mass relative-drift agreement: "
                  f"baseline={rd_a:.3e}, candidate={rd_b:.3e}, "
                  f"|diff|={abs(rd_a-rd_b):.3e}  "
                  f"({'OK' if agree else 'DIVERGED'})")


if __name__ == "__main__":
    main()
