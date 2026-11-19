"""Copy a checkpoint with the snow-layer temperature shifted where snow exists.

Usage: perturb_snow.py <in.npz> <out.npz> --delta-K <float>
       perturb_snow.py --verify <pert_day277.npz> <ctrl_day277.npz>
           exit 0 only if T_snow is finite in both, differs, and is colder on
           average over the control's snow columns (the cooling was read and
           persisted); 1 otherwise.
Only ``land_ml_T_snow`` changes, and only in columns with
``land_ml_snow_depth > 0``; every other entry is copied byte for byte.
Refuses a checkpoint without the snow layer, a zero shift, a shift that
reaches no column, and an existing output file.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

KEY, SWE = "land_ml_T_snow", "land_ml_snow_depth"


def perturb(src, dst, delta_K):
    if delta_K == 0.0:
        raise SystemExit("delta-K must be non-zero")
    if Path(dst).exists():
        raise SystemExit(f"{dst} exists")
    z = np.load(src, allow_pickle=False)
    if KEY not in z.files or SWE not in z.files:
        raise SystemExit(f"{src} has no {KEY}/{SWE}: not a snow-layer checkpoint")
    out = {k: z[k] for k in z.files}
    snow = out[SWE] > 0.0
    if not snow.any():
        raise SystemExit("no snow-covered column to perturb")
    out[KEY] = np.where(snow, out[KEY] + delta_K, out[KEY]).astype(out[KEY].dtype)
    np.savez(dst, **out)
    return int(snow.sum())


def colder_after_restart(pert_path, ctrl_path):
    """Mean T_snow change (pert - ctrl) over the control's snow columns, or a reason it fails."""
    p, c = np.load(pert_path, allow_pickle=False), np.load(ctrl_path, allow_pickle=False)
    for z, name in ((p, pert_path), (c, ctrl_path)):
        if KEY not in z.files or SWE not in z.files:
            return None, f"{name} has no {KEY}/{SWE}"
        if not np.isfinite(z[KEY]).all():
            return None, f"{name} has non-finite {KEY}"
    snow = c[SWE] > 0.0
    if not snow.any():
        return None, "no snow column in the control"
    d = (p[KEY].astype(np.float64) - c[KEY].astype(np.float64))[snow]
    if not d.any():
        return None, f"{KEY} identical on all {int(snow.sum())} snow columns"
    return float(d.mean()), None


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["--verify"]:
        if len(sys.argv) != 4:
            raise SystemExit("usage: perturb_snow.py --verify <pert.npz> <ctrl.npz>")
        mean, why = colder_after_restart(sys.argv[2], sys.argv[3])
        if why:
            print(f"FAIL: {why}")
            sys.exit(1)
        print(f"{'PASS' if mean < 0 else 'FAIL'}: mean T_snow change {mean:+.4g} K on snow columns")
        sys.exit(0 if mean < 0 else 1)
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--delta-K", type=float, required=True)
    a = ap.parse_args()
    print(f"perturbed {perturb(a.src, a.dst, a.delta_K)} snow columns by {a.delta_K} K")
