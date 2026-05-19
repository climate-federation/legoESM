#!/usr/bin/env python
"""Concatenate diagnostics and monthly means from a chained AMIP run.

Walks ``{base}/y*_s*/`` subdirectories in chronological order (sorted by
year then segment number), loads ``diagnostics.npz`` and
``monthly_means.npz`` from each, and concatenates along the time axis into
combined files in ``{base}/``.

Usage::

    python scripts/concat_amip_output.py --base /scratch/b/b309178/amip_chain

    # Only concatenate completed segments (skip missing files silently):
    python scripts/concat_amip_output.py --base /scratch/b/b309178/amip_chain --skip-missing

Output files
------------
``{base}/diagnostics_combined.npz``
    All per-step scalar diagnostics concatenated along the time axis.
    Arrays keyed by ``days`` (time coordinate) plus T_atm, T_low,
    max_wind, mass, cwv, energy_toa_net, energy_column, etc.

``{base}/monthly_means_combined.npz``
    All monthly means concatenated.  Arrays keyed by ``years``,
    ``month_nums``, ``lat``, and all 2D/3D field arrays.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _sorted_seg_dirs(base: Path) -> list[Path]:
    """Return y{year}_s{seg} subdirs in chronological order."""
    dirs = sorted(
        base.glob("y[0-9][0-9][0-9][0-9]_s[0-9][0-9]"),
        key=lambda p: (p.name[1:5], p.name[6:8]),
    )
    return dirs


def _concat_npz(paths: list[Path], time_key: str = "days") -> dict[str, np.ndarray]:
    """Load and concatenate NPZ files along a shared time axis."""
    parts: list[dict[str, np.ndarray]] = []
    for p in paths:
        d = dict(np.load(str(p), allow_pickle=False))
        parts.append(d)

    if not parts:
        return {}

    # Keys present in all files
    common_keys = set(parts[0].keys())
    for d in parts[1:]:
        common_keys &= set(d.keys())

    n_time = [len(d[time_key]) for d in parts if time_key in d]
    time_len_consistent = all(len(d[time_key]) > 0 for d in parts if time_key in d)

    result: dict[str, np.ndarray] = {}
    for key in sorted(common_keys):
        arrays = [d[key] for d in parts]
        first = arrays[0]

        if first.ndim == 0:
            # Scalar — keep from first file
            result[key] = first
            continue

        try:
            result[key] = np.concatenate(arrays, axis=0)
        except ValueError:
            # Shape mismatch on non-time axes — keep first only with a warning
            print(f"  WARNING: cannot concat key '{key}' (shape mismatch), keeping first segment")
            result[key] = first

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base", type=str, required=True,
                        help="Base directory containing y*_s* segment subdirs")
    parser.add_argument("--skip-missing", action="store_true", default=False,
                        help="Skip segments where diagnostics.npz is absent")
    args = parser.parse_args(argv)

    base = Path(args.base)
    if not base.is_dir():
        print(f"ERROR: {base} is not a directory")
        return 1

    seg_dirs = _sorted_seg_dirs(base)
    if not seg_dirs:
        print(f"No y*_s* subdirectories found in {base}")
        return 1

    print(f"Found {len(seg_dirs)} segments in {base}")

    # --- diagnostics.npz ---
    diag_paths = []
    for d in seg_dirs:
        p = d / "diagnostics.npz"
        if p.exists():
            diag_paths.append(p)
        elif not args.skip_missing:
            print(f"  MISSING: {p}  (use --skip-missing to ignore)")
            return 1
        else:
            print(f"  SKIP (missing): {p}")

    if diag_paths:
        print(f"Concatenating {len(diag_paths)} diagnostics files...")
        diag = _concat_npz(diag_paths, time_key="days")
        out = base / "diagnostics_combined.npz"
        np.savez(str(out), **diag)
        n = len(diag.get("days", []))
        print(f"  Saved {out}  ({n} time steps, "
              f"days {float(diag['days'][0]):.1f}–{float(diag['days'][-1]):.1f})")

    # --- monthly_means.npz ---
    mm_paths = []
    for d in seg_dirs:
        p = d / "monthly_means.npz"
        if p.exists():
            mm_paths.append(p)
        elif not args.skip_missing:
            print(f"  MISSING: {p}  (use --skip-missing to ignore)")
            return 1
        else:
            print(f"  SKIP (missing): {p}")

    if mm_paths:
        print(f"Concatenating {len(mm_paths)} monthly_means files...")
        mm = _concat_npz(mm_paths, time_key="years")
        out = base / "monthly_means_combined.npz"
        # 'lat' is the same in every file — keep scalar (already handled above)
        np.savez(str(out), **mm)
        n = len(mm.get("years", []))
        if n:
            y0, y1 = int(mm["years"][0]), int(mm["years"][-1])
            m0, m1 = int(mm["month_nums"][0]), int(mm["month_nums"][-1])
            print(f"  Saved {out}  ({n} months, {y0}-{m0:02d} to {y1}-{m1:02d})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
