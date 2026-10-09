"""Score the two-leaf canopy iteration-cap A/B (cap 10 vs 60), one variable.

Pre-registered (fv3_duo_gaps/q_review/deck_ledgers.md, 2026-10-04): the cap
binds every step in production and the vmapped solve runs to the slowest
column, so lowering it is a COST lever; the A/B asks whether the day-1 land
state and fluxes move.  Reuses the land/ocean masks and area means of
``amip_land_calibrated_ab`` and adds what a day-mean RMS hides (GLM): the
cell-wise p99 / max / signed-mean difference over land, the held-column
counts the land step logs, the wall time, and the checkpoint's own land
state (skin T, top soil T and water, canopy state).

Usage::

    python scripts/validate/land_canopy_cap_ab.py \\
        --control <run dir, cap 60> --arm <run dir, cap 10> \\
        [--control-log <slurm .out>] [--arm-log <slurm .out>]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from scripts.validate.amip_land_calibrated_ab import _masks, _mean, _open  # noqa: E402

_WALL_RE = re.compile(r"MPAS run COMPLETED in ([0-9.]+)s")
_HELD_RE = re.compile(r"land: (\d+) column-steps held in the last (\d+) steps "
                      r"\((\d+) of them on LAND columns")
_CKPT_LAND_FIELDS = (
    ("skin T (T_sfc override)", "physstate_surface_T_sfc_override", "K"),
    ("top soil T", "land_ml_T_soil", "K"),
    ("top soil water", "land_ml_theta_soil", "m3/m3"),
    ("canopy state", "land_ml_canopy_x", "-"),
)


def cellwise_stats(a, b, mask):
    """p99, max and RMS of |b - a| and the signed mean of (b - a) over ``mask``."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    m = np.asarray(mask, dtype=bool).ravel() & np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return float("nan"), float("nan"), float("nan"), float("nan")
    d = b[m] - a[m]
    return (float(np.percentile(np.abs(d), 99)), float(np.abs(d).max()),
            float(np.sqrt(np.mean(d * d))), float(d.mean()))


def parse_log(text):
    """(wall seconds or None, [(held, window, held_on_land), ...]) from a run log."""
    walls = [float(w) for w in _WALL_RE.findall(text)]
    held = [(int(h), int(w), int(hl)) for h, w, hl in _HELD_RE.findall(text)]
    return (max(walls) if walls else None), held


def _top_layer(arr):
    """Top soil layer of an (ncol, nz) field; (ncol,) fields pass through."""
    arr = np.asarray(arr, dtype=float)
    return arr[:, 0] if arr.ndim == 2 else arr.reshape(arr.shape[0], -1)[:, 0]


def _checkpoint(run):
    hits = sorted(glob.glob(os.path.join(run, "checkpoint_day_*.npz")))
    return np.load(hits[-1]) if hits else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--control", required=True, help="run dir, cap 60")
    ap.add_argument("--arm", required=True, help="run dir, cap 10")
    ap.add_argument("--control-log")
    ap.add_argument("--arm-log")
    args = ap.parse_args()

    root = os.path.commonpath([os.path.abspath(args.control), os.path.abspath(args.arm)])
    ctl = os.path.relpath(os.path.abspath(args.control), root)
    arm = os.path.relpath(os.path.abspath(args.arm), root)

    area, land, ocean = _masks(root, ctl)
    area_b, land_b, _ = _masks(root, arm)
    if not (np.allclose(area, area_b) and (land == land_b).all()):
        raise SystemExit("the two arms do not share a grid or a land mask.")

    print("== wall time and held (unconverged) column-steps")
    for name, log in (("cap60", args.control_log), ("cap10", args.arm_log)):
        if not log:
            continue
        wall, held = parse_log(open(log).read())
        tail = held[-4:]
        print(f"  {name}: wall {wall} s; held per window (last 4): "
              + ", ".join(f"{h}/{w} ({hl} land)" for h, w, hl in tail))

    print("== CMOR day/Amon means over LAND (area-weighted) and cell-wise |cap10-cap60|")
    print(f"  {'field':<22}{'cap60':>9}{'cap10':>9}{'diff':>8}"
          f"{'p99':>9}{'max':>9}{'rms':>9}{'bias':>9}")
    for label, var, table, scale, unit in (
            ("latent heat", "hfls", "Amon", 1.0, "W/m2"),
            ("sensible heat", "hfss", "Amon", 1.0, "W/m2"),
            ("near-surface T", "tas", "day", 1.0, "K"),
            ("precipitation", "pr", "day", 86400.0, "mm/d")):
        a, b = _open(root, ctl, var, table), _open(root, arm, var, table)
        if a is None or b is None:
            print(f"  {label:<22}(not written)")
            continue
        av = np.asarray(a.mean("time")) * scale
        bv = np.asarray(b.mean("time")) * scale
        p99, mx, rms, bias = cellwise_stats(av, bv, land)
        print(f"  {label:<22}{_mean(av, area, land):9.2f}{_mean(bv, area, land):9.2f}"
              f"{_mean(bv, area, land) - _mean(av, area, land):+8.2f}"
              f"{p99:9.3f}{mx:9.3f}{rms:9.3f}{bias:+9.3f}  {unit}")
    a, b = _open(root, ctl, "tas", "day"), _open(root, arm, "tas", "day")
    if a is not None and b is not None:
        av, bv = np.asarray(a.mean("time")), np.asarray(b.mean("time"))
        print(f"  ocean tas (control: must not move): "
              f"{_mean(bv, area, ocean) - _mean(av, area, ocean):+.3f} K")

    print("== checkpoint land state, cell-wise |cap10-cap60| over LAND")
    ca, cb = _checkpoint(args.control), _checkpoint(args.arm)
    if ca is None or cb is None:
        print("  (no checkpoint in one arm)")
        return 0
    for label, key, unit in _CKPT_LAND_FIELDS:
        if key not in ca.files or key not in cb.files:
            print(f"  {label:<26}(not in checkpoint)")
            continue
        fa, fb = _top_layer(ca[key]), _top_layer(cb[key])
        if fa.shape[0] != land.size:
            # the CMOR fx mask is the REGRIDDED output grid; the checkpoint is
            # native. Without a native land mask only an all-cell statistic
            # is honest (ocean cells carry the land model's discarded output).
            p99, mx, rms, bias = cellwise_stats(fa, fb, np.ones(fa.shape[0], bool))
            print(f"  {label:<26}p99 {p99:9.4f}  max {mx:9.4f}  rms {rms:9.4f}  "
                  f"bias {bias:+9.4f}  {unit}  [ALL native cells, no land mask]")
            continue
        p99, mx, rms, bias = cellwise_stats(fa, fb, land)
        print(f"  {label:<26}p99 {p99:9.4f}  max {mx:9.4f}  rms {rms:9.4f}  "
              f"bias {bias:+9.4f}  {unit}")
    print("\nA pass is DAY-1 equivalence only; a bias here accumulates in soil water "
          "over weeks (both decks must then carry the same cap).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
