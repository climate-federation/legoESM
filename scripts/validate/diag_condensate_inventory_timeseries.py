#!/usr/bin/env python
"""Condensate inventory versus time, read from a run's own checkpoints.

WHY THIS EXISTS.  A reservoir difference between two arms can arise two ways,
and they call for completely different instruments:

  * a SLOW LEAK -- one arm loses water steadily, so the gap grows with time and
    only a time-averaged in-run budget can attribute it;
  * a FIXED-POINT OFFSET -- both arms equilibrate quickly at different levels,
    so the gap is established within the reservoir's turnover time and never
    grows.  Cloud water turns over in hours, so this is the common case.

Telling them apart costs nothing once checkpoints exist, and gets the question
right before any budget is built.  Reading the deficit as a leak and dividing
it by the run length produces a target rate that is hundreds of times smaller
than every instantaneous budget term, which is what sent an earlier hunt for a
cause that did not exist.

The inventory is the shared column integral, not a re-derived sum
(CLAUDE.md shared-utilities rule).  Layer masses come from each checkpoint's
own stored vertical table, so hybrid and terrain-following arms are handled
without the caller having to know which is which.

Usage
-----
    diag_condensate_inventory_timeseries.py RUN_DIR [RUN_DIR ...]
        [--days 5,10,15] [--species trc_q_c,trc_q_i]

With two or more run directories the first is the reference and the ratio of
each other run to it is printed per date.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from legoesm import constants
from legoesm.diagnostics.column_integrals import column_mass_integral

#: Prognostic water species as the checkpoint writer names them.
DEFAULT_SPECIES = ("trc_q_c", "trc_q_i", "trc_q_r", "trc_q_s", "trc_q_g", "trc_q_v")

#: ``meta_vgrid`` holds the interface coefficients (A, B) of p = A*p_ref + B*p_s.
_VGRID_KEY = "meta_vgrid"


def layer_thickness(npz) -> np.ndarray:
    """Layer pressure thickness [Pa], shape (ncell, nlev), from the checkpoint.

    Raises on a table whose interfaces do not increase downward, rather than
    returning a plausible-looking number: a wrong layer mass multiplies both
    sides of a budget and cancels, so it can hide inside a closing budget.
    """
    A, B = np.asarray(npz[_VGRID_KEY], dtype=np.float64)
    p_s = np.asarray(npz["p_s"], dtype=np.float64)
    dp = np.diff(A)[None, :] * constants.p_ref + np.diff(B)[None, :] * p_s[:, None]
    if not (dp > 0.0).all():
        raise ValueError(
            f"{_VGRID_KEY} interfaces are not monotonically increasing downward; "
            "layer masses would be zero or negative"
        )
    p_top = A[0] * constants.p_ref + B[0] * p_s
    if not np.allclose(dp.sum(axis=1), p_s - p_top, rtol=1e-10):
        raise ValueError("layer masses do not sum to the column mass")
    return dp


def inventory(path: Path, species=DEFAULT_SPECIES) -> dict[str, float]:
    """Mesh-mean column burden [kg/m^2] of each species present in a checkpoint.

    Horizontal weighting is uniform.  The mesh is a quasi-uniform spherical
    centroidal tessellation and is identical across the arms this compares, so
    it cancels in any arm-to-arm ratio; it is NOT a valid global mean on a
    stretched or latitude-longitude grid.

    The shared column integral accumulates in the conservation dtype, which
    is single precision unless x64 is enabled, so treat the last digit as
    noise.  That is six orders below the reservoir differences this probe
    exists to resolve.
    """
    npz = np.load(path, allow_pickle=True)
    dp = layer_thickness(npz)
    p_s = np.asarray(npz["p_s"], dtype=np.float64)
    out = {}
    for name in species:
        if name not in npz.files:
            continue
        field = np.asarray(npz[name], dtype=np.float64)
        out[name] = float(np.asarray(column_mass_integral(field, p_s, None, dp=dp)).mean())
    return out


def series(run: Path, days, species=DEFAULT_SPECIES) -> dict[int, dict[str, float]]:
    out = {}
    for day in days:
        ckpt = run / f"checkpoint_day_{day:04d}.npz"
        if ckpt.exists():
            out[day] = inventory(ckpt, species)
    return out


def _drift(values, days) -> float:
    """Least-squares slope [kg/m^2/day]; a two-point difference is noise-prone."""
    return float(np.polyfit(np.asarray(days, float), np.asarray(values, float), 1)[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--days", default=",".join(str(d) for d in range(1, 91)),
                    help="comma-separated simulation days to look for")
    ap.add_argument("--species", default="trc_q_c,trc_q_i,trc_q_v")
    args = ap.parse_args(argv)

    days = [int(d) for d in args.days.split(",")]
    species = tuple(s.strip() for s in args.species.split(","))
    data = {run: series(run, days, species) for run in args.runs}
    missing = [str(r) for r, s in data.items() if not s]
    if missing:
        raise SystemExit(f"no checkpoints found for: {', '.join(missing)}")

    ref = args.runs[0]
    for name in species:
        print(f"\n== {name} [kg/m^2], mesh mean, uniform horizontal weight"
              f"{'; ref = ' + ref.name if len(args.runs) > 1 else ''}")
        header = f"{'day':>5}" + "".join(f"{r.name[:18]:>19}" for r in args.runs)
        if len(args.runs) > 1:
            header += "".join(f"{r.name[:10] + '/ref':>17}" for r in args.runs[1:])
        print(header)
        for day in sorted({d for s in data.values() for d in s}):
            row = f"{day:>5}"
            for run in args.runs:
                v = data[run].get(day, {}).get(name)
                row += f"{v:>19.5f}" if v is not None else f"{'-':>19}"
            for run in args.runs[1:]:
                a = data[run].get(day, {}).get(name)
                b = data[ref].get(day, {}).get(name)
                row += f"{a / b:>17.3f}" if a and b else f"{'-':>17}"
            print(row)
        for run in args.runs:
            got = sorted(d for d in data[run] if name in data[run][d])
            if len(got) >= 2:
                vals = [data[run][d][name] for d in got]
                print(f"      {run.name}: day {got[0]}-{got[-1]} "
                      f"trend {_drift(vals, got):+.2e} kg/m^2/day")
    return 0


if __name__ == "__main__":
    sys.exit(main())
