#!/usr/bin/env python3
"""Does the tropical rain deficit survive refining the mesh?

The campaign has been ranking convection levers on a 379 km mesh. If the rain
biases those levers are tuned against are largely a property of that mesh, the
ranking does not transfer and the tuning is fitted to the grid.

The comparison is not as simple as scoring both runs. A refined run writes to a
finer output grid, and EVERY metric here is resolution dependent: the fraction
of the tropics raining above 5 mm/day rises on a finer grid for the observation
as well as the model, because averaging over a smaller box preserves more of
the extremes. Comparing a fine run's number against a coarse run's is therefore
a confound, not a result.

So both runs are scored TWICE:

* on the COMMON coarse grid, with the fine run block-averaged down to it and
  the reference binned to the same grid -- this isolates the model's mesh,
  which is the variable under test;
* each on its OWN output grid against the reference binned to that grid --
  the honest answer to "how good is the fine model", which is not comparable
  across the pair and is labelled so.

Usage: resolution_pair.py --coarse <run> --fine <run>
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np

_DIR = pathlib.Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tc = _load("tropical_circulation")
bm = _load("bias_maps")
rb = bm.rb


def _rain(run):
    md = rb._load_model(run, "pr")
    if md is None:
        raise SystemExit(f"FATAL: {run} publishes no pr")
    lat = np.asarray(md.lat, dtype=np.float64)
    lon = np.asarray(md.lon, dtype=np.float64)
    months = rb._month_labels(md)
    return np.asarray(md["pr"]).mean(axis=0) * 86400.0, lat, lon, months


def _block_mean(f, lat, lon, factor):
    """Area-weighted block average onto a grid ``factor`` times coarser.

    Exact for an integer factor, which is the only case allowed: a fractional
    coarsening would need interpolation and would not conserve the mean.
    """
    ny, nx = f.shape
    if ny % factor or nx % factor:
        raise SystemExit(
            f"FATAL: {ny}x{nx} does not divide by {factor}; the two runs' grids "
            "are not nested and cannot be compared on a common one")
    w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], f.shape)
    fw = (f * w).reshape(ny // factor, factor, nx // factor, factor)
    ww = w.reshape(ny // factor, factor, nx // factor, factor)
    out = fw.sum(axis=(1, 3)) / ww.sum(axis=(1, 3))
    la = lat.reshape(-1, factor).mean(axis=1)
    lo = lon.reshape(-1, factor).mean(axis=1)
    return out, la, lo


def _score(pr, lat, lon, months):
    ref = rb._ref_clim("pr", months, lat, lon, src=tc.GPCP) * 86400.0
    c, co = tc.concentration(pr, lat, lon), tc.concentration(ref, lat, lon)
    i, io = tc.itcz_metrics(pr, lat, lon), tc.itcz_metrics(ref, lat, lon)
    belt = np.abs(lat) <= 15
    r = float(np.corrcoef(pr[belt].ravel(), ref[belt].ravel())[0, 1])
    return {
        "heavy": c["frac_heavy"], "heavy_o": co["frac_heavy"],
        "mean": c["mean"], "mean_o": co["mean"],
        "peak": i["peak"], "peak_o": io["peak"], "r": r,
        "grid": f"{lat.size}x{lon.size}",
    }


def _row(tag, s):
    return (f"  {tag:<26}{s['grid']:>9}{s['heavy']:>9.3f}{s['heavy_o']:>9.3f}"
            f"{s['mean']:>9.2f}{s['mean_o']:>9.2f}{s['peak']:>9.2f}"
            f"{s['peak_o']:>9.2f}{s['r']:>8.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coarse", required=True)
    ap.add_argument("--fine", required=True)
    args = ap.parse_args(argv)

    pc, latc, lonc, mc = _rain(args.coarse)
    pf, latf, lonf, mf = _rain(args.fine)
    if sorted(set(mc)) != sorted(set(mf)):
        raise SystemExit(
            f"FATAL: the two runs cover different months ({sorted(set(mc))} vs "
            f"{sorted(set(mf))}) -- that is a second variable")

    factor = latf.size // latc.size
    if factor < 1 or latf.size % latc.size:
        raise SystemExit("FATAL: the fine grid is not an integer refinement "
                         "of the coarse one")

    head = (f"  {'':<26}{'grid':>9}{'heavy':>9}{'obs':>9}{'mean':>9}{'obs':>9}"
            f"{'peak':>9}{'obs':>9}{'rain r':>8}")
    print("\n=== LIKE FOR LIKE: both on the coarse grid, only the MESH differs ===")
    print(head)
    print(_row(f"{args.coarse} (coarse mesh)", _score(pc, latc, lonc, mc)))
    if factor > 1:
        pfc, la, lo = _block_mean(pf, latf, lonf, factor)
        print(_row(f"{args.fine} -> coarse grid", _score(pfc, la, lo, mf)))
    else:
        print(_row(f"{args.fine} (same grid)", _score(pf, latf, lonf, mf)))

    print("\n=== EACH ON ITS OWN OUTPUT GRID -- NOT comparable across the pair ===")
    print("  (the observation column moves with the grid too; that is the point)")
    print(head)
    print(_row(args.coarse, _score(pc, latc, lonc, mc)))
    print(_row(args.fine, _score(pf, latf, lonf, mf)))
    print("\n  heavy = fraction of the tropics above 5 mm/day; mean = tropical "
          "mean [mm/day];\n  peak = ITCZ peak [mm/day]; rain r = pattern "
          "correlation with GPCP over |lat|<=15.\n  Read the FIRST table for "
          "the effect of resolution. The second says how each run\n  scores in "
          "its own right, against an observation on its own grid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
