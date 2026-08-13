#!/usr/bin/env python3
"""Tropical rain split into LAND and OCEAN, against GPCP.

A tropical-mean rain bias hides which surface it comes from, and the two have
different causes: over the ocean the surface is prescribed and evaporation is a
bulk-flux problem, while over land the convective trigger has to fire off a
diurnally heated surface the model computes itself. A deficit concentrated on
one side points somewhere very different from a uniform one.

The land mask is the model's OWN published sftlf, so there is no separate
land-sea convention to reconcile. Cells are split at 50 % land fraction and the
FRACTIONAL cell area is used for the weights, which matters at 5 degrees where
a coastal cell is genuinely part of each.

Usage: land_ocean_rain.py <run> [<run> ...] [--belt 30]
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import pathlib
import sys

import numpy as np
import xarray as xr

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
HEAVY = tc.HEAVY


def _sftlf(run, lat, lon):
    fs = sorted(glob.glob(f"{rb.ROOT}/{run}/cmor/fx/sftlf_fx_*.nc"))
    if not fs:
        raise SystemExit(f"FATAL: {run} publishes no sftlf; cannot split by surface")
    d = xr.open_dataset(fs[-1], decode_times=False)
    f = np.asarray(d["sftlf"], dtype=np.float64)
    if f.shape != (lat.size, lon.size):
        raise SystemExit(f"FATAL: sftlf is {f.shape}, rain grid is "
                         f"{(lat.size, lon.size)}")
    return f / 100.0 if np.nanmax(f) > 1.5 else f


def _stats(pr, w):
    tot = w.sum()
    if tot <= 0:
        return {"mean": np.nan, "heavy": np.nan}
    return {"mean": float((pr * w).sum() / tot),
            "heavy": float((w * (pr > HEAVY)).sum() / tot)}


def run(name, belt):
    md = rb._load_model(name, "pr")
    if md is None:
        raise SystemExit(f"FATAL: {name} publishes no pr")
    lat = np.asarray(md.lat, dtype=np.float64)
    lon = np.asarray(md.lon, dtype=np.float64)
    months = rb._month_labels(md)
    pm = np.asarray(md["pr"]).mean(axis=0) * 86400.0
    po = rb._ref_clim("pr", months, lat, lon, src=tc.GPCP) * 86400.0
    land = _sftlf(name, lat, lon)

    area = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], pm.shape)
    inbelt = np.broadcast_to((np.abs(lat) <= belt)[:, None], pm.shape)
    # fractional weights: a coastal cell belongs partly to each surface
    w_land = area * land * inbelt
    w_ocean = area * (1.0 - land) * inbelt

    out = {"run": name, "land_frac": float(w_land.sum() /
                                           (w_land.sum() + w_ocean.sum()))}
    for tag, w in (("land", w_land), ("ocean", w_ocean)):
        m, o = _stats(pm, w), _stats(po, w)
        out[f"{tag}_m"], out[f"{tag}_o"] = m["mean"], o["mean"]
        out[f"{tag}_hm"], out[f"{tag}_ho"] = m["heavy"], o["heavy"]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--belt", type=float, default=30.0)
    args = ap.parse_args(argv)

    rows = [run(n, args.belt) for n in args.runs]
    print(f"\n=== tropical rain by surface, |lat| <= {args.belt:.0f} "
          f"[mm/day, and area fraction above {HEAVY:.0f} mm/day] ===")
    print(f"  {'run':<13}{'land':>8}{'obs':>8}{'ratio':>7}"
          f"{'ocean':>9}{'obs':>8}{'ratio':>7}"
          f"{'heavy L':>9}{'obs':>7}{'heavy O':>9}{'obs':>7}")
    for r in rows:
        print(f"  {r['run']:<13}{r['land_m']:8.2f}{r['land_o']:8.2f}"
              f"{r['land_m']/max(r['land_o'],1e-9):7.2f}"
              f"{r['ocean_m']:9.2f}{r['ocean_o']:8.2f}"
              f"{r['ocean_m']/max(r['ocean_o'],1e-9):7.2f}"
              f"{r['land_hm']:9.3f}{r['land_ho']:7.3f}"
              f"{r['ocean_hm']:9.3f}{r['ocean_ho']:7.3f}")
    print(f"\n  ratio = model / observed. Land is "
          f"{rows[0]['land_frac']:.0%} of the belt's area, so a land deficit "
          f"is diluted\n  by roughly that factor in any tropical mean -- which "
          f"is why it needs its own row.\n  Weights are FRACTIONAL: a coastal "
          f"cell contributes to both surfaces in proportion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
