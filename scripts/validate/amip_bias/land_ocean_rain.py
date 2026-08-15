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

The same split is applied to the SURFACE side, because a rain deficit that is
not a supply deficit has to show up there: screen temperature and evaporation
against ERA5, plus the model's own turbulent-flux partition. The ClimateEval
reference tree carries ERA5 ``tas`` and ``evspsbl`` but NOT the turbulent
fluxes, so the Bowen ratio and evaporative fraction are printed as MODEL-ONLY
numbers with no observational column -- they rank arms against each other, they
do not score the model against observations.

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


def _wmean(field, w):
    tot = w.sum()
    return float((field * w).sum() / tot) if tot > 0 else np.nan


# variable -> (scale to display units, unit label).  evspsbl is published as a
# mass flux and read as a depth rate so it lines up with the rain rows.
_SURF = {"tas": (1.0, "K"), "evspsbl": (86400.0, "mm/day"),
         "hfls": (1.0, "W/m2"), "hfss": (1.0, "W/m2")}


# Land is split again by the OBSERVED rain rate, so the mask carries no model
# quantity: below ARID_MM/day the surface is a desert whatever the model does,
# above HUMID_MM/day it is a rainforest or monsoon.  A whole-belt land mean
# averages the two together and is dominated by the deserts, which are most of
# the tropical land AREA and where a fixed evaporation efficiency is worst.
ARID_MM, HUMID_MM = 1.0, 3.0


def _weights(land, lat, shape, belt, pr_obs=None):
    """cos-lat area weights per surface class, FRACTIONAL in the land fraction."""
    area = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], shape)
    inbelt = np.broadcast_to((np.abs(lat) <= belt)[:, None], shape)
    w_land = area * land * inbelt
    out = {"land": w_land, "ocean": area * (1.0 - land) * inbelt}
    if pr_obs is not None:
        out["arid"] = w_land * (pr_obs < ARID_MM)
        out["humid"] = w_land * (pr_obs > HUMID_MM)
    return out


def surface(name, belt):
    """Screen temperature, evaporation and the flux partition, by surface class.

    ``tas`` and ``evspsbl`` are scored against ERA5.  ``hfls``/``hfss`` have no
    reference in the ClimateEval tree, so their ratio is reported MODEL-ONLY.
    """
    md0 = rb._load_model(name, "tas")
    if md0 is None:
        raise SystemExit(f"FATAL: {name} publishes no tas")
    lat = np.asarray(md0.lat, dtype=np.float64)
    lon = np.asarray(md0.lon, dtype=np.float64)
    months = rb._month_labels(md0)
    land = _sftlf(name, lat, lon)
    shape = (lat.size, lon.size)
    pr_obs = rb._ref_clim("pr", months, lat, lon, src=tc.GPCP) * 86400.0
    w = _weights(land, lat, shape, belt, pr_obs)

    out = {"run": name, "rain_obs": {t: _wmean(pr_obs, wt)
                                     for t, wt in w.items()}}
    for var, (scale, _unit) in _SURF.items():
        md = rb._load_model(name, var)
        fm = None if md is None else np.asarray(md[var]).mean(axis=0) * scale
        ref = None if md is None else rb._ref_clim(var, months, lat, lon)
        fo = None if ref is None else ref * scale
        for tag, wt in w.items():
            out[f"{var}_{tag}_m"] = np.nan if fm is None else _wmean(fm, wt)
            out[f"{var}_{tag}_o"] = np.nan if fo is None else _wmean(fo, wt)
    return out


def _ef(r, tag):
    le, h = r[f"hfls_{tag}_m"], r[f"hfss_{tag}_m"]
    return le / (le + h) if np.isfinite(le + h) and le + h != 0 else np.nan


def print_surface(rows, tags):
    for tag in tags:
        print(f"\n=== tropical {tag.upper()}: surface state "
              f"(tas/evap vs ERA5; fluxes MODEL-ONLY, no reference exists) ===")
        print(f"  {'run':<13}{'tas':>8}{'d_obs':>7}{'evap':>8}{'obs':>7}"
              f"{'ratio':>7}{'LE':>8}{'H':>7}{'EF':>6}{'rain_obs':>10}")
        for r in rows:
            em, eo = r[f"evspsbl_{tag}_m"], r[f"evspsbl_{tag}_o"]
            print(f"  {r['run']:<13}{r[f'tas_{tag}_m']:8.2f}"
                  f"{r[f'tas_{tag}_m'] - r[f'tas_{tag}_o']:+7.2f}"
                  f"{em:8.2f}{eo:7.2f}{em / max(eo, 1e-9):7.2f}"
                  f"{r[f'hfls_{tag}_m']:8.1f}{r[f'hfss_{tag}_m']:7.1f}"
                  f"{_ef(r, tag):6.2f}{r['rain_obs'][tag]:10.2f}")
    print(f"\n  d_obs = model - ERA5 [K].  evap in mm/day so it reads against "
          f"the rain rows\n  above: evaporation that matches while rain does "
          f"not is a conversion fault,\n  not a supply fault.  EF = LE/(LE+H), "
          f"the latent share of the turbulent flux --\n  model-only, ranks arms "
          f"against each other and NOT against observations.\n  arid / humid "
          f"are the land cells GPCP puts below {ARID_MM:.0f} and above "
          f"{HUMID_MM:.0f} mm/day.")


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

    ws = _weights(land, lat, pm.shape, belt, po)
    w_land, w_ocean = ws["land"], ws["ocean"]

    out = {"run": name, "land_frac": float(w_land.sum() /
                                           (w_land.sum() + w_ocean.sum())),
           "humid_frac": float(ws["humid"].sum() / max(w_land.sum(), 1e-30))}
    for tag, w in ws.items():
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
    print(f"  {'run':<13}{'humid':>8}{'obs':>8}{'ratio':>7}"
          f"{'land':>8}{'obs':>8}{'ratio':>7}"
          f"{'ocean':>9}{'obs':>8}{'ratio':>7}"
          f"{'heavy L':>9}{'obs':>7}{'heavy O':>9}{'obs':>7}")
    for r in rows:
        print(f"  {r['run']:<13}{r['humid_m']:8.2f}{r['humid_o']:8.2f}"
              f"{r['humid_m']/max(r['humid_o'],1e-9):7.2f}"
              f"{r['land_m']:8.2f}{r['land_o']:8.2f}"
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
    print_surface([surface(n, args.belt) for n in args.runs],
                  ("humid", "arid", "ocean"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
