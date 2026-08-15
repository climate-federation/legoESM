#!/usr/bin/env python3
"""Tropical rain split into LAND and OCEAN, against GPCP.

A tropical-mean rain bias hides which surface it comes from, and the two have
different causes: over the ocean the surface is prescribed and evaporation is a
bulk-flux problem, while over land the convective trigger has to fire off a
diurnally heated surface the model computes itself. A deficit concentrated on
one side points somewhere very different from a uniform one.

The land mask is the model's OWN published sftlf, so there is no separate
land-sea convention to reconcile. There is no 50 % cut anywhere: the weights
are FRACTIONAL throughout, so a coastal cell contributes to each surface in
proportion, which matters at 5 degrees where such a cell is genuinely part of
both.

The same split is applied to the SURFACE side, because a rain deficit that is
not a supply deficit has to show up there: screen temperature and evaporation
against ERA5, plus the model's own turbulent-flux partition. The ClimateEval
reference tree carries ERA5 ``tas`` and ``evspsbl`` but NOT the turbulent
fluxes, so the Bowen ratio and evaporative fraction are printed as MODEL-ONLY
numbers with no observational column -- they rank arms against each other, they
do not score the model against observations.

WHAT THE arid / humid CLASSES REALLY ARE. They are cut on the observed rain
rate, but that rate has already been box-averaged onto the model grid, so a
coastal box is classified from a mixture of its land and its ocean. At 5
degrees that mixture is not a detail: an ITCZ coastline would enter the humid
class on the strength of the ocean rain beside it. The classes are therefore
restricted to boxes that are at least ``LAND_PURE`` land, and the surviving
land area fraction is printed next to the table so a shrunken sample is
visible rather than assumed away. The looser all-boxes number is printed
alongside as a sensitivity, because a conclusion that only holds on one of the
two masks is not a conclusion.

EVERY SCORED FIELD MUST COVER THE SAME MONTHS. The surface rows and the rain
rows are compared against each other -- "it rains what it evaporates" is the
whole point -- so a variable published for a different span than the rest is
FATAL, not silently averaged over its own window.

Usage: land_ocean_rain.py <run> [<run> ...] [--belt 30] [--land-pure 0.8]
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


def _same_grid(run, var, d, lat, lon):
    """FATAL unless ``d`` sits on exactly the grid every other field uses.

    A shape check alone passes a reversed or half-cell-shifted latitude axis,
    which would silently score the northern hemisphere against the southern.
    """
    for name, ref in (("lat", lat), ("lon", lon)):
        got = np.asarray(d[name], dtype=np.float64)
        if name == "lon":
            got = got % 360.0
        if got.shape != ref.shape or not np.allclose(got, ref, atol=1e-6):
            raise SystemExit(f"FATAL: {run} {var} is on a different {name} axis "
                             f"than the rest of the run")


def _sftlf(run, lat, lon):
    fs = sorted(glob.glob(f"{rb.ROOT}/{run}/cmor/fx/sftlf_fx_*.nc"))
    if not fs:
        raise SystemExit(f"FATAL: {run} publishes no sftlf; cannot split by surface")
    d = xr.open_dataset(fs[-1], decode_times=False)
    _same_grid(run, "sftlf", d, lat, lon)
    f = np.asarray(d["sftlf"], dtype=np.float64)
    f = f / 100.0 if np.nanmax(f) > 1.5 else f
    if not np.all(np.isfinite(f)) or f.min() < -1e-6 or f.max() > 1.0 + 1e-6:
        raise SystemExit(f"FATAL: {run} sftlf is not a finite fraction in [0,1] "
                         f"(min {np.nanmin(f):.3g}, max {np.nanmax(f):.3g})")
    return f


def _months(run, var, d):
    """Calendar months, rejecting a repeated one.

    A duplicated timestamp double-weights that month in the model mean AND in
    the reference selection, and nothing downstream can see it happened.
    """
    m = rb._month_labels(d)
    if len(set(m)) != len(m):
        raise SystemExit(f"FATAL: {run} {var} publishes a repeated month {m}")
    return m


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
# the ones printed under an "obs" header: a missing reference for these is a
# blank in a scored column, so it is FATAL rather than a nan
_REFERENCED = ("tas", "evspsbl")


# Land is split again by the OBSERVED rain rate, so the mask carries no model
# quantity: below ARID_MM/day the surface is a desert whatever the model does,
# above HUMID_MM/day it is a rainforest or monsoon.  A whole-belt land mean
# averages the two together and is dominated by the deserts, which are most of
# the tropical land AREA and where a fixed evaporation efficiency is worst.
#
# LAND_PURE is what keeps the cut honest.  The observed rain has already been
# box-averaged onto the model grid, so in a coastal box it is a land/ocean
# mixture; classifying only boxes that are almost entirely land removes that
# mixture at the cost of sample size.  Both masks are reported.
ARID_MM, HUMID_MM = 1.0, 3.0
LAND_PURE = 0.8


def _weights(land, lat, shape, belt, pr_obs=None, land_pure=LAND_PURE):
    """cos-lat area weights per surface class, FRACTIONAL in the land fraction."""
    area = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], shape)
    inbelt = np.broadcast_to((np.abs(lat) <= belt)[:, None], shape)
    w_land = area * land * inbelt
    out = {"land": w_land, "ocean": area * (1.0 - land) * inbelt}
    if pr_obs is not None:
        pure = w_land * (land >= land_pure)
        out["arid"] = pure * (pr_obs < ARID_MM)
        out["humid"] = pure * (pr_obs > HUMID_MM)
        # same cut with every box allowed in, so the coastal-mixture
        # sensitivity is a printed number rather than an assumption
        out["humid_all"] = w_land * (pr_obs > HUMID_MM)
        out["arid_all"] = w_land * (pr_obs < ARID_MM)
    return out


def surface(name, belt, land_pure=LAND_PURE):
    """Screen temperature, evaporation and the flux partition, by surface class.

    ``tas`` and ``evspsbl`` are scored against ERA5.  ``hfls``/``hfss`` have no
    reference in the ClimateEval tree, so their ratio is reported MODEL-ONLY.
    """
    md0 = rb._load_model(name, "tas")
    if md0 is None:
        raise SystemExit(f"FATAL: {name} publishes no tas")
    lat = np.asarray(md0.lat, dtype=np.float64)
    lon = np.asarray(md0.lon, dtype=np.float64) % 360.0
    months = _months(name, "tas", md0)
    land = _sftlf(name, lat, lon)
    shape = (lat.size, lon.size)
    pr_obs = rb._ref_clim("pr", months, lat, lon, src=tc.GPCP) * 86400.0
    w = _weights(land, lat, shape, belt, pr_obs, land_pure)

    out = {"run": name, "rain_obs": {t: _wmean(pr_obs, wt)
                                     for t, wt in w.items()},
           "months": months,
           "pure_frac": float(w["humid"].sum() / max(w["humid_all"].sum(),
                                                     1e-30))}
    for var, (scale, _unit) in _SURF.items():
        md = rb._load_model(name, var)
        if md is None:
            raise SystemExit(f"FATAL: {name} publishes no {var}; the surface "
                             f"table compares these rows against each other "
                             f"and cannot be built from a subset")
        _same_grid(name, var, md, lat, lon)
        if _months(name, var, md) != months:
            raise SystemExit(
                f"FATAL: {name} publishes {var} for months "
                f"{_months(name, var, md)} but tas for {months}; comparing "
                f"them would put different seasons in the same row")
        fm = np.asarray(md[var]).mean(axis=0) * scale
        ref = rb._ref_clim(var, months, lat, lon)
        if ref is None and var in _REFERENCED:
            raise SystemExit(f"FATAL: no reference climatology for {var}; it is "
                             f"printed under an 'obs' header and must not be "
                             f"blank")
        fo = None if ref is None else ref * scale
        for tag, wt in w.items():
            out[f"{var}_{tag}_m"] = _wmean(fm, wt)
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
          f"are boxes at least {100 * LAND_PURE:.0f}% land whose GPCP rain is "
          f"below {ARID_MM:.0f} / above {HUMID_MM:.0f} mm/day;\n  that keeps "
          f"{100 * rows[0]['pure_frac']:.0f}% of the humid land area and drops "
          f"the coastal boxes whose observed\n  rain is a land-ocean mixture. "
          f"The _all rows are the same cut with every box\n  allowed in -- if "
          f"the two disagree, the coastline is doing the talking.\n  Months "
          f"scored: {rows[0]['months']}, identical for every field in the "
          f"table.")


def run(name, belt, land_pure=LAND_PURE):
    md = rb._load_model(name, "pr")
    if md is None:
        raise SystemExit(f"FATAL: {name} publishes no pr")
    lat = np.asarray(md.lat, dtype=np.float64)
    lon = np.asarray(md.lon, dtype=np.float64) % 360.0
    months = _months(name, "pr", md)
    pm = np.asarray(md["pr"]).mean(axis=0) * 86400.0
    po = rb._ref_clim("pr", months, lat, lon, src=tc.GPCP) * 86400.0
    land = _sftlf(name, lat, lon)

    ws = _weights(land, lat, pm.shape, belt, po, land_pure)
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
    ap.add_argument("--land-pure", type=float, default=LAND_PURE,
                    help="minimum land fraction for a box to be classified "
                         "arid/humid, so the observed rain doing the "
                         "classifying is not a land-ocean mixture")
    args = ap.parse_args(argv)

    rows = [run(n, args.belt, args.land_pure) for n in args.runs]
    print(f"\n=== tropical rain by surface, |lat| <= {args.belt:.0f} "
          f"[mm/day, and area fraction above {HEAVY:.0f} mm/day] ===")
    print(f"  {'run':<13}{'humid':>8}{'obs':>8}{'ratio':>7}"
          f"{'land':>8}{'obs':>8}{'ratio':>7}"
          f"{'ocean':>9}{'obs':>8}{'ratio':>7}"
          f"{'humid_all':>11}{'ratio':>7}"
          f"{'heavy L':>9}{'obs':>7}{'heavy O':>9}{'obs':>7}")
    for r in rows:
        print(f"  {r['run']:<13}{r['humid_m']:8.2f}{r['humid_o']:8.2f}"
              f"{r['humid_m']/max(r['humid_o'],1e-9):7.2f}"
              f"{r['land_m']:8.2f}{r['land_o']:8.2f}"
              f"{r['land_m']/max(r['land_o'],1e-9):7.2f}"
              f"{r['ocean_m']:9.2f}{r['ocean_o']:8.2f}"
              f"{r['ocean_m']/max(r['ocean_o'],1e-9):7.2f}"
              f"{r['humid_all_m']:11.2f}"
              f"{r['humid_all_m']/max(r['humid_all_o'],1e-9):7.2f}"
              f"{r['land_hm']:9.3f}{r['land_ho']:7.3f}"
              f"{r['ocean_hm']:9.3f}{r['ocean_ho']:7.3f}")
    print(f"\n  ratio = model / observed. Land is "
          f"{rows[0]['land_frac']:.0%} of the belt's area, so a land deficit "
          f"is diluted\n  by roughly that factor in any tropical mean -- which "
          f"is why it needs its own row.\n  Weights are FRACTIONAL: a coastal "
          f"cell contributes to both surfaces in proportion.\n  humid = boxes "
          f"at least {100 * args.land_pure:.0f}% land with GPCP above "
          f"{HUMID_MM:.0f} mm/day; humid_all is the same cut\n  with the "
          f"coastal boxes left in, where the observed rain that classifies the "
          f"box\n  is partly the ocean's.")
    print_surface([surface(n, args.belt, args.land_pure) for n in args.runs],
                  ("humid", "arid", "ocean"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
