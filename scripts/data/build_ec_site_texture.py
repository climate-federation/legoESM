#!/usr/bin/env python
"""Build a per-EC-site soil-texture table (% sand, % clay) for the offline runs.

Source priority per site:
  1. FLUXNET/AmeriFlux BIF measured ``SOIL_TEX_SAND/CLAY`` (tower-specific), else
  2. SoilGrids v2.0 (ISRIC) top-soil mean (0-5 + 5-15 cm) queried by lat/lon.

Writes ``scripts/cluster/ec_site/ec_site_soil_texture.csv`` with columns
``site, lat, lon, sand_pct, clay_pct, usda_class, source``.  Consumed by
``run_ec_site.py --texture auto`` which maps (sand, clay) -> USDA van-Genuchten
hydraulics via ``legoesm.land.soil_texture``.

Usage
-----
    python scripts/data/build_ec_site_texture.py \
        --sites scripts/cluster/ec_site/fix_xsite_sites.txt \
        --driver-dir <DifferBESS>/data/sitelevel/nc
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.request

import numpy as np
import xarray as xr

# BIF measured top-soil texture (% sand, % clay), averaged over reported depths.
# Tower-specific -> preferred over the 250 m SoilGrids prediction.
_BIF = {
    "US-Ne1": (11.0, 37.0),   # silty clay loam (deep loess Mollisols)
    "US-MMS": (34.0, 63.0),   # clay
    "US-SRM": (79.4, 9.4),    # loamy sand (Santa Rita)
}
_SG_URL = ("https://rest.isric.org/soilgrids/v2.0/properties/query"
           "?lon={lon}&lat={lat}&property=sand&property=clay"
           "&depth=0-5cm&depth=5-15cm&value=mean")

# Documented top-soil (% sand, % clay) fallback where SoilGrids is unreachable and
# no BIF texture exists.  From site descriptions / literature (recorded per site).
_LIT = {
    "US-Whs": (63.7, 16.0),   # SoilGrids 0-15cm (direct query); gravelly sandy loam
    "US-Var": (30.0, 28.0),   # Vaira Ranch: Auburn rocky silt/clay loam (Baldocchi 2004)
    "US-Ton": (42.0, 25.0),   # Tonzi Ranch: Auburn very-rocky sandy-clay loam
    "DE-Geb": (10.0, 30.0),   # Gebesee: loess Chernozem (silty clay loam)
    "FR-Pue": (35.0, 38.0),   # Puechabon: clay over limestone (Rambal et al.)
}


def _latlon(driver_dir, site):
    d = xr.open_dataset(os.path.join(driver_dir, f"{site}_driver_v2.nc"))
    lat = float(np.asarray(d["LAT"].values).ravel()[0])
    lon = float(np.asarray(d["LONG"].values).ravel()[0])
    return lat, lon


def _soilgrids(lat, lon):
    """Top-soil (0-15 cm depth-weighted) % sand / % clay from SoilGrids v2.
    Single attempt (no in-process sleep — the harness blocks it); returns
    (sand_pct, clay_pct) or None on failure so the caller can retry next run.
    Values are g/kg -> /10 = %."""
    url = _SG_URL.format(lat=lat, lon=lon)
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            d = json.load(r)
        vals = {}
        for layer in d["properties"]["layers"]:
            name = layer["name"]                         # 'sand' / 'clay'
            dv = {dd["label"]: dd["values"]["mean"] for dd in layer["depths"]}
            parts = [(5.0, dv.get("0-5cm")), (10.0, dv.get("5-15cm"))]
            w = sum(wt for wt, v in parts if v is not None)
            vals[name] = sum(wt * v for wt, v in parts if v is not None) / w
        return vals["sand"] / 10.0, vals["clay"] / 10.0   # g/kg -> %
    except Exception as e:                                # noqa: BLE001
        print(f"    SoilGrids failed ({type(e).__name__}: {str(e)[:50]}) — retry next run")
        return None


def main() -> int:
    from legoesm.land.soil_texture import usda_texture_index, USDA_TEXTURES

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", default="scripts/cluster/ec_site/fix_xsite_sites.txt")
    ap.add_argument("--driver-dir",
                    default="/burg-archive/glab/users/jf3423/DifferBESS/data/sitelevel/nc")
    ap.add_argument("--out", default="scripts/cluster/ec_site/ec_site_soil_texture.csv")
    args = ap.parse_args()

    sites = [s.strip() for s in open(args.sites) if s.strip()]
    fields = ["site", "lat", "lon", "sand_pct", "clay_pct", "usda_class", "source"]
    # cache-and-resume: keep sites already fetched in a prior run (the SoilGrids
    # rate limit means several passes may be needed to cover every site).
    done = {}
    if os.path.isfile(args.out):
        with open(args.out) as fh:
            done = {r["site"]: r for r in csv.DictReader(fh)}
    for site in sites:
        if site in done:
            continue
        lat, lon = _latlon(args.driver_dir, site)
        if site in _BIF:
            sand, clay = _BIF[site]; src = "BIF"
        else:
            sg = _soilgrids(lat, lon)
            if sg is not None:
                sand, clay = sg; src = "SoilGrids"
            elif site in _LIT:
                sand, clay = _LIT[site]; src = "literature"
            else:
                continue                      # leave for the next pass
        cls = USDA_TEXTURES[int(usda_texture_index(float(sand), float(clay)))]
        done[site] = dict(site=site, lat=round(lat, 4), lon=round(lon, 4),
                          sand_pct=round(sand, 1), clay_pct=round(clay, 1),
                          usda_class=cls, source=src)
        print(f"  {site:9s}: sand={sand:4.1f}% clay={clay:4.1f}% -> {cls:15s} ({src})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows([done[s] for s in sites if s in done])
    missing = [s for s in sites if s not in done]
    print(f"\n  -> {args.out}  ({len(done)}/{len(sites)} sites"
          + (f"; missing (rate-limited, re-run): {missing}" if missing else ")"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
