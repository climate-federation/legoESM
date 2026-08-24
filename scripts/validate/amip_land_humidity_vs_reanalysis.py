#!/usr/bin/env python
"""Is the coupled model's near-surface air over land drier than the reanalysis?

WHY THIS MATTERS.  The land model gets good latent heat and photosynthesis when
it is driven by reanalysis (the LMIP simulations), and much less when the SAME
land model is driven by the model's own atmosphere.  So the difference is in how
the land is driven, not in the land model.  Near-surface humidity is the obvious
suspect because it enters the land TWICE:

  * the coupled cold start seeds soil water from it directly
    (``aridity_theta_init``: availability at day 0 EQUALS near-surface RH), and
  * it sets the vapour-pressure deficit the stomata respond to.

A dry model boundary layer would therefore depress soil water and close stomata
at the same time, which is the observed signature.  This measures whether it is.

WHAT IS COMPARED, and the honest caveats.  RELATIVE humidity, because it is what
both consumers above actually read, and because it is far less sensitive than
specific humidity to the height mismatch below.  Land only, area-weighted by the
model's own cell area and masked by its own land fraction.

  * HEIGHT: the model value is its LOWEST PRESSURE LEVEL (1000 hPa) and the
    reanalysis value is its near-surface (~2 m) field.  These are not the same
    height.  Named because comparing across a staggering is a documented way to
    manufacture a result; RH varies far less with height in the surface layer
    than q does, but the mismatch is real and bounds how finely this reads.
  * The model's pressure axis runs SURFACE FIRST (100000 Pa) to top last, so the
    surface level is index 0.  Taking index -1 gives the stratosphere and an
    apparent relative humidity of 0.002 — which is how the first version of this
    script failed, loudly enough to catch.
  * YEAR: the coupled run is January 1979 and the staged reanalysis ends in 1978,
    so this compares January 1979 against January 1978 — different weather, same
    season.  Read the LARGE-SCALE difference, not a few percent.
  * The saturation humidity on BOTH sides comes from the model's own thermo, so
    no re-derived saturation curve can drift between them.
  * The reanalysis is a LAND-ONLY dataset: 57 % of its cells are missing, and
    those missing cells ARE the ocean.  They are masked explicitly on ``isfinite``
    rather than swept up by a ``nanmean``, so a genuinely broken field would show
    as an empty band instead of a plausible number.

Run: PYTHONPATH=. python scripts/validate/amip_land_humidity_vs_reanalysis.py \
         --arm iso_wet_off
"""

from __future__ import annotations

import argparse
import glob
import sys

import numpy as np

_ROOT = "/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs"
_LAND_MIN_FRAC = 0.8
_JAN_STEPS = 4 * 31          # CRU-JRA is 6-hourly: January = 124 records


def _rh(q, T, p):
    """Relative humidity from the MODEL's own saturation law (one source)."""
    from legoesm.thermo import saturation_mixing_ratio
    q_sat = np.asarray(saturation_mixing_ratio(np.asarray(T), np.asarray(p)))
    return np.asarray(q) / np.maximum(q_sat, 1e-12)


def main() -> int:
    import xarray as xr

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=_ROOT)
    ap.add_argument("--arm", required=True, help="coupled run directory name")
    ap.add_argument("--reanalysis", default="data/crujra",
                    help="CRU-JRA directory (the forcing LMIP drives the land with)")
    args = ap.parse_args()

    def _v(name, table="Amon"):
        hits = sorted(glob.glob(
            f"{args.root}/{args.arm}/cmor/{table}/{name}_{table}_*.nc"))
        if not hits:
            raise SystemExit(f"{args.arm}: no {name} in cmor/{table}")
        return xr.open_dataset(hits[0])[name]

    sftlf = np.asarray(_v("sftlf", "fx")).astype(float)
    if np.nanmax(sftlf) > 1.5:
        sftlf = sftlf / 100.0
    area = np.asarray(_v("areacella", "fx")).astype(float)
    land = sftlf >= _LAND_MIN_FRAC

    # Model: the LOWEST pressure level, which is index 0 on this axis (it runs
    # surface-first).  Humidity, temperature and pressure all taken AT that same
    # level, so the three are mutually consistent — pairing a 1000 hPa humidity
    # with a surface pressure would be a different kind of wrong.
    hus = _v("hus").mean("time")
    lev_name = [d for d in hus.dims if d not in ("lat", "lon", "ncol")][0]
    plev = np.asarray(hus[lev_name])
    if plev[0] < plev[-1]:
        raise SystemExit(
            f"pressure axis runs top-first ({plev[0]} .. {plev[-1]}); this script "
            "assumes surface-first and would otherwise read the stratosphere.")
    q_m = np.asarray(hus.isel({lev_name: 0}))
    T_m = np.asarray(_v("ta").mean("time").isel({lev_name: 0}))
    p_m = float(plev[0])
    rh_m = _rh(q_m, T_m, np.full_like(q_m, p_m))
    print(f"model level: {p_m/100:.0f} hPa\n")

    # Reanalysis: the same three fields, January mean.
    hits = sorted(glob.glob(f"{args.reanalysis}/*TPQWL*.nc"))
    if not hits:
        raise SystemExit(f"no TPQWL files under {args.reanalysis}")
    d = xr.open_dataset(hits[-1], decode_times=False)
    sel = dict(time=slice(0, _JAN_STEPS))
    q_r = np.asarray(d["QBOT"].isel(**sel).mean("time"))
    T_r = np.asarray(d["TBOT"].isel(**sel).mean("time"))
    p_r = np.asarray(d["PSRF"].isel(**sel).mean("time"))
    rh_r = _rh(q_r, T_r, p_r)
    lat_r = np.asarray(d["LATIXY"])[:, 0]

    # Compared as LAND ZONAL MEANS, not cell by cell: the two are on different
    # grids and a band mean needs no interpolation to get wrong.  Each side uses
    # its OWN land definition — the model its land fraction, the reanalysis its
    # missing-data pattern, which for this land-only dataset is the ocean.
    lat_m = np.asarray(_v("tas").lat)
    lat_m = (np.broadcast_to(lat_m.reshape(-1, 1), rh_m.shape)
             if lat_m.ndim == 1 and rh_m.ndim == 2 else lat_m)
    land_r = np.isfinite(rh_r)

    def _band(lo, hi):
        m = land & (lat_m >= lo) & (lat_m < hi) & np.isfinite(rh_m)
        model = (float((rh_m[m] * area[m]).sum() / area[m].sum())
                 if m.any() else np.nan)
        w2 = np.broadcast_to(
            np.cos(np.deg2rad(lat_r)).reshape(-1, 1), rh_r.shape)
        band = np.broadcast_to(
            ((lat_r >= lo) & (lat_r < hi)).reshape(-1, 1), rh_r.shape)
        k = band & land_r
        rean = float((rh_r[k] * w2[k]).sum() / w2[k].sum()) if k.any() else np.nan
        return model, rean

    print(f"Near-surface RELATIVE humidity over land, {args.arm} vs reanalysis")
    print("(model = lowest model level, reanalysis = near-surface; see caveats)\n")
    print(f"{'band':<22}{'model':>8}{'reanalysis':>12}{'difference':>12}")
    for lo, hi, name in ((-23, 23, "tropics"), (23, 50, "N subtropics/mid"),
                         (50, 90, "N high lat"), (-50, -23, "S subtropics/mid"),
                         (-90, 90, "all land")):
        m, r = _band(lo, hi)
        print(f"{name:<22}{m:8.3f}{r:12.3f}{m - r:+12.3f}")

    m_all, r_all = _band(-90, 90)
    print(f"\nModel land air is {'DRIER' if m_all < r_all else 'MOISTER'} than the "
          f"reanalysis by {abs(m_all - r_all):.3f} in relative humidity.")
    print("Soil water at the coupled cold start is seeded AS this relative")
    print("humidity, so a difference here transfers one-for-one into the")
    print("fraction of plant-available water the land model starts with.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
