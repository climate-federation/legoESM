"""Tropical moisture scorecard for AMIP runs: water budget, column vapour,
rain and the humidity profile against ERA5 for the same month.

Reads each run's CMOR monthly output (``<run>/cmor/{Amon,fx}``) and the
ClimateEval ERA5 input tree.  Per run, prints:

- GLOBAL area-weighted P, E, P - E and prw (a closed atmosphere has
  P - E + dW/dt ~ 0; prw from two consecutive months gives dW/dt).
- 20S-20N ocean (sftlf < 10 %) and land (> 90 %) means of prw, pr, evspsbl,
  and the q / RH / T profile beside ERA5 (ERA5 coarsened to the model grid).

RAIN IS SCORED AGAINST GPCP ONLY (PI directive 2026-09-26: reanalysis rain is
not trusted), same month AND year, area-weighted onto the model grid.  Outside
the GPCP record rain is printed as NOT SCORED -- never a climatology or ERA5
substitute.

NUMBERS ONLY -- no verdict.  RH uses ``legoesm.thermo.relative_humidity`` on
the monthly-mean q and T (not the mean of instantaneous RH) on both sides.
"""
from __future__ import annotations

import argparse
import glob

import numpy as np

ERA5_ROOT = "/work/bd1179/b309141/climateeval_input/reanalysis_ERA5/mon"
GPCP_ROOT = "/work/bd1179/b309141/climateeval_input/observation_GPCP/mon/pr"
_DAY = 86400.0


def area_mean(x, w, mask):
    x = np.asarray(x, float)
    ok = mask & np.isfinite(x)
    return float((x * w)[ok].sum() / w[ok].sum())


def budget_gap(p_mm_d, e_mm_d, prw_prev, prw_now, days):
    """P - E + dW/dt [mm/d]; zero for a closed atmosphere."""
    return (p_mm_d - e_mm_d) + (prw_now - prw_prev) / days


def band_means(x, lat, area, mask, edges):
    """Area-weighted mean of x in each [lo, hi) latitude band under mask."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = mask & ((lat >= lo) & (lat < hi))[:, None]
        out.append(area_mean(x, area, m) if m.any() else float("nan"))
    return out


def print_bands(run, lat, lon, area, lf, P, prw, P_ref, prw_ref):
    edges = list(range(-40, 41, 10))
    full = np.ones((lat.size, lon.size), bool)
    for name, mk in (("ocean", full & (lf < 0.1)), ("land", full & (lf > 0.9))):
        for v, x, r, src in (("pr", P, P_ref, "GPCP"), ("prw", prw, prw_ref, "ERA5")):
            if r is None:
                print(f"[{run}] bands {v} {name}: NOT SCORED (no {src} for this month)")
                continue
            m, o = band_means(x, lat, area, mk, edges), band_means(r, lat, area, mk, edges)
            print(f"[{run}] bands {v} {name} model-{src} " + " ".join(
                f"{lo:+d}..{hi:+d}:{mm - oo:+.2f}" for lo, hi, mm, oo in zip(edges[:-1], edges[1:], m, o)))


def block_area_mean(x, lat_fine, factor):
    """cos(lat)-weighted mean of factor x factor blocks: the area-conservative
    average of nested regular cells onto the coarse grid."""
    w = np.cos(np.deg2rad(lat_fine))[:, None] * np.ones((1, x.shape[1]))
    ny, nx = x.shape[0] // factor, x.shape[1] // factor
    xs = (x * w).reshape(ny, factor, nx, factor).sum(axis=(1, 3))
    return xs / w.reshape(ny, factor, nx, factor).sum(axis=(1, 3))


def gpcp_pr_mm_day(lat, lon, year, month):
    """GPCP monthly rain for year-month on the model grid [mm/d], or None when
    the month is outside the record.  Requires the model cells to be exact
    unions of GPCP cells (asserted)."""
    import xarray as xr
    g = xr.open_dataset(sorted(glob.glob(f"{GPCP_ROOT}/*.nc"))[0]).pr.sortby("lat")
    if g.attrs.get("units") != "kg m-2 s-1":
        raise ValueError(f"GPCP pr units {g.attrs.get('units')!r}, expected kg m-2 s-1")
    sel = g.isel(time=np.flatnonzero((g.time.dt.year == year).values
                                     & (g.time.dt.month == month).values))
    if sel.time.size == 0:
        return None
    f = int(round(float(lat[1] - lat[0]) / float(g.lat[1] - g.lat[0])))
    coarse_lat = g.lat.values.reshape(-1, f).mean(axis=1)
    coarse_lon = g.lon.values.reshape(-1, f).mean(axis=1)
    if not (np.allclose(coarse_lat, lat) and np.allclose(coarse_lon, lon)):
        raise ValueError("model grid is not a nesting of the GPCP grid")
    return block_area_mean(sel.isel(time=0).values, g.lat.values, f) * _DAY


def _coarsen_to(x, lat, lon):
    x = x.sortby("lat")
    if x.lat.size > lat.size:
        n = x.lat.size // lat.size
        x = x.isel(lat=slice(0, lat.size * n)).coarsen(
            lat=n, lon=x.lon.size // lon.size, boundary="trim").mean()
        return x.assign_coords(lat=lat, lon=lon[:x.lon.size])
    return x.interp(lat=lat, lon=lon)


def main(argv=None):
    import xarray as xr
    import jax.numpy as jnp
    from legoesm.thermo import relative_humidity, specific_humidity_to_mixing_ratio

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--month", type=int, default=2, help="1-based month index in the CMOR files")
    ap.add_argument("--year", type=int, default=1979)
    ap.add_argument("--bands", action="store_true",
                    help="also print 10-degree zonal-band means (40S-40N): pr vs GPCP, prw vs ERA5, ocean and land")
    a = ap.parse_args(argv)

    def era5(v, lat, lon, plev=None):
        f = sorted(glob.glob(f"{ERA5_ROOT}/{v}/*.nc"))
        x = xr.open_dataset(f[0])[v].sel(time=f"{a.year}-{a.month:02d}").isel(time=0)
        if plev is not None:
            x = x.sel(plev=plev)
        return _coarsen_to(x.load(), lat, lon)

    for run in a.runs:
        def mod(v, t=a.month - 1, table="Amon"):
            return xr.open_dataset(glob.glob(f"{run}/cmor/{table}/{v}_*.nc")[0])[v].isel(time=t)
        area = xr.open_dataset(glob.glob(f"{run}/cmor/fx/areacella_*.nc")[0]).areacella.values
        lf = xr.open_dataset(glob.glob(f"{run}/cmor/fx/sftlf_*.nc")[0]).sftlf.values
        lf = lf / 100.0 if lf.max() > 1.5 else lf
        hus = mod("hus")
        lat, lon = hus.lat.values, hus.lon.values
        allm = np.ones_like(area, dtype=bool)
        P, E = mod("pr").values * _DAY, mod("evspsbl").values * _DAY
        prw = mod("prw").values
        line = (f"[{run}] month {a.month} GLOBAL P {area_mean(P, area, allm):.3f} "
                f"E {area_mean(E, area, allm):.3f} P-E {area_mean(P - E, area, allm):+.3f} mm/d "
                f"prw {area_mean(prw, area, allm):.2f} kg/m2")
        if a.month > 1:
            dW = area_mean(prw, area, allm) - area_mean(mod("prw", a.month - 2).values, area, allm)
            line += f"  dW/dt {dW / 30.0:+.3f} mm/d (30-day month approximation)"
        print(line)
        if a.bands:
            print_bands(run, lat, lon, area, lf, P, prw,
                        gpcp_pr_mm_day(lat, lon, a.year, a.month), era5("prw", lat, lon).values)
        trop = (np.abs(lat) <= 20)[:, None] & np.ones((1, lon.size), bool)
        regions = {"trop_ocean": trop & (lf < 0.1), "trop_land": trop & (lf > 0.9)}
        gp = gpcp_pr_mm_day(lat, lon, a.year, a.month)
        if gp is None:
            print(f"[{run}] RAIN NOT SCORED: no GPCP for {a.year}-{a.month:02d} "
                  f"(record 1983-01..2024-09)")
            gp = np.full_like(P, np.nan)
        ref = {"prw": era5("prw", lat, lon).values, "pr": gp,
               "evspsbl": era5("evspsbl", lat, lon).values * _DAY}
        for reg, mk in regions.items():
            print(f"[{run}] {reg}: prw {area_mean(prw, area, mk):.2f} (ERA5 {area_mean(ref['prw'], area, mk):.2f})"
                  f"  pr {area_mean(P, area, mk):.2f} (GPCP {area_mean(ref['pr'], area, mk) if np.isfinite(ref['pr']).any() else float('nan'):.2f})"
                  f"  E {area_mean(E, area, mk):.2f} (ERA5 {area_mean(ref['evspsbl'], area, mk):.2f})")
            print(f"[{run}] {reg}  p[hPa] q_model q_ERA5 [g/kg] | RH_model RH_ERA5 | dT[K]")
            for p in hus.plev.values:
                if p < 20000:
                    continue
                qm, tm = hus.sel(plev=p).values, mod("ta").sel(plev=p).values
                qe, te = era5("hus", lat, lon, p).values, era5("ta", lat, lon, p).values
                rh = [np.asarray(relative_humidity(jnp.asarray(t), jnp.full(t.shape, p),
                                                   specific_humidity_to_mixing_ratio(jnp.asarray(q))))
                      for q, t in ((qm, tm), (qe, te))]
                print(f"[{run}] {reg} {p / 100:6.0f} {area_mean(qm * 1e3, area, mk):7.2f} "
                      f"{area_mean(qe * 1e3, area, mk):7.2f} | {area_mean(rh[0], area, mk):5.2f} "
                      f"{area_mean(rh[1], area, mk):5.2f} | {area_mean(tm - te, area, mk):+5.2f}")


if __name__ == "__main__":
    main()
