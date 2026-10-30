#!/usr/bin/env python
"""MACv2-SP anthropogenic aerosol on the Kinne MACv1 pre-industrial background
-> monthly per-band column aerosol optics files for 1850-2014.

Port of the REFERENCE routine ``sp_aop_profile`` (``mo_simple_plumes_v1.f90``,
input4MIPs ``MPI-M-MACv2-SP-1-0`` v20170201; Stevens et al. 2017, GMD 10,
433-452) in COLUMN form, plus the background composition MPI-ESM1.2 used in
CMIP6 (plumes superimposed on the Kinne 2013 MACv1 natural aerosol).

Kept from the reference, term by term:
- per-feature rotated Gaussians with East/West widths selected on the sign of
  the (wrapped) longitude offset; one +-360 wrap when |dlon| exceeds 180 deg
  (260 deg for plume 1, Europe).  Longitudes are taken in the reference's
  0-360 convention (its driver reads a T63 0-360 grid): the 260 deg wrap is not
  convention-invariant, so the -180..180 MACv1 grid is mapped to 0-360 first.
- time weights ``year_weight(FLOOR(year_fr)-1849) * ann_cycle(feature,
  FLOOR(frac(year_fr)*52)+1)``, applied per feature before the feature sum.
- AOD(lambda) = cw_an * exp(-angstrom*ln(lambda/550)); ssa(lambda) and
  asy(lambda) from ``lfactor = min(1, 700/lambda)``; ssa AOD-weighted and asy
  scattering-weighted across plumes.
Dropped, each because it is a no-op for a column product: the beta-function
vertical profile (normalised over the host's levels, so it sums to 1 on any
terrain-following column whose levels sit above the surface) and the
orography mask (all such levels are above ground).  Not produced: the Twomey
factor ``dNovrN``; the model derives CCN from the fine-mode 550 nm AOD instead.

Products (layout of the staged ``aeropt_kinne_sw_b14_{tot,fin}_*_MACv1.nc``):
``aod/ssa/asy(time, lnwl=14, lat, lon)`` with ``wl1_sun/wl2_sun`` band edges.
- fin(Y, m) = MACv1 pre-industrial fine (g30_pre_9999) + MACv2-SP plumes(Y, m)
- tot(Y, m) = fin(Y, m) + MACv1 coarse (g30_cor_2000)
Each month's plume time weights are the mean of daily samples taken at 12 UTC.
Optics are evaluated at each band's representative wavelength ``lnwl``.

Usage::

    python scripts/data/adapt_cmip6_aerosol.py --macv2sp MACv2.0-SP_v1.nc \\
        --macv1-dir raw_macv1 --band-template aeropt_kinne_sw_b14_fin_1979-2014_MACv1.nc \\
        --out-dir data/cmip6_forcing --years 1850 2014
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import hashlib
from pathlib import Path

import numpy as np

NTIMES = 52            # reference: annual cycle resolved weekly
FIRST_YEAR = 1850      # reference: year_weight(1) is 1850
_LAMBDA_REF_NM = 550.0
_LFACTOR_NM = 700.0
_PER_PLUME = ("plume_lat", "plume_lon", "aod_spmx", "ssa550", "asy550", "angstrom")
_PER_FEATURE = ("sig_lon_E", "sig_lon_W", "sig_lat_E", "sig_lat_W", "theta", "ftr_weight")


def read_plumes(ds) -> dict:
    """Plume parameters from an opened MACv2.0-SP Dataset, oriented by dimension
    NAME: per-plume ``(np,)``, per-feature ``(nf, np)``, ``year_weight (nyears,
    np)`` with masked years as NaN, ``ann_cycle (nf, 52, np)``."""
    def arr(name, *dims):
        return np.asarray(ds[name].transpose(*dims).values, dtype=np.float64)

    p = {k: arr(k, "plume_number") for k in _PER_PLUME}
    p.update({k: arr(k, "plume_feature", "plume_number") for k in _PER_FEATURE})
    p["year_weight"] = arr("year_weight", "years", "plume_number")
    p["ann_cycle"] = arr("ann_cycle", "plume_feature", "year_fr", "plume_number")
    p["years"] = np.asarray(ds["years"].values)
    if p["years"][0] != FIRST_YEAR or np.any(np.diff(p["years"]) != 1):
        raise ValueError(f"years must run 1850,1851,... (got {p['years'][:3]}...)")
    if p["ann_cycle"].shape[1] != NTIMES:
        raise ValueError(f"ann_cycle has {p['ann_cycle'].shape[1]} times, expected {NTIMES}")
    for k, v in p.items():
        if k != "year_weight" and not np.all(np.isfinite(v)):
            raise ValueError(f"{k} has non-finite values")
    for k in ("sig_lon_E", "sig_lon_W", "sig_lat_E", "sig_lat_W"):
        if not np.all(p[k] > 0):
            raise ValueError(f"{k} must be positive (a zero width divides into NaN AOD)")
    return p


def time_weights(p: dict, year_fr: float) -> np.ndarray:
    """Reference ``set_time_weight``: anthropogenic weight ``(nf, np)``."""
    iyear = int(np.floor(year_fr)) - FIRST_YEAR
    iweek = int(np.floor((year_fr - np.floor(year_fr)) * NTIMES))
    if not (0 <= iyear < p["year_weight"].shape[0] and 0 <= iweek < NTIMES):
        raise ValueError(f"year_fr {year_fr} outside the MACv2-SP time axis")
    yw = p["year_weight"][iyear]
    if not np.all(np.isfinite(yw)):
        raise ValueError(f"year {int(np.floor(year_fr))} has no MACv2-SP year_weight")
    return yw[None, :] * p["ann_cycle"][:, iweek, :]


def month_time_weights(p: dict, year: int, month: int) -> np.ndarray:
    """Monthly mean of the reference time weights sampled daily at 12 UTC
    (``year_fr`` = year + elapsed fraction of the Gregorian year)."""
    ndays = 366 if calendar.isleap(year) else 365
    doy0 = dt.date(year, month, 1).timetuple().tm_yday - 1
    nd = calendar.monthrange(year, month)[1]
    return np.mean([time_weights(p, year + (doy0 + d + 0.5) / ndays)
                    for d in range(nd)], axis=0)


def feature_gaussians(p: dict, lat, lon) -> np.ndarray:
    """Per-feature, per-plume Gaussian shapes ``(nf, np, *lat.shape)``;
    ``lat``/``lon`` in degrees, any longitude convention (mapped to 0-360)."""
    lat = np.asarray(lat, dtype=np.float64)[None, None]
    lon = np.mod(np.asarray(lon, dtype=np.float64), 360.0)[None, None]
    ex = (Ellipsis,) + (None,) * (lat.ndim - 2)
    dlat = lat - p["plume_lat"][None, :][ex]
    dlon = lon - p["plume_lon"][None, :][ex]
    lim = np.where(np.arange(p["plume_lat"].size) == 0, 260.0, 180.0)[None, :][ex]
    dlon = np.where(np.abs(dlon) > lim, dlon - np.copysign(360.0, dlon), dlon)
    east = dlon > 0.0
    a = 0.5 / np.where(east, p["sig_lon_E"][ex], p["sig_lon_W"][ex]) ** 2
    b = 0.5 / np.where(east, p["sig_lat_E"][ex], p["sig_lat_W"][ex]) ** 2
    ct, st = np.cos(p["theta"][ex]), np.sin(p["theta"][ex])
    lon_r = ct * dlon + st * dlat
    lat_r = -st * dlon + ct * dlat
    return np.exp(-(a * lon_r ** 2 + b * lat_r ** 2))


def column_aod550(p: dict, tw: np.ndarray, gauss: np.ndarray) -> np.ndarray:
    """Reference ``cw_an``: per-plume column 550 nm AOD ``(np, ...)``."""
    ex = (Ellipsis,) + (None,) * (gauss.ndim - 2)
    return p["aod_spmx"][ex] * np.sum((tw * p["ftr_weight"])[ex] * gauss, axis=0)


def plume_band_sums(p: dict, cw: np.ndarray, lambdas_nm) -> tuple:
    """Plume-summed ``(aod, aod*ssa, aod*ssa*asy)`` per band ``(nb, ...)``."""
    ex = (Ellipsis,) + (None,) * (cw.ndim - 1)
    lam = np.asarray(lambdas_nm, dtype=np.float64)[:, None]               # (nb, 1)
    lf = np.minimum(1.0, _LFACTOR_NM / lam)
    ssa = p["ssa550"] * lf ** 4 / (p["ssa550"] * lf ** 4 + (1.0 - p["ssa550"]) * lf)
    asy = p["asy550"] * np.sqrt(lf)
    lfac = np.exp(-p["angstrom"] * np.log(lam / _LAMBDA_REF_NM))           # (nb, np)
    aod = lfac[ex] * cw[None]                                              # (nb, np, ...)
    return (aod.sum(1), (aod * ssa[ex]).sum(1), (aod * (ssa * asy)[ex]).sum(1))


def column_optics(sums) -> tuple:
    """``(aod, ssa, asy)`` from summed ``(aod, sca, asc)``; reference zero-AOD
    limits ssa -> 1, asy -> 0."""
    aod, sca, asc = sums
    ssa = np.where(aod > 0, sca / np.where(aod > 0, aod, 1.0), 1.0)
    asy = np.where(sca > 0, asc / np.where(sca > 0, sca, 1.0), 0.0)
    return aod, ssa, asy


def component_sums(aod, ssa, asy) -> tuple:
    return aod, aod * ssa, aod * ssa * asy


def _read_macv1(path: Path, lnwl) -> tuple:
    """MACv1 ``(aod, ssa, asy)`` as ``(12, nb, lat, lon)`` float64, its lat/lon;
    the record order must be the template's ``lnwl`` (from the ncecat history)."""
    import re
    import netCDF4

    with netCDF4.Dataset(path) as nc:
        wl = [int(w) for w in re.findall(r"_(\d{5})nm\.nc", nc.history)]
        if wl != [int(round(w)) for w in lnwl]:
            raise ValueError(f"{path}: record wavelengths {wl} != band template {list(lnwl)}")
        out = tuple(np.asarray(nc[k][:], dtype=np.float64).transpose(1, 0, 2, 3)
                    for k in ("aod", "ssa", "asy"))
        return out, np.asarray(nc["lat"][:]), np.asarray(nc["lon"][:])


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(macv2sp: Path, macv1_dir: Path, band_template: Path, out_dir: Path,
          first_year: int, last_year: int) -> tuple[Path, Path]:
    import netCDF4
    import xarray as xr

    with xr.open_dataset(macv2sp, decode_times=False) as ds:
        p = read_plumes(ds)
    if first_year > last_year:
        raise ValueError(f"--years {first_year} {last_year}: first after last")
    for y in range(first_year, last_year + 1):
        time_weights(p, y + 0.5)        # refuses years off the MACv2-SP axis or masked
    with netCDF4.Dataset(band_template) as t:
        lnwl = np.asarray(t["lnwl"][:], dtype=np.float64)
        edges = {k: np.asarray(t[k][:]) for k in ("wl1_sun", "wl2_sun")}
    pre, lat, lon = _read_macv1(macv1_dir / "g30_pre_9999.nc", lnwl)
    cor, lat_c, lon_c = _read_macv1(macv1_dir / "g30_cor_2000.nc", lnwl)
    if not (np.array_equal(lat, lat_c) and np.array_equal(lon, lon_c)):
        raise ValueError("MACv1 pre-industrial and coarse grids differ")
    gauss = feature_gaussians(p, *np.meshgrid(lat, lon, indexing="ij"))

    span = f"{first_year}-{last_year}"
    paths = {k: out_dir / f"aeropt_macv2sp_sw_b14_{k}_{span}_MACv1bg.nc" for k in ("fin", "tot")}
    # Written under a temporary name and renamed on completion: the launchers
    # select these filenames, so a failed build must not leave a partial file there.
    tmp = {k: v.with_name(v.name + ".partial") for k, v in paths.items()}
    files = {k: netCDF4.Dataset(v, "w", format="NETCDF4") for k, v in tmp.items()}
    ref = dt.date(first_year, 1, 1)
    src = {"macv2sp": (macv2sp, _sha256(macv2sp)),
           "pre": (macv1_dir / "g30_pre_9999.nc", None), "cor": (macv1_dir / "g30_cor_2000.nc", None)}
    var = {}
    for kind, nc in files.items():
        nc.createDimension("time", None)
        nc.createDimension("lnwl", lnwl.size)
        nc.createDimension("lat", lat.size)
        nc.createDimension("lon", lon.size)
        v = nc.createVariable("time", "f8", ("time",))
        v.units, v.calendar, v.long_name = f"days since {first_year}-01-01 00:00:00", "standard", "mid-month time"
        for name, val, units in (("lnwl", lnwl, "nm"), ("lat", lat, "degrees_north"),
                                 ("lon", lon, "degrees_east")):
            nc.createVariable(name, "f4", (name,))[:] = val
            nc[name].units = units
        for name in ("wl1_sun", "wl2_sun"):
            nc.createVariable(name, "f4", ("lnwl",))[:] = edges[name]
            nc[name].units = "um"
        for name in ("aod", "ssa", "asy"):
            var[kind, name] = nc.createVariable(
                name, "f4", ("time", "lnwl", "lat", "lon"), zlib=True, complevel=2,
                chunksizes=(1, lnwl.size, lat.size, lon.size))
            var[kind, name].units = "1"
        nc.title = (f"MACv2-SP anthropogenic plumes (Stevens et al. 2017) on the Kinne MACv1 "
                    f"pre-industrial {'fine' if kind == 'fin' else 'fine + coarse'} background, "
                    f"{lnwl.size} RRTMG-SW bands, monthly {span}")
        nc.source = (f"MACv2.0-SP_v1.nc (input4MIPs MPI-M-MACv2-SP-1-0 v20170201, sha256 "
                     f"{src['macv2sp'][1]}) + MACv1 g30_pre_9999.nc"
                     + (" + g30_cor_2000.nc" if kind == "tot" else ""))
        nc.history = f"built {dt.date.today()} by scripts/data/adapt_cmip6_aerosol.py"
    k = 0
    for y in range(first_year, last_year + 1):
        for m in range(1, 13):
            cw = column_aod550(p, month_time_weights(p, y, m), gauss)
            fin = [a + b for a, b in zip(plume_band_sums(p, cw, lnwl),
                                         component_sums(*(x[m - 1] for x in pre)))]
            tot = [a + b for a, b in zip(fin, component_sums(*(x[m - 1] for x in cor)))]
            a0, b0 = dt.date(y, m, 1), dt.date(y + (m == 12), m % 12 + 1, 1)
            for kind, sums in (("fin", fin), ("tot", tot)):
                files[kind]["time"][k] = (a0 - ref).days + (b0 - a0).days / 2.0
                for name, val in zip(("aod", "ssa", "asy"), column_optics(sums)):
                    var[kind, name][k] = val
            k += 1
        print(y, flush=True)
    for nc in files.values():
        nc.close()
    for k in paths:
        tmp[k].replace(paths[k])
    return paths["fin"], paths["tot"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--macv2sp", type=Path, required=True, help="MACv2.0-SP_v1.nc")
    p.add_argument("--macv1-dir", type=Path, required=True,
                   help="directory with MACv1 g30_pre_9999.nc and g30_cor_2000.nc")
    p.add_argument("--band-template", type=Path, required=True,
                   help="staged aeropt_kinne_sw_b14_*_MACv1.nc: lnwl + wl1_sun/wl2_sun")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--years", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    for path in build(a.macv2sp, a.macv1_dir, a.band_template, a.out_dir, *a.years):
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
