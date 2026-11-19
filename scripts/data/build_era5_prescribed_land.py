#!/usr/bin/env python
"""Build the prescribed-land boundary file from ERA5 monthly means (Levante pool).

Writes a 12-month climatology (``--years``) of skin temperature ``ts`` [K],
sensible heat ``hfss`` and latent heat ``hfls`` [W m-2, POSITIVE UPWARD, the
model's land-flux convention] and broadband albedo ``albedo`` [1] on a regular
1-degree lat-lon grid, with a mid-month ``time`` axis in days of a 365-day
year, for ``run_amip.py --land-prescribed-channels ... --land-prescribed-flux-file``
(``legoesm.forcing.prescribed_land`` reads it).

ERA5 sources (DKRZ /pool/data/ERA5/E5, GRIB, N320 reduced Gaussian):
  sf/an/1M/235  skt   monthly mean of analyses [K]
  sf/an/1M/243  fal   forecast albedo, monthly mean of analyses [-] (snow/ice aware)
  sf/fc/1M/146  sshf  monthly mean of DAILY forecast accumulations [J m-2 per 24 h],
  sf/fc/1M/147  slhf  ECMWF sign: positive DOWNWARD (checked 2026-11-06: Jan 1979
                      global mean slhf -6.89e6 J m-2 -> -79.7 W m-2 downward =
                      80 W m-2 up).
Conversion: W m-2 upward = -(J m-2 per day) / 86400.
Regridding: cdo remapnn onto r360x180 (as prep_levante_era5_ic.py).

The output is runtime data (data/ is gitignored), not committed.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

from legoesm.forcing.time_utils import NOLEAP_DAYS_PER_MONTH, NOLEAP_MONTH_STARTS
from scripts.data.prep_levante_era5_ic import CDO, ERA5_ROOT

SEC_PER_DAY = 86400.0
# mid-month day of a 365-day year (Jan 15.5, Feb 45.0, ...): the reader's axis
NOLEAP_MID_MONTH = (np.asarray(NOLEAP_MONTH_STARTS[:12], dtype=np.float64)
                    + 0.5 * np.asarray(NOLEAP_DAYS_PER_MONTH, dtype=np.float64))
# (kind, ECMWF code) of each output variable
SOURCES = {"ts": ("an", 235), "albedo": ("an", 243),
           "hfss": ("fc", 146), "hfls": ("fc", 147)}
# January global-mean plausibility window of the built file (units/sign guard)
PLAUSIBLE_JAN = {"ts": (280.0, 292.0), "hfss": (12.0, 25.0),
                 "hfls": (60.0, 100.0), "albedo": (0.12, 0.25)}


def _era5_file(kind: str, code: int, year: int) -> Path:
    prefix = "E5sf00" if kind == "an" else "E5sf12"
    return ERA5_ROOT / "sf" / kind / "1M" / str(code) / f"{prefix}_1M_{year}_{code}.grb"


def _ymonmean(kind: str, code: int, years: range, grid: str, out: Path) -> None:
    """cdo: concatenate the yearly monthly-mean files, regrid, climatological month mean."""
    srcs = [str(_era5_file(kind, code, y)) for y in years]
    for s in srcs:
        if not Path(s).exists():
            raise FileNotFoundError(s)
    cat = out.with_suffix(".cat.grb")
    subprocess.run([CDO, "-s", "cat", *srcs, str(cat)], check=True)
    subprocess.run([CDO, "-s", "-f", "nc4", "-ymonmean", f"-remapnn,{grid}",
                    str(cat), str(out)], check=True)
    cat.unlink()


def upward_flux_w_m2(daily_accum_j_m2) -> np.ndarray:
    """ERA5 daily accumulation [J m-2 per day, positive DOWNWARD] -> W m-2
    positive UPWARD."""
    return -np.asarray(daily_accum_j_m2, dtype=np.float64) / SEC_PER_DAY


def _the_3d_var(ds) -> np.ndarray:
    return ds[[v for v in ds.data_vars if ds[v].ndim == 3][0]].values.astype(np.float64)


def build_dataset(raw: dict, lat, lon, years: str) -> xr.Dataset:
    """The reader's file from the cdo climatologies ``raw`` (ERA5 units)."""
    f = {"ts": raw["ts"], "albedo": raw["albedo"],
         "hfss": upward_flux_w_m2(raw["hfss"]), "hfls": upward_flux_w_m2(raw["hfls"])}
    if f["ts"].shape[0] != 12:
        raise ValueError(f"expected 12 climatological months, got {f['ts'].shape[0]}")
    w = np.cos(np.deg2rad(np.asarray(lat)))[:, None]
    gm = {k: float((v[0] * w).sum() / (w * np.ones_like(v[0])).sum()) for k, v in f.items()}
    print("global means Jan: " + " ".join(f"{k}={gm[k]:.3f}" for k in f))
    bad = [k for k, (lo, hi) in PLAUSIBLE_JAN.items() if not lo < gm[k] < hi]
    if bad:
        raise ValueError(f"January global means outside the plausible range for {bad}: "
                         f"{ {k: gm[k] for k in bad} }; check units/sign")
    up = {"units": "W m-2", "positive": "up"}
    dims = ("time", "lat", "lon")
    return xr.Dataset(
        {"ts": (dims, f["ts"], {"units": "K", "long_name": "ERA5 skin temperature, climatological month mean"}),
         "hfss": (dims, f["hfss"], dict(up, long_name="ERA5 surface sensible heat flux")),
         "hfls": (dims, f["hfls"], dict(up, long_name="ERA5 surface latent heat flux")),
         "albedo": (dims, f["albedo"], {"units": "1", "long_name": "ERA5 forecast albedo (fal), climatological month mean"})},
        coords={"time": ("time", NOLEAP_MID_MONTH, {"units": "days since 0001-01-01", "calendar": "noleap",
                                                     "long_name": "mid-month day of a 365-day climatological year"}),
                "lat": np.asarray(lat), "lon": np.asarray(lon)},
        attrs={"source": "ERA5 monthly means, DKRZ /pool/data/ERA5/E5 (skt 235, fal 243 an; sshf 146, slhf 147 fc)",
               "years": years, "sign_convention": "fluxes positive upward (model land convention); ERA5 sign flipped",
               "built_by": "scripts/data/build_era5_prescribed_land.py"})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", default="1979-1998")
    ap.add_argument("--grid", default="r360x180",
                    help="cdo target grid (default 1 deg; nearest-neighbour from the "
                         "N320 reduced Gaussian grid, as prep_levante_era5_ic.py)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    y0, y1 = (int(v) for v in a.years.split("-"))
    years = range(y0, y1 + 1)
    raw = {}
    with tempfile.TemporaryDirectory() as td:
        for name, (kind, code) in SOURCES.items():
            p = Path(td) / f"{name}.nc"
            _ymonmean(kind, code, years, a.grid, p)
            with xr.open_dataset(p) as ds:
                raw[name] = _the_3d_var(ds)
                lat, lon = ds["lat"].values, ds["lon"].values
    build_dataset(raw, lat, lon, a.years).to_netcdf(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
