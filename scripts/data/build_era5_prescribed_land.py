#!/usr/bin/env python
"""Build the prescribed-land boundary file from ERA5 monthly means (Levante pool).

Writes a 12-month climatology (``--years``) of skin temperature ``ts`` [K], sensible
heat ``hfss`` and latent heat ``hfls`` [W m-2, POSITIVE UPWARD, the model's land-flux
convention] on a regular 1-degree lat-lon grid, with a mid-month ``time`` axis in
days of a 365-day year, for ``run_amip.py --land-prescribed-flux-file``.

ERA5 sources (DKRZ /pool/data/ERA5/E5, GRIB, N320 reduced Gaussian):
  sf/an/1M/235  skt   monthly mean of analyses [K]
  sf/fc/1M/146  sshf  monthly mean of DAILY forecast accumulations [J m-2 per 24 h],
  sf/fc/1M/147  slhf  ECMWF sign: positive DOWNWARD (checked 2026-11-06: Jan 1979 global
                      mean slhf -6.89e6 J m-2 -> -79.7 W m-2 downward = 80 W m-2 up).
Conversion: W m-2 upward = -(J m-2 per day) / 86400.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

ERA5 = Path("/pool/data/ERA5/E5/sf")
CDO = "/sw/spack-levante/cdo-2.2.2-4z4icb/bin/cdo"
SEC_PER_DAY = 86400.0
NOLEAP_MID_MONTH = np.array([15.5, 45.0, 74.5, 105.0, 135.5, 166.0, 196.5, 227.5, 258.0,
                             288.5, 319.0, 349.5])  # day of a 365-day year, mid-month


def _era5_file(kind: str, code: int, year: int) -> Path:
    prefix = "E5sf00" if kind == "an" else "E5sf12"
    return ERA5 / kind / "1M" / str(code) / f"{prefix}_1M_{year}_{code}.grb"


def _ymonmean(kind: str, code: int, years: range, grid: str, out: Path) -> None:
    """cdo: concatenate the yearly monthly-mean files, regrid, climatological month mean."""
    srcs = [str(_era5_file(kind, code, y)) for y in years]
    for s in srcs:
        if not Path(s).exists():
            raise FileNotFoundError(s)
    cat = out.with_suffix(".cat.grb")
    subprocess.run([CDO, "-s", "cat", *srcs, str(cat)], check=True)
    subprocess.run([CDO, "-s", "-f", "nc4", "-ymonmean", f"-remapnn,{grid}", str(cat), str(out)],
                   check=True)
    cat.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", default="1979-1998")
    ap.add_argument("--grid", default="r360x180", help="cdo target grid (default 1 deg; nearest-neighbour from the N320 reduced Gaussian grid, as prep_levante_era5_ic.py)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    y0, y1 = (int(v) for v in a.years.split("-"))
    years = range(y0, y1 + 1)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _ymonmean("an", 235, years, a.grid, td / "skt.nc")
        _ymonmean("fc", 146, years, a.grid, td / "sshf.nc")
        _ymonmean("fc", 147, years, a.grid, td / "slhf.nc")
        skt = xr.open_dataset(td / "skt.nc")
        sshf = xr.open_dataset(td / "sshf.nc")
        slhf = xr.open_dataset(td / "slhf.nc")
        v_skt = [v for v in skt.data_vars if skt[v].ndim == 3][0]
        v_sh = [v for v in sshf.data_vars if sshf[v].ndim == 3][0]
        v_lh = [v for v in slhf.data_vars if slhf[v].ndim == 3][0]
        ts = skt[v_skt].values.astype(np.float64)
        hfss = -sshf[v_sh].values.astype(np.float64) / SEC_PER_DAY
        hfls = -slhf[v_lh].values.astype(np.float64) / SEC_PER_DAY
        lat = skt["lat"].values
        lon = skt["lon"].values
    if ts.shape[0] != 12:
        raise ValueError(f"expected 12 climatological months, got {ts.shape[0]}")
    w = np.cos(np.deg2rad(lat))[:, None]
    gm = lambda f: float((f * w).sum() / (w * np.ones_like(f)).sum())  # noqa: E731
    print(f"global means Jan: ts={gm(ts[0]):.2f} K hfss={gm(hfss[0]):.1f} hfls={gm(hfls[0]):.1f} W/m2 (up)")
    if not (12.0 < gm(hfss[0]) < 25.0 and 60.0 < gm(hfls[0]) < 100.0 and 280.0 < gm(ts[0]) < 292.0):
        raise ValueError("global means outside the plausible range; check units/sign")
    ds = xr.Dataset(
        {"ts": (("time", "lat", "lon"), ts, {"units": "K", "long_name": "ERA5 skin temperature, climatological month mean"}),
         "hfss": (("time", "lat", "lon"), hfss, {"units": "W m-2", "positive": "up", "long_name": "ERA5 surface sensible heat flux"}),
         "hfls": (("time", "lat", "lon"), hfls, {"units": "W m-2", "positive": "up", "long_name": "ERA5 surface latent heat flux"})},
        coords={"time": ("time", NOLEAP_MID_MONTH, {"units": "days since 0001-01-01", "calendar": "noleap",
                                                     "long_name": "mid-month day of a 365-day climatological year"}),
                "lat": lat, "lon": lon},
        attrs={"source": "ERA5 monthly means, DKRZ /pool/data/ERA5/E5 (skt 235 an; sshf 146, slhf 147 fc)",
               "years": a.years, "sign_convention": "fluxes positive upward (model land convention); ERA5 sign flipped",
               "built_by": "scripts/data/build_era5_prescribed_land.py"})
    ds.to_netcdf(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
