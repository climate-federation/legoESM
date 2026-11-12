"""Daily ERA5 surface forcing for the ocean-WB lane, from DKRZ's on-disk ERA5 (no download).

One call = one month: each parameter's daily GRIB (N320 reduced Gaussian) is regridded with
``cdo remapcon`` (after ``setgridtype,regular``) onto the 1-degree lat-lon ocean's cells (centres
lat -89.5..89.5 S->N, lon 0..359), then one float32 ``(17, 180, 360)`` ``.npy`` per day is written.

Units in the store (vote 2026-11-11, 2-1): analyses as ERA5 gives them; forecast fields are the
DAILY SUMS divided by 86400 s, i.e. mean rates (W m-2, m s-1 of water, N m-2).  Conversion to
model inputs (kg m-2 s-1, specific humidity) happens in ``legoesm.training.era5_ocean_forcing``.
Sign convention is ERA5's: fluxes positive DOWNWARD (into the ocean).

    python build_era5_ocean_forcing.py MONTH_INDEX --out DIR --work DIR   (index 0 = 1993-01)
Needs ``cdo`` on PATH (module load cdo/2.6.0-gcc-11.2.0).
"""
import argparse
import calendar
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

POOL = Path("/pool/data/ERA5/E5/sf")
# (name, pool kind, param code, divide by 86400 s)
CHANNELS = (
    ("u10", "an", "165", False), ("v10", "an", "166", False), ("t2m", "an", "167", False),
    ("d2m", "an", "168", False), ("msl", "an", "151", False), ("sp", "an", "134", False),
    ("ssrd", "fc", "169", True), ("strd", "fc", "175", True), ("tp", "fc", "228", True),
    ("sf", "fc", "144", True), ("sshf", "fc", "146", True), ("slhf", "fc", "147", True),
    ("ssr", "fc", "176", True), ("str", "fc", "177", True), ("ewss", "fc", "180", True),
    ("nsss", "fc", "181", True), ("e", "fc", "182", True),
)
UNITS = {"u10": "m s-1", "v10": "m s-1", "t2m": "K", "d2m": "K", "msl": "Pa", "sp": "Pa",
         "ssrd": "W m-2", "strd": "W m-2", "tp": "m s-1", "sf": "m s-1", "sshf": "W m-2",
         "slhf": "W m-2", "ssr": "W m-2", "str": "W m-2", "ewss": "N m-2", "nsss": "N m-2",
         "e": "m s-1"}
NLAT, NLON = 180, 360
GRID_TXT = ("gridtype = lonlat\nxsize = 360\nysize = 180\nxfirst = 0\nxinc = 1\n"
            "yfirst = -89.5\nyinc = 1\n")


def month_span(index):
    y, m = 1993 + index // 12, index % 12 + 1
    return y, m, calendar.monthrange(y, m)[1]


def source_file(kind, code, y, m):
    lvl = "00" if kind == "an" else "12"
    return POOL / kind / "1D" / code / f"E5sf{lvl}_1D_{y}-{m:02d}_{code}.grb"


def stack_day(fields, it):
    """Per-channel (n_days, 180, 360) arrays -> one day's (17, 180, 360) float32."""
    a = np.stack([fields[name][it] for name, *_ in CHANNELS]).astype(np.float32)
    assert a.shape == (len(CHANNELS), NLAT, NLON), a.shape
    return a


def done_ok(path):
    try:
        a = np.load(path, mmap_mode="r")
    except (OSError, ValueError):
        return False
    return a.shape == (len(CHANNELS), NLAT, NLON) and a.dtype == np.float32 and bool(
        np.isfinite(a).all())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("month", type=int)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--work", required=True, type=Path)
    a = p.parse_args(argv)
    import xarray as xr

    y, m, nd = month_span(a.month)
    days = [f"{y}-{m:02d}-{d:02d}" for d in range(1, nd + 1)]
    marker = a.out / "done" / f"{y}-{m:02d}.DONE"
    if marker.exists() and (a.out / "meta.json").exists() and all(
            done_ok(a.out / f"{d}.npy") for d in days):
        print(f"{y}-{m:02d} already done", flush=True)
        return
    work = a.work / f"{y}-{m:02d}"
    work.mkdir(parents=True, exist_ok=True)
    grid = work / "grid.txt"
    grid.write_text(GRID_TXT)
    fields = {}
    for name, kind, code, per_day in CHANNELS:
        nc = work / f"{name}.nc"
        if not nc.exists():
            tmp = work / f"{name}_part.nc"
            subprocess.run(["cdo", "-s", "-f", "nc4", f"-remapcon,{grid}", "-setgridtype,regular",
                            str(source_file(kind, code, y, m)), str(tmp)], check=True)
            tmp.rename(nc)
        with xr.open_dataset(nc) as ds:
            v = ds[list(ds.data_vars)[0]]
            lat = ds["lat"].values
            if v.shape[0] != nd:
                raise SystemExit(f"{y}-{m:02d} {name}: {v.shape[0]} records, expected {nd}")
            if not np.allclose(lat, np.arange(-89.5, 90.0)):
                raise SystemExit(f"{name}: unexpected latitudes after remap")
            x = np.asarray(v.values, np.float64).reshape(nd, NLAT, NLON)
        fields[name] = x / 86400.0 if per_day else x
    a.out.mkdir(parents=True, exist_ok=True)
    meta = a.out / "meta.json"
    if not meta.exists():
        tmp = meta.with_name(f"meta.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({
            "channels": [c[0] for c in CHANNELS], "units": UNITS,
            "source": {c[0]: str(source_file(c[1], c[2], 1993, 1)).replace("1993-01", "YYYY-MM")
                       for c in CHANNELS},
            "regrid": "cdo remapcon after setgridtype,regular onto lon 0..359, lat -89.5..89.5",
            "forecast_fields": "daily sums / 86400 s", "sign": "ERA5: positive downward",
            "dtype": "float32"}, indent=1))
        tmp.replace(meta)
    for it, d in enumerate(days):
        dst = a.out / f"{d}.npy"
        tmp = dst.with_name(dst.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.save(fh, stack_day(fields, it))
        tmp.replace(dst)
    bad = [d for d in days if not done_ok(a.out / f"{d}.npy")]
    if bad:
        raise SystemExit(f"{y}-{m:02d}: invalid day files {bad}")
    shutil.rmtree(work)
    marker.parent.mkdir(exist_ok=True)
    marker.write_text("\n".join(days) + "\n")
    print(f"ERA5_OCEAN_DONE {y}-{m:02d}", flush=True)


if __name__ == "__main__":
    main()
