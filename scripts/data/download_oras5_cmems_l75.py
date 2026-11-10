"""Daily ORAS5 at all 75 depths, coarsened to the 1-degree lat-lon ocean grid (ocean WB data).

Source: CMEMS ``cmems_mod_glo_phy-all_my_0.25deg_P1D-m``, the ORAS5 member (``*_oras``), 0.25 deg.
One call = one month: eight ``copernicusmarine subset`` requests (one per variable, all depths)
into a transient raw directory, then one ``.npy`` per day on the model's tracer grid, then the raw
month is deleted and a ``DONE`` marker is written once every day file of the month validates.

Horizontal remap is the 14-depth S2S store's own ``_block_index``/``_remap`` (cos-lat weighted
mean of the 0.25 deg points nearest each destination centre, NaN where no finite source),
reused unchanged and pointed at the model's cell centres: lat -89.5..89.5 (S->N), lon 0..359.

Day file: float32 (304, 180, 360) = thetao, so, uo, vo at 75 levels (variable axis slowest), then
zos, mlotst, siconc, sithick.  NaN on land.  Velocities stay cell-centre means; C-grid faces are
the loader's job.  ``meta.json`` in the store records depths, grid and channel names.

Run inside the Copernicus client's venv (needs xarray + the ``copernicusmarine`` CLI):
    python download_oras5_cmems_l75.py MONTH_INDEX --out DIR --raw DIR --s2s DIR
MONTH_INDEX counts months from 1993-01.
"""
import argparse
import calendar
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

DATASET = "cmems_mod_glo_phy-all_my_0.25deg_P1D-m"
MEMBER = "oras"
VARS_3D = ("thetao", "so", "uo", "vo")
VARS_2D = ("zos", "mlotst", "siconc", "sithick")
NLEV, NLAT, NLON = 75, 180, 360
NCHAN = len(VARS_3D) * NLEV + len(VARS_2D)
LAT0, LON0 = -89.5, 0.0                  # model tracer centres: row 0 = 89.5S, col 0 = 0E


def channel_names(depths):
    return [f"{v}_{z:.4f}" for v in VARS_3D for z in depths] + list(VARS_2D)


def month_span(index):
    y, m = 1993 + index // 12, index % 12 + 1
    return f"{y}-{m:02d}-01", f"{y}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}"


def fetch(var, start, end, raw, client):
    out = raw / f"{var}.nc"
    if not out.exists():
        cmd = [client, "subset", "-i", DATASET, "-t", start, "-T", end, "-v", f"{var}_{MEMBER}",
               "-o", str(raw), "-f", f"{var}_part.nc", "--overwrite"]   # no -z: all depths
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        (raw / f"{var}_part.nc").rename(out)
    return out


def day_stack(ds3, ds2, it, idx, w, remap):
    """One day of the raw month -> (304, 180, 360) float32 on the model grid."""
    out = []
    for v in VARS_3D:
        x = ds3[v][f"{v}_{MEMBER}"].isel(time=it).transpose("depth", "latitude", "longitude").values
        assert x.shape[0] == NLEV, (v, x.shape)
        out.append(remap(x, idx, w, NLAT, NLON))
    for v in VARS_2D:
        x = ds2[v][f"{v}_{MEMBER}"].isel(time=it)
        x = x.isel(depth=0) if "depth" in x.dims else x
        out.append(remap(x.transpose("latitude", "longitude").values, idx, w, NLAT, NLON)[None])
    a = np.concatenate(out, 0).astype(np.float32)
    assert a.shape == (NCHAN, NLAT, NLON), a.shape
    return a


def done_ok(path):
    try:
        a = np.load(path, mmap_mode="r")
    except (OSError, ValueError):
        return False
    return (a.shape == (NCHAN, NLAT, NLON) and a.dtype == np.float32
            and bool(np.isfinite(a[0]).any()))          # an all-NaN day is not done


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("month", type=int)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--raw", required=True, type=Path)
    p.add_argument("--s2s", required=True, type=Path, help="S2S_model checkout (s2s.cmems remap)")
    p.add_argument("--client", default="copernicusmarine")
    a = p.parse_args(argv)

    import pandas as pd
    import xarray as xr
    sys.path.insert(0, str(a.s2s))
    from s2s.cmems import _block_index, _remap

    start, end = month_span(a.month)
    days = pd.date_range(start, end, freq="D")
    a.out.mkdir(parents=True, exist_ok=True)
    marker = a.out / "done" / f"{start[:7]}.DONE"
    if marker.exists():
        if (a.out / "meta.json").exists() and all(
                done_ok(a.out / f"{t.date()}.npy") for t in days):
            print(f"{start[:7]} already done", flush=True)
            return
        marker.unlink()                      # a day file went missing/bad: rebuild the month
    raw = a.raw / start[:7]
    raw.mkdir(parents=True, exist_ok=True)
    ds3 = {v: xr.open_dataset(fetch(v, start, end, raw, a.client)) for v in VARS_3D}
    ds2 = {v: xr.open_dataset(fetch(v, start, end, raw, a.client)) for v in VARS_2D}
    d0 = ds3["thetao"]
    depths = [float(z) for z in d0.depth.values]
    meta = a.out / "meta.json"
    if not meta.exists():
        tmp = meta.with_name(f"meta.{os.getpid()}.tmp")   # array tasks race here
        tmp.write_text(json.dumps({
            "source": DATASET, "member": MEMBER, "depths_m": depths,
            "lat_centres": [LAT0 + j for j in range(NLAT)],
            "lon_centres": [LON0 + i for i in range(NLON)],
            "channels": channel_names(depths), "dtype": "float32", "land": "NaN",
            "remap": "S2S_model s2s.cmems._block_index/_remap, lat0=-89.5 lon0=0.0"}, indent=1))
        tmp.replace(meta)
    elif json.loads(meta.read_text())["depths_m"] != depths:
        raise SystemExit(f"{start[:7]}: depth axis differs from {meta}")
    for v, d in {**ds3, **ds2}.items():
        if not (np.array_equal(d.latitude.values, d0.latitude.values)
                and np.array_equal(d.longitude.values, d0.longitude.values)):
            raise SystemExit(f"{start[:7]}: {v} horizontal axes differ from thetao")
        if v in VARS_3D and not np.array_equal(d.depth.values, d0.depth.values):
            raise SystemExit(f"{start[:7]}: {v} depth axis differs from thetao")
    idx, w = _block_index(d0.latitude.values, d0.longitude.values, NLAT, NLON, LAT0, LON0)
    stamps = {v: pd.to_datetime(d.time.values).normalize() for v, d in {**ds3, **ds2}.items()}
    for t in days:
        dst = a.out / f"{t.date()}.npy"
        if done_ok(dst):
            continue
        its = {v: np.flatnonzero(s == t) for v, s in stamps.items()}
        missing = [v for v, i in its.items() if len(i) != 1]
        if missing:
            raise SystemExit(f"{t.date()}: not exactly one record for {missing}")
        it = int(its["thetao"][0])
        if any(int(i[0]) != it for i in its.values()):
            raise SystemExit(f"{t.date()}: variables disagree on the time index")
        tmp = dst.with_name(dst.name + ".tmp")
        with open(tmp, "wb") as fh:          # np.save appends .npy to a bare name
            np.save(fh, day_stack(ds3, ds2, it, idx, w, _remap))
        tmp.replace(dst)
        print(f"{t.date()} ok", flush=True)
    bad = [str(t.date()) for t in days if not done_ok(a.out / f"{t.date()}.npy")]
    if bad:
        raise SystemExit(f"{start[:7]}: invalid day files {bad}")
    for d in (*ds3.values(), *ds2.values()):
        d.close()
    shutil.rmtree(raw)
    marker.parent.mkdir(exist_ok=True)
    marker.write_text("\n".join(str(t.date()) for t in days) + "\n")
    print(f"ORAS5_L75_DONE {start[:7]}", flush=True)


if __name__ == "__main__":
    main()
