"""Heat drift of a legoESM run, from its restart snapshots (one per model year).

For each snapshot: volume-mean T, the implied mean surface heat uptake since the previous
snapshot [W m-2 of ocean area], and the horizontal-mean T - WOA at 173, 460, 1000 m, globally
and in three bands (south of 35 S, 35 S - 35 N, north of 35 N).  The first line is WOA itself.

Usage: python drift_by_depth.py RUN_DIR WOA_ON_MODEL_GRID.nc GRID.npz
"""
import glob
import sys
from pathlib import Path

import numpy as np

RHO_CP = 1026.0 * 3990.0
R_E = 6.371e6


def main():
    run, woa_path, grid_path = sys.argv[1:4]
    import xarray as xr
    g = np.load(grid_path)
    lat, lon = np.asarray(g["lat"], float), np.asarray(g["lon"], float)
    if np.abs(lat).max() <= np.pi / 2 + 1e-6:
        lat, lon = np.rad2deg(lat), np.rad2deg(lon)
    z, zh = np.abs(np.asarray(g["z"], float)), np.abs(np.asarray(g["z_half"], float))
    nlon = lon.size
    lat_e = np.concatenate([[-90.0], 0.5 * (lat[1:] + lat[:-1]), [90.0]])
    dlon = np.deg2rad(np.median(np.diff(lon)))
    area = (R_E ** 2 * (np.sin(np.deg2rad(lat_e[1:])) - np.sin(np.deg2rad(lat_e[:-1]))) * dlon)[:, None] * np.ones((1, nlon))
    da = xr.open_dataset(woa_path)["T"]
    if "time" in da.dims:
        da = da.mean("time")
    Tw = np.asarray(da.transpose(*[d for d in da.dims if d.startswith("lat")],
                                 *[d for d in da.dims if d.startswith("lon")], ...).values, float)
    files = sorted(glob.glob(f"{run}/restart_day*.npz"))
    yearly = [f for f in files if round(float(Path(f).stem.replace("restart_day", ""))) % 365 == 0]
    files = sorted(set(files[:1] + yearly)) if yearly else files
    if not files:
        sys.exit(f"no restart_day*.npz in {run}")
    r0 = np.load(files[0])
    ocean = r0["land_mask"] > 0.5
    H = np.asarray(r0["H_bathy"], float)
    dz = np.clip(H[..., None] - zh[None, None, :-1], 0.0, (zh[1:] - zh[:-1])[None, None, :]) * ocean[..., None]
    ok = (dz > 1e-3) & np.isfinite(Tw)
    vol = np.where(ok, dz * area[..., None], 0.0)
    A_ocean = np.sum(area[ocean])
    LA = lat[:, None] * np.ones((1, nlon))
    bands = {"glob": ocean, "S<35S": ocean & (LA < -35), "35S-35N": ocean & (np.abs(LA) <= 35), "N>35N": ocean & (LA > 35)}
    ks = sorted({int(np.argmin(np.abs(z - d))) for d in (173, 460, 1000)})

    def stats(T):
        vm = np.sum(np.where(ok, T, 0) * vol) / np.sum(vol)
        out = []
        for k in ks:
            for b, m in bands.items():
                s = ok[..., k] & m & np.isfinite(T[..., k])
                out.append(np.sum((T[..., k] - Tw[..., k])[s] * area[s]) / np.sum(area[s]))
        return vm, out

    hdr = "".join(f"{f'{z[k]:.0f}m':>{8 * len(bands)}s}" for k in ks)
    sub = "".join("".join(f"{b:>8s}" for b in bands) for _ in ks)
    print(f"T - WOA [degC], horizontal means; Q = mean heat uptake since previous line [W m-2]\n"
          f"{'day':>8s} {'volT':>7s} {'Q':>6s} {hdr}\n{'':>23s}{sub}")
    vm, out = stats(Tw)
    print(f"{'WOA':>8s} {vm:7.3f} {'':>6s} " + "".join(f"{x:+8.2f}" for x in out))
    prev = None
    for f in files:
        day = float(Path(f).stem.replace("restart_day", ""))
        vm, out = stats(np.asarray(np.load(f)["T"], float))
        Q = "" if prev is None else f"{(vm - prev[1]) * RHO_CP * np.sum(vol) / A_ocean / ((day - prev[0]) * 86400):6.1f}"
        print(f"{day:8.0f} {vm:7.3f} {Q:>6s} " + "".join(f"{x:+8.2f}" for x in out))
        prev = (day, vm)


if __name__ == "__main__":
    main()
