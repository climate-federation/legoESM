"""Zonal-mean u on the model's top (pure-pressure) levels vs ERA5, same dates.

For each run ``NAME=RUN_DIR:DAY`` it reads ``RUN_DIR/checkpoint_day_DDDD.npz``
(cell u, ``meta_vgrid`` = (A_half, B_half)) and ``experiment_config.json``
(grid, start_year/start_day), takes the zonal mean on the requested top levels
(refused unless B_half = 0 there, so their pressure is p_ref * A, the same in
every column; full level = mean of its half levels), and compares it with the
ERA5 zonal mean at the same instant, interpolated linearly in log p to those
pressures and in latitude to the model rows. Prints, per run and level: the
cos-weighted RMS difference, and the max |zonal-mean u| with its latitude in
each hemisphere, for model and ERA5.

    python scripts/validate/strat_zonal_wind_vs_era5.py --levels 0 1 2 3 \\
        --run 0.5h=out_T30_05:30 --run 8h=out_T30_8h:30
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ARCO_ERA5_37 = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"


def top_level_pressures_pa(meta_vgrid, levels, p_ref):
    a_half, b_half = np.asarray(meta_vgrid[0]), np.asarray(meta_vgrid[1])
    levels = np.asarray(levels, dtype=int)
    if np.any(b_half[: levels.max() + 2] != 0.0):
        raise ValueError("requested levels are not pure-pressure (B_half != 0)")
    return 0.5 * (a_half[levels] + a_half[levels + 1]) * p_ref


def model_zonal_mean(u_cells, n_lat, n_lon, levels):
    u = np.asarray(u_cells)
    if u.shape[0] != n_lat * n_lon:
        raise ValueError(f"{u.shape[0]} cells != {n_lat} x {n_lon}")
    return u.reshape(n_lat, n_lon, -1)[:, :, levels].mean(axis=1)  # (n_lat, nk)


def interp_logp(field_p, p_src, p_dst):
    """field_p (..., n_src) on pressures p_src -> (..., n_dst) at p_dst, linear in
    ln p; p_dst outside [min p_src, max p_src] is refused (no extrapolation)."""
    p_src = np.asarray(p_src, dtype=float)
    order = np.argsort(p_src)
    lp, f = np.log(p_src[order]), np.asarray(field_p)[..., order]
    lq = np.log(np.asarray(p_dst, dtype=float))
    if lq.min() < lp[0] or lq.max() > lp[-1]:
        raise ValueError("target pressure outside the source levels")
    return np.stack([np.apply_along_axis(lambda c: np.interp(q, lp, c), -1, f)
                     for q in lq], axis=-1)


def compare(model_zm, era_zm, lat_deg):
    w = np.cos(np.deg2rad(lat_deg))[:, None]
    rms = np.sqrt((w * (model_zm - era_zm) ** 2).sum(0) / w.sum())
    out = []
    for k in range(model_zm.shape[1]):
        row = {"rms": float(rms[k])}
        for tag, z in (("model", model_zm[:, k]), ("era5", era_zm[:, k])):
            for hemi, m in (("nh", lat_deg > 0), ("sh", lat_deg < 0)):
                i = int(np.argmax(np.abs(z) * m))
                row[f"{tag}_{hemi}_max"] = float(z[i])
                row[f"{tag}_{hemi}_lat"] = float(lat_deg[i])
        out.append(row)
    return out


def era5_zonal_mean(src, when, p_dst_pa, lat_dst_deg):
    import xarray as xr
    anon = {"storage_options": {"token": "anon"}} if src.startswith("gs://") else {}
    ds = xr.open_zarr(src, chunks=None, **anon)
    lev = ds["level"].values
    u = ds["u_component_of_wind"].sel(time=np.datetime64(when), level=lev[lev <= 50]).load()
    if not np.all(np.isfinite(u.values)):
        raise ValueError("non-finite ERA5 u")
    zm = u.mean("longitude").transpose("latitude", "level")
    lat = zm["latitude"].values
    o = np.argsort(lat)
    zm_lat = np.stack([np.interp(lat_dst_deg, lat[o], zm.values[o, j])
                       for j in range(zm.shape[1])], axis=-1)
    return interp_logp(zm_lat, zm["level"].values * 100.0, p_dst_pa)


def main(argv=None):
    from legoesm import constants
    from legoesm.grids.factory import create_grid
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="append", required=True, help="NAME=RUN_DIR:DAY")
    p.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--src", default=ARCO_ERA5_37)
    a = p.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    sha = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(f"STRAT provenance sha {sha} argv {argv}")
    for spec in a.run:
        name, rest = spec.split("=", 1)
        rdir, day = rest.rsplit(":", 1)
        cfg = json.loads((Path(rdir) / "experiment_config.json").read_text())
        z = np.load(Path(rdir) / f"checkpoint_day_{int(day):04d}.npz")
        if abs(float(z["day"]) - int(day)) > 1e-6:
            raise ValueError(f"{spec}: checkpoint day {float(z['day'])}")
        g = cfg["grid"]
        grid = create_grid(g["grid_type"], g["resolution"])
        lat2d = np.rad2deg(np.asarray(grid.lat2d))
        lat = lat2d[:, 0]
        if not np.allclose(lat2d, lat[:, None]):
            raise ValueError("grid rows are not latitude circles")
        p_pa = top_level_pressures_pa(z["meta_vgrid"], a.levels, constants.p_ref)
        mzm = model_zonal_mean(z["u"], lat2d.shape[0], lat2d.shape[1], a.levels)
        when = datetime(int(cfg["start_year"]), 1, 1) + timedelta(
            days=float(cfg["start_day"]) + float(z["day"]))
        ezm = era5_zonal_mean(a.src, when, p_pa, lat)
        for k, row in zip(a.levels, compare(mzm, ezm, lat)):
            print(f"STRAT {name} {when:%Y-%m-%d} lev {k} p {p_pa[a.levels.index(k)] / 100:.2f} hPa "
                  f"rms {row['rms']:.1f} | NH model {row['model_nh_max']:.1f}@{row['model_nh_lat']:.0f} "
                  f"era5 {row['era5_nh_max']:.1f}@{row['era5_nh_lat']:.0f} | SH model "
                  f"{row['model_sh_max']:.1f}@{row['model_sh_lat']:.0f} era5 "
                  f"{row['era5_sh_max']:.1f}@{row['era5_sh_lat']:.0f}")


if __name__ == "__main__":
    main()
