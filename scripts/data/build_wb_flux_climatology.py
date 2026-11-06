#!/usr/bin/env python
"""Build the ERA5 surface-flux climatology for the WB CAM6 arm, or (``--field
sic``) the ERA5 sea-ice climatology for the learned arms
(``legoesm.training.wb_flux_climatology``).

Run under SLURM, never on a login node:
  --init                 create the empty store (one job, before the array);
  --task K --n-tasks N   per-(day, hour) five-year means of every N-th day;
  --finalize             61-day running mean of the raw means, mark complete;
  --check                print how many raw days are done (exit 1 if any is not).
Already-done days are skipped, so a killed task is simply re-run.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_REPO = Path(__file__).resolve().parents[2]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--init", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--finalize", action="store_true")
    g.add_argument("--task", type=int)
    p.add_argument("--n-tasks", type=int, default=1, dest="n_tasks")
    p.add_argument("--field", choices=("flux", "sic"), default="flux")
    p.add_argument("--training-core", default="spectral", dest="training_core",
                   choices=("spectral", "latlon", "cubed_sphere"),
                   help="the grid the store is built on (sic only for the "
                        "grid-point cores)")
    return p


def sic_day_mean(ds, day, grid):
    """``(n_columns,)`` training-years mean sea-ice fraction of noleap
    ``day``, regridded exactly as the AIMIP sea-ice forcing is (land NaN
    filled from the nearest ocean value, then clipped to [0, 1]; vote
    2026-11-06, reuse over a second regridder)."""
    from legoesm.training.aimip_amip_forcing import regrid_era5_field_2d
    from legoesm.training.wb_flux_climatology import SIC_VAR, sic_day_times

    lat = np.asarray(ds.lat.values, np.float64)
    lon = np.asarray(ds.lon.values, np.float64)
    out = []
    for t in sic_day_times(day):
        raw = np.asarray(ds[SIC_VAR].sel(time=np.datetime64(t, "ns"))
                         .transpose("lat", "lon").values, np.float64)
        out.append(np.clip(regrid_era5_field_2d(raw, lat, lon, grid).reshape(-1),
                           0.0, 1.0))
    return np.mean(out, axis=0)


def main_sic(a, yml, grid, root):
    import zarr

    from legoesm.training import wb_flux_climatology as wfc
    from legoesm.training.era5_to_state import open_era5_zarr

    source = yml["era5_zarr"]
    path = wfc.sic_store_path(root, grid, source)
    ncol = int(np.asarray(grid.grid_lat).size)
    shape = (wfc.N_DAYS, ncol)
    print(f"sea-ice climatology {path}", flush=True)
    if a.init:
        if not path.exists():
            g = zarr.open_group(str(path), mode="w")
            for name in ("raw", "clim"):
                g.create_array(name, shape=shape, dtype=np.float32,
                               chunks=(1, ncol), fill_value=np.nan)
            g.create_array("raw_done", data=np.zeros(wfc.N_DAYS, bool), chunks=(1,))
            g.attrs.update({"complete": False, "var": wfc.SIC_VAR,
                            "years": list(wfc.YEARS),
                            "window_days": wfc.SIC_WINDOW_DAYS,
                            "source": source})
        print("created", flush=True)
        return 0
    g = zarr.open_group(str(path), mode="r+")
    done = np.asarray(g["raw_done"])
    if a.check:
        print(f"raw days done {int(done.sum())}/{wfc.N_DAYS}; "
              f"complete={g.attrs['complete']}", flush=True)
        return 0 if done.all() else 1
    if a.finalize:
        if not done.all():
            raise SystemExit(f"{int((~done).sum())} raw days are not done")
        g["clim"][:] = wfc.smooth_days(np.asarray(g["raw"]),
                                       window=wfc.SIC_WINDOW_DAYS).astype(np.float32)
        sha = subprocess.run(["git", "-C", str(_REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        g.attrs.update({"complete": True, "git_sha": sha})
        print("finalized", flush=True)
        return 0
    if not 0 <= a.task < a.n_tasks:
        raise SystemExit(f"--task {a.task} outside 0..{a.n_tasks - 1}")
    ds = open_era5_zarr(source)
    if wfc.SIC_VAR not in ds:
        raise SystemExit(f"{source} has no {wfc.SIC_VAR}")
    for d in range(a.task, wfc.N_DAYS, a.n_tasks):
        if done[d]:
            continue
        g["raw"][d] = sic_day_mean(ds, d, grid).astype(np.float32)
        g["raw_done"][d] = True
        print(f"day {d} done", flush=True)
    return 0


def hourly_planes(fds, t, grid, acc):
    """``(7, n_columns)`` model-grid planes (``FLUX_KEYS`` order) of the
    ERA5 means at hour ``t``: the training loader's signs and regridder."""
    from legoesm.training.era5_to_state import (
        ERA5_RADIATION_FLUX_VARS,
        ERA5_TURBULENT_FLUX_VARS,
        era5_radiation_flux_planes,
        era5_turbulent_flux_planes,
        regrid_2d_to_gaussian,
    )
    from legoesm.training.wb_flux_climatology import FLUX_KEYS

    snap = fds.sel(time=np.datetime64(t, "ns"))
    lat = np.deg2rad(np.asarray(fds.lat.values, np.float64))
    lon = np.deg2rad(np.asarray(fds.lon.values, np.float64))
    raw = [np.asarray(snap[v].values, np.float64).squeeze()
           for v in ERA5_TURBULENT_FLUX_VARS + ERA5_RADIATION_FLUX_VARS]
    if lat[1] < lat[0]:
        lat, raw = lat[::-1], [r[::-1] for r in raw]
    planes = (era5_turbulent_flux_planes(*raw[:4], acc)
              + era5_radiation_flux_planes(*raw[4:], acc))
    out = []
    for name, plane in zip(FLUX_KEYS, planes):
        if not np.isfinite(plane).all():
            raise ValueError(f"ERA5 {name} at {t} is not finite: store defect.")
        out.append(regrid_2d_to_gaussian(plane, lat, lon, grid).reshape(-1))
    return np.stack(out).astype(np.float64)


def day_means(fds, day, grid, acc):
    """``(24, 7, n_columns)`` five-year means of noleap ``day``."""
    from legoesm.training import wb_flux_climatology as wfc

    hours = wfc.climatology_hours(day)
    return np.stack([np.mean([hourly_planes(fds, t, grid, acc) for t in ts],
                             axis=0) for ts in hours])


def main(argv=None):
    a = build_parser().parse_args(argv)
    import jax
    jax.config.update("jax_enable_x64", True)
    import yaml
    import zarr

    from legoesm.training import scale_build as sb
    from legoesm.training import wb_flux_climatology as wfc
    from legoesm.training.era5_to_state import open_era5_zarr

    yml = yaml.safe_load(open(a.config))
    sb.validate_wb_campaign_yaml(yml)
    root = sb.wb_state_cache_root(yml)
    if root is None:
        raise SystemExit(f"{a.config} names no state_cache_dir")
    cfg = SimpleNamespace(mode="physics", training_core=a.training_core,
                          smoke=False, windows=None, n_days=None,
                          multi_step_hours=None)
    if a.field != "sic" and a.training_core != "spectral":
        raise SystemExit("the flux climatology is built for the spectral core only")
    _, grid, *_ = sb.build_mode_components(cfg, yml)
    if a.field == "sic":
        return main_sic(a, yml, grid, root)
    _, sst_cfg, _ = sb.wb_state_cache_configs(cfg, yml)
    path = wfc.store_path(root, grid)
    ncol = int(grid.n_lat) * int(grid.n_lon)
    shape = (wfc.N_DAYS, wfc.N_HOURS, len(wfc.FLUX_KEYS), ncol)
    print(f"flux climatology {path}", flush=True)
    if a.init:
        if not path.exists():
            g = zarr.open_group(str(path), mode="w")
            for name in ("raw", "clim"):
                g.create_array(name, shape=shape, dtype=np.float32,
                               chunks=(1, *shape[1:]), fill_value=np.nan)
            g.create_array("raw_done", data=np.zeros(wfc.N_DAYS, bool), chunks=(1,))
            g.attrs.update({"complete": False, "keys": list(wfc.FLUX_KEYS),
                            "years": list(wfc.YEARS),
                            "window_days": wfc.WINDOW_DAYS,
                            "flux_zarr": sst_cfg.flux_zarr})
        print("created", flush=True)
        return 0
    g = zarr.open_group(str(path), mode="r+")
    done = np.asarray(g["raw_done"])
    if a.check:
        print(f"raw days done {int(done.sum())}/{wfc.N_DAYS}; "
              f"complete={g.attrs['complete']}", flush=True)
        return 0 if done.all() else 1
    if a.finalize:
        if not done.all():
            raise SystemExit(f"{int((~done).sum())} raw days are not done")
        for h in range(wfc.N_HOURS):
            g["clim"][:, h] = wfc.smooth_days(np.asarray(g["raw"][:, h])
                                              ).astype(np.float32)
        sha = subprocess.run(["git", "-C", str(_REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        g.attrs.update({"complete": True, "git_sha": sha})
        print("finalized", flush=True)
        return 0
    if not 0 <= a.task < a.n_tasks:
        raise SystemExit(f"--task {a.task} outside 0..{a.n_tasks - 1}")
    if not sst_cfg.flux_zarr:
        raise SystemExit("the deck names no flux_zarr (hourly ERA5 means)")
    fds = open_era5_zarr(sst_cfg.flux_zarr)
    acc = float(sst_cfg.flux_accum_seconds)
    for d in range(a.task, wfc.N_DAYS, a.n_tasks):
        if done[d]:
            continue
        g["raw"][d] = day_means(fds, d, grid, acc).astype(np.float32)
        g["raw_done"][d] = True
        print(f"day {d} done", flush=True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(_REPO))
    raise SystemExit(main())
