"""Direct test for scripts/validate/compare_dino_nemo.py: synthetic
legoESM snapshots + NEMO-layout NetCDFs -> all 5 figures written."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

_SPEC = importlib.util.spec_from_file_location(
    "compare_dino_nemo",
    Path(__file__).resolve().parents[2]
    / "scripts" / "validate" / "compare_dino_nemo.py")
cdn = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cdn)


@pytest.mark.parametrize("vertical", ["zstar", "masked_zco"])
def test_writes_all_figures(tmp_path, monkeypatch, vertical):
    import dataclasses

    from legoesm.ocean.experiments.dino import (
        DINOConfig, create_dino_z_star, dino_lat_lon_grid,
        dino_lat_lon_vertical,
    )
    cfg = dataclasses.replace(DINOConfig(), vertical_coordinate=vertical)
    grid = dino_lat_lon_grid(cfg, n_lon=50)
    z = (dino_lat_lon_vertical(grid, cfg) if vertical == "masked_zco"
         else create_dino_z_star(cfg))
    n_lat, n_lon, nlev = grid.n_lat, 50, z.n_levels
    rng = np.random.default_rng(0)

    snap_dir = tmp_path / "lego" / "snapshots"
    snap_dir.mkdir(parents=True)
    for i in range(3):
        np.savez_compressed(
            snap_dir / f"snapshot_{i:05d}.npz",
            T=rng.uniform(2, 20, (n_lat, n_lon, nlev)),
            S=rng.uniform(34, 36, (n_lat, n_lon, nlev)),
            u=rng.uniform(-0.5, 0.5, (n_lat, n_lon + 1, nlev)),
            v=rng.uniform(-0.5, 0.5, (n_lat + 1, n_lon, nlev)),
            eta=np.zeros((n_lat, n_lon)),
            land_mask=np.ones((n_lat, n_lon)),
            H_bathy=np.full((n_lat, n_lon), 4000.0),
            time_days=np.asarray(30.0 * i),
            time_seconds=np.asarray(30.0 * 86400 * i),
        )

    lat1d = np.degrees(np.asarray(grid.lat))
    lat199 = np.concatenate([lat1d, [lat1d[-1] + 1.0]])
    # NEMO files carry jpk = wet+1 levels (dummy bottom) in the
    # masked-zco convention; match the legacy count otherwise.
    nlev_nemo = nlev + 1 if vertical == "masked_zco" else nlev
    dz_n = np.concatenate([np.asarray(z.dz_ref),
                           np.asarray(z.dz_ref)[-1:]])[:nlev_nemo]
    depth = np.cumsum(dz_n) - 0.5 * dz_n
    nt, nx = 3, n_lon + 2
    lat2d = np.repeat(lat199[:, None], nx, axis=1)
    gridt = tmp_path / "grid_T.nc"
    gridu = tmp_path / "grid_U.nc"
    for path, vars3d in ((gridt, ("toce", "soce", "e3t")),
                         (gridu, ("uoce", "e3u"))):
        with netCDF4.Dataset(path, "w") as ds:
            ds.createDimension("time_counter", nt)
            ds.createDimension("deptht", nlev_nemo)
            ds.createDimension("y", 199)
            ds.createDimension("x", nx)
            v = ds.createVariable("nav_lat", "f8", ("y", "x")); v[:] = lat2d
            v = ds.createVariable("deptht", "f8", ("deptht",)); v[:] = depth
            for name in vars3d:
                v = ds.createVariable(
                    name, "f8", ("time_counter", "deptht", "y", "x"))
                if name.startswith("e3"):
                    v[:] = np.broadcast_to(
                        dz_n[None, :, None, None],
                        (nt, nlev_nemo, 199, nx))
                else:
                    v[:] = rng.uniform(1, 20, (nt, nlev_nemo, 199, nx))

    out = tmp_path / "figs"
    monkeypatch.setattr(sys, "argv", [
        "compare_dino_nemo.py",
        "--legoesm-dir", str(tmp_path / "lego"),
        "--nemo-gridt", str(gridt), "--nemo-gridu", str(gridu),
        "--month", "2", "--out-dir", str(out)])
    cdn.main()
    for f in ("maps", "sections", "timeseries", "profiles", "mld"):
        assert (out / f"dino_compare_{f}.png").exists(), f
