"""The host JRA55 block preload must deliver per-CELL forcing stacks for an MPAS
mesh (it fed lat-lon slices to the per-cell zenith/bulk builder; first MPAS
SPMD GPU smoke, job 27326306: broadcast [655364] vs [180,360]).

One block is preloaded from a tiny synthetic lat-lon cache onto an ico2 mesh
using the same regrid weights the driver builds; every atmosphere leaf and the
runoff stack must come out (n_steps, nCells).  Fails on the old code with the
broadcast error.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

jax.config.update("jax_enable_x64", True)


def _synthetic_cache():
    spec = importlib.util.spec_from_file_location(
        "_jra55_dispatch_harness", _ROOT / "tests" / "unit" / "test_run_omip_jra55_dispatch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._make_synthetic_cache


@pytest.mark.parametrize("preload", ["host"])
def test_block_preload_regrids_to_mpas_cells(tmp_path, preload):
    import xarray as xr
    from legoesm.grids.regridding import compute_latlon_to_voronoi_weights
    from legoesm.grids.voronoi import create_voronoi_mesh
    from scripts.run import run_omip

    n_lat, n_lon, dt, n_steps = 8, 16, 10800.0, 3
    cache = _synthetic_cache()(tmp_path, n_lat=n_lat, n_lon=n_lon, n_records=8)
    ds = xr.open_zarr(str(cache), consolidated=True)
    mesh = create_voronoi_mesh(2)
    rw = compute_latlon_to_voronoi_weights(
        np.deg2rad(np.asarray(ds["lat"])), np.deg2rad(np.asarray(ds["lon"])),
        np.asarray(mesh.latCell), np.asarray(mesh.lonCell))
    jra55_state = {
        "cache_path": str(cache), "ref_year": int(ds.attrs.get("ref_year", 1958)),
        "lat_2d": np.asarray(mesh.latCell), "lon_2d": np.asarray(mesh.lonCell),
        "co2_ppmv": 400.0, "grid_type": "mpas", "cycle": True,
        "regrid_weights": rw,
    }
    atm_stack, runoff_stack = run_omip._preload_jra55_forcing_block(
        0, n_steps, dt, jra55_state)
    for k, v in atm_stack.items():
        assert tuple(v.shape) == (n_steps, mesh.nCells), (k, v.shape)
        assert bool(np.all(np.isfinite(np.asarray(v)))), k
    assert tuple(runoff_stack.shape) == (n_steps, mesh.nCells)
