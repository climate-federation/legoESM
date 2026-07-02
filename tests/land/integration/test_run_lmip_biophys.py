"""End-to-end smoke for the CRU-JRA-forced biophysics-only land driver (M3).

Runs ``scripts/run/run_lmip_biophys.py`` on a tiny latlon grid with synthetic
forcing (no external data) and a synthetic surfdata, and asserts the
forcing -> regrid -> disaggregate -> lax.scan land-step pipeline produces finite
fields over land and writes its time-series NetCDF.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from legoesm.land.surface_data.schema import write_surfdata
from legoesm.land.surface_params import N_PFT_CLM5

pytest.importorskip("xarray")

_ROOT = Path(__file__).resolve().parents[3]
_DRIVER = _ROOT / "scripts" / "run" / "run_lmip_biophys.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("run_lmip_biophys", _DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_surfdata(path, nlat=8, nlon=16, nlev=7):
    lat = np.linspace(-85, 85, nlat); lon = np.linspace(0, 337.5, nlon)
    pft = np.zeros((1, N_PFT_CLM5, nlat, nlon)); pft[0, 4] = 100.0
    lai = np.zeros((12, N_PFT_CLM5, nlat, nlon)); lai[:, 4] = 4.0
    soil = lambda v: np.full((nlev, nlat, nlon), v)
    write_surfdata(
        path, lat=lat, lon=lon, soil_dz=np.full(nlev, 0.2),
        sand_pct=soil(40.0), clay_pct=soil(20.0), organic=soil(5.0),
        bulk_density=soil(1300.0), soil_color=np.full((nlat, nlon), 8.0),
        year=np.array([2015.0]),
        f_land=np.full((1, nlat, nlon), 100.0),
        f_lake=np.zeros((1, nlat, nlon)), f_glacier=np.zeros((1, nlat, nlon)),
        pft_frac=pft, monthly_lai=lai, monthly_sai=np.zeros_like(lai),
        monthly_height_top=np.zeros_like(lai), monthly_height_bot=np.zeros_like(lai),
    )


def test_biophys_driver_synthetic_smoke(tmp_path):
    mod = _load_driver()
    sd = tmp_path / "sd.nc"; _write_surfdata(str(sd))
    out = tmp_path / "out"
    args = mod.build_parser().parse_args([
        "--surfdata", str(sd),
        "--grid-type", "latlon", "--resolution", "4",
        "--surface-scheme", "simple_seb", "--land-mode", "multilayer",
        "--bulk", "constant",               # constant Cd/Ch — hermetic pipeline smoke,
                                            # not a MOST-vs-constant comparison
        "--dt", "3600", "--n-steps", "4",
        "--forcing-dir", "",                # force synthetic (hermetic, ignore any staged data/crujra)
        "--output", str(out),
    ])
    rc = mod.run(args)
    assert rc == 0                                            # PASS (no NaN over land)

    import xarray as xr
    ds = xr.open_dataset(out / "lmip_biophys.nc")
    assert ds.sizes["time"] == 4
    assert ds.attrs["carbon"] == "none"                      # biophysics-only
    # latlon output uses the (time, lat, lon) rectangular layout — never (time, ncol)
    # (the driver reshape gate for latlon; non-rectangular grids keep ncol)
    assert ds["T_sfc"].dims == ("time", "lat", "lon")
    assert ds.sizes["lat"] == 4 and ds.sizes["lon"] == 8    # --resolution 4 -> 4 x 8
    assert np.all(np.isfinite(ds["T_sfc"].values))           # all-land surfdata
    # surface temperature stays physical under the (synthetic) forcing
    assert float(ds["T_sfc"].min()) > 200.0
    assert float(ds["T_sfc"].max()) < 360.0


def test_build_model_times_synthetic_starts_at_zero():
    mod = _load_driver()
    t = mod.build_model_times(196.0, 3600.0, 5, synthetic=True)
    assert t[0] == 0.0                                        # synthetic ignores start-doy
    t2 = mod.build_model_times(10.0, 3600.0, 5, synthetic=False)
    assert t2[0] == 10.0 * 86400.0                           # real forcing honours start-doy


def _run_driver(mod, sd, out, extra_args=()):
    args = mod.build_parser().parse_args([
        "--surfdata", str(sd),
        "--grid-type", "latlon", "--resolution", "4",
        "--surface-scheme", "simple_seb", "--land-mode", "multilayer",
        "--bulk", "constant",
        "--dt", "3600", "--n-steps", "4",
        "--forcing-dir", "",
        "--output", str(out),
        *extra_args,
    ])
    return mod.run(args)


def test_cold_run_auto_saves_restart(tmp_path):
    """A cold run always writes restart_end.npz next to lmip_biophys.nc."""
    mod = _load_driver()
    sd = tmp_path / "sd.nc"; _write_surfdata(str(sd))
    out = tmp_path / "cold"
    assert _run_driver(mod, sd, out) == 0
    assert (out / "restart_end.npz").exists()

    # Metadata survives the round-trip and records the run's config.
    from legoesm.land.restart import load_land_restart
    _, meta = load_land_restart(out / "restart_end.npz",
                                expected_land_mode="multilayer",
                                expected_ncol=32, expected_n_layers=None)
    assert meta["metadata"]["surface_scheme"] == "simple_seb"
    assert meta["n_steps_completed"] == 4


def test_warm_start_uses_loaded_state(tmp_path):
    """When --restart-from is set, the driver's initial state IS the loaded
    state — not the cold-init default.  Verified with a hand-crafted seed
    at a distinctive T_soil (250 K uniform) that a cold start would never
    reach in 4 h of physics under synthetic forcing (~285 K air)."""
    import jax.numpy as jnp
    from legoesm.land.restart import save_land_restart
    from legoesm.land.state import MultiLayerLandState

    mod = _load_driver()
    sd = tmp_path / "sd.nc"; _write_surfdata(str(sd))
    ncol, nlay = 32, 8              # 4x8 latlon at --resolution 4
    seed = MultiLayerLandState(
        T_soil=jnp.full((ncol, nlay), 250.0),   # unmistakable cold column
        psi_soil=jnp.full((ncol, nlay), -1.0),
        theta_soil=jnp.full((ncol, nlay), 0.30),
        runoff_surface=jnp.zeros(ncol),
        runoff_subsurface=jnp.zeros(ncol),
        snow_depth=jnp.zeros(ncol),
        snow_age=jnp.zeros(ncol),
    )
    seed_path = tmp_path / "seed.npz"
    save_land_restart(seed_path, seed, land_mode="multilayer",
                      t_end_s=0.0, n_steps_completed=0)

    out = tmp_path / "warm"
    assert _run_driver(mod, sd, out,
                       extra_args=("--restart-from", str(seed_path))) == 0

    # Load the run's END-of-scan state (auto-saved).  Deep soil layers are
    # thermally slow; after 4 h from a 250 K seed they must still be near 250 K
    # (a fresh cold start would have deep T at ~280 K, the T_init default in
    # init_multilayer_land_state).
    from legoesm.land.restart import load_land_restart
    st_end, _ = load_land_restart(out / "restart_end.npz",
                                  expected_land_mode="multilayer",
                                  expected_ncol=32, expected_n_layers=None)
    T_deep = np.asarray(st_end.T_soil)[:, -1]                    # deepest layer
    assert float(T_deep.mean()) < 255.0                          # near seed
    assert float(T_deep.mean()) > 245.0
