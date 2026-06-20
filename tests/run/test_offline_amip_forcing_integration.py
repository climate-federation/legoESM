"""End-to-end: an offline ERA5-built AMIP forcing drives a real ModelDriver run (iter 420).

iter-419 added ``build_era5_amip_forcing`` (the FORCING-side twin of
``open_local_era5_dataset``) and validated it through ``load_amip_forcing`` in ISOLATION.
But no test ran an actual ``ModelDriver.setup().run()`` with a custom forcing — so the
composition (build forcing -> wire into ExperimentConfig -> run a model integration ->
extract the column state) was unproven.  This locks that path with a DIFFERENTIAL that
proves the run CONSUMES the prescribed SST (not merely loads the forcing object): two AMIP
runs that differ ONLY by a uniform +15 K SST bump must yield a warmer boundary layer
(``get_sst_sic`` is a closure over the forcing, so a finite/gradiented value there does not
prove the segment step read it — the standard-atmosphere IC already carries an equator-pole
gradient — codex-review iter 420).  CI-portable (synthetic archive, coarse grid, gray
radiation, 1 day).
"""

from __future__ import annotations

import numpy as np
import pytest


def _write_synthetic_amip_archive(tmp_path, *, nt=48, sst_add=0.0):
    """A monthly-style RDA ``sstk`` (a latitudinal SST gradient, Kelvin, optionally bumped by
    ``sst_add``) + ``ci`` (zero sea-ice) so ``build_era5_amip_forcing`` produces a
    NON-trivial, regrid-testing forcing.  Overwrites in place (called twice for the bump)."""
    import xarray as xr

    nlat, nlon = 16, 32
    lat = np.linspace(90.0, -90.0, nlat)         # ERA5 order: descending
    lon = np.linspace(0.0, 348.75, nlon)
    t = (np.datetime64("2020-01-01T00")
         + np.arange(nt) * np.timedelta64(1, "h")).astype("datetime64[ns]")
    sst2d = (300.0 - 25.0 * np.abs(lat) / 90.0)[:, None] * np.ones((1, nlon))  # 275..300 K
    sst = np.broadcast_to(sst2d + sst_add, (nt, nlat, nlon)).astype("f4")
    xr.Dataset({"SSTK": (("time", "latitude", "longitude"), sst, {"units": "K"})},
               coords={"time": t, "latitude": lat, "longitude": lon}).to_netcdf(
        tmp_path / "e5.oper.an.sfc.128_034_sstk.ll025sc.2020010100_2020013123.nc")
    xr.Dataset({"CI": (("time", "latitude", "longitude"),
                       np.zeros((nt, nlat, nlon), "f4"), {"units": "(0-1)"})},
               coords={"time": t, "latitude": lat, "longitude": lon}).to_netcdf(
        tmp_path / "e5.oper.an.sfc.128_031_ci.ll025sc.2020010100_2020013123.nc")


def _run_amip(tmp_path, forcing_name, fcfg):
    """Wire an AMIPForcingConfig onto an ExperimentConfig via the SHARED field map
    (``apply_amip_forcing_to_config`` — the same the campaign applies) and run a tiny
    hydrostatic gray-radiation AMIP ModelDriver for one day."""
    from legoesm.driver.config import (
        DycoreConfig,
        ExperimentConfig,
        GridConfig,
        OutputConfig,
    )
    from legoesm.driver.model_driver import ModelDriver

    from scripts.data.load_local_era5 import apply_amip_forcing_to_config

    base = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        output=OutputConfig(diag_days=1), radiation="gray", days=1)
    cfg = apply_amip_forcing_to_config(base, fcfg)
    driver = ModelDriver(cfg)
    driver.setup()
    driver.run()
    return driver


@pytest.mark.slow
def test_offline_era5_forcing_drives_amip_modeldriver_run(tmp_path):
    from legoesm.training.run_to_column_mean import amip_column_state

    from legoesm import constants
    from scripts.data.load_local_era5 import build_era5_amip_forcing

    # --- Base run (the gradient SST). ---
    _write_synthetic_amip_archive(tmp_path)
    fcfg = build_era5_amip_forcing(
        str(tmp_path), "20200101", str(tmp_path / "amip_forcing.nc"), hour_stride=24)
    assert fcfg.dataset == "custom" and fcfg.sst_var == "SSTK" and fcfg.sic_var == "CI"
    driver_a = _run_amip(tmp_path, "amip_forcing.nc", fcfg)

    # The prescribed ERA5 SST loaded + regridded: finite, the equator->pole gradient survived
    # the regrid to the (coarser) model grid, and the seawater freeze floor was applied.
    sst_applied = np.asarray(driver_a.get_sst_sic(0.0)[0])
    assert bool(np.all(np.isfinite(sst_applied)))
    assert float(sst_applied.max()) > float(sst_applied.min())        # gradient, not constant
    assert float(sst_applied.min()) >= constants.T_freeze_ocean       # freeze floor applied
    assert float(sst_applied.max()) <= 301.0
    # The AMIP column state (atm + prescribed SST) extracts finite — the compare-side input.
    col_a = amip_column_state(driver_a, day=0.0)
    assert bool(np.all(np.isfinite(np.asarray(col_a.T))))
    assert bool(np.all(np.isfinite(np.asarray(col_a.q_v))))
    t_bot_a = float(np.asarray(col_a.T)[..., -1].mean())

    # --- DIFFERENTIAL run: the SAME setup with a uniform +15 K SST bump. ---
    # If the run merely LOADED the forcing (without the segment step reading it), the two
    # boundary layers would be identical; a warmer ocean MUST warm the BL — proving the SST
    # is consumed.  The smoke measured ~+6.8 K mean bottom-T for +15 K SST; assert >> noise.
    _write_synthetic_amip_archive(tmp_path, sst_add=15.0)
    fcfg_warm = build_era5_amip_forcing(
        str(tmp_path), "20200101", str(tmp_path / "amip_forcing_warm.nc"), hour_stride=24)
    driver_b = _run_amip(tmp_path, "amip_forcing_warm.nc", fcfg_warm)
    t_bot_b = float(np.asarray(amip_column_state(driver_b, day=0.0).T)[..., -1].mean())

    assert t_bot_b > t_bot_a + 2.0, (
        f"a +15 K SST forcing should warm the boundary layer, but mean bottom-level T went "
        f"{t_bot_a:.3f} -> {t_bot_b:.3f} K (delta {t_bot_b - t_bot_a:.3f}); the run is not "
        "consuming the prescribed SST.")
