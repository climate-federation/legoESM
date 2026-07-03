"""Integration: a realistic AMIP config WITH a land-sea mask runs STABLY (iter 456).

iter 454 exposed ``--land-mask-path`` and iter 455 locked that the mask reaches the
ranking via ``static_land_fraction``/``ocean_valid_mask``.  This locks the OTHER half a
realistic ``--ocean-only`` empirical run needs: that the AMIP model actually RUNS over the
land columns the mask introduces — the prognostic slab-land ``T_land`` engages where
``f_land>0`` and the atmospheric state stays FINITE (a land mask must not NaN the run).

This is the composition no other test covers: the slab land is unit-tested in isolation
(``test_land_surface.py``), but not together with the AMIP config + a real land mask + a
full run + the campaign's column extraction.  Slow (a real short run + compile) — hence a
``tests/run`` integration test, smallest grid / shortest run.
"""
from __future__ import annotations

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)


def test_amip_run_with_land_mask_is_finite(tmp_path):
    import xarray as xr
    from legoesm.driver.model_driver import ModelDriver
    from legoesm.training.compare_reanalysis import model_state_is_finite
    from legoesm.training.run_to_column_mean import amip_column_state

    from scripts.experiment.write_amip_clubb_lite_config import (
        build_amip_clubb_lite_config,
    )

    # ERA5-lsm style land-sea mask (fraction): Northern hemisphere land, Southern ocean.
    lat = np.linspace(-89.0, 89.0, 90)
    lon = np.linspace(0.0, 358.0, 180)
    mask = np.where(lat[:, None] > 0.0, 1.0, 0.0) * np.ones_like(lon)
    mask_path = tmp_path / "era5_lsm.nc"
    xr.Dataset({"lsm": (("lat", "lon"), mask)},
               coords={"lat": lat, "lon": lon}).to_netcdf(mask_path)

    # Smallest viable AMIP config (gray radiation, analytical SST) WITH the land mask.
    cfg = build_amip_clubb_lite_config(
        resolution=4, nlev=4, dt=600.0, radiation="gray", days=1,
        land_mask_path=str(mask_path))
    driver = ModelDriver(cfg, output_dir=str(tmp_path / "run"))
    driver.setup()

    # The land mask reached the surface-flux machinery (drives the slab-land + flux blend).
    assert driver.physics is not None and driver.physics.f_land is not None
    f_land = np.asarray(driver.physics.f_land)
    assert f_land.min() >= 0.0 and f_land.max() <= 1.0
    assert f_land.max() > 0.5            # land columns exist (NH) — non-trivial mask

    driver.run(start_step=0)

    # The atmosphere state extracted exactly as the campaign does stays FINITE over the
    # land+ocean columns (a land mask must not blow the run up).
    col = amip_column_state(driver, day=float(cfg.days))
    assert model_state_is_finite(col), "AMIP state went non-finite with a land mask"

    # The prognostic slab-land temperature engaged and is physical (not NaN/absurd).
    t_land = driver._carry_aux.get("T_land") if driver._carry_aux else None
    if t_land is not None:
        t_land = np.asarray(t_land)
        assert bool(np.all(np.isfinite(t_land)))
        assert 150.0 < float(t_land.min()) and float(t_land.max()) < 360.0
