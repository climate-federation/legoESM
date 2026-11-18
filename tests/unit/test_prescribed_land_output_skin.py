"""Prescribed land: the published surface temperature must carry the file's
land skin temperature, not the ocean anchor the output falls back to."""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import xarray as xr

from legoesm.driver.config import (
    ExperimentConfig, GridConfig, DycoreConfig,
)
from legoesm.driver.model_driver import ModelDriver
from legoesm.forcing.prescribed_land import prescribed_land_at_day

_MID = np.array([15.5, 45.0, 74.5, 105.0, 135.5, 166.0, 196.5, 227.5, 258.0,
                 288.5, 319.0, 349.5])


def _write_file(tmp_path):
    lat = np.arange(-89.5, 90.0, 1.0)
    lon = np.arange(0.5, 360.0, 1.0)
    shape = (_MID.size, lat.size, lon.size)
    # A skin temperature far from any ocean anchor, so a fallback is visible.
    ts = np.full(shape, 330.0)
    ds = xr.Dataset(
        {"ts": (("time", "lat", "lon"), ts, {"units": "K"}),
         "hfss": (("time", "lat", "lon"), np.full(shape, 20.0),
                  {"units": "W m-2", "positive": "up"}),
         "hfls": (("time", "lat", "lon"), np.full(shape, 60.0),
                  {"units": "W m-2", "positive": "up"})},
        coords={"time": _MID, "lat": lat, "lon": lon})
    p = tmp_path / "presc.nc"
    ds.to_netcdf(p)
    return str(p)


def test_prescribed_land_skin_reaches_published_surface_temperature(monkeypatch, tmp_path):
    def _half_land(self):
        n = self.grid.nCells
        self._phis_data = jnp.zeros(n)
        self._f_land = jnp.asarray((np.arange(n) < n // 2).astype(np.float64))

    monkeypatch.setattr(ModelDriver, "_create_topography", _half_land)
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=3, nlev=5),
        dycore=DycoreConfig(dt=600.0, discretization="mpas"),
        dataset="analytical", radiation="gray",
        days=1200.0 / 86400.0, use_multilayer_land=False,
        land_prescribed_flux_file=_write_file(tmp_path),
        land_prescribed_channels="ts_lh")
    driver = ModelDriver(cfg, output_dir=tmp_path / "run")
    driver.setup()
    assert driver.run() == "COMPLETED"
    got = getattr(driver, "_land_T_skin_last", None)
    assert got is not None, "published ts/tas never see the prescribed land skin"
    want = prescribed_land_at_day(driver._land_presc, float(np.floor(cfg.start_day)))[0]
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=0, atol=1e-9)
