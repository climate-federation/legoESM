"""Prescribed land on the MPAS run loop (ModelDriver._run_column): what the
atmosphere is handed each step, against the file, and that ocean cells and the
feature-off run are untouched."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

jax.config.update("jax_enable_x64", True)

from legoesm.driver.config import (  # noqa: E402
    DycoreConfig, ExperimentConfig, GridConfig,
)
from legoesm.driver.model_driver import ModelDriver  # noqa: E402
from legoesm.forcing.time_utils import (  # noqa: E402
    NOLEAP_DAYS_PER_MONTH, NOLEAP_MONTH_STARTS,
)
from legoesm.thermo import latent_heat_vaporization  # noqa: E402

_MID = np.asarray(NOLEAP_MONTH_STARTS[:12]) + 0.5 * np.asarray(NOLEAP_DAYS_PER_MONTH)
# far from any ocean anchor, so a fallback to the ocean value is visible
_TS, _SH, _LH, _ALB = 330.0, -25.0, 60.0, 0.55


def _write_file(tmp_path):
    lat = np.arange(-89.5, 90.0, 1.0)
    lon = np.arange(0.5, 360.0, 1.0)
    shape = (_MID.size, lat.size, lon.size)
    up = {"units": "W m-2", "positive": "up"}
    ds = xr.Dataset(
        {"ts": (("time", "lat", "lon"), np.full(shape, _TS), {"units": "K"}),
         "hfss": (("time", "lat", "lon"), np.full(shape, _SH), up),
         "hfls": (("time", "lat", "lon"), np.full(shape, _LH), up),
         "albedo": (("time", "lat", "lon"), np.full(shape, _ALB), {"units": "1"})},
        coords={"time": _MID, "lat": lat, "lon": lon})
    p = tmp_path / "presc.nc"
    ds.to_netcdf(p)
    return str(p)


def _run(tmp_path, monkeypatch, name, **kw):
    """3 steps of the MPAS loop; returns (driver, [forcing dict per step])."""
    def _half_land(self):
        n = self.grid.nCells
        self._phis_data = jnp.zeros(n)
        f = np.zeros(n)
        f[: n // 3] = 1.0
        f[n // 3: n // 2] = 0.5
        self._f_land = jnp.asarray(f)

    monkeypatch.setattr(ModelDriver, "_create_topography", _half_land)
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=3, nlev=5),
        dycore=DycoreConfig(dt=600.0, discretization="mpas"),
        dataset="analytical", radiation="gray", turbulence="louis",
        topography="synthetic.nc", days=1800.0 / 86400.0,
        use_multilayer_land=False, **kw)
    driver = ModelDriver(cfg, output_dir=tmp_path / name)
    driver.setup()
    seen = []
    step0 = driver.model.step

    def _spy(state, dt, *a, forcing=None, **k):
        seen.append({key: np.asarray(v) for key, v in (forcing or {}).items()
                     if hasattr(v, "shape")})
        return step0(state, dt, *a, forcing=forcing, **k)

    monkeypatch.setattr(driver.model, "step", _spy)
    assert driver.run() == "COMPLETED"
    monkeypatch.undo()
    return driver, seen


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("presc")
    try:
        on = _run(tmp, mp, "on", land_prescribed_channels="ts_fluxes",
                  land_prescribed_flux_file=_write_file(tmp),
                  land_prescribed_ice_sheet_bulk_sensible=False)
        off = _run(tmp, mp, "off")
    finally:
        mp.undo()
    return on, off


def test_atmosphere_receives_the_file_over_land_only(runs):
    (drv, seen_on), (_, seen_off) = runs
    assert len(seen_on) == len(seen_off) == 3
    f = np.asarray(drv._f_land).reshape(-1)
    land, ocean = f == 1.0, f == 0.0
    # the run loop is fp32: 1e-6 relative
    L = float(latent_heat_vaporization(jnp.asarray(_TS)))
    for on, off in zip(seen_on, seen_off):
        # skin temperature: the file's over the land fraction, ocean untouched
        np.testing.assert_allclose(on["T_sfc"][land], _TS, rtol=1e-6)
        np.testing.assert_array_equal(on["T_sfc"][ocean], off["T_sfc"][ocean])
        mixed = f == 0.5
        np.testing.assert_allclose(
            on["T_sfc"][mixed], 0.5 * off["T_sfc"][mixed] + 0.5 * _TS,
            rtol=1e-6)
        # fluxes: the file's values, positive up, unscaled; water = hfls/L_v
        np.testing.assert_allclose(on["shflx_land"], _SH, rtol=1e-6)
        np.testing.assert_allclose(on["lhflx_land"], _LH, rtol=1e-6)
        np.testing.assert_allclose(on["evap_land"] * L, _LH, rtol=1e-6)
        assert "shflx_land_bulk_weight" not in on
        for k in ("shflx_land", "lhflx_land", "evap_land"):
            assert k not in off, k
        assert "sfc_albedo" in on and "sfc_albedo" in off
        np.testing.assert_allclose(on["sfc_albedo"][land], _ALB, rtol=1e-6)
        np.testing.assert_array_equal(on["sfc_albedo"][ocean],
                                      off["sfc_albedo"][ocean])


def test_published_skin_is_the_file_value(runs):
    (drv, _), _ = runs
    got = np.asarray(drv._land_T_skin_last)
    np.testing.assert_allclose(got, _TS, rtol=0, atol=1e-9)


def test_off_run_has_no_prescribed_state(runs):
    _, (drv_off, _) = runs
    assert drv_off._land_presc is None
    assert getattr(drv_off, "_land_T_skin_last", None) is None


def test_ice_sheet_option_auto_on_with_ts_fluxes_marks_only_ice_sheets(
        tmp_path, monkeypatch):
    """Auto under 'ts_fluxes': the run hands the turbulence the glacier mask
    of the surface data (static, every step) with the snow roughness; the
    marked cells are Antarctica, Greenland and (at most) other polar ice caps."""
    from tests._land_surfdata import require_surfdata
    drv, seen = _run(tmp_path, monkeypatch, "sheet",
                     land_prescribed_channels="ts_fluxes",
                     land_prescribed_flux_file=_write_file(tmp_path),
                     surfdata_path=require_surfdata())
    lat = np.rad2deg(np.asarray(drv.grid.latCell))
    lon = np.rad2deg(np.asarray(drv.grid.lonCell)) % 360.0
    w0 = seen[0]["shflx_land_bulk_weight"]
    assert set(np.unique(w0)) <= {0.0, 1.0}
    marked = w0 == 1.0
    assert (marked & (lat < -60.0)).sum() > 0                 # Antarctica
    greenland = (lat > 58.0) & (lon > 285.0) & (lon < 350.0)
    assert (marked & greenland).sum() > 0
    polar = (lat < -60.0) | (lat > 58.0)
    assert not (marked & ~polar).any(), lat[marked & ~polar]
    for f in seen:
        np.testing.assert_array_equal(f["shflx_land_bulk_weight"], w0)
        np.testing.assert_allclose(f["shflx_land_bulk_z0"], 0.0024)


def test_refresh_follows_each_steps_own_day_across_midnight(tmp_path, monkeypatch):
    """A run crossing 1 January 24:00 hands step k the file's value at
    floor(day_k) -- the same rule a restart's seed uses -- with the
    December->January wrap of a time-varying file."""
    from legoesm.forcing.prescribed_land import prescribed_land_at_day
    path = _write_file(tmp_path)
    with xr.open_dataset(path) as ds:
        ds = ds.load()
    ds["ts"] = ds["ts"] + 10.0 * np.arange(12)[:, None, None]   # Jan 330 ... Dec 440
    p2 = str(tmp_path / "presc_t.nc")
    ds.to_netcdf(p2)
    dt_day = 600.0 / 86400.0
    drv, seen = _run(tmp_path, monkeypatch, "midnight",
                     land_prescribed_channels="ts",
                     land_prescribed_flux_file=p2,
                     start_day=1.0 - dt_day)
    f = np.asarray(drv._f_land).reshape(-1)
    land = f == 1.0
    want = [float(prescribed_land_at_day(drv._land_presc, d)[0][0]) for d in (0.0, 1.0, 1.0)]
    assert want[0] != want[1]
    for k, fk in enumerate(seen):
        np.testing.assert_allclose(fk["T_sfc"][land], want[k], rtol=1e-6, err_msg=str(k))   # fp32 lane


def test_restart_at_midnight_sees_the_unbroken_runs_channels(tmp_path, monkeypatch):
    """Restart identity of the prescribed channels: a run started at day 1.0
    (the seed a restart uses: floor of its first step's day) hands its first
    step the same skin temperature as the unbroken run's step at day 1.0.
    The campaign rule (refresh AFTER the step, from that step's day) gave the
    unbroken run day-0 values there and the restart day-1 values."""
    path = _write_file(tmp_path)
    with xr.open_dataset(path) as ds:
        ds = ds.load()
    ds["ts"] = ds["ts"] + 10.0 * np.arange(12)[:, None, None]
    p2 = str(tmp_path / "presc_r.nc")
    ds.to_netcdf(p2)
    dt_day = 600.0 / 86400.0
    kw = dict(land_prescribed_channels="ts", land_prescribed_flux_file=p2)
    drv, unbroken = _run(tmp_path, monkeypatch, "unbroken",
                         start_day=1.0 - dt_day, **kw)
    _, restart = _run(tmp_path, monkeypatch, "restart", start_day=1.0, **kw)
    land = np.asarray(drv._f_land).reshape(-1) == 1.0
    np.testing.assert_array_equal(restart[0]["T_sfc"][land],
                                  unbroken[1]["T_sfc"][land])
    assert not np.array_equal(unbroken[0]["T_sfc"][land],
                              unbroken[1]["T_sfc"][land])
