"""Tests for ERA5 surface-flux and land-fraction loading in era5_to_state.

Covers: sign conventions (positive-up fluxes, stress on the atmosphere),
latitude flipping of the flux store, land-sea mask lookup order
(state store first, then flux store), the default-off path returning
None fields, and error messages for missing variables.
"""
from __future__ import annotations

import legoesm.training.era5_to_state as e2s
import numpy as np
import pytest
import xarray as xr
from legoesm.training.era5_to_state import TrainingERA5Config

from tests.unit.test_era5_load_slice import _config, _synthetic_era5

LAT = np.linspace(-90, 90, 5)  # flux store: opposite of state store


def _flux_ds(var_fill=None, drop=(), with_land=False):
    """Flux store with opposite-order latitude and distinct lat-dependent data."""
    if var_fill is None:
        var_fill = {}
    n_lat, n_lon = 5, 6
    data_vars = {}
    defaults = {
        "mean_surface_sensible_heat_flux": 10.0,
        "mean_surface_latent_heat_flux": 20.0,
        "mean_eastward_turbulent_surface_stress": 30.0,
        "mean_northward_turbulent_surface_stress": 40.0,
        "mean_surface_downward_short_wave_radiation_flux": 50.0,
        "mean_surface_net_short_wave_radiation_flux": 60.0,
        "mean_surface_downward_long_wave_radiation_flux": 70.0,
        "mean_surface_net_long_wave_radiation_flux": 80.0,
    }
    for name, base in defaults.items():
        if name in drop:
            continue
        # lat-dependent so a missing latitude flip changes results
        data_vars[name] = (
            ("time", "lat", "lon"),
            (base + 0.01 * np.arange(n_lat)[:, None]
             + 0.001 * np.arange(n_lon)[None, :]).astype(np.float32)[None],
        )
    for name, val in var_fill.items():
        if name in drop:
            continue
        data_vars[name] = (("time", "lat", "lon"),
                           np.broadcast_to(np.float32(val), (1, n_lat, n_lon)))
    if with_land:
        data_vars["land_sea_mask"] = (("lat", "lon"), np.linspace(0.0, 1.0, n_lat * n_lon).reshape(n_lat, n_lon))
    return xr.Dataset(
        data_vars=data_vars,
        coords={"time": [0], "lat": LAT, "lon": np.linspace(0, 300, 6)},
    )


def _flux_config(**kw):
    return TrainingERA5Config(
        zarr_store="dummy", levels=(1000.0, 500.0, 100.0), **kw
    )


def _load(config, flux_ds):
    return e2s.load_era5_slice(
        config, 0,
        ds=_synthetic_era5("long"),
        flux_ds=flux_ds,
    )


def test_flags_off_gives_none_fields():
    sl = e2s.load_era5_slice(_config(), 0, ds=_synthetic_era5("long"))
    for f in ("sfc_shf", "sfc_lhf", "sfc_tau_x", "sfc_tau_y",
              "sfc_sw_up", "sfc_sw_down", "sfc_lw_up", "land_frac"):
        assert getattr(sl, f) is None


def test_sign_conventions_and_lat_flip():
    cfg = _flux_config(load_surface_fluxes=True, flux_accum_seconds=2.0)
    sl = _load(cfg, _flux_ds(with_land=True))
    ds = _flux_ds()

    def flipped(name):
        # flux store lat is -90..90; flip to state store order (90..-90)
        return ds[name].values[0][::-1] / 2.0

    np.testing.assert_allclose(sl.sfc_shf, -flipped("mean_surface_sensible_heat_flux"))
    np.testing.assert_allclose(sl.sfc_lhf, -flipped("mean_surface_latent_heat_flux"))
    np.testing.assert_allclose(
        sl.sfc_tau_x, -flipped("mean_eastward_turbulent_surface_stress"))
    np.testing.assert_allclose(sl.sfc_tau_y, -flipped("mean_northward_turbulent_surface_stress"))
    np.testing.assert_allclose(sl.sfc_sw_down, flipped("mean_surface_downward_short_wave_radiation_flux"))
    np.testing.assert_allclose(
        sl.sfc_sw_up,
        flipped("mean_surface_downward_short_wave_radiation_flux")
        - flipped("mean_surface_net_short_wave_radiation_flux"),
    )
    np.testing.assert_allclose(
        sl.sfc_lw_up,
        flipped("mean_surface_downward_long_wave_radiation_flux")
        - flipped("mean_surface_net_long_wave_radiation_flux"),
    )
    # non-uniform values were loaded (not accidentally constant)
    assert np.ptp(sl.sfc_shf) > 0.01


def test_land_frac_from_state_store_takes_precedence():
    state = _synthetic_era5("long", extra2d={"land_sea_mask": 0.75})
    cfg = _flux_config(load_land_frac=True)
    sl = e2s.load_era5_slice(cfg, 0, ds=state, flux_ds=_flux_ds(with_land=True))
    assert sl.land_frac.shape == (5, 6)
    np.testing.assert_allclose(sl.land_frac, 0.75)


def test_land_frac_from_flux_store_is_flipped():
    cfg = _flux_config(load_land_frac=True)
    sl = _load(cfg, _flux_ds(with_land=True))
    expected = _flux_ds(with_land=True)["land_sea_mask"].values[::-1]
    np.testing.assert_allclose(sl.land_frac, expected)
    # check the flip actually matters
    np.testing.assert_raises(AssertionError, np.testing.assert_allclose,
                             sl.land_frac,
                             _flux_ds(with_land=True)["land_sea_mask"].values)


def test_land_frac_alone_loads_without_fluxes():
    cfg = _flux_config(load_land_frac=True)
    state = _synthetic_era5("long", extra2d={"land_sea_mask": 0.5})
    sl = e2s.load_era5_slice(cfg, 0, ds=state)
    np.testing.assert_allclose(sl.land_frac, 0.5)
    assert sl.sfc_shf is None


def test_missing_flux_variable_mentions_flag():
    cfg = _flux_config(load_surface_fluxes=True)
    ds_bad = _flux_ds(drop=("mean_surface_latent_heat_flux",))
    with pytest.raises(ValueError, match="load_surface_fluxes=True"):
        _load(cfg, ds_bad)


def test_land_frac_absent_everywhere_mentions_flag():
    cfg = _flux_config(load_land_frac=True)
    with pytest.raises(ValueError, match="load_land_frac"):
        _load(cfg, _flux_ds())


def test_surface_fluxes_imply_land_frac():
    # load_surface_fluxes=True pulls land_frac too
    cfg = _flux_config(load_surface_fluxes=True)
    state = _synthetic_era5("long", extra2d={"lsm": 0.25})
    sl = e2s.load_era5_slice(cfg, 0, ds=state, flux_ds=_flux_ds())
    np.testing.assert_allclose(sl.land_frac, 0.25)


def test_non_finite_flux_plane_is_refused_with_the_field_named():
    cfg = _flux_config(load_surface_fluxes=True)
    bad = _flux_ds(with_land=True)
    v = bad["mean_surface_latent_heat_flux"].values.copy()
    v[0, 2, 3] = np.nan
    bad["mean_surface_latent_heat_flux"].values[:] = v
    with pytest.raises(ValueError, match="sfc_lhf.*non-finite"):
        _load(cfg, bad)


def test_flux_store_without_the_state_time_is_refused():
    """No nearest-hour substitution: the flux snapshot must be AT the state time."""
    cfg = _flux_config(load_surface_fluxes=True)
    ds = _flux_ds(with_land=True).assign_coords(time=[7])
    with pytest.raises(ValueError, match="no snapshot at the state time"):
        _load(cfg, ds)


def test_flux_store_on_a_shifted_grid_is_refused():
    """Same shape and latitude sense but different coordinates: refused."""
    cfg = _flux_config(load_surface_fluxes=True)
    ds = _flux_ds(with_land=True)
    shifted = ds.assign_coords(lon=ds.lon.values + 15.0)
    with pytest.raises(ValueError, match="grid coordinates differ"):
        _load(cfg, shifted)
