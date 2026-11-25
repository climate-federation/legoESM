"""Coupling seam: 10 m wind, axis conventions, remap grid guards."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.coupler.grid_remap import make_latlon_remapper, remap_field
from legoesm.coupler.wave_coupling import (
    neutral_wind_10m,
    remap_to_wave,
    wave_remap_grid,
    wave_to_remap,
)
from legoesm.wave.config import WaveConfig, WaveGridConfig
from tests.wave.unit.conftest import tiny_grid, tiny_network

_POLE_LAT_DEG = 90.0


class _RegularGrid(NamedTuple):
    lon: jnp.ndarray
    lat_v: jnp.ndarray
    n_lat: int
    n_lon: int


def test_neutral_wind_matches_log_profile():
    tau_x = jnp.asarray([[0.1, 0.0]])
    tau_y = jnp.asarray([[0.0, 0.1]])
    rho = jnp.asarray([[1.2, 1.2]])
    z0 = jnp.asarray([[1e-4, 1e-4]])
    u10, v10 = neutral_wind_10m(tau_x, tau_y, rho, z0)
    speed = np.sqrt(0.1 / 1.2)
    expected = speed / constants.kappa * np.log(10.0 / 1e-4)
    assert np.isclose(float(np.asarray(u10)[0, 0]), -expected, rtol=1e-5)
    assert np.isclose(float(np.asarray(u10)[0, 1]), 0.0, atol=1e-6)
    assert np.isclose(float(np.asarray(v10)[0, 1]), -expected, rtol=1e-5)


def test_zero_stress_returns_zero_wind():
    """A calm cell must be exactly calm, not a ~0.03 m/s floor wind."""
    zeros = jnp.zeros((2, 3))
    u10, v10 = neutral_wind_10m(
        zeros, zeros, jnp.full((2, 3), 1.2), jnp.full((2, 3), 1e-4))
    np.testing.assert_array_equal(np.asarray(u10), np.zeros((2, 3)))
    np.testing.assert_array_equal(np.asarray(v10), np.zeros((2, 3)))


def test_wind_direction_opposes_atmospheric_drag():
    u10, v10 = neutral_wind_10m(
        jnp.asarray([[0.05]]), jnp.asarray([[-0.05]]),
        jnp.asarray([[1.2]]), jnp.asarray([[1e-4]]))
    assert float(np.asarray(u10)[0, 0]) < 0.0
    assert float(np.asarray(v10)[0, 0]) > 0.0


def test_axis_transpose_round_trip():
    field = jnp.arange(30, dtype=jnp.float32).reshape(6, 5)
    converted = wave_to_remap(field)
    assert converted.shape == (5, 6)
    assert np.array_equal(np.asarray(remap_to_wave(converted)), np.asarray(field))


def _global_wave_config(lon_step=4.0, lat_step=4.0):
    n_lon = int(round(360.0 / lon_step))
    n_lat = int(round(180.0 / lat_step))
    lon = tuple(2.0 + lon_step * i for i in range(n_lon))
    lat = tuple((90.0 - 0.5 * lat_step) - lat_step * i for i in range(n_lat))
    grid = WaveGridConfig(lon_deg=lon, lat_deg=lat,
                          freq_hz=tiny_grid().freq_hz,
                          theta_rad=tiny_grid().theta_rad)
    return WaveConfig(grid=grid, network=tiny_network())


def test_regional_wave_grid_is_rejected():
    config = WaveConfig(grid=tiny_grid(), network=tiny_network())
    with pytest.raises(NotImplementedError, match="global in latitude"):
        wave_remap_grid(config)


def test_ascending_latitude_is_rejected():
    config = _global_wave_config()
    ascending = config._replace(grid=config.grid._replace(
        lat_deg=tuple(reversed(config.grid.lat_deg))))
    with pytest.raises(ValueError, match="descending"):
        wave_remap_grid(ascending)


def test_non_global_longitude_is_rejected():
    config = _global_wave_config()
    partial = config._replace(grid=config.grid._replace(
        lon_deg=config.grid.lon_deg[:-1]))
    with pytest.raises(NotImplementedError, match="360 deg"):
        wave_remap_grid(partial)


def test_global_wave_grid_remaps_a_constant_field():
    dst = wave_remap_grid(_global_wave_config(lon_step=4.0, lat_step=4.0))
    # Coarser 10-degree source grid, ascending latitude.
    src_lon = jnp.deg2rad(jnp.linspace(5.0, 355.0, 36))
    src_lat_v = jnp.deg2rad(jnp.linspace(-_POLE_LAT_DEG, _POLE_LAT_DEG, 19))
    src = _RegularGrid(lon=src_lon, lat_v=src_lat_v, n_lat=18, n_lon=36)

    weights = make_latlon_remapper(src, dst)
    field = jnp.ones((18, 36))
    out = remap_field(field, weights)
    assert out.shape == (45, 90)
    assert np.allclose(np.asarray(out), 1.0, atol=1e-6)


def test_masked_remap_excludes_land_bathymetry_fill():
    """Coastal depth must be the WET mean, not sea floor blended with land."""
    from legoesm.coupler.grid_remap import masked_conservative_mean

    # 2 rows x 8 uniform 45-degree source cells, ascending latitude.
    src_lon = jnp.deg2rad(jnp.asarray(
        [22.5 + 45.0*i for i in range(8)], dtype=jnp.float64))
    src_lat_v = jnp.deg2rad(jnp.asarray(
        [-_POLE_LAT_DEG, 0.0, _POLE_LAT_DEG], dtype=jnp.float64))
    src = _RegularGrid(lon=src_lon, lat_v=src_lat_v, n_lat=2, n_lon=8)
    # Destination: two 180-degree cells per row.
    grid = WaveGridConfig(
        lon_deg=(90.0, 270.0), lat_deg=(45.0, -45.0),
        freq_hz=tiny_grid().freq_hz, theta_rad=tiny_grid().theta_rad)
    dst = wave_remap_grid(WaveConfig(grid=grid, network=tiny_network()))
    weights = make_latlon_remapper(src, dst)

    mask = jnp.ones((2, 8), dtype=jnp.float64)
    mask = mask.at[0, 0].set(0.0)          # one dry source cell
    depth = jnp.where(mask > 0, 100.0, 5500.0)  # ocean_H_max land fill

    mean, fraction = masked_conservative_mean(depth, mask, weights)
    naive = remap_field(depth, weights)
    # The first destination cell covers four source cells: 3 wet, 1 dry.
    assert np.isclose(float(fraction[0, 0]), 0.75, atol=1e-6)
    assert np.isclose(float(mean[0, 0]), 100.0, atol=1e-6)
    # Non-vacuity: the plain remap shows the contaminated value.
    assert np.isclose(float(naive[0, 0]), 0.25*5500.0 + 0.75*100.0,
                      atol=1e-3)


def test_actual_bulk_producer_recovers_wind_direction():
    from legoesm.core.bulk_flux import simple_bulk_fluxes
    u = jnp.asarray([8.], dtype=jnp.float32)
    zero = jnp.zeros_like(u)
    one = jnp.ones_like(u)
    tx, ty, _, _ = simple_bulk_fluxes(u, zero, one*290, one*.005,
                                     one*290, one*.005, one*1.2, u, .001, .001)
    u10, v10 = neutral_wind_10m(tx, ty, one*1.2, one*.0001)
    assert bool(jnp.all(u10*u > 0))
    np.testing.assert_array_equal(v10, zero)
