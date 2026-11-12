"""Static wave environment: metrics, dispersion attachment and depth guards."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.constants import R_earth_neuralswm
from legoesm.wave.grid import (
    build_wave_env,
    frequency_cell_widths,
    prepare_env,
    spectral_integral,
    wave_cell_area,
    with_dispersion,
)


def test_metrics_are_the_spherical_ones(tiny_config):
    env = build_wave_env(tiny_config)
    lat = np.asarray(tiny_config.grid.lat_deg, dtype=float)
    expected_x = R_earth_neuralswm * np.round(np.cos(np.deg2rad(lat)), 5)
    assert np.allclose(np.asarray(env.x_metric).ravel(), expected_x, rtol=1e-6)
    assert np.allclose(np.asarray(env.y_metric), R_earth_neuralswm, rtol=1e-6)
    assert env.lon.shape == (1, 6, 1, 1, 1)
    assert env.lat.shape == (1, 1, 5, 1, 1)


def test_prepare_env_attaches_dispersion(tiny_config, tiny_state):
    env = prepare_env(tiny_config, tiny_state.depth)
    assert env.wavenum.shape == (1, 6, 5, 1, 4)
    assert bool(jnp.all(env.wavenum > 0))
    assert bool(jnp.all(env.group_velocity > 0))


def test_depth_with_two_spatial_axes_is_rejected(tiny_config):
    env = build_wave_env(tiny_config)
    bad_depth = jnp.ones((1, 6, 5, 3, 1))
    with pytest.raises(ValueError, match="wave depth"):
        with_dispersion(env, bad_depth)


def test_depth_spatial_shape_must_match_the_environment(tiny_config):
    env = build_wave_env(tiny_config)
    with pytest.raises(ValueError, match="spatial shape"):
        with_dispersion(env, jnp.ones((1, 5, 6)))


def test_build_rejects_an_invalid_config(tiny_config):
    bad = tiny_config._replace(
        grid=tiny_config.grid._replace(freq_hz=(0.0, 0.1, 0.2, 0.3)))
    with pytest.raises(ValueError, match="freq_hz"):
        build_wave_env(bad)


def test_frequency_widths_include_both_endpoint_cells(tiny_config):
    freq = np.asarray(tiny_config.grid.freq_hz, dtype=float)
    widths = np.asarray(frequency_cell_widths(jnp.asarray(freq)))
    assert widths.shape == freq.shape
    assert np.all(widths > 0)
    first_edge = freq[0] - .5 * (freq[1] - freq[0])
    last_edge = freq[-1] + .5 * (freq[-1] - freq[-2])
    assert np.isclose(widths.sum(), last_edge - first_edge, rtol=1e-12)


def test_spectral_integral_ignores_land(tiny_config, tiny_state):
    env = prepare_env(tiny_config, tiny_state.depth)
    sea = jnp.ones_like(tiny_state.mask)
    land = jnp.zeros_like(tiny_state.mask)
    total = spectral_integral(tiny_state.spectra, env, sea)
    assert bool(jnp.all(total > 0))
    assert bool(jnp.all(spectral_integral(tiny_state.spectra, env, land) == 0))
    assert wave_cell_area(env).shape[:3] == tiny_state.mask.shape
