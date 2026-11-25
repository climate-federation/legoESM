"""Source-network feature maps: direction projection, gradients and shapes."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.wave.grid import prepare_env
from legoesm.wave.networks.embedding import (
    data_direction_radians,
    depth_gradients,
    sdn_features,
    wind_features,
    ww_inter_features,
)


def test_wind_features_project_onto_the_wave_direction(tiny_config, tiny_state):
    env = prepare_env(tiny_config, tiny_state.depth)
    theta = np.asarray(data_direction_radians(env)).reshape(-1)
    speed = 20.0
    u = jnp.full(tiny_state.mask.shape, speed)
    v = jnp.zeros_like(u)
    features = np.asarray(wind_features(env, u, v))
    cg = np.asarray(env.group_velocity)[0, 0, 0, 0, :]  # one spatial column
    # float32 feature arithmetic: 1e-3 still rejects a sin/cos swap or a
    # missing 1/20 scale, which change the value by O(1).
    expected = speed * np.sin(theta)[:, None] * cg[None, :] / 20.0
    assert np.allclose(features[0, 0], expected, rtol=1e-3, atol=1e-6)

    v_only = np.asarray(wind_features(env, jnp.zeros_like(u), u))[0, 0]
    expected_v = speed * np.cos(theta)[:, None] * cg[None, :] / 20.0
    assert np.allclose(v_only, expected_v, rtol=1e-3, atol=1e-6)


def test_depth_gradients_recover_a_constant_slope(tiny_config, tiny_state):
    env = prepare_env(tiny_config, tiny_state.depth)
    lon_rad = np.deg2rad(np.asarray(tiny_config.grid.lon_deg))
    slope = 0.25  # metres per radian of longitude
    bathymetry = jnp.broadcast_to(
        jnp.asarray(50.0 + slope * lon_rad)[None, :, None],
        tiny_state.depth.shape)
    dddx, dddy = depth_gradients(env, bathymetry)
    # The wrap pad only disturbs the two edge columns; float32 cancellation
    # on adjacent ~50 m depths limits the agreement to ~1e-4 relative.
    assert np.allclose(np.asarray(dddx)[0, 1:-1, :, 0, 0], slope, rtol=1e-3, atol=1e-6)
    assert np.allclose(np.asarray(dddy), 0.0, atol=1e-6)


def test_sdn_features_use_the_network_frequency_spacing(tiny_config, tiny_state):
    env = prepare_env(tiny_config, tiny_state.depth)
    lon_rad = np.deg2rad(np.asarray(tiny_config.grid.lon_deg))
    slope = 0.25
    bathymetry = jnp.broadcast_to(
        jnp.asarray(50.0 + slope * lon_rad)[None, :, None],
        tiny_state.depth.shape)
    features = np.asarray(sdn_features(env, bathymetry))
    theta = np.asarray(data_direction_radians(env)).reshape(-1)
    dfreq = np.asarray(env.dfreq).reshape(-1)
    expected = slope * np.sin(theta)[:, None] * dfreq[None, :] / 20.0
    # Longitude index 1 is away from the wrap-pad edge; the field is uniform
    # in latitude, so the flat index is Nlat.
    assert np.allclose(features[tiny_state.depth.shape[2], 0], expected,
                       rtol=1e-3, atol=1e-6)


def test_flat_bathymetry_has_zero_sdn_features(tiny_config, tiny_state):
    env = prepare_env(tiny_config, tiny_state.depth)
    features = np.asarray(sdn_features(env, tiny_state.depth))
    assert features.shape == (tiny_state.mask.size, 1, env.n_theta, env.n_freq)
    assert np.allclose(features, 0.0, atol=1e-6)


def test_ww_inter_features_are_finite(tiny_config, tiny_state):
    env = prepare_env(tiny_config, tiny_state.depth)
    features = np.asarray(ww_inter_features(env))
    assert features.shape == (tiny_state.mask.size, 1, env.n_theta, env.n_freq)
    assert np.all(np.isfinite(features))


def test_spatial_arrays_must_be_3_to_5_dimensional():
    from legoesm.wave.networks.embedding import _spatial_3d

    with pytest.raises(ValueError, match="3D/4D/5D"):
        _spatial_3d(jnp.zeros((2, 3)))
