"""Config validation and deck loading for the wave component."""

from __future__ import annotations

import json

import numpy as np
import pytest

from legoesm.wave.config import (
    WaveConfig,
    WaveNetworkConfig,
    WaveTimeConfig,
)
from tests.wave.unit.conftest import tiny_grid, tiny_network


def test_default_config_validates(tiny_config):
    tiny_config.validate()


def test_unknown_parameterization_raises():
    config = WaveConfig(
        grid=tiny_grid(),
        network=tiny_network(parameterization="not-a-scheme"),
    )
    with pytest.raises(ValueError, match="parameterization"):
        config.validate()


def test_none_parameterization_ignores_transport_network_settings():
    config = WaveConfig(
        grid=tiny_grid(), network=tiny_network(parameterization="none"))
    assert config.network.use_transport_networks
    config.validate()
    # Inactive network dimensions must not constrain a physics-only grid.
    config._replace(network=config.network._replace(
        transport_input_channels=1, transport_output_vectors=1)).validate()


def test_unknown_checkpoint_level_raises():
    from legoesm.wave.config import WavePhysicsConfig

    config = WaveConfig(
        grid=tiny_grid(),
        physics=WavePhysicsConfig(checkpoint_level="sometimes"),
    )
    with pytest.raises(ValueError, match="checkpoint_level"):
        config.validate()


def test_network_dimension_must_match_grid():
    config = WaveConfig(grid=tiny_grid(), network=tiny_network(mlp_input_channels=7))
    with pytest.raises(ValueError, match="mlp_input_channels"):
        config.validate()


def test_transport_input_must_match_frequencies():
    config = WaveConfig(
        grid=tiny_grid(), network=tiny_network(transport_input_channels=9))
    with pytest.raises(ValueError, match="transport_input_channels"):
        config.validate()


def test_frequency_grid_must_be_positive_and_increasing():
    decreasing = WaveConfig(
        grid=tiny_grid()._replace(freq_hz=(0.2, 0.1, 0.05, 0.04)))
    with pytest.raises(ValueError, match="freq_hz"):
        decreasing.validate()
    zero = WaveConfig(grid=tiny_grid()._replace(freq_hz=(0.0, 0.1, 0.2, 0.3)))
    with pytest.raises(ValueError, match="freq_hz"):
        zero.validate()


def test_direction_grid_must_be_uniform():
    config = WaveConfig(
        grid=tiny_grid()._replace(theta_rad=(0.0, 0.1, 0.3, 0.6)))
    with pytest.raises(ValueError, match="theta_rad"):
        config.validate()


def test_direction_grid_may_start_anywhere_on_the_circle():
    """A uniform grid whose last spacing wraps through 0 is still uniform.

    The archived WW3 cases store directions from 90 deg, so the straight
    diff of the last pair is -270 deg; only the circular spacing is 90 deg.
    """
    theta = tuple(np.deg2rad(np.asarray([90.0, 180.0, 270.0, 0.0])))
    config = WaveConfig(
        grid=tiny_grid()._replace(theta_rad=theta), network=tiny_network())
    config.validate()


def test_direction_grid_must_cover_the_full_circle():
    half_circle = tuple(np.linspace(0.0, np.pi, 4, endpoint=False))
    config = WaveConfig(grid=tiny_grid()._replace(theta_rad=half_circle))
    with pytest.raises(ValueError, match="full circle"):
        config.validate()


def test_spatial_grid_needs_two_cells():
    config = WaveConfig(grid=tiny_grid()._replace(lon_deg=(100.0,)))
    with pytest.raises(ValueError, match="at least two longitudes"):
        config.validate()


def test_negative_time_step_raises():
    config = WaveConfig(grid=tiny_grid(), time=WaveTimeConfig(dt_global_s=-1),
                        network=tiny_network())
    with pytest.raises(ValueError, match="dt_global_s"):
        config.validate()


def test_from_json_round_trip(tmp_path, tiny_config):
    deck = {
        "grid": {
            "lon_deg": list(tiny_config.grid.lon_deg),
            "lat_deg": list(tiny_config.grid.lat_deg),
            "freq_hz": list(tiny_config.grid.freq_hz),
            "theta_rad": list(tiny_config.grid.theta_rad),
        },
        "time": {"dt_global_s": 3600},
        "network": {"mlp_input_channels": tiny_network().mlp_input_channels,
                    "mlp_hidden_channels_0": 8,
                    "mlp_hidden_channels_1": 8,
                    "mlp_hidden_channels_2": 8,
                    "transport_input_channels": tiny_network().transport_input_channels,
                    "transport_hidden_channels": 8,
                    "transport_output_vectors": tiny_network().transport_output_vectors},
    }
    path = tmp_path / "wave.json"
    path.write_text(json.dumps(deck))
    config = WaveConfig.from_json(str(path))
    assert tuple(config.grid.lon_deg) == tuple(tiny_config.grid.lon_deg)
    assert config.time.dt_global_s == 3600


def test_from_json_unknown_key_raises(tmp_path, tiny_config):
    deck = {
        "grid": {
            "lon_deg": list(tiny_config.grid.lon_deg),
            "lat_deg": list(tiny_config.grid.lat_deg),
            "freq_hz": list(tiny_config.grid.freq_hz),
            "theta_rad": list(tiny_config.grid.theta_rad),
            "spacing": 0.5,
        },
    }
    path = tmp_path / "wave.json"
    path.write_text(json.dumps(deck))
    with pytest.raises(TypeError):
        WaveConfig.from_json(str(path))


def test_from_json_rejects_unknown_top_level_keys(tmp_path, tiny_config):
    deck = {"grid": tiny_config.grid._asdict(),
            "network": tiny_config.network._asdict(),
            "physic": {"facvx": 0.0}}
    path = tmp_path / "typo.json"
    path.write_text(json.dumps(deck))
    with pytest.raises(TypeError, match="physic"):
        WaveConfig.from_json(str(path))



def test_json_config_advances_with_cached_jit(tmp_path, tiny_config, tiny_state):
    import jax
    import jax.numpy as jnp
    from legoesm.wave import WaveModel, prepare_env

    config = tiny_config._replace(gse=tiny_config.gse._replace(enabled=False))
    path = tmp_path / "jit.json"
    path.write_text(json.dumps({
        name: value._asdict() for name, value in config._asdict().items()}))
    loaded = WaveConfig.from_json(str(path))
    model = WaveModel(loaded, key=jax.random.key(0))
    env = prepare_env(loaded, tiny_state.depth)
    wind = jnp.ones_like(tiny_state.depth)
    expected = model.step(tiny_state.spectra, env, tiny_state.depth,
                          tiny_state.mask, wind, wind)
    actual = model.advance(tiny_state, env, wind, wind)
    np.testing.assert_allclose(actual.spectra, expected, rtol=1e-5, atol=1e-7)
    # Callers may construct WaveState with its Python-zero default. Preserve
    # the scan carry dtype even when x64 promotes that zero to int64.
    def advance(current, _):
        return model.advance(current, env, wind, wind), None

    final, _ = jax.lax.scan(advance, tiny_state._replace(spatial_phase=0),
                           None, length=2)
    assert final.spatial_phase.dtype == jnp.asarray(0).dtype
