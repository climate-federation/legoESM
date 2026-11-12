"""Shared fixtures for the wave-component unit tests."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.wave.config import (
    WaveConfig,
    WaveGridConfig,
    WaveNetworkConfig,
    WavePhysicsConfig,
    WaveTimeConfig,
)

N_DIR = 4
N_FREQ = 4
SPEC = N_DIR * N_FREQ


def tiny_grid() -> WaveGridConfig:
    # Latitude DESCENDING: the oracle's y-tendency sign assumes it.
    return WaveGridConfig(
        lon_deg=tuple(100.0 + 0.5 * i for i in range(6)),
        lat_deg=tuple(30.0 - 0.5 * i for i in range(5)),
        freq_hz=tuple(0.05 * 1.1**i for i in range(N_FREQ)),
        # Directions are equally spaced over the full circle (the model's
        # quadrature dtheta = 2*pi/N assumes that).
        theta_rad=tuple(
            float(jnp.pi * (180.0 - (7.5 + (360.0 / N_DIR) * i)) / 180.0)
            for i in range(N_DIR)
        ),
    )


def tiny_network(**overrides) -> WaveNetworkConfig:
    values = dict(
        mlp_input_channels=4 * SPEC,
        mlp_hidden_channels_0=8,
        mlp_hidden_channels_1=8,
        mlp_hidden_channels_2=8,
        transport_input_channels=3 * N_FREQ,
        transport_hidden_channels=8,
        transport_output_vectors=2 * N_FREQ,
    )
    values.update(overrides)
    return WaveNetworkConfig(**values)


@pytest.fixture
def tiny_config() -> WaveConfig:
    return WaveConfig(
        grid=tiny_grid(),
        time=WaveTimeConfig(),
        physics=WavePhysicsConfig(),
        network=tiny_network(),
    )


@pytest.fixture
def tiny_depth():
    return jnp.full((1, 6, 5), 50.0)


@pytest.fixture
def tiny_state(tiny_depth):
    from legoesm.wave.state import init_wave_state

    state = init_wave_state(tiny_depth, n_dir=N_DIR, n_freq=N_FREQ)
    return state._replace(spectra=jnp.ones_like(state.spectra) * 0.01)
