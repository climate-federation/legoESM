"""Differentiable spectral ocean-wave component.

This package vendors the WADEN dynamics (MIT, commit 46aadc7) and the
NeuralSWM parameterization glue. The component is independent of the
atmosphere/ocean/land/ice packages; all coupling lives in
``legoesm.coupler``.

Quick start::

    import jax
    from legoesm.wave import (
        WaveConfig, WaveGridConfig, WaveModel, init_wave_state,
        load_wave_weights, prepare_env,
    )

    config = WaveConfig(grid=WaveGridConfig(lon, lat, freq, theta))
    model = WaveModel(config, key=jax.random.key(0))
    # Production runs must load the released weights; a fresh model has
    # randomly initialised networks.
    model = load_wave_weights(model, "converted_weights.npz")
    state = init_wave_state(depth, n_dir=24, n_freq=30)
    env = prepare_env(config, state.depth)
    state = model.advance(state, env, u10, v10)

For a physics-only run without external weights, select
``WaveNetworkConfig(parameterization="none", use_transport_networks=False)``;
the two switches together are required by :meth:`WaveConfig.validate`.
"""

from legoesm.wave.checkpoint import load_wave_weights
from legoesm.wave.config import (
    WaveConfig,
    WaveGSEConfig,
    WaveGridConfig,
    WaveNetworkConfig,
    WavePhysicsConfig,
    WaveTimeConfig,
)
from legoesm.wave.grid import WaveEnv, build_wave_env, prepare_env, with_dispersion
from legoesm.wave.indices import stokes_drift, wave_parameters
from legoesm.wave.model import WaveModel
from legoesm.wave.state import WaveState, init_wave_state

__all__ = [
    "WaveConfig",
    "WaveEnv",
    "WaveGSEConfig",
    "WaveGridConfig",
    "WaveModel",
    "WaveNetworkConfig",
    "WavePhysicsConfig",
    "WaveState",
    "WaveTimeConfig",
    "build_wave_env",
    "init_wave_state",
    "load_wave_weights",
    "prepare_env",
    "stokes_drift",
    "wave_parameters",
    "with_dispersion",
]
