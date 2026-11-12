"""Differentiable spectral ocean-wave component.

This package vendors the WADEN dynamics (MIT, commit 46aadc7) and the
NeuralSWM parameterization glue. The component is independent of the
atmosphere/ocean/land/ice packages; all coupling lives in
``legoesm.coupler``.

This first slice provides the static grid, configuration, state and
diagnostics. The transport kernels, integrator, networks and coupler
wiring land in the following commits of the same series.

Quick start::

    import jax
    from legoesm.wave import WaveConfig, WaveGridConfig, init_wave_state, prepare_env

    config = WaveConfig(grid=WaveGridConfig(lon, lat, freq, theta))
    state = init_wave_state(depth, n_dir=24, n_freq=30)
    env = prepare_env(config, state.depth)
"""

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
from legoesm.wave.state import WaveState, init_wave_state

__all__ = [
    "WaveConfig",
    "WaveEnv",
    "WaveGSEConfig",
    "WaveGridConfig",
    "WaveNetworkConfig",
    "WavePhysicsConfig",
    "WaveState",
    "WaveTimeConfig",
    "build_wave_env",
    "init_wave_state",
    "prepare_env",
    "stokes_drift",
    "wave_parameters",
    "with_dispersion",
]
