# legoESM wave component

Optional spectral ocean-wave component. It is a substrate-only federation
member: it imports `legoesm.core` and never imports the atmosphere, ocean,
land or ice packages. All field exchange with the rest of the model lives in
`legoesm.coupler`.

The dynamics are vendored from WADEN (`46aadc7`, MIT) and the learned
source-term architecture from NeuralSWM. The original MIT notice is in
`LICENSE`.

## Public surface

This first slice provides:

- `WaveConfig` and its nested grid, time, physics, network and GSE configs;
- `WaveState` and `init_wave_state`;
- `build_wave_env`, `prepare_env`, `with_dispersion`;
- `wave_parameters` (SWH, mean direction, mean period) and `stokes_drift`.

Transport kernels, the operator-split integrator, the Equinox
parameterizations and the coupler wiring land in the following commits of
the same series.

## Example

```python
import jax
from legoesm.wave import (
    WaveConfig, WaveGridConfig, init_wave_state, prepare_env,
)

config = WaveConfig(grid=WaveGridConfig(lon, lat, freq, theta))
state = init_wave_state(depth, n_dir=24, n_freq=30)
env = prepare_env(config, state.depth)
```

## Install

```bash
python scripts/experiment/install_federation.py wave
```

## Licence

MIT. The vendored WADEN notice is in `LICENSE`.
