# legoESM wave component

Optional spectral ocean-wave component. It is a substrate-only federation
member: it imports `legoesm.core` and never imports the atmosphere, ocean,
land or ice packages. All field exchange with the rest of the model lives in
`legoesm.coupler`.

The dynamics are vendored from WADEN (`46aadc7`, MIT) and the learned
source-term architecture from NeuralSWM. The original MIT notice is in
`LICENSE`.

## Public surface

- `WaveConfig` and its nested grid, time, physics, network and GSE configs;
- `WaveState` and `init_wave_state`;
- `WaveEnv`, `build_wave_env`, `prepare_env`, `with_dispersion`;
- `WaveModel` (operator-split integrator, learned source, optional GSE);
- `wave_parameters` (SWH, mean direction, mean period) and `stokes_drift`;
- `load_wave_weights` and the converted NeuralSWM checkpoint loader.

The coupled driver owns the atmosphere/ocean side: 10 m neutral wind from the
coupler stress, the Stokes-drift magnitude into KPP Langmuir, and restart of
the spectrum and spatial-order phase.

## Example

```python
import jax
from legoesm.wave import (
    WaveConfig, WaveGridConfig, WaveModel, init_wave_state,
    load_wave_weights, prepare_env,
)

config = WaveConfig(grid=WaveGridConfig(lon, lat, freq, theta))
model = WaveModel(config, key=jax.random.key(0))
model = load_wave_weights(model, "converted_weights.npz")
state = init_wave_state(depth, n_dir=24, n_freq=30)
env = prepare_env(config, state.depth)
state = model.advance(state, env, u10, v10)
```

A physics-only run selects
`WaveNetworkConfig(parameterization="none", use_transport_networks=False)`.

## Validation

- `tests/wave/unit/` — config, grid, transport, refraction, integrator, GSE,
  networks, checkpoint and coupling-seam tests;
- `tests/wave/integration/test_coupled_wave_run.py` — short dynamic-ocean,
  KPP, wave and restart continuation run;
- `scripts/validate/validate_wave_ww3.py` — archived WW3 propagation cases
  (evidence in `docs/wave/results/waden_20261112/`);
- `scripts/validate/validate_wave_parity.py` and
  `validate_wave_standalone.py` — optional NeuralSWM reference lanes.

See `docs/wave/waden_port_validation.md` for the protocol, gates and the
recorded results.

## Install

```bash
python scripts/experiment/install_federation.py wave
```

## Licence

MIT. The vendored WADEN notice is in `LICENSE`.
