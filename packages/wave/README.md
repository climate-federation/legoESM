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

The optional ocean feedback extends the upstream LMD94-style KPP with
VR12-AL (aligned wind/waves): Li et al. (2016), Table 2 / Eq. (12),
doi:10.1016/j.ocemod.2015.07.020, following Van Roekel et al. (2012),
doi:10.1029/2011JC007516. Both diffusivity velocity scales and the unresolved
shear used to diagnose boundary-layer depth receive the same multiplier.
The fixed published coefficients are 3.1 and 5.4; `--langmuir-coeff` and
`KPPConfig.langmuir_coeff` are removed. `None` Stokes retains prescribed-La
mixing when enabled; use `enable_langmuir=False` for a no-feedback control.

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
`WaveNetworkConfig(parameterization="none")`. This always disables neural
transport, including when `use_transport_networks=True`; the analytic theta
and frequency tendencies are selected without constructing network weights.
With this configuration, coupled runs omit `--wave-weights`; neural runs require
it. With no source terms, wind is passed through the interface but does not
generate waves. A calm initial spectrum stays calm. Checkpoints record an
empty weights hash for a run without weights.

For a regular latitude–longitude ocean, use the coupled CLI with
`--grid latlon --ocean dynamic --wave-mode active --wave-config wave.json --kpp-langmuir`.
The wave deck's global time step must match the coupling interval (3600 s by
default). Add a converted `--wave-weights` file for a neural source term.
`--kpp-langmuir` explicitly selects KPP with Langmuir enhancement; without
that flag the regular latitude–longitude CLI retains its existing mixing
configuration. Wave starts with zero Stokes on the ocean grid and thereafter
supplies the previous coupling step's diagnostic. With Wave off and Langmuir
explicitly enabled, KPP instead uses its configured prescribed Langmuir number.
The coupled Wave path requires the coupler-owned surface exchange and excludes
Voronoi atmosphere or ocean grids.

`WaveModel.advance` caches JIT compilation while accepting changing spectra,
winds, environments and network weights as array inputs. GSE requires the
concrete environment preflight `validate_wave_environment(config, env)` from
`legoesm.wave.model` before advancing; the coupled driver performs this check.

## Validation

- `tests/wave/unit/` — config, grid, transport, refraction, integrator, GSE,
  networks, checkpoint and coupling-seam tests;
- `tests/wave/integration/test_coupled_wave_run.py` — short dynamic-ocean,
  KPP, wave and restart continuation run;
- `scripts/validate/validate_wave_ww3.py` — archived WW3 propagation cases
  (evidence in `docs/wave/evidence/waden_20261112/`);
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
