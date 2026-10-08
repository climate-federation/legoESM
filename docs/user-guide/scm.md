# Single-Column Model (SCM)

The legoESM Single-Column Model is a **dycore-free** driver that
exercises the full atmospheric physics pipeline on a single
`(lat, lon)` column. It is the canonical tool for:

- parameterization integration tests,
- stability and timestep sweeps,
- idealized process studies (RCE, GABLS, BOMEX-style cases),
- swap-matrix experiments that cross every parameterization with
  every time integrator.

Issue: [#277](https://github.com/climate-federation/legoESM/issues/277).

Source: `packages/atmosphere/legoesm/atmosphere/forcing/scm/scm.py` (`SingleColumnModel`).
Demo:   `scripts/matrix/run_scm_test_matrix.py rce` (per-case modules under `scripts/matrix/scm/`).

## Why an SCM?

The SCM reuses the canonical hydrostatic physics factory
(`legoesm.atmosphere.physics.combined.make_physics`) — so any scheme
that runs in the full 3-D model also runs in the SCM **without
modification**. There is no horizontal advection, no
pressure-gradient force, and no Coriolis: only column tendencies from
radiation, convection, turbulence, microphysics, and gravity-wave
drag, advanced by the chosen time integrator.

This lets you isolate column physics from dycore behaviour. A
parameterization bug that manifests on the cubed-sphere is almost
always reproducible in the SCM in a few seconds.

## Quick start

```bash
# 50-day tropical RCE with gray rad + Louis BL + Kessler + simple convection
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python scripts/matrix/run_scm_test_matrix.py rce --days 50
```

Surface temperature should settle near the prescribed value
(~300 K); stratospheric T should approach the gray-radiation
equilibrium near 200 K.

## API

```python
import jax.numpy as jnp
from legoesm.atmosphere.forcing.scm.scm import SingleColumnModel
from legoesm.atmosphere.physics import (
    PhysicsConfig, RadiationConfig, TurbulenceConfig,
    ConvectionConfig, MicrophysicsConfig,
)

cfg = PhysicsConfig(
    radiation=RadiationConfig(scheme="gray"),
    turbulence=TurbulenceConfig(scheme="louis"),
    convection=ConvectionConfig(scheme="mass_flux"),
    microphysics=MicrophysicsConfig(scheme="kessler"),
)

scm = SingleColumnModel.create(
    physics_config=cfg,
    nlev=40,
    dt=300.0,
    latitude_deg=0.0,
    T_profile=jnp.linspace(220.0, 295.0, 40),
    q_v_profile=1.8e-2 * jnp.exp(-jnp.linspace(0, 10, 40)),
    time_integrator="forward_euler",   # or "rk2", "rk4"
)

final_state, history = scm.run(nsteps=288, save_every=12)
```

## Time integrators

The SCM ships with three integrators registered in
`atmosphere.scm.TIME_INTEGRATORS`:

| Name | Order | Notes |
|---|---|---|
| `forward_euler` | 1 | Default. Compatible with every scheme. |
| `rk2`           | 2 | Heun's method. PhysicsState carry advanced once per outer step. |
| `rk4`           | 4 | Classical RK4. Carry advanced from stage-1 only. |

Custom integrators can be registered with
`register_time_integrator(name, step_fn)`.

`SingleColumnModel.create` rejects integrator/physics combinations
that would silently lie about the order — for example, stateful
throttled convection schemes that rely on the prior PhysicsState carry
are restricted to `forward_euler` (see commit `3cbc57e7` "narrow RK
gate" and `a91f6e5b` "widen RK gating").

## Swap matrix

`scripts/` ships a swap-matrix sweep that crosses every
parameterization with every integrator (commit `36b87553` "SCM swap
matrix: every parameterization × every integrator"). Use it to verify
new schemes don't regress the physics suite.

## Behaviour notes

- Tracer species introduced by physics tendencies (`q_c`, `q_r`,
  `q_i`, …) are materialised from zero — the SCM never silently drops
  a tracer that a microphysics scheme emits.
- No positivity clipping is applied at the integrator level. Schemes
  that need positive mixing ratios must enforce that themselves;
  clipping at integrator-level hides instability and breaks AD
  smoothness.
- The SCM grid is `(1, 1)` so column reshapes give `ncol = 1`. All
  shapes are `(face=1, x=1, y=1, level=nlev)` to match the canonical
  `_DIMS_3D` convention.

## See also

- [docs/user-guide/amip.md](amip.md) — full 3-D AMIP driver.
- [docs/science/ml_physics_parameterization.md](../science/ml_physics_parameterization.md)
  — joint ML/physics workflows.
- `scripts/matrix/run_scm_test_matrix.py rce` — RCE demo and sanity test.
- `scripts/matrix/run_scm_test_matrix.py {gabls1,ekman,wangara}` — stable, neutral, and convective BL benchmarks.
- `scripts/matrix/run_scm_test_matrix.py oracle` — regenerate jax_scm reference NetCDFs (needs `.venv-jax-scm`).
