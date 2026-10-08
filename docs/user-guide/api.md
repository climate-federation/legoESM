# API reference

A curated map of the public entry points. Each component package
(`legoesm.atmosphere`, `legoesm.ocean`, `legoesm.land`, `legoesm.ice`,
`legoesm.coupler`, `legoesm.ml`) installs independently; the `legoesm`
meta-package re-exports the cross-cutting helpers below. For full signatures read
the source under `packages/<pkg>/legoesm/`, or the
[technical specification](https://github.com/climate-federation/legoESM/blob/main/docs/science/specs/SPECIFICATION.md).

## Grids

`legoesm.grids.factory`

```python
from legoesm.grids.factory import create_grid

create_grid("cubed_sphere", 48)                          # global C48 (FV3-faithful)
create_grid("cubed_sphere", 48, stretch_fac=3.0,         # Schmidt-stretched / refined
            target_lat=0.7, do_cube_transform=True)
create_grid("mpas", 4, density_fn=...)                   # variable-resolution Voronoi (TRiSK)
create_grid("latlon", ...)                               # lat-lon finite-volume
create_grid("gaussian", ...)                             # spectral Gaussian
```

One grid object feeds every component (atmosphere, ocean, …).

## Composable physics

`legoesm.atmosphere.physics.combined` — build a single `physics_fn` from any subset
of **radiation, convection, turbulence, microphysics, gravity-wave drag**; disable a
process with `scheme="none"`. Composition is exactly additive within a step. See
[Composable physics tendencies](../architecture/composability.md).

```python
from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
```

## Components & complexity

`legoesm.components` — Dycore / Forced / Surface / Prescribed bricks; run a component
standalone or fully coupled. `legoesm.components.complexity` selects the model-complexity
tier (fixed-SST → slab → multilayer → full-3-D ocean; shallow-water → hydrostatic →
nonhydrostatic atmosphere).

## Machine learning

`legoesm.ml` — SFNO neural cores (`ml/sfno.py`), losses (`ml/loss.py`:
`area_weighted_mse`, `spectral_loss`, `per_variable_mse`), the optimizer
(`ml.training.create_optimizer`), and channel packing (`ml/channel_packing.py`).
Training modes (physics-param tuning, neural GCM, SFNO+dycore) build on
`build_segment_fn(...).raw` inside `eqx.filter_value_and_grad`.

## Shared scientific utilities

| Module | Provides |
|--------|----------|
| `legoesm.constants` | physical constants (`T_freeze`, `R_d`, `c_pd`, `L_v`, `g`, `p_ref`, …) |
| `legoesm.thermo` | `saturation_vapor_pressure`, `saturation_mixing_ratio`, `saturation_mixing_ratio_ice` |
| `legoesm.diagnostics.column_integrals` | `column_water_vapor` and other column integrals |
| `legoesm.ocean.eos` | `compute_ocean_rho`, `compute_ocean_rho_and_pressure` |

## Command-line interface

`legoesm` (entry point `legoesm.cli:main`):

| Command | Purpose |
|---------|---------|
| `legoesm run <config.yaml>` | run a simulation from a config |
| `legoesm test` | run standard validation test cases |
| `legoesm benchmark` | run performance benchmarks |
| `legoesm reproduce <manifest> --check` | re-run a manifest and assert a bit-identical result |
