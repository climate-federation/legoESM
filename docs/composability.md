# Composable physics tendencies

legoESM physics is **composable**: you choose any subset of the available
process parameterizations and the model runs exactly that subset. The full
stack and any partial stack are both first-class — e.g. for the atmosphere you
can run the full `radiation + convection + turbulence + microphysics +
gravity-wave drag`, or just `radiation + turbulence`, or radiation alone, or
none at all. The same holds for the ocean, and the land surface is composed by
model-complexity selection.

## How it works

Each component builds a single `physics_fn(state, ...) -> tendencies` by
evaluating every **enabled** process on the *same input state* and **summing**
their tendencies. A process is disabled by setting its scheme to `"none"`.
Because every enabled process reads the same input state and the results are
added, composition is **exactly additive within a step**:

```
tendency(A + B + C) == tendency(A) + tendency(B) + tendency(C)
```

to machine precision (process coupling happens across timesteps via the state
update, not within a single physics evaluation). This is enforced by tests
(see *Guarantees* below), so it is a contract, not an accident.

## Atmosphere

Five processes: **radiation, convection, turbulence, microphysics,
gravity-wave drag**.

Standalone API — `legoesm.atmosphere.physics.combined`:

```python
from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
from legoesm.atmosphere.physics.radiation.config import RadiationConfig
from legoesm.atmosphere.physics.convection.config import ConvectionConfig
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig

# radiation + turbulence only (everything else disabled):
cfg = PhysicsConfig(
    radiation=RadiationConfig(scheme="gray"),
    convection=ConvectionConfig(scheme="none"),
    turbulence=TurbulenceConfig(scheme="smagorinsky"),
    microphysics=MicrophysicsConfig(scheme="none"),
    gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
)
physics_fn = make_physics(cfg, model_type="hydrostatic", dt=300.0)
tendency, _ = physics_fn(state, grid, sigma)
```

Driver API — the same composition is exposed on `ExperimentConfig` as
per-process scheme fields, each independently settable (including `"none"`):
`radiation`, `convection`, `turbulence`, `microphysics`,
`gravity_wave_drag`, `cloud_scheme`. `validate_strict()` rejects an unknown
scheme for any of these (so a typo fails at config validation, not at run time),
and every factory raises `ValueError` on an unknown scheme rather than silently
falling back to a default.

## Ocean

Five processes: **vertical_mixing, lateral_mixing, surface_forcing,
convection, shortwave_penetration** (physics-level `bottom_drag` is deprecated
in favour of the dynamics-level `bottom_drag_r`).

API — `legoesm.ocean.physics.combined`:

```python
from legoesm.ocean.physics.combined import OceanPhysicsConfig, make_ocean_physics
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig

# vertical mixing + convection only:
cfg = OceanPhysicsConfig(
    vertical_mixing=VerticalMixingConfig(scheme="kpp"),
    lateral_mixing=LateralMixingConfig(scheme="none"),
    surface_forcing=SurfaceForcingConfig(scheme="none"),
    convection=OceanConvectionConfig(scheme="enhanced_diffusion"),
    shortwave_penetration=None,   # disable solar penetration heating
)
ocean_physics_fn = make_ocean_physics(cfg)
tendency = ocean_physics_fn(state, grid, z_coord, surface_forcing)
```

Set any sub-config's `scheme="none"` to drop that process, and
`shortwave_penetration=None` to drop solar-penetration heating. With every
process off, the tendency is identically zero.

## Land

The land surface is a single-column model, so its composability is
**model-complexity selection** rather than a sum of independent process
tendencies. Choose the complexity rung (slab vs. discretized multi-layer soil)
and the surface-flux (bulk) scheme via the land component factory /
`ExperimentConfig` land fields; the surface-energy-balance, soil/snow, and
optional carbon processes are wired per rung. Sea ice, lakes, and snow are
selected the same way through their respective configs.

## Guarantees (tests)

- **Atmosphere** — `tests/unit/test_physics_combined.py::test_tendency_additivity`
  proves `combined == sum of individual` to `1e-10` for radiation + turbulence +
  gravity-wave drag, plus all-schemes-on and convection×radiation smoke tests.
- **Ocean** — `tests/ocean/unit/test_ocean_physics_composability.py` proves, for
  the five ocean processes: all-off → identically zero, each single process runs
  finite, `combined(subset) == exact sum` for several subsets, and disabling a
  process removes exactly its contribution.

These tests are the contract: any change that breaks additive composition (e.g.
a scheme that silently runs when set to `"none"`, or cross-process coupling
leaking into a single evaluation) fails them.
