# Composability & architecture

legoESM is assembled from interchangeable **bricks**. Four axes vary
independently, and the same codebase runs anything from a single-column RCE
experiment to a CMIP-class coupled spin-up:

| Axis | What varies | Selected by |
|------|-------------|-------------|
| **Extent / grid** | global, regional (limited-area), or idealized plane; uniform or refined | `legoesm.grids.factory` |
| **Model complexity** | shallow-water → hydrostatic → nonhydrostatic atmosphere; fixed-SST → slab → multilayer → full-3-D ocean; SCM / LES / CRM | `legoesm.components.complexity` |
| **Composable physics** | any subset of process parameterizations, summed | per-process `scheme=` config |
| **Fidelity** | research-idealized → intermediate → operational (AMIP/OMIP/CMIP) | test tier + forcing |

---

## Axis 1 — Extent & grid

One factory builds every grid; the grid is then shared by every component
(atmosphere and ocean run on the *same* grid object).

### Global grids — `create_grid(grid_type, resolution, **kw)`

`legoesm.grids.factory.create_grid` accepts five global families:

```python
from legoesm.grids.factory import create_grid

create_grid("cubed_sphere", 48)   # C48 FV3-faithful cube (~200 km)
create_grid("gaussian", 42)       # T42 spectral Gaussian
create_grid("latlon", 90)         # 90 lats × 180 lons (n_lon defaults to 2·n_lat)
create_grid("mpas", 5)            # icosahedral/Voronoi SCVT, 10 242 cells
create_grid("tripole", grid_file="mesh_mask.nc")  # NEMO ORCA tripolar (file-backed)
```

`resolution` means the natural knob per family: cube face size *N* (C\ *N*),
spectral truncation, number of latitudes, or icosahedral subdivision level
(`nCells = 10·4^level + 2`). Unknown type → `ValueError` listing the valid set.

### Refinement — telescoping resolution where you need it

```python
# Schmidt-stretched cube: high resolution telescoped over a target point
fine = create_grid("cubed_sphere", 48,
                   stretch_fac=3.0, target_lat=0.7, target_lon=0.0,
                   do_cube_transform=True)

# Variable-resolution MPAS mesh: density-weighted Lloyd relaxation,
# finer where density_fn returns larger values (here: near the equator)
import numpy as np
mpas = create_grid("mpas", 4,
                   density_fn=lambda lat, lon: 1.0 + 5.0 * np.exp(-(lat**2) / 0.2))
```

(Schmidt stretch requires the default `gnomonic="equiangular"`; the operational
`"ed"` gnomonic rejects stretching.)

### Regional / limited-area — `create_regional_grid(grid_type, **kw)`

```python
from legoesm.grids.factory import create_regional_grid

# Closed basin or zonally-periodic channel (lat-lon C-grid):
grid, wall_mask = create_regional_grid(
    "latlon", n_lat=50, n_lon=100,
    lat_south=-5.0, lat_north=5.0, lon_west=0.0, lon_east=360.0,
    periodic_x=True)          # E/W wrap, N/S walls — e.g. an Eady channel

create_regional_grid("mercator", n_lon=144, lat_max_deg=60.0)   # isotropic dx≈dy
create_regional_grid("mpas", lon_range=(280., 360.), lat_range=(-30., 30.),
                     resolution_km=100.0, periodic_x=True)
create_regional_grid("cubed_sphere", n=96, face_id=0)           # single panel
```

Supported regional families: `latlon`, `mercator`, `mpas`, `cubed_sphere`
(single panel). `periodic_x=True` gives a channel; the default closed basin
returns a `wall_mask` (1 = wet, 0 = wall).

### Idealized plane — LES / CRM / RCE

The doubly-periodic plane is built directly (it has no global/regional factory
entry) via `legoesm.grids.plane.create_plane_grid`:

```python
from legoesm.grids.plane import create_plane_grid

# Classic non-rotating RCE / CRM column-stack:
grid = create_plane_grid(nx=128, ny=128, nlev=60, dx=4_000.0, dy=4_000.0,
                         coriolis_mode="none")

# Rotating LES (f-plane or beta-plane):
grid = create_plane_grid(nx=64, ny=64, nlev=80, dx=50.0, dy=50.0,
                         coriolis_mode="f_plane", f0=1e-4)
```

`coriolis_mode` is `"none"`, `"f_plane"`, or `"beta_plane"`. The plane is
uniform-resolution (no density refinement).

---

## Axis 2 — Model complexity

The complexity ladders live in
`legoesm.components.complexity` as small `StrEnum`s. Each rung is a *physical*
fidelity level; the numerical method (the `discretization`) is an **orthogonal**
choice, so e.g. shallow-water can be solved on a cube, spectrally, on an
MPAS mesh, or by an SFNO.

### Atmosphere — single column to nonhydrostatic 3-D

`AtmosphereComplexity` has three rungs, whose values *equal* the dycore
`model_type` strings:

| Rung (`model_type`) | Equations | Typical use |
|---------------------|-----------|-------------|
| `shallow_water` | barotropic single-layer SW | Williamson / Galewsky dycore tests |
| `hydrostatic` | hydrostatic primitive equations (3-D) | AMIP, Held–Suarez, CMIP-class |
| `nonhydrostatic` | fully-compressible Euler (3-D) | CRM / LES, mountain waves, rising thermal |

Selection in the driver is a `(model_type, discretization)` pair on
`ExperimentConfig.dycore`:

```python
from legoesm.driver.config import DycoreConfig
DycoreConfig(model_type="hydrostatic", discretization="cdgrid")
```

`discretization` choices include `cdgrid` (FV3 C-D cube), `spectral`, `sfno`
(neural), `u_cast`, `mpas`, `latlon_cgrid`, and `plane`. The solver name is
resolved by `legoesm.atmosphere.dynamics.resolve_solver_name(...)`; an unknown
combination raises rather than silently falling back.

**Single-column model (SCM).** The fastest brick — one atmospheric column with
full physics, ideal for RCE and boundary-layer cases:

```python
import jax.numpy as jnp
from legoesm.atmosphere.scm import SingleColumnModel
from legoesm.atmosphere.physics import PhysicsConfig, RadiationConfig, TurbulenceConfig

scm = SingleColumnModel.create(
    physics_config=PhysicsConfig(
        radiation=RadiationConfig(scheme="gray"),
        turbulence=TurbulenceConfig(scheme="louis")),
    nlev=40, dt=300.0,
    T_profile=jnp.linspace(220.0, 295.0, 40), latitude_deg=0.0)
final_state, history = scm.run(nsteps=288, save_every=12)
```

**LES / CRM.** Large-eddy and cloud-resolving runs use the nonhydrostatic
**plane** dycore (`model_type="nonhydrostatic", discretization="plane"`) on a
`create_plane_grid` domain. The scale-dependent dynamic Smagorinsky closure
(LASD; Bou-Zeid–Meneveau–Parlange) lives in
`legoesm.atmosphere.physics.turbulence.lasd_core`. Canonical cases ship as
scripts/tests: GABLS1 stable boundary layer (`scripts/matrix/scm/gabls1.py`), Wangara
diurnal cycle, and the Skamarock–Klemp rising thermal
(`scripts/run/run_plane_rising_thermal.py`).

### Ocean — fixed SST to full 3-D

`OceanComplexity` spans four rungs:

| Rung | Model |
|------|-------|
| `fixed_sst` | prescribed SST, no ocean state |
| `slab` | single mixed-layer energy balance |
| `slab_multilayer` | mixed layer + deep layer with vertical mixing |
| `full_3d` | prognostic baroclinic + barotropic primitive-equation ocean |

The slab rungs are built with `legoesm.ocean.simple_ocean.make_ocean(SimpleOceanConfig(mode=...))`;
the full-3-D prognostic ocean is wired through the component factory (it is a
model, not a slab mode). Land (`slab` / `multilayer`) and sea ice
(`thermodynamic` / `dynamic`) have their own two-rung ladders.

### Model-wide complexity

`ModelComplexity` dials every component at once:

```python
from legoesm.driver.coupled_config import preset_complexity
cfg = preset_complexity("idealized")      # SW atm + fixed-SST + slab land + thermo ice
cfg = preset_complexity("intermediate")   # hydrostatic + slab ocean + multilayer land
# "full" (full-3-D ocean + dynamic ice) is built via the driver factory directly
```

Ready-made coupled presets cover the common combinations:
`aquaplanet`, `slab_simple`, `slab_pft`, `slab_richards`, `slab_carbon`,
`full_coupled` (in `legoesm.driver.coupled_config`).

---

## Axis 3 — Composable physics

legoESM physics is **composable**: you choose any subset of the available
process parameterizations and the model runs exactly that subset. The full
stack and any partial stack are both first-class — e.g. for the atmosphere you
can run the full `radiation + convection + turbulence + microphysics +
gravity-wave drag`, or just `radiation + turbulence`, or radiation alone, or
none at all. The same holds for the ocean; the land surface composes by
model-complexity selection.

### How it works

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

### Atmosphere

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

### Ocean

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

**Tidal mixing** rides on `VerticalMixingConfig.tidal` but is *not* part of this
composition: it is a separate caller-applied additive step (precompute
`K_tidal` with `vertical_mixing.tidal.compute_tidal_diffusivity`, apply with
`ocean.coupler.tidal_mixing_apply.apply_tidal_mixing_step`). To keep the
contract honest, `make_ocean_physics` **raises** if `tidal.enabled=True` rather
than silently ignoring it.

### Land

The land surface is a single-column model, so its composability is
**model-complexity selection** rather than a sum of independent process
tendencies. Choose the complexity rung (slab vs. discretized multi-layer soil)
and the surface-flux (bulk) scheme via the land component factory /
`ExperimentConfig` land fields; the surface-energy-balance, soil/snow, and
optional carbon processes are wired per rung. Sea ice, lakes, and snow are
selected the same way through their respective configs.

### Guarantees (tests)

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

---

## The packages (high-level)

Source lives under `packages/<member>/legoesm/...` as a uv workspace of
independently-installable federation packages. The dependency layering is
one-way:

```
core (substrate)  <  {atmosphere, ocean, land, ice}  <  coupler  <  driver
```

The four Earth-system components never import one another or the orchestration
layers above them; this is enforced in CI by `import-linter` contracts (see
`pyproject.toml`) and `tests/test_federation_plan.py`. That one-way rule is
exactly what lets `pip install legoesm-ocean` run the ocean standalone.

| Package | Owns |
|---------|------|
| **legoesm-core** | the shared substrate: physical `constants`, `thermo`, grids + operators, state containers, time-stepping, runtime/backend, MPI `parallel` (halo exchange + AD-safe reductions), I/O, and the component protocol. Imports nothing above it. |
| **legoesm-atmosphere** | dynamical cores (FV3 cubed-sphere C-D, lat-lon C, spectral, SFNO, MPAS) across shallow-water / hydrostatic / nonhydrostatic, plus the physics suite (radiation, convection, turbulence, microphysics, clouds, GWD) and the SCM. |
| **legoesm-ocean** | Boussinesq primitive-equation + barotropic solvers on multiple grids, EOS, vertical/lateral mixing (KPP, GM–Redi), tracer transport, biogeochemistry, and the slab ocean. |
| **legoesm-land** | soil thermal + hydrology (Richards), snow budget, stomatal conductance / photosynthesis, the carbon cycle, and slab/lake surface models. |
| **legoesm-ice** | sea-ice thermodynamics + EVP/mEVP dynamics, multi-category ice-thickness distribution, ridging, brine, melt ponds. |
| **legoesm-coupler** | surface-flux exchange (bulk Monin–Obukhov, radiative, sensible/latent), tile-fraction blending, flux accumulation, lakes — *and* the `driver/` orchestration (`ModelDriver`, `CoupledESMDriver`, config, checkpoint/restart, physics pipeline, component factory). |
| **legoesm-ml** | SFNO neural operators, learned physics, the three training drivers (physics-param tuning, neural-GCM, SFNO+dycore), and data assimilation. Components opt in via the `[ml]` extra. |
| **legoesm-tools** | external forcing ingestion (ERA5/JRA55/AMIP), diagnostics (energy/angular-momentum/conservation budgets, monthly means), experiment templates, and visualization. |

Components share the FV3 shallow-water barotropic solver **without importing
each other**: atmosphere registers `fv3sw`/`fv3edge` providers on the
`legoesm.sw_barotropics` entry-point group, and the ocean barotropic solver
resolves them by name through `legoesm.registry`.

---

## Axis 4 — The research → operational ladder

The same bricks carry a run from a quick idealized sanity check all the way to
CMIP-style production. The progression is encoded as **test tiers** (pytest
markers in `pyproject.toml`), each adding fidelity *and* a stricter conservation
gate:

| Tier | Scope | Gates |
|------|-------|-------|
| **tier0** | unit / operator kernels — no model integration | numerical invariants |
| **tier1** | research: idealized / single-column / shallow-water | mass + energy + angular-momentum gates, analytic benchmarks (Williamson, Galewsky, Jablonowski–Williamson) |
| **tier2** | intermediate: hydrostatic 3-D + slab surfaces | mass + energy + moisture closure (Held–Suarez, aquaplanet) |
| **tier3** | operational: full complexity + real forcing | budget closure under AMIP / OMIP / ERA5 |

Run a rung explicitly with e.g. `pytest -m "tier1 and not slow"`.

**Operationalization (AMIP/OMIP → CMIP).** At the top of the ladder the model
ingests real forcing and writes CMOR / CF-1.8 output:

```bash
# Atmosphere-only AMIP (prescribed SST):
JAX_ENABLE_X64=1 .venv/bin/python scripts/run/run_amip.py \
    --grid-type cubed_sphere --resolution 16 --days 365

# Ocean-only OMIP spin-up:
JAX_ENABLE_X64=1 .venv/bin/python scripts/run/run_omip.py

# Fully coupled:
JAX_ENABLE_X64=1 .venv/bin/python scripts/run/run_coupled.py --preset full_coupled --days 365
```

`legoesm.io` writes the standard CMIP tables (Amon / Lmon / Omon / Oyr / Ofx /
SImon / SIyr), and every run emits a reproducibility manifest that
`legoesm reproduce <manifest> --check` re-runs into a fresh directory and
asserts a bit-identical result. The production-readiness status by component is
tracked in [CMIP readiness](../validation/cmip_readiness.md); the tiered test framework is
described in [Testing](../validation/TESTING.md).
