# legoESM

**A differentiable Earth System Model in JAX.**

legoESM is a from-scratch Earth System Model built so that the *entire* simulation
— atmosphere, ocean, land, sea ice, and their coupling — is a pure JAX program you
can differentiate end to end with `jax.grad`. That makes data assimilation,
parameter estimation, and hybrid AI–physics modelling first-class rather than
bolted on.

It is assembled from interchangeable **bricks**: dynamical cores, physics schemes,
grids, vertical coordinates, time integrators, and model complexity are all swappable
independently — so the same codebase runs anything from a single-column RCE column to
a fully coupled centennial spin-up.

```{note}
**Requirements:** Python ≥ 3.11 and JAX ≥ 0.4.35. The whole model is JIT-compiled,
so the same code runs on CPU, GPU, TPU, and Apple Silicon, and scales to MPI clusters
via `mpi4jax`. Apple Silicon users who want the spectral solver must set
`JAX_PLATFORMS=cpu` (the Metal backend has no `float64`).
```

## Installation

Install from a clone (development install — the supported path today):

```bash
git clone https://github.com/climate-federation/legoESM.git
cd legoESM
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[ml,mesh,viz,data]"               # neural cores, MPAS meshes, plotting, data IO
pip install "mpi4py>=4.1,<5" "mpi4jax>=0.9,<0.10"   # multi-node MPI runs
pip install -e ".[docs]"                           # build this documentation site
```

Sanity check:

```bash
.venv/bin/python -c "import legoesm, jax; print('jax', jax.__version__, 'backend', jax.default_backend())"
```

See [Getting started](user-guide/getting_started.md) for the full top-to-bottom walkthrough.

## Quick start

One grid feeds every component; pick a complexity and run:

```python
from legoesm.grids.factory import create_grid

# One grid, any component. Global cubed-sphere at C48:
grid = create_grid("cubed_sphere", 48)

# Refined: a Schmidt-stretched cube telescoped over a target region ...
fine = create_grid("cubed_sphere", 48, stretch_fac=3.0, target_lat=0.7, do_cube_transform=True)

# ... or a variable-resolution MPAS mesh, finer where density is high:
import numpy as np
mpas = create_grid("mpas", 4, density_fn=lambda lat, lon: 1.0 + 5.0 * np.exp(-(lat**2) / 0.2))
```

From the command line:

```bash
legoesm run    config/examples/<experiment>.yaml   # run a simulation from a config
legoesm test   --case held_suarez                  # standard validation cases
legoesm reproduce <manifest> --check               # bit-identical reproducibility gate
```

## The three Lego axes

Everything is assembled from interchangeable bricks along three independent axes —
inspect the full capability map via [`legoesm.taxonomy`](#capability-taxonomy):

| Axis | What varies | Where it lives |
|------|-------------|----------------|
| **Complexity** | fixed-SST → slab → multilayer → full-3-D ocean; shallow-water → hydrostatic → nonhydrostatic atmosphere; … | `legoesm.components.complexity` |
| **Bricks** | run a component standalone **or** fully coupled | `legoesm.components` (Dycore/Forced/Surface/Prescribed bricks) |
| **Extent** | global, regional (limited-area), or idealized; uniform or **refined** | `legoesm.grids.factory` |

Grids are shared between components: lat-lon FV, spectral Gaussian, cubed-sphere
(FV3-faithful), MPAS/Voronoi (TRiSK), and SFNO all feed both the atmosphere and the
ocean.

See [Composability & architecture](architecture/composability.md) for how to instantiate each
axis (grids, regional/idealized extent, the SCM/LES/CRM/shallow-water/3-D complexity
ladder), a high-level tour of the packages, and the research → operational
(AMIP/OMIP/CMIP) progression.

(capability-taxonomy)=
## Capability taxonomy

```python
from legoesm import taxonomy
taxonomy.capability_report()   # the full complexity × extent × grid/core map
taxonomy.grid_capability("mpas").variable_resolution   # True
```

## Reproducibility

Every run writes a manifest (resolved config + RNG seeds + state digest); the
`legoesm reproduce <manifest> --check` gate re-runs into a fresh directory and
asserts a bit-identical result.

## Citation

See `CITATION.cff` (GitHub "Cite this repository") and `.zenodo.json` — a
Zenodo-linked release mints the DOI. Full details on the [References](references.md) page.

```{toctree}
:hidden:
:caption: Getting started

Home <self>
user-guide/getting_started
user-guide/wizard
architecture/composability
```

```{toctree}
:hidden:
:caption: Examples

user-guide/scm
user-guide/amip
user-guide/rcemip1_crm
science/ml_physics_parameterization
```

```{toctree}
:hidden:
:caption: Validation & performance

validation/dycore_validation_catalog
validation/PHYSICS_PARAMETERIZATION_TESTS
validation/cmip_readiness
validation/TESTING
performance/REAL_HARDWARE_SCALING
```

```{toctree}
:hidden:
:caption: Architecture & development

architecture/DISTRIBUTED_ARCHITECTURE
user-guide/developers
user-guide/api
```

```{toctree}
:hidden:
:caption: Reference

references
```
