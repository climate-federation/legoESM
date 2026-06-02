# legoESM

**A differentiable Earth System Model in JAX.**

legoESM is a from-scratch Earth System Model built so that the *entire* simulation
— atmosphere, ocean, land, sea ice, and their coupling — is a pure JAX program you
can differentiate end to end with `jax.grad`. That makes data assimilation,
parameter estimation, and hybrid AI–physics modelling first-class rather than
bolted on.

## The three Lego axes

Everything is assembled from interchangeable bricks along three independent axes —
inspect the full capability map in [`legoesm.taxonomy`](#capability-taxonomy):

| Axis | What varies | Where it lives |
|------|-------------|----------------|
| **Complexity** | fixed-SST → slab → multilayer → full-3-D ocean; shallow-water → hydrostatic → nonhydrostatic atmosphere; … | `legoesm.components.complexity` |
| **Bricks** | run a component standalone **or** fully coupled | `legoesm.components` (Dycore/Forced/Surface/Prescribed bricks) |
| **Extent** | global, regional (limited-area), or idealized; uniform or **refined** | `legoesm.grids.factory` |

Grids are shared between components: lat-lon FV, spectral Gaussian, cubed-sphere
(FV3-faithful), MPAS/Voronoi (TRiSK), and SFNO all feed both the atmosphere and the
ocean.

## Quickstart

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

## Capability taxonomy

```python
from legoesm import taxonomy
taxonomy.capability_report()   # the full complexity × extent × grid/core map
taxonomy.grid_capability("mpas").variable_resolution   # True
```

## Reproducibility

Every run writes a manifest (resolved config + RNG seeds + state digest); the
`legoesm reproduce <manifest> --check` gate re-runs into a fresh directory and
asserts a bit-identical result. See [Getting started](getting_started.md).

## Where to go next

- [Getting started](getting_started.md) — install, first run, the reproduce gate
- [Single-column model](scm.md) — the cheapest standalone brick + gradient check
- [Dycore validation catalogue](dycore_validation_catalog.md) — Williamson,
  Galewsky, Jablonowski–Williamson, DCMIP, Held–Suarez
- [CMIP readiness](cmip_readiness.md) and [real-hardware scaling](REAL_HARDWARE_SCALING.md)

## Citation

See `CITATION.cff` (GitHub "Cite this repository") and `.zenodo.json` — a
Zenodo-linked release mints the DOI.
