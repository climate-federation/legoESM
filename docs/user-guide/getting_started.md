# Getting Started with legoESM

Welcome. This is the **fastest** path from a clean checkout to a
running simulation — and a map for "where do I look next?"

If you have never touched legoESM before, read this top-to-bottom; it
takes about 20 minutes including the first run.

For technical depth, see [SPECIFICATION.md](../science/specs/SPECIFICATION.md). For
context on what changed recently, see [CHANGELOG.md](../../CHANGELOG.md).

---

## 1. What legoESM is (in one paragraph)

legoESM is a **differentiable Earth System Model**, written from
scratch in [JAX](https://github.com/google/jax). It couples
atmosphere, ocean, land, sea ice, lakes, and a tile-based surface
coupler through a single interface, and the entire coupled model is
end-to-end differentiable via `jax.grad`. The "lego" in the name is
literal: dynamical cores, physics schemes, grids, vertical
coordinates, time integrators, and model complexity are all swappable
independently, so you can run anything from a single-column RCE
experiment to a fully coupled centennial spin-up using the same
codebase.

Three things that make it different from a traditional ESM:

- **Differentiable**: parameters, initial conditions, and even
  parameterizations are gradient-trainable.
- **JIT-compiled**: every time-step kernel is compiled by JAX, so
  the same code runs on CPU, GPU, TPU, and Apple Silicon, and scales
  to MPI clusters via mpi4jax.
- **AI-ready**: SFNO neural cores and ML parameterizations live next
  to classical physics under the same interface.

---

## 2. Install (5 commands)

legoESM is a uv workspace of independently-installable members, so the install
command depends on whether you have `uv` (see the README "Installing" section for
the full why):

```bash
git clone https://github.com/climate-federation/legoESM.git
cd legoESM
python -m venv .venv
source .venv/bin/activate

# With uv (resolves the workspace natively):
uv sync --extra dev

# …or pip-only (no uv) — the helper resolves the inter-member dependency DAG:
python scripts/experiment/install_federation.py --all --extras dev
```

> A bare `pip install -e ".[dev]"` **fails** here (`Could not find a version that
> satisfies the requirement legoesm-core~=1.0.0`): plain pip cannot resolve the
> unpublished workspace members. Use `uv` or the helper above. To install just one
> component standalone: `python scripts/experiment/install_federation.py atmosphere`
> (or `ocean` / `land` / `ice`).

Optional MPI extras (only if you want multi-node runs):

```bash
pip install "mpi4py>=4.1,<5" "mpi4jax>=0.9,<0.10"
```

Requirements: **Python ≥ 3.11, JAX ≥ 0.4.35**. Apple Silicon users
who want the spectral solver should set `JAX_PLATFORMS=cpu` (the
Metal backend has no `float64`).

Quick sanity check:

```bash
.venv/bin/python -c "import legoesm, jax; print('jax', jax.__version__, 'backend', jax.default_backend())"
```

---

## 3. Your first runs

### a. A 30-second shallow-water test

```bash
legoesm test williamson --case 2 --resolution 48 --days 5
```

This runs Williamson Test Case 2 (geostrophically balanced flow)
on a C48 cubed-sphere. It is the canonical sanity check that
your install and JAX backend are working.

### b. A radiative-convective equilibrium column (SCM)

```bash
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python scripts/matrix/run_scm_test_matrix.py rce --days 50
```

A 50-day RCE single-column run with gray radiation, Louis turbulence,
Kessler microphysics, and a simple mass-flux convection scheme. See
[docs/user-guide/scm.md](scm.md) for the full SCM API.

### c. A 1-year AMIP smoke test

```bash
JAX_ENABLE_X64=1 .venv/bin/python scripts/run/run_amip.py \
    --grid-type cubed_sphere --resolution 16 --days 365
```

Atmosphere-only 365-day AMIP at C16/L40 with the bare-CLI default physics —
a smoke test, not the production configuration (that is
`--config config/amip/amip_production.yaml`; see [amip.md](amip.md)).
Output lands under `results/amip/` unless `--output` is given.

### d. (Optional) A coupled aquaplanet

```bash
JAX_ENABLE_X64=1 .venv/bin/python scripts/run/run_coupled.py --preset slab_simple --days 365
```

Fully coupled atmosphere + slab ocean + bucket land. Other presets:
`aquaplanet`, `slab_pft`, `slab_richards`, `slab_carbon`,
`full_coupled`.

---

## 4. Repository layout (where things live)

```
src/legoesm/
├── atmosphere/        # dycores, physics packages, SCM
│   ├── dynamics/      # cubed-sphere CD, lat-lon C, spectral, Voronoi
│   ├── physics/       # radiation, convection, turbulence, microphysics, GWD
│   └── scm.py         # single-column model
├── ocean/             # 3D ocean: dynamics, physics, experiments, BGC
│   ├── dynamics/      # split-explicit baroclinic + barotropic
│   ├── physics/       # KPP/Ri, GM-Redi, Leith, tidal mixing, bottom drag
│   ├── experiments/   # idealized testbeds (Eady, ACC, DINO, Munk, ...)
│   ├── fidelity/      # Veros peer-comparison adapters
│   └── spinup.py      # centennial spin-up + AMOC + RPE tracking
├── ice/               # sea-ice: EVP/mEVP, ITD, ridging, ponds, brine
├── land/              # slab + multilayer land, carbon (DALEC-990)
├── coupler/           # tile coupler, lake, ice-shelf cavity
├── grids/             # cubed-sphere, lat-lon, Mercator, tripolar, MPAS
├── driver/            # ModelDriver, CoupledESMDriver, EarthSystemDriver
├── parallel/          # ParallelRuntime, MPI halo, sendrecv VJP
├── ml/                # SFNO, channel packing, training (research)
├── da/                # variational data assimilation (4D-Var)
├── training/          # differentiable training driver + ERA5 ingestion
├── io/                # CMOR/CF-1.8 output (Amon/Lmon/Omon/Oyr/Ofx/SImon/SIyr)
├── diagnostics/       # energy budget, monthly means, column integrals
├── constants.py       # ALL physical constants live here — never inline literals
├── thermo.py          # saturation thermodynamics (use, do not re-derive)
└── supported_matrix.py  # canonical dispatch entries

scripts/               # CLI runners (run_amip.py, run_omip.py, run_scm_test_matrix.py, ...)
tests/                 # pytest suite + distributed/ + validation/
docs/                  # this folder
```

A few principles to internalise:

- **Constants are centralised.** `legoesm.constants` is authoritative
  for `g`, `R_d`, `c_pd`, `L_v`, `T_freeze`, etc. Never write
  literal `9.80616` or `273.15` anywhere.
- **Thermo is centralised.** `legoesm.thermo.saturation_vapor_pressure`
  and friends — never re-derive Tetens/Magnus.
- **Every dispatch branch has a test.** Adding a new scheme means
  adding a direct unit test that exercises it; see CLAUDE.md.

---

## 5. The core entry points

| You want to... | Use |
|---|---|
| Run a 3-D atmosphere | `ModelDriver` / `scripts/run/run_amip.py` |
| Run a fully coupled simulation | `CoupledESMDriver` / `scripts/run/run_coupled.py` |
| Run an OMIP ocean spin-up | `scripts/run/run_omip.py` |
| Run a single column | `SingleColumnModel.create()` / `scripts/matrix/run_scm_test_matrix.py` |
| Run a Williamson / DCMIP test | `scripts/matrix/run_atmosphere_test_matrix.py` |
| Run the ocean test matrix | `scripts/matrix/run_ocean_test_matrix.py` |
| Multi-device or MPI | `ParallelRuntime.create()` |
| Differentiable training | `legoesm.training.training_driver` |

All drivers are thin: read the relevant CLI script, then jump into
the `driver/` package to see what they call. For how the grid, complexity,
and physics bricks that these drivers assemble are chosen, see
[docs/architecture/composability.md](../architecture/composability.md).

---

## 6. The differentiability story

Everything inside the time-step kernel is pure JAX, so you can do:

```python
import jax
from legoesm.driver.compiled_segments import build_segment_fn

segment_fn = build_segment_fn(...)
loss, grads = jax.value_and_grad(loss_fn)(params, segment_fn.raw, ...)
```

A few rules that matter when you write your own loss:

- Use `.raw` (the non-donating variant) inside `jax.grad` /
  `eqx.filter_value_and_grad`. The JIT-compiled donating variant
  frees inputs and conflicts with reverse-mode AD.
- Pass time-varying forcing as an **explicit argument**
  (`SegmentForcing`), not via Python closure capture — otherwise
  every change recompiles.
- For MPI runs, only `allreduce(SUM)` is differentiable.
  `max`/`min`/`allgather`/`bcast` are diagnostic-only.

Full picture: [docs/science/ml_physics_parameterization.md](../science/ml_physics_parameterization.md).

---

## 7. Testing your changes

The single most useful command after editing physics:

```bash
JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/<area>/ -x
```

After touching the dycore on cubed-sphere, always run the atmosphere
test matrix in quick mode and **eyeball the v-wind snapshots**:

```bash
JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_atmosphere_test_matrix.py \
    --only sw --grid cubed_sphere --quick
```

Passing pytest is *necessary but not sufficient* on the cubed-sphere
— edge artifacts only reliably show up in field snapshots. See
[docs/dev-notes/cubed_sphere_edge_artifacts.md](../dev-notes/cubed_sphere_edge_artifacts.md).

Ocean: `scripts/matrix/run_ocean_test_matrix.py` (per-case PASS/FAIL is
written to its `summary.json`).

Fast static guardrails (constants/saturation/dispatch/contracts/federation
tripwires — seconds, no GPU):

```bash
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
    tests/test_no_hardcoded_constants.py tests/test_no_saturation_reimpl.py \
    tests/test_dispatch_hardening.py tests/test_physics_contracts.py \
    tests/test_import_boundaries.py tests/test_federation_plan.py -q
```

For the full picture — pytest tiers, the matrix framework, the guardrail harness,
the `scripts/validate/` scientific validators, and the adversarial-review agents
(Codex `/codex:adversarial-review`, `physics-validator`, `lego-modularity-tester`,
…) — see [docs/validation/TESTING.md](../validation/TESTING.md).

Sea ice: `scripts/matrix/run_sea_ice_test_matrix.py` (15 standard benchmarks).

---

## 8. Common pitfalls

- **Forgetting `JAX_ENABLE_X64=1`.** Most scientific tests need
  `float64`. Spectral solvers *require* it.
- **`Buffer has been deleted` errors.** Donated input was reused.
  Use the `.raw` variant (non-donating) inside `jax.grad`.
- **A new function that already exists.** Search before writing.
  Constants in `constants.py`, thermo in `thermo.py`, column
  integrals in `diagnostics/column_integrals.py`, losses in
  `ml/loss.py`, atmosphere column helpers in
  `atmosphere/physics/_shared.py`.
- **NumPy in source.** Use `jax.numpy` everywhere. NumPy breaks
  JIT and AD.
- **Apple Silicon + spectral solver.** Force CPU:
  `JAX_PLATFORMS=cpu`.

---

## 9. Where to go next

- [docs/architecture/composability.md](../architecture/composability.md) — **how the model is assembled**: instantiating each axis (grids, regional/idealized extent, the SCM/LES/CRM/shallow-water/3-D complexity ladder), a high-level package tour, and the research → operational (AMIP/OMIP/CMIP) ladder.
- [README.md](../../README.md) — feature inventory and platform matrix.
- [SPECIFICATION.md](../science/specs/SPECIFICATION.md) — full technical specification.
- [docs/validation/cmip_readiness.md](../validation/cmip_readiness.md) — CMIP production checklist.
- [docs/user-guide/amip.md](amip.md) — AMIP workflow.
- [docs/user-guide/scm.md](scm.md) — single-column model.
- [docs/dev-notes/ocean_experiments_reference.md](../dev-notes/ocean_experiments_reference.md) — ocean testbeds.
- [docs/performance/REAL_HARDWARE_SCALING.md](../performance/REAL_HARDWARE_SCALING.md) — multi-GPU / MPI.
- [docs/science/ml_physics_parameterization.md](../science/ml_physics_parameterization.md) — joint ML / physics workflow.
- [docs/validation/dycore_validation_catalog.md](../validation/dycore_validation_catalog.md) — dycore validation inventory.

When you are ready to contribute: read `CLAUDE.md` at the repo root.
It encodes the engineering rules (constants discipline, audit
expectations, naming conventions) that the codebase is enforced
against.

Welcome aboard.
