# CLM-ML-JAX — State, Test, and Usability Report (2026-07-12)

Session goal: verify the CLM-ML-JAX canopy code on `main`, test whether it works,
review how it works now, and assess "readily available for anyone to use." The
advisor had reported "issues with the code."

## Bottom line

**The code on `main` works. The advisor's "issues" were a broken environment, not
broken code.** After fixing the environment, the interface passes end-to-end.

- Native JAX two-leaf canopy: **112/113 unit tests pass** (1 stale test, see below).
- CLM-ML-JAX external interface: **31/32 unit tests pass** (1 expected xfail =
  not JAX-differentiable).
- CLM-ML integration tests: **31 passed, 1 XPASS(strict) in 14m40s** — the lone
  "failure" is a *stale xfail marker*, not a regression (see below).

## Architecture (how it works now on main)

Two canopy schemes selected by config **type** via `isinstance` dispatch in
`packages/land/legoesm/land/multilayer_land.py`:

1. **Native JAX two-leaf canopy** — `canopy/{solver,photosynthesis,radiative_transfer,
   sif,energy_balance,stability}.py` (~3.7k LOC). Fully JAX/JIT/`grad`-compatible.
   Sun/shade big-leaf, canonical FvCB C3+C4 (consolidated #897), optional SIF.
   **Dependency-free, differentiable — the default recommended path.**
   Selected via `MultiLayerLandConfig(surface_scheme=TwoLeafCanopyConfig(...))`.
2. **CLM-ML-JAX external interface** — `canopy/clm_ml_interface.py` (1302 LOC).
   Wraps the Fortran-ported `clm-ml-jax` package (Bonan 2021, CLM-ML v2, 9-layer
   multilayer canopy). **NOT** JAX-differentiable (`differentiable: False`,
   forward-only). Lazy imports keep base legoESM importable without it.
   Selected via `MultiLayerLandConfig(surface_scheme=CLMMLCanopyConfig(...))`.
   Entry: `compute_clm_ml_canopy_fluxes(...)`.

Both return the shared `SurfaceFluxOutput`; energy contract:
`Rnet = shflx + lhflx + G_soil + stflx_air + stflx_veg`.

## Environment setup that makes it run (THE reproducible fix)

The repo `.venv` (Python **3.14**, arm64) was broken for the CLM-ML path:

| Problem | Fix |
|---|---|
| `.venv/bin/pip` missing; `legoesm` meta-pkg not installed editable | run tests with `PYTHONPATH` = all 8 `packages/*` roots (see below), or `pip install -e ".[dev]"` |
| `netCDF4` installed as **x86_64** `.so` under arm64 Python → `dlopen` fails in CLM-ML RSL psihat lookup (`MLCanopyTurbulenceMod.LookupPsihatINI`) | rebuild from source against Homebrew HDF5 (arm64) |
| No cp314 netCDF4 wheel exists (any arch) | source build, or **use Python 3.11/3.12** |
| `f90nml` missing (clm-ml-jax dep) | `pip install f90nml` |
| `clm-ml-jax` package not installed | `pip install -e ./clm-ml-jax` (it is a **separate, gitignored git repo** at repo root) |

Commands used (from repo root):
```bash
PP="packages/core:packages/atmosphere:packages/ocean:packages/land:packages/ice:packages/coupler:packages/ml:packages/tools"
.venv/bin/python -m pip install -e clm-ml-jax/ --no-deps    # NOTE: python -m pip (no pip binary)
.venv/bin/python -m pip install f90nml
HDF5_DIR="$(brew --prefix hdf5)" NETCDF4_DIR="$(brew --prefix netcdf)" \
  .venv/bin/python -m pip install --force-reinstall --no-cache-dir --no-binary netCDF4 netCDF4
# run canopy tests:
JAX_ENABLE_X64=1 PYTHONPATH="$PP" .venv/bin/python -m pytest tests/land/unit/test_canopy.py -q
```

## The one real code defect (stale test on main)

`tests/land/unit/test_canopy_soil_evap_resistance.py::test_higher_exp_reduces_bare_soil_evaporation`
**fails**: `LE_soil` is bit-identical (77.382) regardless of `soil_evap_resistance_exp`.
Root cause: the EC-site work made `soil_evap_series_resistance=True` the **default**
(`packages/land/legoesm/land/config.py:132`), which *by design* bypasses the
`S_top**soil_evap_resistance_exp` beta-efficiency path (mutually exclusive). The test
builds its config **without** `soil_evap_series_resistance=False`, so the exponent is
inert. **Model correct; test stale.** Fix: add `soil_evap_series_resistance=False` to
the `_le_soil` config in that test.

## Second stale test (integration) — do NOT auto-flip

`tests/land/integration/test_canopy_carbon_coupling.py::test_prognostic_lai_jax_grad_through_feedback`
is decorated `xfail(strict=True)` documenting a NaN in reverse-mode grad through
`C_fol → LAI → canopy Newton IFT adjoint` (interaction of `jacfwd` in `solve_bwd` with
the `LAI_CRIT=2` hard clip in `compute_aerodynamics`; tracked for Phase 7). This session
it **XPASSed** — the grad ran clean, so strict-xfail flags it as a failure. Meaning: the
documented limitation may now be fixed, OR the NaN is nondeterministic/platform-dependent.
**Action: do not blindly remove the marker** — first confirm the grad is reliably
finite across seeds/platforms; if so, drop the xfail and keep the test as a real
differentiability guard. (Contrast the soil-evap test, which was an unambiguous 1-line fix.)

## "Readily available for anyone" — gaps

1. **`[canopy]` extra is unresolvable.** `pyproject.toml` pins
   `canopy = ["clm-ml-jax>=0.1.0"]`, but that package is **not on PyPI** — it is the
   local gitignored `clm-ml-jax/` repo. `pip install ".[canopy]"` fails for anyone.
   Fix: publish clm-ml-jax, point extra at a git URL, or document `pip install -e ./clm-ml-jax`.
2. **Pin a supported Python** (3.11/3.12). 3.14 has no netCDF4 wheel → source build.
3. **Document the real install recipe** (above). The native two-leaf path needs none
   of this and should be the default for anyone not needing CLM-ML v2 fidelity.

## Test timings (measured)

- Native canopy unit tests (112): ~3m17s (JAX x64 compile-bound).
- CLM-ML interface unit tests (32): ~1m08s.
- CLM-ML integration tests (32): >8 min (pure-Python CLM-ML forward, not JIT).

## Git handling this session (nothing destroyed)

- Branch `aya/mergeMLC_experiments` = 0 commits ahead of main, 125 behind. Untouched.
- WIP stashed: `stash@{0}` = `.claude/settings.json` + `scripts/validate/validate_clm_ml_canopy.py`.
- 16 untracked experiment files (fluxnet/chats offline runners, plots) left in place.
- Checked out `origin/main` **detached** (branch pointer never moved). Restore: `git checkout aya/mergeMLC_experiments && git stash pop`. Pre-session HEAD: `774f672ed`.

## Validation data present (for CHATS7 Fortran-fidelity check)

- Fortran reference: `docs/output_files_clm_ml-v2/CHATS7_2007-*.out` (present, gitignored).
- JAX outputs: `clm-ml-jax/src/output_files/JAX_outputs_05_2007_31days/` (present).
- Harness: `scripts/validate/validate_clm_ml_canopy.py` (plot/compare, not a live run).

## Open items / recommended next actions

- [ ] Fix the stale soil-evap test (1 line).
- [ ] Make `[canopy]` extra installable (git URL or publish) + write install doc.
- [ ] Add a numeric CI regression test pinning CHATS7 validation (memory notes this gap).
- [ ] Known physics debt (from prior memory, still open): snow/frozen-soil fluxes
      (`h2osoi_ice=0, snl=0`), `soilresis_col` fixed 2000 s/m, `thk_col` 0.5 default,
      single-PFT tiles, C4 path untested, geographic `lon` not passable (virtual-lon only).
