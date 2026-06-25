# Ocean model-config recipe audit & consolidation plan

**Status:** audit complete 2026-06-16. Companion to the recipe-architecture vision
(#388) and the Veros-stepping-duplication ticket (#433); extends Pierre's
ocean-experiment refactor (#376). First consolidation already landed: the
global-overturning model-config factories (branch
`cleanup/global-overturning-recipe-factory`).

## Why

The ocean code predates the "recipe" abstraction, so model-config assembly is
scattered. The *same* experiment's recipe can exist in up to three places —
the matrix runner, the production drivers, and a Layer-1 builder — assembled
three different ways and free to drift. This audit enumerates every ocean
model-config assembly, clusters them into distinct recipes, maps where each is
used, and flags divergence. Goal: one `create_model_config` factory per
experiment, consumed by matrix + drivers + fidelity harnesses (the pattern DINO
already follows).

## Scope of the sweep

Every construction of `LatLonCGridOceanConfig` / `MPASOceanConfig` /
`OceanConfig` (+ the cube `OceanConfig` and `SpectralOceanConfig`) across
`packages/ocean`, `scripts/run`, `scripts/matrix`. Raw counts: ~125 lat-lon +
~42 MPAS + ~26 cube files; the bulk (~325 sites) is in `tests/` (mostly minimal
unit-test fixtures — a separate, lower-priority category, not covered here).

## Recipe inventory (~11 families)

| # | Family | Defined in | Config class(es) | Assembly status |
|---|---|---|---|---|
| 1 | Veros ACC (faithful) | `ocean/fidelity/veros_acc_recipe.py` | LatLon | builder `build_acc_recipe` |
| 2 | Veros acc_basic | `ocean/fidelity/veros_acc_basic_recipe.py` | LatLon | builder (EKE off, const Prandtl) |
| 3 | Veros global_4deg | `ocean/fidelity/veros_global_4deg_recipe.py` | LatLon | builder (EKE `isopycnal_diffusion=False`, gsw EOS) |
| 4 | Veros global_1deg | `ocean/fidelity/veros_global_1deg_recipe.py` | LatLon | builder (sync dt, EKE back on, channel closures) |
| 5 | Veros global_flexible | `ocean/fidelity/veros_global_flexible_recipe.py` | LatLon | builder (Vinokur-stretched, EKE on) |
| 6 | Global overturning (latlon) | `experiments/global_overturning.py` | LatLon | ✅ factory `global_overturning_model_config` |
| 7 | Global overturning (MPAS) | `experiments/global_overturning.py` | MPAS | ✅ factory `global_overturning_mpas_model_config` |
| 8 | DINO (latlon + MPAS) | `experiments/dino.py` | LatLon + MPAS | ✅ factory pair — **gold standard** (matrix + driver both call it) |
| 9 | Eady idealized | `experiments/eady_uniform.py` | LatLon | inline-in-`build_eady_uniform_setup` |
| 10 | Silvestri jet | `experiments/silvestri_baroclinic_jet.py` | LatLon | inline-in-`build_silvestri_baroclinic_jet_setup` |
| 11 | bare/default | model `__init__` fallbacks | all | defaults |
| — | production tunings: Drake band, Tripole, comparison(latlon/mpas), OMIP, mpas_etopo | `scripts/run/` | LatLon/MPAS | hand-rolled |

Note on fingerprints: family 6/7 (global overturning) use **linear T-only EOS**
(not Wright) and the factory sets **no** momentum-advection scheme (model
default); DINO uses Wright EOS + Visbeck GM/Redi. (Some script-level fingerprint
guesses in the raw audit conflated these — trust the package-level reading.)

## Divergence findings (the actionable core)

### 1. Veros builders copy-paste the faithful-stepping bundle (= #433)
The bundle
`outer_integrator="ab2" + barotropic_solver="rigid_lid" + coriolis_scheme="explicit_ab2" + ab2_scope="advective" + momentum_friction_additive=True + implicit_vmix_dzw_slot=True`
is re-spelled in all 5 Veros builders; `acc_basic` keeps its **own copy** rather
than importing from `acc`. (`dt_mom_ratio` legitimately varies: 9 / 48 / 1 / 8.)
The *physics* blocks are properly factored (`ACC_TKE_CONFIG`/`ACC_GM_REDI_CONFIG`
reused via `._replace()`). **Fix:** one shared `VEROS_STEPPING_FAITHFUL` dict
imported by all five.

### 2. Drake-band scripts are unflagged global-overturning duplicates
`run_drake_momentum_budget{,_implicit,_divdamp}.py` + `run_drake_sensitivity.py`
(4 files) build `GlobalOverturningConfig` then hand-roll the exact config the GO
factory now produces, differing by a single knob (`barotropic_solver="implicit_cn"`
/ `barotropic_div_damp=0.1`). **Fix:** call `global_overturning_model_config(...)`
with overrides — same pattern as the GO consolidation, ~300 lines, bit-identical.

### 3. Matrix tests a *simpler* recipe than production runs (silent divergence)
`_run_experiment_via_registry` scrapes only a fixed subset of fields into
`setup_kw`:
`{A_h, A_v, K_h, K_v, K_bih, B_h, C_smag, bottom_drag_r, tracer_advection, barotropic_diffusion_alpha, barotropic_div_damp}`
(+ optional `eos_linear`/`gm_redi` via hooks). Structural fields **never
expressible** by the matrix: `pgf_scheme`, `momentum_advection`,
`barotropic_solver`, `slope_foot`, and the MPAS PV params
(`pv_scheme`/`apvm_dt`/`pv_alpha`/`K_zeta_bih`/`C_leith`). Consequence for
`global_overturning`: the matrix validates a Laplacian+GM/Redi recipe while the
production driver runs the full OMIP biharmonic stack — **the matrix is not
testing what production runs.** **DINO is the only experiment that avoids this**
(its `run_dino` matrix runner calls `dino_*_model_config` directly). **Fix:**
teach `_run_experiment_via_registry` to prefer
`EXPERIMENT_CONFIG["create_model_config"]` when present (now registered for GO),
then re-run the affected matrix cases to validate (NOT bit-identical — a genuine
fidelity improvement).

## Matrix dispatch summary

24 matrix runners: 7 use the generic registry path; 17 hand-roll inline; only
`dino` calls the experiment's own factory. `_create_ocean_setup` (modular,
`ocean_test_matrix/setup.py`) is strictly richer than the registry scrape (~22
vs ~13 expressible params), but only inline runners can leverage it.

## Consolidation plan (prioritized)

1. **Drake band → GO factory** (HIGH, low-risk, bit-identical). 4 scripts.
2. **Matrix → `create_model_config`** for GO (HIGH value; needs matrix re-run).
   Generalize the registry helper to call the factory; roll out per experiment.
3. **`VEROS_STEPPING_FAITHFUL` shared dict** across the 5 Veros builders (= #433).
4. **Migrate the remaining Layer-2 experiments** (eady, silvestri, and the
   bespoke matrix runners) to expose a `create_model_config` factory, so matrix
   + drivers converge on one recipe each (the DINO end-state).
5. **Tests** (separate track): point unit-test fixtures at shared factory/fixtures
   instead of ~325 inline constructions.

## Deprecation / cleanup candidates

- Fold `run_drake_momentum_budget_{implicit,divdamp}.py` into the base runner +
  a flag; delete the single-knob variants.
- Archive/document research one-offs if their campaigns are closed:
  `run_geometric_stage0*.py`, `mpas_realistic_geometry/run_mpas_*.py`.
- Move `diagnose_*.py` probes out of `scripts/run/global_overturning/` to
  `scripts/tmp/` (per the File Layout rules).
- **Keep hand-rolled (correctly — not recipe copies):** `run_comparison_{latlon,mpas}.py`
  (intentional A_h variance for cross-grid comparison), `run_omip.py` (CLI sweep
  harness), `run_tripole_20yr.py` (distinct tripolar/Wright tuning). `mpas_etopo`
  is closer to the MPAS factory and *could* adopt it (MEDIUM, optional).

## End-state target

Each experiment owns one `create_model_config` factory; matrix, production
drivers, and fidelity harnesses all consume it (DINO already does). We are ~2/11
families there (GO + DINO). This is the ocean instantiation of the
recipe-as-single-source-of-truth axis in #388.
