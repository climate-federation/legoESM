# BENCH — NEMO-BENCH-inspired ocean performance benchmark

**Status:** implemented on branch `bench/ocean-bench` (2026-10-10).
**Source of truth:** Irrmann, G., Masson, S., Maisonnave, É., Guibert, D.,
& Raffin, E. (2022), "Improving ocean modeling software NEMO 4.0
benchmarking and communication efficiency", GMD 15, 1567–1582,
https://doi.org/10.5194/gmd-15-1567-2022, Sect. 2.2; NEMO sources
`tests/BENCH/{MY_SRC,EXPREF}` (forge.nemo-ocean.eu, main).

## What BENCH is (and is not)

A performance-benchmark ocean configuration that keeps the *full production
complexity* of the model while requiring **zero input files** and producing
**physically meaningless** results. From the paper (Sect. 2.2.1): results
"are meaningless from a physical point of view and should be used only for
benchmarking purposes". The design goals, in order:

1. trivial to deploy (source checkout only — no data fetch);
2. every NEMO scheme/parameterization exercisable (not a dwarf);
3. no I/O perturbation of timing (no input, no output by default);
4. **unique value at every grid point** → halo-exchange / sharding bugs
   are detectable in the fields themselves.

## NEMO BENCH reference (formulas ported, with provenance)

| Element | NEMO source | Port in legoESM |
|---|---|---|
| Domain sizing by parameters (positive = global, negative = per-subdomain for weak scaling) | `usrdef_nam.F90:103-114` (`nn_isize/jsize/ksize`) | `BenchConfig` fields + `--weak-lat-rows` in `run_bench.py` |
| Flat-bottom closed box, no land, uniform z-grid `dz = H/(jpk-1)` | `usrdef_zgr.F90:139-165` | `bench_uniform_z_star` (explicit-thickness z*) |
| Per-point ramp `z2d` in ±0.1 with hemisphere mirroring | `usrdef_istate.F90:44-54` | `build_z2d_ramp` (exact index construction, incl. the doubled numerator) |
| T/S/velocity ICs from `z2d` + depth factor `f = (k)/(jpk-1)` | `usrdef_istate.F90:63-77` | `create_initial_conditions` |
| ssh from the plain point index, ±0.05 m | `usrdef_istate_ssh` | `create_initial_conditions` |
| Zero surface forcing (ocean) | `usrdef_sbc.F90:56-92` | `create_forcings` → `SurfaceForcingConfig(scheme="none")` |
| ORCA-like size presets 360×331 / 1440×1206 / 4320×3146 × 75, `nn_itend=1000` | `EXPREF/namelist_cfg_orca{1,025,12}_like` | `BENCH_PRESETS` (n_lon kept; latlon analog rows) + `--steps 1000` default |
| North-Fold (`ln_NFold=T/F`) in every preset | namelist `namusr_def` | `--grid tripole` → `create_synthetic_tripole` (active fold, no mesh file) |

## legoESM divergences (deliberate, recorded)

1. **Spherical grids, not the flat 100-km beta-plane.** NEMO BENCH fixes
   dx = 100 km so one dt fits all sizes (5400/1440/480 s per preset). On
   the sphere dt must scale with resolution; the presets below carry
   MEASURED stability values for the `legoesm_nemo_like_v1` recipe
   (RK3 momentum + explicit barotropic substeps).
2. **IC backgrounds temperate, not near-freezing.** NEMO's T = −1 →
   −1.5 °C targets ~50% sea-ice cover under SI3 (`rn_thres_sst`);
   legoESM BENCH runs no SI3, so the light stratification
   ("almost constant everywhere with a light stratification", paper
   Sect. 2.2.1) is placed around T = 10 → 6 °C, S = 34 → 35 psu, with
   the `z2d` amplitudes and depth trends preserved verbatim.
3. **No SI3/PISCES lanes** (follow-up); **no `ln_perpetual_ts`** (an
   Asselin-filter-specific NEMO device, inapplicable to legoESM
   integrators); **no bi-periodic option** (`nn_perio=7`).
4. **Physics mapping.** BENCH runs the production-like
   `legoesm_nemo_like_v1` recipe (EEN vector-invariant momentum,
   Hollingsworth KE gradient, PPM/FCT tracers, veros_gsw EOS ≈ TEOS-10,
   smc03 PGF, RK3 momentum) + TKE vertical mixing + enhanced-diffusion
   convection (NEMO `ln_zdftke` + `ln_zdfevd`); no GM/Redi/MLE (NEMO's
   BENCH orca1 namelist enables EIV/MLE, but on a 100-km grid with
   meaningless fields they add cost without benchmarking value —
   turning them on is a one-line `physics=` override).
   Bottom drag: the recipe default (linear `bottom_drag_r=0`); NEMO's
   `ln_non_lin` quadratic law is available via
   `bottom_drag_scheme="nemo_quadratic"` override.

## Pre-registered stability gates (before any long run)

Written down 2026-10-10, before the first multi-day run. **Stability only**
(the configuration is physically meaningless). A run is valid iff:

* every field finite at the final step;
* max speed < 1.0 m/s (ICs are O(0.01) m/s);
* max |eta| < 1.0 m (ICs are O(0.05) m).

Planted-violation controls (`tests/ocean/unit/test_bench.py::
TestValidateGates`): NaN state, overspeed (5 m/s), overshot eta (5 m)
each FAIL the gate — proving the gates fire. The gates are not to be
relaxed to make a run pass; a FAIL means the preset dt (or recipe) is
wrong for that size, and the fix belongs in `BENCH_PRESETS`.

## Measured stability boundaries (10-step probes, 2026-10-10, CPU, fp64)

| Grid (latlon) | dt (s) | stable? |
|---|---|---|
| 24×48×10 | 3600 | yes (12 steps) |
| 36×72×20 | 3600 | **NO — NaN on step 1** (also with vmix off) |
| 36×72×20 | 1800 | yes (60 steps, gates PASS: max|u|=0.14, max|eta|=0.09) |
| 36×72×20 | 1200 | yes (10 steps) |
| 48×96×20 | 1200 | yes (10 steps) |

Presets therefore carry dt = 1800/450/120 s for orca{1,025,12}_like
(halved first guesses, stability margin). The 36×72 boundary shows the
RK3+explicit-barotropic stack needs roughly half the OMIP forward-Euler
dt at 2° spacing; re-measure before trusting any preset beyond 10× the
probe length.

## Surfaces

* Experiment module: `packages/ocean/legoesm/ocean/experiments/bench.py`
* Standalone timing harness (run_dino-compatible JSONL):
  `scripts/run/run_bench.py` — `--grid {latlon,tripole}`, `--preset`,
  `--weak-lat-rows`, timing record fields `compile_ms`, `steady_mean_ms`,
  `sypd`, `mcells_s`, `gates_pass`, ...
* CI matrix lane: `run_ocean_test_matrix.py --only bench --grid latlon`
  (36×72×20, dt=300 s default, PASS 2026-10-10, 3.7 s)
* Template: `config/templates/ocean/bench.yaml` (setup-style, `data: []`)
* Tests: `tests/ocean/unit/test_bench.py` (21 tests incl. planted
  violations); `test_recipe_map.py` updated (bench →
  `legoesm_nemo_like_v1`, coverage set explicitly extended)

## Follow-ups (out of scope on this branch)

* MPAS Voronoi lane (grid_support currently False);
* runners-repo template + HPC scaling campaigns (after merge + re-pin);
* SI3 sea-ice lane (paper's ~1/5 ice-cover IC design);
* MPI waiting-time instrumentation (NEMO's Sect. 2.3 tool).
