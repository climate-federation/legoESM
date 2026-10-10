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
| Domain sizing by parameters (positive = global, negative = per-subdomain for weak scaling) | `usrdef_nam.F90:103-114` (`nn_isize/jsize/ksize`) | `BenchConfig` fields; weak scaling (negative semantics) requires SPMD sharding — runners-repo follow-up, NOT wired here |
| Flat-bottom closed box, no land, uniform z-grid `zd = 5000/jpkm1` (74 wet levels for `nn_ksize=75`) | `usrdef_zgr.F90:139-165`, `usrdef_nam.F90` | `bench_uniform_z_star(n_levels=74, ...)` — `n_levels` is the WET count |
| Per-point ramp `z2d` with hemisphere mirroring (south `0.1*(p-0.5)`, north `0.1*(1.5-p')`, doubled numerator, split `mjg < Nj0glo/2._wp` REAL division) | `usrdef_istate.F90:41-54` | `build_z2d_ramp` (exact index construction incl. the doubled numerator and the REAL-division hemisphere split) |
| T/S/velocity ICs from `z2d` + depth factor `zfact = (jk-1)/(jpk-1)` | `usrdef_istate.F90:63-77` | `create_initial_conditions` (`f = k/n_levels`, max `(nlev-1)/nlev` = 73/74) |
| ssh from the plain point index, ±0.05 m | `usrdef_istate_ssh` | `create_initial_conditions` |
| Zero surface forcing (ocean) | `usrdef_sbc.F90:56-92` | `create_forcings` → `SurfaceForcingConfig(scheme="none")` |
| ORCA-like size presets 360×331 / 1440×1206 / 4320×3146 × `nn_ksize=75`, `nn_itend=1000` | `EXPREF/namelist_cfg_orca{1,025,12}_like` | `BENCH_PRESETS` (tripole keeps NEMO's exact j-count; latlon is the Mercator analog) + `--steps 1000` default |
| North-Fold (`ln_NFold`) in every preset | namelist `namusr_def` | `--grid tripole` → `create_synthetic_tripole` (active T-fold, no mesh file) |

## legoESM divergences (deliberate, recorded)

1. **Spherical grids, not the flat 100-km beta-plane — and the latlon
   lane is Mercator-truncated at 80°.** NEMO BENCH fixes dx = 100 km so
   one dt fits all sizes (5400/1440/480 s per preset). On the sphere dt
   scales with resolution. A full-sphere equirectangular latlon grid is
   **structurally unusable** for BENCH: its pole-row dx (~110 m at
   89.75° for 1° resolution) makes the explicit barotropic substep
   CFL-unstable at ANY preset dt (measured: 180×360 non-finite at step
   1-2 for dt down to 450 s / dt_baro down to 1 s; see the probe). The
   latlon analog therefore truncates at `lat_max_deg=80` with isotropic
   Mercator row placement (`create_mercator_grid`, the DINO placement),
   keeping NEMO's n_lon; n_lat = 2K is derived (278/1116/3350). NEMO
   BENCH never runs a full-sphere equirectangular grid either (its box
   is closed; the ORCA-like presets fold).
2. **IC backgrounds temperate, not near-freezing.** NEMO's T = −1 →
   −1.5 °C targets ~50% sea-ice cover under SI3 (`rn_thres_sst`);
   legoESM BENCH runs no SI3, so the light stratification
   ("almost constant everywhere with a light stratification", paper
   Sect. 2.2.1) is placed around T = 10 → 6 °C, S = 34 → 35 psu, with
   the `z2d` amplitudes and depth trends preserved verbatim.
3. **u/v face ramps.** NEMO evaluates z2d at the T-grid point and
   multiplies by umask/vmask (`usrdef_istate.F90:74-75`); the C-grid
   port builds each FACE axis its own face-count ramp
   (`build_z2d_ramp(n_lat, n_lon+1)` / `(n_lat+1, n_lon)`) so every
   face value stays unique — preserving the MPI-bug-detection property
   (values at the same physical location differ from NEMO's).
4. **Fold type.** NEMO's orca1_like uses an F-point fold
   (`cn_NFtype='F'`), orca025/12 a T-point fold; the synthetic tripole
   is a T-fold for every preset.
5. **No SI3/PISCES lanes** (follow-up); **no `ln_perpetual_ts`** (an
   Asselin-filter-specific NEMO device, inapplicable to legoESM
   integrators); **no bi-periodic option** (`nn_perio=7`).
6. **Physics mapping.** BENCH runs the production-like
   `legoesm_nemo_like_v1` recipe (EEN vector-invariant momentum,
   Hollingsworth KE gradient, PPM/FCT tracers, veros_gsw EOS ≈ TEOS-10,
   smc03 PGF, RK3 momentum) + TKE vertical mixing + enhanced-diffusion
   convection (NEMO `ln_zdftke` + `ln_zdfevd`) — APPLIED.  NOT mapped:
   MLE/EIV (orca1 namelist), bilaplacian tracer diffusion
   (orca025/12), UBS momentum + no lateral dyn. diffusion (orca12),
   double-diffusion and internal-wave mixing (all presets), and
   `ln_non_lin` quadratic bottom drag (the default keeps linear
   `bottom_drag_r=0`; pass `bottom_drag_scheme="nemo_quadratic"` via
   overrides to enable). On the 100-km meaningless-field grid these add
   cost without benchmarking value; turning each on is a one-line
   `physics=`/override.
7. **Weak scaling is NOT wired.** NEMO's negative `nn_jsize` semantics
   require actual SPMD domain sharding; legoESM's BENCH harness is
   single-process (each process would run the full domain). Real weak
   scaling belongs to the runners-repo SPMD harness (follow-up); no
   misleading flag is shipped.

## Pre-registered stability gates (before any long run)

Written down 2026-10-10, before the first multi-day run. **Stability only**
(the configuration is physically meaningless). A run is valid iff:

* every field finite at the final step;
* max pointwise speed (cell-centred) < 1.0 m/s (ICs are O(0.005) m/s);
* max |eta| < 1.0 m (ICs are O(0.05) m).

Planted-violation controls (`tests/ocean/unit/test_bench.py::
TestValidateGates`): NaN state, overspeed (5 m/s), overshot eta (5 m)
each FAIL the gate — including via the matrix scalar function's
`max_abs_eta` key — proving the gates fire in every consumer. The gates
are not to be relaxed to make a run pass; a FAIL means the preset dt (or
recipe) is wrong for that size, and the fix belongs in `BENCH_PRESETS`.

## Measured stability (committed probe, 2026-10-10, CPU, fp64)

Source: `scripts/validate/ocean_bench/stability_probes.py` (provenance-
stamped JSONL; git SHA, machine, JAX version per record). Lane values:

| Grid | dt (s) | steps | verdict |
|---|---|---|---|
| eq 24×48×10 | 3600 | 12 | STABLE (max_speed 0.0684, max_eta 0.2009) |
| eq 36×72×20 | 3600 | 12 | **UNSTABLE (non-finite; expected-FAIL control)** |
| eq 36×72×20 | 1800 | 60 | STABLE (max_speed 0.1599, max_eta 0.1090) |
| eq 36×72×20 | 1200 | 12 | STABLE (max_speed 0.0715, max_eta 0.2963) |
| eq 48×96×20 | 1200 | 12 | STABLE (max_speed 0.0631, max_eta 0.2922) |
| merc 278×360×74 (orca1_like shape) | 1200 | 60 | STABLE (max_speed 0.0528, max_eta 0.3369) |

Boundary measurements (not pre-registered, same session): the 1°
Mercator analog is stable at dt ≤ 1500 s (60 steps) and non-finite at
dt = 1650 s (step 13); the 0.5° analog (n_lon=720) is stable at
dt ≤ 600 s and non-finite at dt = 750 s. The full-sphere equirectangular
180×360 grid is non-finite at step 1-2 at every dt ≥ 450 s tested
(barotropic-pole CFL; divergence 1).

Presets therefore carry dt = 1200/300/120 s for orca{1,025,12}_like
(measured boundary halved for margin at 1°; the finer presets follow the
measured dx-scaling — re-measure before long runs).

**TKE-alive control** (same probe): the TKE lane is stable and its
60-step wall cost is 1.07× the `vmix=none` control (3.87 s vs 3.61 s) —
the closure solve runs and its cost is inside the measured step. On
these ICs the closure's *trajectory* effect is below 4-decimal detection
over 60 steps (K converges to background under zero forcing — the same
degeneracy NEMO BENCH's TKE has, by design).

## Surfaces

* Experiment module: `packages/ocean/legoesm/ocean/experiments/bench.py`
* Standalone timing harness (run_dino-compatible JSONL):
  `scripts/run/run_bench.py` — `--grid {latlon,tripole}`, `--preset`,
  timing record fields `compile_ms`, `steady_mean_ms`, `sypd`,
  `mcells_s`, `gates_pass`, ...
* Committed stability probe (the citable source for every number
  above): `scripts/validate/ocean_bench/stability_probes.py`
* CI matrix lane: `run_ocean_test_matrix.py --only bench --grid latlon`
  (36×72×74, dt=300 s default, PASS 2026-10-10: max_speed=0.0514 m/s,
  max_eta=0.1112 m, 3.4 s)
* Template: `config/templates/ocean/bench.yaml` (setup-style, `data: []`)
* Tests: `tests/ocean/unit/test_bench.py` (incl. planted violations);
  `test_recipe_map.py` (bench → `legoesm_nemo_like_v1`)

## Follow-ups (out of scope on this branch)

* MPAS Voronoi lane (grid_support currently False);
* runners-repo template + HPC scaling campaigns, incl. REAL weak
  scaling via SPMD sharding (after merge + re-pin);
* SI3 sea-ice lane (paper's ~1/5 ice-cover IC design);
* MPI waiting-time instrumentation (NEMO's Sect. 2.3 tool);
* orca025/orca12-shape stability probes on GPU/HPC (dt values are
  dx-scaled extrapolations from the measured 1°/0.5° boundaries).
