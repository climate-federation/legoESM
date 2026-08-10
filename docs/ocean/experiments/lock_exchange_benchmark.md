# Lock exchange: the standard cross-dycore spurious-mixing benchmark

The lock exchange (Petersen et al. 2015, DOI:10.1016/j.ocemod.2014.12.004) is
the standard benchmark every legoESM ocean dycore must pass. It measures the
one thing tracer advection must not do: mix water masses that the resolved
flow did not mix.

## Setup (identical across arms — protocol invariants)

- `LockExchangeConfig`: flat 20 m basin, 20 levels, T = 5 °C west / 30 °C east
  of `front_longitude = 0°`, S uniform, rest start, land poleward of 80°.
- `DEFAULT_DT = 300 s`, 5 days, matched `A_v = 1e-4`, `K_v = 1e-5` m²/s.
- Barotropic solver: `implicit_cn` on the three configurable arms (latlon,
  mpas, tripole) — the explicit split's eta/advecting-flux inconsistency
  breaks flux-form constancy preservation and over/undershoots ANY limiter
  (uniform-T probe: ±0.05 K/day explicit vs ±1e-10 implicit). FESOM builds
  its own implicit CG SSH operator inside fesom_jax (not selectable here).
- Tracer advection: Zalesak FCT class on latlon + tripole (`fct2`,
  `LOCKEX_CGRID_TRACER_ADV`) and FESOM (internal); MPAS `tvd` (no FCT in the
  port — bounded to 3e-12/day at ico3). Dim-split TVD is not multi-D
  monotone on distorted curvilinear cells; certified FCT is.
- Documented per-arm geometry difference: the **tripole (eORCA1)** arm keeps
  Earth's continents (latitude mask ∩ NEMO tmask: 0.534 wet on the native
  332x362 mesh; 0.607 on the regridded 1° artifact, vs 0.889 for the pure
  latitude mask) because NEMO-closed cells carry degenerate metrics that
  cannot be opened. Depth stays flat 20 m.

## Metric

`RPE_rel` = drift of the **sorted reference potential energy on moving
z-star volumes** (`RPE_mov`: parcels = `area·h(eta)`, densest packed at the
bottom; kernel `legoesm.ocean.rpe.pack_sorted_rpe`). Spurious mixing can
only RAISE it; a negative drift means the metric or the dycore is wrong.
Fixed-reference-volume RPE was retracted: it drifts negative even on
bounded, exactly conserving arms (reversible eta / linfs
concentration–dilution term).

## Gates (`run_lock_exchange`, all whole-run)

| Gate | Threshold | Catches |
|---|---|---|
| `T_min_front` / `T_max_front` | IC bounds ± 1e-6 K, ocean-masked, per-sample extremes | limiter overshoot, flux/thickness inconsistency |
| `RPE_rel_mixing_sign` | ≥ −1e-12 | sign-inverted metric, anti-diffusive dycore |
| `heat_rel_drift` | ≤ 1e-6 relative (moving volumes) | flux leaks, clipping |
| `PE_rel_final < 0` | — | missing PE→KE conversion |
| flat-bottom / positive-volume / spectral tripwires | raise | metric misuse |

## How to run

Prerequisites: the tripole arm needs the eORCA1 mesh file
`data/grids/eORCA1.2_mesh_mask.nc` (Zenodo record 4436658, 462 MB, not
tracked); the FESOM arm needs the packaged `pi` mesh (fesom_jax install).
The `sbatch` wrapper is Levante-specific (account/partition/paths); on any
other machine use the per-arm command below.

```bash
# all four arms (Levante; one GPU each, ~1-30 min):
sbatch --array=0-3 scripts/cluster/lock_exchange_dycore_comparison.sbatch
# single arm:
JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_ocean_test_matrix.py \
    --only lock_exchange --grid tripole
# comparison figure (below) from the finished runs:
.venv/bin/python scripts/plot/plot_lock_exchange_benchmark.py
# per-step diagnosis (RPE_fix/RPE_mov, conservation, violation locations,
# scheme A/B, uniform-T constancy probe):
.venv/bin/python scripts/validate/lockex_rpe_trace.py --grid latlon --uniform-t
```

Figure: `results/lock_exchange_dycore/lock_exchange_benchmark.png` — per
arm SST + equatorial T sections (day 1 / day 5) + the RPE_rel overlay and
gate summary.

## Current reference numbers (5 days, 2026-08-10, PR #1549 code)

| Arm | Status | RPE_rel (mixing) | T range (whole run) |
|---|---|---|---|
| latlon 36x72 | PASS | +1.711e-8 | [5, 30] ± 1e-9 |
| mpas ico3 | PASS | +6.201e-8 | [5, 30] ± 1e-9 |
| fesom pi | PASS | +1.153e-8 | [5, 30] ± 1e-12 |
| tripole eORCA1 | PASS | +1.357e-8 | [5, 30] ± 1e-9 |

MPAS's larger (still tiny) mixing is the tvd-vs-FCT scheme difference, not
a defect. History of how these gates were earned (explicit-split
inconsistency, FCT h_new certification bug, TVD corner overshoot):
`docs/dev-notes/fesom_lockexchange_session_state.md`.

## The rest of the standard ocean-grid suite

Lock exchange is one case in a family. The other standard cases and the
grids they cover (`sbatch --array=0-8
scripts/cluster/ocean_grid_benchmark_suite.sbatch`, results 2026-08-10,
job 26854231):

| Case | cube | latlon | mpas | tripole | fesom |
|---|---|---|---|---|---|
| rest_state_stratified_with_land | PASS | PASS | PASS | PASS | PASS |
| rest_state_uniform_with_land | PASS | PASS | PASS | PASS | PASS |
| rest_state_stratified_no_land | PASS | PASS | PASS | —¹ | —² |
| rest_state_uniform_no_land | PASS | PASS | PASS | —¹ | —² |
| barotropic_wave | PASS | FAIL³ | PASS | PASS | FAIL⁴ |
| geostrophic_adjustment | FAIL³ | PASS | PASS | PASS | PASS |
| phillips_two_layer | FAIL³ | PASS | PASS | FAIL⁵ | FAIL⁵ |
| inertia_gravity_wave | PASS⁶ | PASS⁶ | PASS⁶ | PASS⁶ | PASS⁶ |
| lock_exchange | —⁷ | PASS | PASS | PASS | PASS |

Every case now RUNS on every registered grid (no ERROR rows).

### Gates are stratified by MODELLING MODE, never by arm

A benchmark that loosens a gate "for the FESOM arm" is an exemption list.
Conservation tolerances are therefore keyed on the mode the arm integrates
in, and every model in that mode is held to the same number
(`_MODE_TRACER_DRIFT_TOL_PER_DAY` in the matrix):

| Mode | Tracer-content drift | Basis |
|---|---|---|
| `moving_thickness` (z-star / ALE) | 1e-8 /day | measured lat-lon z-star: 1.2e-15 over 2 days |
| `fixed_thickness` (linfs, NEMO key_linssh) | 5e-6 /day | measured FESOM linfs: 2.6e-6 over 2 days |

Per **day**, not absolute: this suite runs 1- to 10-day cases, and an
absolute tolerance would let a short run hide a leak a longer run fails on.

### RETRACTED: "FESOM does not conserve heat"

An earlier revision of this document claimed FESOM had a real heat-
conservation defect. That is not supported. Decomposing the drift as
`ΔH = Σ(T(t)−T(0))·V(t) + ΣT(0)·(V(t)−V(0))` (tracer term A, thickness
term B) gives, over 2 days of geostrophic adjustment:

| Arm | A (tracer) | B (thickness) | residual |
|---|---|---|---|
| lat-lon z-star | −1.925e-7 | +1.925e-7 | 1.2e-15 (they cancel) |
| FESOM z-star ALE | −2.366e-7 | +6.812e-7 | 4.4e-7 |
| FESOM linfs | −3.288e-6 | +6.815e-7 | 2.6e-6 |

Term B is identical to three digits between the two FESOM modes *even
though one moves its layer thicknesses and the other does not* — which can
only happen if B is computed from `eta` by the **diagnostic** rather than
from the model's own thicknesses (FESOM carries `hnode`). So the leading
candidate is a diagnostic-vs-model volume mismatch on the FESOM arm. The
open action is to read `hnode` in the diagnostic and re-measure.

Also refuted along the way: switching FESOM to z-star ALE was predicted to
move these gates to PASS. It removes ~83% of the linfs drift and no more —
`FesomOceanConfig.vertical_coordinate="zstar"` is now selectable (default
stays `linfs`) so the experiment is repeatable.

1. The tripole basin is defined by the NEMO tmask, so a no-land tripole
   variant does not exist.
2. The FESOM mesh carries the same 80° land threshold as the with-land
   arms (190 of 3140 nodes dry) and the setup exposes no 90° variant, so
   FESOM belongs to the with-land rows only.
3. Pre-existing failure, unchanged by this work.
4. `eta_conservation = 0.449` vs a 0.5 gate. NOT explained by the linfs
   mode (that is a tracer-content approximation, not a free-surface mass
   one) — a separate, open FESOM-on-triangles item.
5. Baroclinic-instability growth exceeds the `eta_growth < 10` gate
   (tripole 33.9, FESOM 20.8; cube 26.4 has always failed it). The ICs
   run correctly on both arms — this is a physics/gate question, not
   wiring.
6. UNGATED on L2. `eta_exact` in this case is an **f-plane plane wave**
   (constant f0 = 1e-4, i.e. an f-plane at 43.3°, with zonal wavenumber
   `kx/a`) while every arm integrates the full sphere (f = 2Ω sin(lat)
   spanning ±1.46e-4; true wavenumber `kx/(a cos lat)`, 2× larger at
   60°). The comparison is unpassable by construction — measured L2 ≈ 1.0
   on latlon, mpas, tripole and FESOM alike. A gate no arm can pass
   discredits the whole matrix, so L2 and amplitude are now reported as
   diagnostics and the case gates on stability: peak |eta| over the run
   ≤ 5× the IC amplitude (calibrated, not guessed — measured peaks are
   1.001 latlon / 1.000 mpas / 1.000 FESOM / 3.428 tripole). Rebuilding
   the case on an f-plane channel, where a Poincaré wave is defined, is
   the open action. (Note: the previously reported "ω matches on every
   grid" was never a measurement — ω is computed by the harness from one
   formula and never read from the model. Retracted.)
7. Lock exchange excludes cubed_sphere; see the exclusion note above.
