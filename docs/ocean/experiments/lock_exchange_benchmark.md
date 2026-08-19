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
grids they cover (`sbatch --array=0-6
scripts/cluster/ocean_grid_benchmark_suite.sbatch`, results 2026-08-10,
job 26850849):

| Case | latlon | mpas | fesom | tripole |
|---|---|---|---|---|
| rest_state_stratified_with_land | PASS | PASS | PASS | PASS |
| rest_state_uniform_with_land | PASS | PASS | PASS | PASS |
| rest_state_stratified_no_land | PASS | PASS | —¹ | —² |
| rest_state_uniform_no_land | PASS | PASS | —¹ | —² |
| geostrophic_adjustment | PASS | PASS | —³ | PASS |
| phillips_two_layer | PASS | PASS | —³ | —⁴ |
| inertia_gravity_wave | FAIL⁵ | FAIL⁵ | —³ | —⁴ |
| lock_exchange | PASS | PASS | PASS | PASS |

Rest-state drifts are at machine precision on every arm (eta 0 to 1e-31,
T ≤ 1.5e-14, S ≤ 7.6e-15); geostrophic adjustment settles to
max|u| = 0.014 (latlon) / 0.027 (mpas) / 0.446 m/s (tripole, the
continental-boundary arm) against a 1 m/s gate.

1. The FESOM mesh is built with the same 80° land threshold as the
   with-land arms (190 of 3140 nodes dry), and the setup exposes no 90°
   variant — so FESOM belongs to the WITH-LAND rows only. (An earlier
   version of this table registered it as "no land"; that was a
   mislabel, caught in review.)
2. The tripole basin is defined by the NEMO tmask, so a no-land tripole
   variant does not exist.
3. FESOM has IC builders for the rest state and the lock exchange only;
   the perturbation cases would each need a node-based analytic IC.
4. The IC writes the analytic u/v EDGE fields from 1-D `grid.lat`/`grid.lon`
   and a uniform `dlon`/`dlat` — rectilinear-only; on the curvilinear
   eORCA1 mesh both now raise `NotImplementedError` naming the gap
   (previously a bare shape mismatch). Needs a curvilinear edge IC.
5. Pre-existing failure on every grid (analytic L2 0.91 latlon / 1.01 mpas
   / 0.23 cubed_sphere vs a 0.1 gate) — an unresolved case, not an arm
   problem. `cubed_sphere` also fails geostrophic_adjustment (2.05 m/s)
   and phillips (eta_growth 26.4).
