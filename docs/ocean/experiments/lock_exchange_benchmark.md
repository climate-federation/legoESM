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

Lock exchange is one case in a family. Run the family with
`sbatch --array=0-9 scripts/cluster/ocean_grid_benchmark_suite.sbatch`.
Results 2026-08-13 (jobs 26917739 + 26917853), on the re-centred
split-explicit barotropic averaging window:

| Case | latlon | mpas | fesom | tripole |
|---|---|---|---|---|
| rest_state_stratified_with_land | PASS | PASS | PASS | PASS |
| rest_state_uniform_with_land | PASS | PASS | PASS | PASS |
| rest_state_stratified_no_land | PASS | PASS | —¹ | —² |
| rest_state_uniform_no_land | PASS | PASS | —¹ | —² |
| geostrophic_adjustment | PASS | PASS | PASS | PASS |
| phillips_two_layer | PASS | PASS | PASS | PASS |
| inertia_gravity_wave | PASS³ | PASS³ | PASS³ | PASS³ |
| barotropic_wave | PASS | PASS | PASS | PASS |
| lock_exchange | PASS | PASS | PASS | PASS |
| inertia_gravity_wave_channel | —⁴ | —⁴ | —⁴ | —⁴ |
| lock_exchange (Petersen channel) | PASS⁵ | — | — | — |

Rest-state drifts are at machine precision on every arm (eta 0 to 1e-31,
T ≤ 1.5e-14, S ≤ 7.6e-15).

1. The FESOM mesh is built with the same 80° land threshold as the
   with-land arms (190 of 3140 nodes dry), and the setup exposes no 90°
   variant — so FESOM belongs to the WITH-LAND rows only.
2. The tripole basin is defined by the NEMO tmask, so a no-land tripole
   variant does not exist.
3. PASS here means only: not frozen, not exploding, not extinguished.
   This case's initial condition is an f-plane plane wave imposed on a
   sphere where the model integrates `f = 2Ω sin(lat)`, so it is not an
   eigenmode anywhere and no wave-speed gate is constructible on it. The
   case that DOES verify wave behaviour is the channel one below.
4. `inertia_gravity_wave_channel` runs on `latlon_channel` only — a
   Cartesian f-plane box, which is the one geometry where the Poincaré
   channel mode is exact. A second grid would need a PLANAR hexagonal
   mesh with TRiSK metrics and a constant Coriolis parameter; the repo
   has no such mesh (`create_beta_plane_cgrid_geometry` is lat-lon only,
   Voronoi meshes carry `f = 2Ω sin(lat)`, and `grids/plane.py` is a quad
   C-grid for CRM work). Putting the mode on a spherical patch would make
   it a non-eigenmode and repeat exactly the error in note 3.
5. Was the ONLY failing case in the suite until 2026-08-13: it finished
   with water at −3.00 °C from an initial range of exactly [5, 30]. See
   "The Petersen channel arm's advection scheme" below.

## Is the cross-grid agreement real, or is the instrument blunt?

`scripts/validate/ocean_grid_consistency.py` compares the arms against a
MEASURED tolerance (each arm against itself at 2× resolution). Every case
passed — with the tolerance 1.5× to 245× LARGER than the difference it was
judging, so the test could not fail.

`--refinement` asks the question that can: refine BOTH arms one step and
see whether the difference shrinks. `--refined-budget` measures each arm's
own discretisation error at the refined level too, so the difference has a
tolerance there as well. Without it every row is labelled UNBUDGETED TREND,
because a direction is not a convergence claim.

Measured 2026-08-13 (jobs 26917825/26917826/26918225/26918425; latlon
36x72→72x144 with the tolerance from 144x288, MPAS ico4→ico5 with the
tolerance from ico6).

A D/E ratio alone is not readable, because refining moves TWO things: how
far apart the arms are (D) and how well each arm knows its own answer (E).
Only one combination is diagnostic on its own — **E falling while D rises**
means each arm is converging and they are converging to DIFFERENT limits.
D/E improving because the tolerance loosened is not evidence of anything.

| Case | D/E coarse | D/E refined | D ratio | E ratio | what that combination means |
|---|---|---|---|---|---|
| barotropic_wave | 0.50 | 0.70 | 0.93 | 0.66 | converging together |
| geostrophic_adjustment | 0.74 | 0.57 | 0.85 | 1.11 | tolerance loosened faster than the difference shrank |
| inertia_gravity_wave | 0.31 | 0.41 | 0.62 | 0.46 | converging together |
| lock_exchange (front, km) | 0.56 | 0.24 | 1.50 | 3.58 | case de-settling — **not adjudicable here** |
| phillips_two_layer | 0.20 | **1.08** | 3.30 | 0.60 | **converging to DIFFERENT limits** |

**`phillips_two_layer` is the one case with the different-limits
signature**: each arm's own resolution sensitivity FALLS by 0.60× while
the two arms move 3.30× further apart, ending at 1.08× of their own
tolerance. Each dycore is settling down; they are settling on different
answers. That is an algorithmic inconsistency, and it was invisible at the
coarse resolution alone, where the difference was a fifth of the tolerance.

What the other rows do and do NOT say:

- `inertia_gravity_wave` and `barotropic_wave` converge together — both D
  and E fall. These two are cross-grid consistent as far as this test goes.
- `geostrophic_adjustment`'s D/E improves from 0.74 to 0.57, but its
  tolerance LOOSENED (1.11×) while the difference fell only to 0.85×. The
  improvement is partly the tolerance moving, so read it as "inside budget
  at both levels", not as a demonstration of convergence.
- `lock_exchange` is **unadjudicated, not passing**. Its front-displacement
  disagreement grows (1.0 → 1.5 km) and its D/E falls to 0.24 only because
  each arm's own front self-error grew 3.58× (1.88 → 6.73 km). A tolerance
  outrunning the thing it bounds is not agreement. This row's numbers are a
  scalar displacement in km and are NOT comparable with the field-norm
  numbers in the other rows.

The phillips result is a delta between two resolutions, not a trend, and
the case's own amplitude doubles between them, so it is reported as
"BUDGET EXCEEDED (1.08x)" rather than as a convergence or divergence
result. The competing reading — that an unstable case simply amplifies any
difference at its own growth rate — was tested with a log-linear fit of the
difference against time and is NOT settled either way (R² 0.43–0.76, and a
NEGATIVE slope: the difference appears early and does not grow through the
run). A poor fit refutes nothing in either direction.

Next measurement for phillips, in order: decompose D by field at its peak
time (SSH vs temperature vs barotropic velocity localises which term the
two discretisations represent differently), then re-run both arms from a
single high-resolution analytic initial condition regridded to each native
mesh, which controls the initial-condition projection that the
evolution-difference construction does not cancel on the native meshes.

## The Petersen channel arm's advection scheme

The 64 km × 4 km, 1 km-resolution channel is the only lock-exchange arm
that resolves its own front. It ran WENO5, chosen for sharpness, and
finished with water at −3.00 °C. WENO5 is essentially-non-oscillatory, not
monotonicity-preserving.

One variable, everything else in the registration fixed. Final T range
against the initial [5.00, 30.00], and the spurious-mixing number:

| scheme | T range [°C] | RPE_rel | |
|---|---|---|---|
| `weno5` | −3.00 … 31.77 | 6.536e-6 | FAIL |
| `tvd` | −0.40 … 30.00 | 9.832e-6 | FAIL |
| `superbee` | −2.99 … 30.00 | 8.889e-6 | FAIL |
| `ppm_fct` | 5.00 … 30.00 | 6.261e-6 | PASS |
| `fct2` | 5.00 … 30.00 | 6.852e-6 | PASS |

Every flux LIMITER undershoots; only the flux-CORRECTED schemes are
bounded — the same reason the global lat-lon and tripole arms already run
`LOCKEX_CGRID_TRACER_ADV`, and the channel arm now runs that same
constant. `tvd` and `superbee` produce no overshoot above 30.00 while
still undershooting: the limiter works within each directional sweep and
the multi-dimensional combination of monotone sweeps does not.

NOT a vertical-CFL artefact. With dz = 1 m and dt = 30 s, CFL_v reaches 1
at only 0.033 m/s, so an unstable vertical operator was the competing
explanation. Perturbation test, `weno5`, only dt changed: 30 s → −3.00,
15 s → −2.92, 7.5 s → −2.88 °C. A 4× smaller dt moves the undershoot by
4%, where a CFL mechanism predicts it falling roughly with dt.
