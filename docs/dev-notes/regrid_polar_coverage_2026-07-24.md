# Conservative-regrid polar coverage: fracarea + polar-fill weights

**Date:** 2026-07-24 (updated 2026-07-25) · **Status:** shipped — the polar gap is
fixed in the WEIGHTS (`fracarea` + `polar_fill` at the forcing callers); the
coverage check is strict `sum == 1` again and on at all three callers

## The invariant

A conservative lat-lon remap normalises each destination cell by its **own**
area (`weights = overlap_area / dst_cell_area`, no renormalisation by the
weight sum), so a destination cell the source only partially covers gets
`sum(weights) < 1` and the field comes back **reduced**.
`compute_overlap_weights(..., require_full_coverage=True)` raises on that.

The catch: `sum(weights) == 1` is only *attainable* when the source truly spans
the destination. **Every real forcing dataset stops short of the pole**, so its
outermost destination latitude row legitimately under-covers:

Quote the **cell centre** and the **inferred edge** separately — they are not the
same number, and the deficit is set by the edge:

| source | outermost cell centre | inferred outer edge |
|---|---|---|
| CORE-II NYF (94 rows) | ±88.542° | ≈±89.5° |
| JRA55-do TL319 (320 rows) | ±89.570° | ≈±89.85° |

The edges are approximate because `_edges_from_centers_deg` /
`_grid_edges_from_centers` infer them as `c[-1] + (c[1]-c[0])/2` — the *first*
spacing, which on a Gaussian grid is not the local one — so the exact value
depends on the real latitude array (not captured from the run). What is robust,
and all the argument needs: **both land short of ±90**, so `np.clip(..., ±π/2)`
is inert on real forcing. See "What was kept" for when the clamp does bite.

Model→model remaps (`coupler/grid_remap.py`) use pole-clamped v-faces, so for
**global** model grids `sum == 1` *is* attainable and the check stays strict. Note
this does not hold for every grid that path accepts: a regional Mercator/DINO frame
also satisfies `_is_regular_latlon`, its outer row against a global destination is
*not* fully coverable, and it is correctly rejected (that path takes neither
`fracarea` nor `polar_fill`, so nothing papers over it).

## What happened

The M8 hardening enabled `require_full_coverage=True` on the two
external-forcing remappers on the premise that "lat edges are clamped to the
pole, so any residual deficit is a bug". That premise is false for a
non-polar source. A real-forcing smoke test (Ginsburg job **9176233**, CORE-II
NYF) raised at **every** production target resolution:

18×36, 60×120 and 90×180 — always in lat row 0 or `n_lat-1`. The only worst-case
figure recorded from the run is **`|sum-1| = 6.6e-2` at 90×180**; the other two
were not captured.

The geometry reproduces that measurement exactly. CORE-II T62 is the 94-point
Gaussian grid, so the latitude array is fully determined —
`asin(leggauss(94))` gives outermost centre **−88.541950°** and, through
`_edges_from_centers_deg`, outer edge **−89.486342°** (gap 0.513658°). The
resulting deficit for the 2°-wide polar row of a 90×180 target is **6.5967e-2**,
i.e. the recorded 6.6e-2. (A cruder reconstruction assuming *uniform* 94 rows gives
6.4e-2 — close, but the Gaussian one is exact, so quote that.)

So it would have **aborted production OMIP preprocessing**. The deficit is
physical (no data poleward of the source's outer edge), not a seam/ghost defect
— it persists at `dd/ds ~ 1.1`, where the ghost-column logic is irrelevant.

JRA55-do was reverted on the same geometric argument rather than on a
measurement: no cached JRA55 was available at the time to reproduce it.

## What was kept

- The **pole clamp** on inferred lat edges (`np.clip(..., ±π/2)`). It prevents an
  extrapolated edge from spilling *past* the pole, where `sin` caps below 1 and
  fabricates a spurious polar gap. Note this is a **coarse-grid** guard, not a
  rounding guard, and it is inert on the real forcing grids above: it bites when
  uniform-spacing inference overshoots by degrees, as in the 8-row test fixture
  (outer centre 86° → inferred edge ≈98°). Pinned by
  `tests/unit/test_jra55_do.py::test_cache_polar_coverage_after_lat_clamp`
  (still RED without it, though now via the polar-gap budget: the unclamped 8-row
  fixture infers edges at ±98.3°, a 8.3° gap, which the treatments refuse to
  extrapolate across).
- The **`ceil(dd/ds)` ghost columns** in `omip2_applicator`. That fix is correct
  and closes the genuine 0/360 seam deficit (a single ghost left the
  seam-straddling column HALVED, and the 10-m pressure iteration then NaN'd).

## SHIPPED (final): fix the WEIGHTS, and the check goes back to being strict

The coverage check is `require_full_coverage` again and compares each destination
cell against a hard **1.0**, exactly as it originally did. That is possible because
the polar gap is now handled where it always belonged — in the weights:

| option | fixes | mechanism |
|---|---|---|
| `normalization='fracarea'` | a **partly** covered row | divide by the covered **latitude fraction** — never by the row sum → the area-weighted mean of the overlapping source |
| `polar_fill=True` | a row with **no** overlap | give it the source's outermost row, zonally resolved (zeroth-order poleward extrapolation), bounded by `polar_fill_max_gap_deg` (default 2°) |

Both are on at the two forcing callers; `coupler/grid_remap.py` takes neither.
With them, every destination cell sums to 1, a constant field survives on **every**
row at **any** resolution, and a longitude seam/ghost deficit still raises.

`normalization` is ESMF's vocabulary (`DSTAREA` vs `FRACAREA`), deliberately, so the
choice is recognisable rather than bespoke. Default is `dstarea` — unchanged,
strictly conservative behaviour for every existing caller.

### The trade-off, stated plainly

Both operations **break strict global conservation**: they fill the uncovered
fraction with real values instead of zero, so the remapped integral exceeds the
source's by exactly the uncovered cap. For a band symmetric about the equator
carrying a constant field that excess is `1/sin(band) − 1` (pinned by
`test_treatment_trades_strict_conservation_for_correct_magnitude`); for a band
`[a, b]` it is `2/(sin b − sin a) − 1`, and for a non-constant field only the sign
is guaranteed.

That is the right trade for **forcing**, where every channel is INTENSIVE (`tas`,
`huss`, `psl`, winds, and the radiative/precip flux **densities**) and the shortfall
is a DATA GAP rather than a region of genuine zero flux — a diluted air temperature
is simply wrong. It is the WRONG trade for the conservative **flux** direction,
which is why `grid_remap.py` keeps `dstarea` and no fill: the ESM energy and
freshwater budgets depend on that path conserving.

### What this deleted

Fixing the weights made the previous iteration's machinery unnecessary, so it is
gone: `lat_shortfall_floor`, `_MIN_LAT_FLOOR`, `_required_lat_coverage` and the
per-row floor guard. `_attainable_lat_coverage` **stays** — it is now load-bearing
for the safety property below: `fracarea` scales by `1 / lat_frac[j]`, NEVER by the
row sum. Those existed only to
decide *how much dilution to tolerate* — a question that stops being asked once the
row carries the correct value. Two consequences worth noting:

- The production 1° polar rows no longer ship at 0.736 coverage (`T_air ≈ 184 K`).
  They now carry the field's own value. The earlier note recorded that dilution as
  disclosed-but-blessed; it is now simply fixed.
- The fine-resolution abort is gone. A destination finer than the source's polar
  gap (CORE-II below ~0.51°, JRA55-do below ~0.15°) used to raise because its
  outermost rows had no data; `polar_fill` supplies them, so `--latlon-res 360x720`
  and finer now build.

### Iteration history — three designs were rejected before this one

Recorded because each looked obviously right, and the first two shipped:

1. **Relax the reference to attainable coverage.** A row matches its own attainable
   value by construction, so a regionally-limited source passed while emitting
   zeros — strictly worse than the bug being fixed.
2. **Bound the shortfall POSITIONALLY** ("only the outermost row may be short").
   Assumes the polar gap is narrower than one destination row; CORE-II's 0.514° gap
   spans 4 rows at 0.5°, so it hard-aborted legitimate targets.
3. **Bound it GEOMETRICALLY.** Algebraically *identical* to (2): a contiguous source
   band's outer edge cuts exactly ONE destination row and every row beyond is
   exactly zero-covered. Its own regression test failed, which is how it was caught.

Then a floor (`lat_shortfall_floor`), which worked but only ever chose how much
wrongness to accept. The lesson the sequence teaches: **a check cannot repair a
representation defect.** Every attempt to express "this row is allowed to be wrong"
either admitted something worse or aborted something valid. Fixing the weights
removed the question.

Two further defects caught in review of the weight fix itself:

- `fracarea` was **not** the no-op on a fully covered remap that its docstring
  claimed: a full cell can sum to `1 ± 1 ulp`, and `1/(1+ε) ≠ 1` perturbed an
  already-correct weight. Now only cells deviating by more than `_COVERAGE_TOL` are
  rescaled, which makes the claim true. Caught only because the test asserted
  `assert_array_equal` rather than `allclose`.
- `polar_fill` must reproduce the source's edge row **column by column**, not its
  zonal mean, or it would smear away polar longitude structure
  (`test_polar_fill_gives_uncovered_rows_the_outermost_source_row_zonally`).

## Related

- `packages/core/legoesm/grids/conservative_regrid.py` — the helper
- `packages/ocean/legoesm/ocean/coupler/omip2_applicator.py` — OMIP2 forcing
- `packages/tools/legoesm/forcing/jra55_do.py` — JRA55-do cache build
- `packages/coupler/legoesm/coupler/grid_remap.py` — model→model remap; keeps
  `dstarea` and no `polar_fill` (the conservative flux direction), strict check

### The safety property, stated once

`fracarea` scales a short row by `1 / lat_frac[j]` — the **latitude** factor —
never by the row sum. The weights are separable (`row_sum[j,i] = lat_frac[j] ·
lon_frac[i]` exactly), so dividing by the row sum would normalise a **longitude**
deficit away too, silently repairing the seam/ghost gap this module's coverage
check exists to catch. That is not theoretical: it shipped in the first draft of
this change, and measured, the historical single-ghost bug (`lon_frac = 0.625`,
the one that left a forcing column HALVED and NaN'd the 10-m pressure iteration)
came back at 1.000 and passed the strict check, while two pre-existing regression
tests went green-but-vacuous.

The polar-gap budget (`max_polar_gap_deg`, default 2°) applies to **both**
treatments, not just `polar_fill`: `fracarea` spreads a partly covered row's data
over the part the source never reached, which is the same extrapolation in
miniature. Checked on the source geometry alone, so `polar_fill=True` can never
pass where `polar_fill=False` fails. Real sources are far inside it (CORE-II 0.51°,
JRA55-do 0.15°); a ±10° regional band, or a latitude axis misread as radians
(±1.57°), is refused.

## CONFIRMED downstream: this was the ½° latlon polar blow-up

The zero-forcing regime was not hypothetical — it was crashing a real configuration,
and the crash had been worked around rather than diagnosed.

`scripts/cluster/omip_nemo/_diag_llh_polarcap.sbatch` records a half-degree
(`--latlon-res 360x720`) `latlon_bathy` run failing at step 3, `max|u|` going
`0.08 → 0.86 → 177` **at lat 89.75**, and the response was
`--mask-polar-cap-lat 89.0` — masking the offending cell out.

That latitude is exactly the row this note is about. At 360×720 the destination
rows are 0.5° and CORE-II's inferred outer edge is 89.4863°, so:

| dst row | centre | `lat_frac` |
|---|---|---|
| [89.5, 90] | **89.75** | **0.000000** |
| [89.0, 89.5] | 89.25 | 0.981540 |
| [88.5, 89.0] | 88.75 | 1.000000 |

The blow-up sat on the *only* row with zero coverage — which under `dstarea`
returns `0 × field`, i.e. `T_air = 0 K`, zero winds, zero radiation, against an
SST near 271 K. Its neighbour at 89.25 was 98% covered and behaved.

**Controlled A/B** (jobs 9188278 / 9188279), same tree, same command, one variable
— the forcing polar treatment — and `--mask-polar-cap-lat` REMOVED so the cell is
live:

| arm | polar row | outcome |
|---|---|---|
| A: `dstarea`, no `polar_fill`, check off (pre-fix) | zeros | **NaN at step 1**, `finite: False`, rc=1 |
| B: `fracarea` + `polar_fill` (shipped) | real values | **210 steps stable**, peak `max\|u\|` 1.089 m/s, SST 13.03 °C, SSS 33.86, `umax` never leaves the tropics (lat −15.3) |

Arm B is 70× past the step-3 failure point with no polar excursion, so
**`--mask-polar-cap-lat` is no longer required for this configuration.** Eight
launchers use `--latlon-res 360x720`; all were exposed.

Scope of the claim, deliberately narrow:
- **CONFIRMED** — the ½° polar NaN/blow-up at lat 89.75 was caused by the
  zero-coverage polar row, and is fixed.
- **NOT claimed** — the separate first-step HANG investigated in
  `_diag_llh_hang.sbatch` (`TotalCPU=0`, a GPU/sync deadlock) is a different
  symptom and was not tested here.
- **REFUTED** — an earlier guess of mine that this bore on the Arctic halocline /
  excess-ice-growth work. It cannot: `_conservative_regrid_to_latlon` is reached
  only for `latlon`/`latlon_regional`, while the tripole and MPAS runs use
  `_nn_interp_to_points`, untouched by any of this. At 1° the affected row is also
  only 0.26% of Arctic ice area.

### Second confirmation: the full OMIP `host` configuration, and what the HANG was not

The blow-up reproduces and clears in the full production-shaped configuration too,
not just the reduced polarcap one. Re-running the `host` variant of
`_diag_llh_hang.sbatch` (polar filter + geothermal + runoff + SSS restoring +
`--prognostic-sea-ice --prognostic-ice-dynamics free_drift`) at 360×720, one
variable — the forcing polar treatment — and no `--mask-polar-cap-lat`:

| arm | job | outcome |
|---|---|---|
| pre-fix (`dstarea`, no `polar_fill`, check off) | 9190293 | **NaN at step 3**, `finite: False`, rc=1 |
| shipped (`fracarea` + `polar_fill`) | 9190223 | **rc=0**, ran to completion, `max\|u\|` ≈ 1.05 m/s, `umax` in the tropics |

Step 3 is exactly where `_diag_llh_polarcap.sbatch` recorded the original failure.
Base drift between the two arms was checked and is inert (a docs commit and an
unrelated Levante submit script; no executed code differs).

**The HANG is a separate matter and remains unexplained.** `_diag_llh_hang.sbatch`
described a first-step *hang* (`TotalCPU=0`, a GPU/sync deadlock or stalled CUDA
alloc), and it did **not reproduce in any arm** — `min`, `pf` and `host` all
completed with `rc=0`, and even the pre-fix arm NaN'd rather than hung. So:

- **CONFIRMED** — the ½° NaN/blow-up at step 3, in both the reduced and the full
  `host` configuration, was the zero-coverage polar row; it is fixed.
- **NOT reproduced, NOT attributed** — the `TotalCPU=0` hang. Nothing here explains
  it; it may have been transient (node/driver) or fixed by something unrelated.
  Do not credit this change with it.

The original hypothesis in that script — that the polar-filter FFT or the host-loop
device-sync at 720 longitudes was the culprit — is not supported: both now run.

The three `--mask-polar-cap-lat 89.0` uses are all in untracked `_diag_llh_*`
probes, so there is nothing to revert in tree; the flag is simply no longer needed
for this failure.
