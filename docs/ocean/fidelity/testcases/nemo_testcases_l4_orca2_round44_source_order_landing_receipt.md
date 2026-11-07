# ORCA2 round 44 — stage-1 EMP/runoff source-order landing

Date: 2026-09-27  
Card: `orca2_vector_een_c2`  
Claim label: **given NEMO's entry** (Decision 52)

## Verdict

**LANDED.**  Preserving NEMO's two source statements as two ordered additions
closes the stage-1 after-SBC boundary for both tracers: temperature moves from
2,514 / 228,641 unequal cells to 0, and salinity from 2,418 / 228,641 to 0.
The literal record replay is also 0 / 228,641 for both tracers.

The exact source boundary exposes the next statement rather than closing the
whole stage.  The following stage-1 QCO/RK combination differs on 57,160
temperature cells (maximum `7.105427357601002e-15 K`) and 57,180 salinity
cells (maximum `1.4210854715202004e-14`).  Under oracle-fidelity Rule 12 this
is a valid exact-operator landing and a downstream handoff, not a claim that
the stage or the whole card is bit-identical.

No configuration, selector, carried-state rule, sea-ice field, score domain,
scientific threshold, or stabiliser changed.  The card's six sea-ice
`unmeasured_features` entries remain exactly frozen.

## Preregistered predictions

| prediction | outcome |
|---|---|
| R44-P1 landed round-43 instrument reproduces | **CONFIRMED**: after-advection T/S are exact; before the landing, after-SBC is 2,514 T and 2,418 S unequal with maxima `4.235164736271502e-22` and `1.6940658945086007e-21` |
| R44-P2 record, execution policy, scorer, and plant bind | **CONFIRMED** |
| R44-P3 literal NEMO EMP-then-runoff replay closes after-SBC | **CONFIRMED**: 0 / 228,641 for T and S |
| R44-P4 the stored density reciprocal alone owns the production residual | **REFUTED**: stored-reciprocal multiplication still leaves 2,214 T and 2,418 S cells unequal; ordered accumulation is also required |
| R44-P5 the source fix closes the complete stage-1 T/S row | **REFUTED**: the downstream QCO/RK combination leaves 57,160 T and 57,180 S cells unequal |

The two failed predictions remain frozen in the preregistration.  Neither was
reworded after measurement.

## Statement walk

All rows in this section are **given NEMO's entry**, scored on the admitted
rank-0 support-safe wet interior with CPU fp64 and scalar libm.

| ordered boundary or discriminator | T unequal / scored | S unequal / scored | maximum T / S |
|---|---:|---:|---:|
| after centered advection | 0 / 228,641 | 0 / 228,641 | 0 / 0 |
| production before round 44: after SBC | 2,514 / 228,641 | 2,418 / 228,641 | `4.235164736271502e-22` / `1.6940658945086007e-21` |
| divide-by-`rho0` replay | 2,484 / 228,641 | 2,768 / 228,641 | `2.117582368135751e-22` / `1.6940658945086007e-21` |
| stored-`r1_rho0` replay | 2,214 / 228,641 | 2,418 / 228,641 | `2.117582368135751e-22` / `1.6940658945086007e-21` |
| literal EMP then runoff replay | 0 / 228,641 | 0 / 228,641 | 0 / 0 |
| landed production: after SBC | 0 / 228,641 | 0 / 228,641 | 0 / 0 |
| landed production: stage-1 output | 57,160 / 228,641 | 57,180 / 228,641 | `7.105427357601002e-15` / `1.4210854715202004e-14` |

The narrow reciprocal-only hypothesis is therefore false.  NEMO first stores
the thickness-scaled density reciprocal and applies EMP to the existing Krhs;
its later runoff statement adds to that already-rounded Krhs.  Combining EMP
and runoff into one source array changes the floating-point association.  The
landing keeps the two terms separate until the Krhs accumulation.  A planted
`1e16` cancellation test distinguishes the ordered result (`1`) from the
regrouped result (`0`).

QNS and SFX are structurally inactive at stage 1; they are not described as
numerical zeros.  The implementation retains the no-runoff fast path without
adding a zero array, preserving GYRE signed zeros.

## Ten-step ORCA2 ladder

These results are also **given NEMO's entry**.  The complete kt=1..10 ladder
contains 200 rows and passes its landing policy.

- No AT-BAR row leaves the bar.
- The first whole-card non-bit checkpoint is unchanged: kt=1 stage-1
  temperature, 233,341 / 399,600 unequal, maximum
  `0.0014770192519700243 K`.
- Four row documents move, all only in the final bits of `mean_abs_over_unequal`:
  kt=4 stage-1/2/3 salinity and kt=5 stage-1 salinity.  Their unequal counts,
  first unequal indices, and maxima are unchanged; by maximum, all four are
  `same_max`, with zero moves toward and zero moves away.
- The source-order landing is therefore an internal exact-boundary repair; it
  does not move the card's first whole-trajectory divergence.

## Shared-model GYRE gate

The package change executes through shared tracer code, so both mandatory GYRE
comparisons were run against the certified pre-round package state.

- Ten-step comparison: 70 / 70 certified rows accepted, largest oracle
  residual worsening 0 ULP.
- `ladder.residuals.npz`: all 210 arrays are `np.array_equal`.
- Thirty-day member: all 30 daily snapshots are byte-identical; day-30 SHA-256
  is `a66143733bcc9e4efaa22fe2b3a831629e4d57ffd701ddd827d5553c3e0b007a`.

Thus GYRE is byte-identical at the ladder and snapshot levels.  This is not a
GYRE physics landing.

## Controls, tests, and review

- Source scorer plant: `REFUSE: planted source-order tracer cell rejected
  through scorer`.
- Outcome plant: `REFUSE: planted GYRE daily snapshot digest rejected`.
- Focused statement-order tests: 2 passed.  An additional full 32-test tracer
  module attempt reached 18 passes and then stalled without a timeout report;
  it was interrupted and is not called green.
- Push gate: 127 passed.
- Card gate: 170 passed (160-test DINO/tank/lock/overflow batch plus the
  10-test round-34 tank batch).
- The one required `tests/ocean/fidelity -n 12` attempt reached 97% with two
  red markers, then repeated the documented xdist-controller stall without a
  failure summary.  It was interrupted and is not called green.  The five
  previously documented red node IDs were rerun in one serial battery and
  remain red with unchanged signatures: GYRE round-129 stale certification,
  round-51 private trace registry, SI3 scalar-math provenance, worktree-stamp
  ratchet, and the `hires_lane_surface` case-board ratchet.
- Default receipt citation gate: 274 citations, zero failures, zero unmapped
  citations, and a clean map audit after the required SequenceMatcher
  re-anchor of every affected model citation.
- Separate read-only `codex exec` review: **independent review unavailable
  in-sandbox** (`failed to initialize in-process app-server client: Read-only
  file system`).

## Evidence

Durable artifacts are under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round44/`.
The decisive files are `source_order_base.json`, `source_order_tip.json`,
`outcome.json`, `orca2_ladder_tip.json`, `gyre_offline_compare.json`,
`gyre_tip_ladder.residuals.npz`, `source_order_plant.log`,
`outcome_plant.log`, `push_gate.log`, `card_gates_160.log`,
`card_gates_tank34.log`, `focused_tracer_isolated.log`,
`fidelity_full.log`, `full_battery_known_reds.log`, and `codex_review.log`.

## OPEN

1. Continue at the first downstream non-bit statement: replay the compiled
   stage-1 QCO/RK combination from the now-exact Krhs, preserving its multiply,
   add, multiply, and divide association.  The registered discriminator is
   57,160 T and 57,180 S cells at one-ULP scale.
2. The first whole-card non-bit checkpoint remains kt=1 stage-1 temperature at
   `0.0014770192519700243 K`; the statement within the full-card production
   path remains unattributed beyond the record-backed internal walk.
3. The round-20 slow-forcing owner and Decision-52 **independent** initial
   T/S/SSH transcription and year comparison remain open.  No independent
   result is mixed into the twin tables above.
4. Sea ice remains out of scope and exactly the card's frozen six-item
   `unmeasured_features` tuple.

## Compiled-source citations

The executing nonlinear-free-surface stage-1 branch stores its reciprocal and
then adds EMP to Krhs; after the stage switch, the runoff block stores its own
depth reciprocal and adds tracer runoff to that rounded Krhs at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/trasbc.f90:275-328`.
This is the exact two-statement order landed in round 44.

The newly exposed downstream statement is NEMO's stage-1/2 QCO tracer update:
it combines the Kbb tracer with the thickness-weighted Krhs and divides by the
Kaa thickness ratio at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:670-681`.
