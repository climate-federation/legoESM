# Preregistration: NEMO-testcases L2 GYRE round 111 residual ranking and FCT accumulation split

Date: 2026-09-18. Frozen at incoming lane tip
`3081ecacc81f6d07b342f91ba8fbbe32efebdab2`, before any round-111
measurement or numerical edit. Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round111/`.

Decision 43 remains the landing rule. The immutable before arm is the landed
round-110 trajectory at
`phase3/round110/candidate/{ladder.json,day_gap.json}`. This round first
re-ranks that residual, then reopens only the same-stage stage-3
FCT/advection-content discriminator named in round 110. It does not resume the
Decision-41 one-ULP stage walk and does not replay the round-70 oracle-output
substitution as an implementation.

## Immutable rows and magnitude target

| row | immutable round-110 value |
|---|---:|
| kt2 T max abs | `1.4210854715202004e-14` K |
| kt2 S max abs | `2.1316282072803006e-14` psu |
| kt2 U max abs | `2.7377110452773967e-12` m/s |
| kt2 V max abs | `3.2849219221489645e-12` m/s |
| kt3 T max abs | `8.600419718618468e-7` K |
| kt3 S max abs | `6.979441735666114e-8` psu |
| day-30 T RMS | `6.890484901489568e-5` K |

The landed run first creates a material temperature gap after step 2: its
step-3 entry has `8.80218929304543e-8 K` RMS. At day 30, 54.0502% of squared
temperature error is in 100--1000 m, 54.9790% in the west third, and 83.9340%
in the wind band; the absolute peak is at level `k=7`, 91.420 m and 26.9779 N.
The daily series has a non-smooth spike at day 23 (`5.2839e-4 K` RMS) before
falling to the immutable day-30 value. These are calibration facts to
reproduce, not new measurements.

## Compiled statement and record boundary

The compiled GYRE stage clears tracer `Krhs` and calls advection before the
other stage-3 sources
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:827-868`). Its
active FCT routine first calls `fct_up1_2stp`, which writes the upstream
concentration tendency into `Krhs`, and the record snapshots that boundary
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:163-173`). The
two-step routine constructs the first upstream fluxes, midpoint tracer,
averaged upstream fluxes, and then adds their divergence divided by
`e3t(Kmm)` to `Krhs`
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:503-540`,
`:570-611`). After `nonosc` limits the anti-diffusive face fluxes, the caller
forms a second divergence and **adds** its separately divided tendency to the
already-written upstream `Krhs`
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:316-329`). ZDF
later consumes the accumulated result in
`e3t(Kbb)*T(Kbb) + p2dt*e3t(Kmm)*Krhs`
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/trazdf.f90:547-565`).

The admitted round-64 stream records both `adv_up1_T/S` (the first writer) and
`after_adv_T/S` (the second writer), as well as the later source boundaries
and final content. legoESM instead reconstructs full limited face fluxes,
differences the combined flux, and carries an advection content directly into
the stage helper. Those forms are algebraically equivalent but do not expose
NEMO's two concentration-tendency writes. The registered diagnostic extends
the existing round-67/round-66 production capture; it does not create a
second stage harness. It will expose the live upstream and anti-diffusive
contributions under the full production step, in eager and JIT modes.

## Frozen predictions and falsifiers

**P1 -- residual calibration and ranking.** The committed owners self-check
will pass. A 60-step replay will reproduce the rounding-scale step-2 entry and
the `8.80218929304543e-8 K` step-3-entry temperature RMS. Days 1--30 and the
day-30 decomposition will reproduce the immutable round-110 JSON, including
day-30 T RMS `6.890484901489568e-5 K`, the 100--1000 m and west-third leading
cuts, and peak `k=7`. REFUTED by any changed immutable value, birth step,
leading cut, peak level, or a failing self-check; that stops the source edit.

**P2 -- day-23 localization.** A new decomposition of the already-recorded
round-110 day-23 state will show the spike concentrated in the west third and
in one of the two upper kilometre bands (0--100 m or 100--1000 m), with at
least 50% of squared temperature error in each leading cut and the absolute
peak no deeper than `k=11`. REFUTED if any threshold fails. This localization
is descriptive and cannot by itself qualify a candidate.

**P3 -- production FCT split discriminator.** The untouched production-JIT
arm will reproduce round 110's native local content T/S maxima
`5.743498263655056e-5` / `7.387909136014059e-6` and kt3 T/S maxima
`8.600420500215478e-7` / `6.979443156751586e-8`, subject only to the already
documented ladder-versus-local final rounding. The live `adv_up1` boundary is
predicted non-bit against the admitted NEMO row, and the live final
`after_adv` boundary is predicted non-bit with a temperature maximum of order
`1e-11 K/s`. REFUTED if `adv_up1` is bit-exact, if the final row is bit-exact,
or if either boundary is not observed through the production step. The exact
counts and maxima are measurements, not fitted predictions.

**P4 -- eligible source candidate.** The only registered numerical candidate
is NEMO's two-write accumulation: retain the upstream tendency as a distinct
materialized result, form the limited anti-diffusive tendency separately, add
them in the compiled order, and only then accumulate SBC/QSR/LDF before ZDF.
It is eligible for a trajectory run only if, from NEMO's recorded stage entry,
both the upstream boundary and complete `after_adv` boundary are bit-exact in
the full production JIT closure and in production eager mode. The production
plant changes one wet upstream/anti operand by one ULP, must change a scored
boundary, print `STATUS PLANT-FIRED`, and exit nonzero. Any unequal cell,
eager/JIT disagreement, inert plant, or need to infer an unrecorded operand
REFUTES eligibility. In particular, deriving an anti term by subtracting the
two oracle outputs is post-hoc and is not accepted as an input proof.

If P4 closes, the local kt3 prediction is T at or below
`6.0e-8 K` and S at or below `7.0e-9`; the preserved oracle-output endpoint
`5.760972143775689e-8 K` is a ceiling discriminator, not an implementation.
If P4 does not close because the first non-bit writer lacks recorded operands,
the round stops for a new write-only NEMO record under a new target name; no
physics is landed.

**P5 -- Decision-43 trajectory.** Reached only after P4. The candidate must
strictly reduce day-30 T RMS below `6.890484901489568e-5 K`, keep the first
over-bar boundary no earlier than kt2, keep every kt1 AT-BAR row at bar, and
register every moved row including regressions. The numerical forecast is a
day-30 T RMS in `1e-6`--`6e-5 K`; outside that band the forecast is REFUTED,
while equality or worsening rejects the landing. LOCK_EXCHANGE, OVERFLOW and
DINO are resolved from their cards and measured if they execute the shared
statement. ORCA2 remains UNMEASURED-WITH-SPEC: resolve its compiled tracer
integrator/FCT selection, record both FCT writes and their operands at
kt=1--10, replay them under fp64 JIT, and compare the next consumed T/S state.

## Measurement order and stopping rules

1. Commit this preregistration before creating round-111 evidence.
2. Run owners `--self-check`, `--step-gap 60`, `--day-gap` for days 1--30,
   and `--decompose` at days 30 and 23 against the immutable candidate run.
3. Extend the existing production capture with the two FCT writer boundaries;
   run untouched JIT/eager arms and a production-JIT one-ULP plant.
4. If and only if both recorded boundaries become bit-exact under the literal
   compiled association, run the full kt1--10 ladder, days 1--30 and the
   shared-card gates. Otherwise keep the diagnostic, identify the first
   non-bit compiled writer, and prepare only the missing passive record.
5. Run a separate read-only Codex refutation pass; run the citation gate and
   shifted-citation plant; run focused tests and the complete ocean fidelity
   and unit trees with twelve workers, quoting every terminal summary and
   diffing failing IDs against the frozen pre-existing list.

## Scope

CPU only, fp64, `JAX_ENABLE_X64=1`. No NEMO source is modified; neither
`makenemo` nor `mpirun` is run. No configuration/default, coefficient,
stabilizer, carried state, restart schema, year harness, reconciliation gate,
freshwater pair, #1484 guard, or immutable artifact is changed. No oracle
output may enter the independent implementation. A missing operand produces
`STOPPED_FOR_RECORD`, never an inferred source-exact claim.
