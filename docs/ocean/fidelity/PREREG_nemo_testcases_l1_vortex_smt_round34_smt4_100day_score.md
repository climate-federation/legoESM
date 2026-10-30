# Preregistration — VORTEX_SMT round 34 (lane round 246): SMT-4 100-day score

Frozen before parsing or scoring the admitted SMT-4 daily trajectory. Base:
`e2a05d99a` (round 245 / VORTEX_SMT round 33). Scientific evidence belongs
under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round246/`; the immutable
NEMO trajectory is the round-237 `smt4vec100d` record.

## Question and existing instruments

Operator note CI requires the final deferred one-rung-per-round 100-day
comparison and movie. This round advances only to the explicit
`VORTEX_SMT4_VEC-zps` card, then stops for the operator's PR decision. No
model, card, option, coefficient, timestep, run length, scoring definition, or
certified trajectory changes.

Pre-implementation search found the shared round-210 scorer and renderer, the
round-237 self-describing admission report, the round-238 certified SMT-4
registry and prior current-program trajectory, and the round-245 fail-closed
score-gate pattern. The current production implementation is byte-identical to
round 238 under `packages/` and `src/`. This round extends the existing
instruments; it does not create another trajectory harness.

The availability-only inventory found exactly 100 files named at steps 30,
60, ..., 3000 and an admission report whose top-level status is `ADMITTED`,
whose reference restart is byte-identical, and whose three planted reports are
`REFUSED`. No trajectory field or score was read before freezing this file.

The compiled executing restart program schedules frequency-based output at
`VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:104-119` and writes
the scored `sshn/un/vn/tn/sn` state at
`VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:176-180`.

## Frozen predictions and falsifiers

1. **R34-P1 — admission.** The round-237 record contains exactly 100 daily
   restarts at steps 30..3000; every required scored field is finite; the main
   admission is `ADMITTED`; the header, field-name, and truncation plants are
   `REFUSED`. Any mismatch refuses all scores.
2. **R34-P2 — current short-run calibration.** Before accepting a long-run
   number, the record's first ten entries reproduce the round-238 certified
   SMT-4 registry exactly. Any mismatch refuses the comparison.
3. **R34-P3 — deterministic reproduction.** Because production model code is
   unchanged since round 238, day-100 wet three-dimensional T RMS reproduces
   `2.552708052e-04 K` to the precision printed in that receipt and remains
   above the current SMT-3 value `1.7729713625071864e-04 K`. Either condition
   failing is retained as a refutation; neither reference is revised after the
   run.
4. **R34-P4 — registered checkpoints.** Days 1/2/5/10/20/30/60/100 are
   registered for T/u/v/ssh RMS and maxima and compared with SMT-3 through the
   same scorer. The curve does not establish a causal owner.
5. **R34-P5 — movie.** The shared renderer produces 100 frames plus the
   day-1/30/60/100 montage with SMT-4's own bathymetry, a fixed day-100
   SSH-difference scale, and a fixed day-1 velocity-arrow scale. Missing input
   or a zero/non-finite scale refuses rendering.
6. **R34-P6 — disposition.** This is measurement only. No physics, card,
   configuration, carried state, or certified short-run registry changes.
   After the score, the receipt compares the four mini-ladder rungs at day 100
   and stops with `DECISION_NEEDED` for the operator's PR work or any new rung.

## Execution and controls

The existing round-210 scorer runs the production closure on CPU in fp64/libm
and writes daily legoESM snapshots plus the score JSON and curve. The existing
renderer consumes those exact snapshots and NEMO restarts. A round-246 gate
checks admission, daily count and cadence, current short-run reproduction,
finiteness, checkpoint completeness, deterministic reproduction, the SMT-3
ordering, and visual artefacts. Its missing-checkpoint, reproduction-bound,
and SMT-3-order plants must print `STATUS PLANT-FIRED` and exit nonzero.

No hidden option, threshold, interpolation, stabiliser, or alternate record is
introduced. **UNASKED list: EMPTY.**
