# Preregistration — VORTEX_SMT round 33 (lane round 245): SMT-3 100-day score

Frozen before parsing or scoring the admitted SMT-3 daily trajectory. Base:
`cb2a145b795c` (round 244 / VORTEX_SMT round 32). Scientific evidence belongs
under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round245/`; the immutable
NEMO trajectory is the round-224 `smt3vec100d` record.

## Question and existing instruments

Operator note CI requires the deferred one-rung-per-round 100-day comparison
and movie sequence. SMT-1 and SMT-2 are complete; this round advances only to
the explicit `VORTEX_SMT3_VEC-zps` card. No model, card, option, coefficient,
timestep, run length, scoring definition, or certified trajectory changes.

Pre-implementation search found the shared round-210 scorer and renderer, the
round-224 self-describing admission report, the round-237 certified post-
landing SMT-3 registry, and the round-244 fail-closed score-gate pattern. They
already define the production-JIT fp64/libm run, wet masks, daily restart
reader, day-1..100 score, eight checkpoint table, fixed movie scales, and
plants. This round extends those instruments; it does not create another
trajectory harness.

The inventory found exactly 100 files named at steps 30, 60, ..., 3000 and an
admission report whose top-level status is `ADMITTED`, whose reference restart
is byte-identical, and whose three planted reports are `REFUSED`. This is an
availability check only; no trajectory field or score was read before this
preregistration.

The compiled executing restart program schedules frequency-based output at
`VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:104-119` and writes
the scored `sshn/un/vn/tn/sn` state at
`VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:176-180`.

## Frozen predictions and falsifiers

1. **R33-P1 — admission.** The round-224 record contains exactly 100 daily
   restarts at steps 30..3000; every required scored field is finite; the main
   admission is `ADMITTED`; the header, field-name, and truncation plants are
   `REFUSED`. Any mismatch refuses all scores.
2. **R33-P2 — current short-run calibration.** Before accepting a long-run
   number, the record's first ten entries reproduce the post-landing round-237
   certified SMT-3 registry exactly. The stale pre-landing round-226 registry
   is not an acceptable reference. Any mismatch refuses the comparison.
3. **R33-P3 — inherited magnitude prediction.** The current-tree day-100 wet
   three-dimensional T RMS is below the pre-landing round-226 value
   `6.813785267886451e-04 K` but above SMT-2's current
   `8.1037591477894766e-06 K`. Either inequality failing is retained as a
   refutation; neither bound is revised after the run.
4. **R33-P4 — registered checkpoints.** Days 1/2/5/10/20/30/60/100 are
   registered for T/u/v/ssh RMS and maxima and compared with SMT-2 through the
   same scorer. The curve does not establish a causal owner.
5. **R33-P5 — movie.** The shared renderer produces 100 frames plus the
   day-1/30/60/100 montage with SMT-3's own bathymetry, a fixed day-100
   SSH-difference scale, and a fixed day-1 velocity-arrow scale. Missing input
   or a zero/non-finite scale refuses rendering.
6. **R33-P6 — disposition.** This is measurement only. No physics, card,
   configuration, carried state, or certified short-run registry changes.
   SMT-4 remains the next one-rung comparison.

## Execution and controls

The existing round-210 scorer runs the production closure on CPU in fp64/libm
and writes daily legoESM snapshots plus the score JSON and curve. The existing
renderer consumes those exact snapshots and NEMO restarts. A round-245 gate
checks admission, daily count and cadence, current short-run reproduction,
finiteness, checkpoint completeness, both magnitude inequalities, and visual
artefacts. Its missing-checkpoint and both bound-violation plants must print
`STATUS PLANT-FIRED` and exit nonzero.

No hidden option, threshold, interpolation, stabiliser, or alternate record is
introduced. **UNASKED list: EMPTY.**
