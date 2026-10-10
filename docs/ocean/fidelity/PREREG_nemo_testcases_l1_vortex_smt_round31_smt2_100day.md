# Preregistration — VORTEX_SMT round 31 (lane round 243): SMT-2 100-day comparison

Frozen before acquiring or scoring a 100-day SMT-2 trajectory. Base:
`cfe1ebf81` (round 242 / VORTEX_SMT round 30). Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round243/`.

## Question and existing instrument

Operator note CI orders the deferred one-rung-per-round 100-day comparison and
movie program. Round 242 completed SMT-1; this round advances only to the
already-certified `VORTEX_SMT2_VEC-zps` card. No model, card, option,
coefficient, timestep, run length, scoring definition, or accepted trajectory
changes.

Pre-implementation search found the shared round-210 100-day scorer and movie
renderer plus the common VORTEX acquisition driver. They already define the
production-JIT fp64/libm run, wet masks, daily restart reader, day-1..100
score, eight checkpoint table, fixed movie scales, and the `smt2vec100d`
acquisition variant. This round extends those tools; it does not build a
second harness.

The daily SMT-2 oracle record is absent. The admitted kt=1..10 record exists
under `vortex_smt/round10`, and the common driver has a hash-checked reuse arm
for its certified plain and instrumented binaries. Its 100-day deck keeps
NEMO's shipped `nn_itend=3000` and `rn_Dt=2880 s` while setting `nn_stock=30`
for daily restarts. The compiled executing restart program schedules
frequency-based output at
`VORTEX_SMT2_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:104-119` and writes
the scored `sshn/un/vn/tn/sn` state at
`VORTEX_SMT2_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:176-180`.

## Frozen predictions and falsifiers

* **R31-P1 — acquisition/admission.** The operator-run wrapper reuses only the
  two binaries whose hashes equal the admitted SMT-2 kt=1..10 manifest,
  creates exactly 100 daily restarts at steps 30..3000 in the round-243
  evidence root, and passes the common driver's self-describing admission
  checks. A hash mismatch, missing or extra day, malformed required field,
  non-finite value, or nonzero NEMO exit refuses the record. Until this
  succeeds the round is `STOPPED_FOR_RECORD`, not a measurement.
* **R31-P2 — certified short-run calibration.** Before accepting any 100-day
  number, the record's first ten step-entry frames reproduce the current
  round-237 certified SMT-2 50-row registry exactly. Any mismatch refuses all
  long-run scores.
* **R31-P3 — inherited magnitude prediction.** The still-open prediction from
  the SMT-2 landing receipt remains binding: day-100 T RMS is within 2x of the
  measured SMT-1 value (`4.3321114781972461e-05 K`), and NEMO's day-100
  maximum absolute U is lower than SMT-1's. A larger T RMS, a non-finite row,
  or a non-reduced NEMO U maximum is retained as a refutation. This round does
  not revise that prediction after seeing the record.
* **R31-P4 — registered checkpoints.** The continuation receipt records
  T/u/v/ssh RMS and maxima at days 1/2/5/10/20/30/60/100, and compares SMT-2
  with SMT-1 through the same scorer. No causal owner is inferred from the
  curve.
* **R31-P5 — movie.** The existing renderer produces 100 frames plus the
  day-1/30/60/100 montage with SMT-2's own bathymetry, a fixed day-100
  SSH-difference colour scale, and a fixed day-1 velocity-arrow scale.
  Missing snapshots or a zero/non-finite scale refuses rendering.
* **R31-P6 — disposition.** This is measurement only. No physics, card,
  configuration, carried state, or certified registry lands or moves. The
  next round proceeds to SMT-3 only after acquisition, scoring, and rendering
  complete.

## No hidden choices

The 100-day length, 30-step daily cadence, checkpoint days, fields, masks,
movie form, and SMT-2/SMT-1 comparison were fixed by Decisions 87/93 and the
round-223 OPEN section. No new scientific choice is made. A missing record
requests acquisition; it is never replaced by interpolation or by the
ten-step record.
