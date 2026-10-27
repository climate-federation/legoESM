# Preregistration — VORTEX_SMT round 29 (lane round 241): SMT-1 100-day comparison

Frozen before acquiring or scoring a 100-day SMT-1 trajectory. Base:
`9cfc4bd0b` (round 240 / VORTEX_SMT round 28). Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round241/`.

## Question and existing instrument

Operator note CI orders the deferred one-rung-per-round 100-day comparison and
movie program, starting with SMT-1. The card is the already-certified
`VORTEX_SMT1_VEC-zps`; no model, card, option, coefficient, timestep, run
length, scoring definition, or accepted trajectory changes.

Pre-implementation search found the shared round-210 100-day scorer and movie
renderer plus the common VORTEX acquisition driver. They already define the
production JIT step, fp64/libm precision, wet masks, daily field reader,
day-1..100 score, eight checkpoint table, fixed movie scales, and the
`smt1vec100d` acquisition variant. This round extends those tools; it does not
build a second harness.

The daily oracle record does not exist at the base commit. The admitted SMT-1
kt=1..10 record exists, and the common driver has a hash-checked reuse arm for
its certified plain/instrumented binaries. Its 100-day deck keeps NEMO's
shipped `nn_itend=3000` and `rn_Dt=2880 s` while setting `nn_stock=30` for
daily restarts. The compiled restart writer schedules frequency-based dumps at
`VORTEX_SMT1_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:104-119` and writes
`sshn/un/vn/tn/sn` at
`VORTEX_SMT1_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:176-180`.

## Frozen predictions and falsifiers

* **R29-P1 — acquisition/admission.** The operator-run wrapper reuses only the
  two binaries whose hashes equal the admitted SMT-1 kt=1..10 manifest, creates
  exactly 100 daily restarts at steps 30..3000 in the round-241 evidence root,
  and passes the common driver's self-describing record/admission checks. A
  hash mismatch, missing/extra day, malformed required field, non-finite value,
  or nonzero NEMO exit refuses the record. Until that succeeds the round is
  `STOPPED_FOR_RECORD`, not a measurement.
* **R29-P2 — certified short-run calibration.** Before scoring day 1, the
  record's first ten step-entry frames reproduce the currently certified
  SMT-1 50-row registry exactly. Any mismatch refuses every 100-day number.
* **R29-P3 — bounded 100-day fidelity.** All T/u/v/ssh RMS and maximum rows are
  finite on days 1..100. The frozen magnitude prediction is day-100 T RMS below
  `1e-3 K`; a larger or unbounded result is retained as a refutation, never
  clipped or omitted.
* **R29-P4 — registered checkpoints.** The receipt records T/u/v/ssh RMS and
  maxima at days 1/2/5/10/20/30/60/100 and compares SMT-1 against the already
  measured SMT-0 vector curve using the same scorer. No causal owner is inferred
  from the curve.
* **R29-P5 — movie.** The existing renderer produces 100 frames plus the
  day-1/30/60/100 montage with the SMT-1 card's own bathymetry on all panels,
  a fixed day-100 SSH-difference colour scale, and a fixed day-1 velocity-arrow
  scale. Missing snapshots or a zero/non-finite scale refuses rendering.
* **R29-P6 — disposition.** This is measurement only. No physics, card,
  configuration, carried state, or certified registry lands or moves. The next
  round proceeds to SMT-2 only after the SMT-1 acquisition, score, and movie
  are complete and controlled.

## No hidden choices

The 100-day length, 30-step daily cadence, checkpoint days, fields, masks, and
movie form are inherited from the user-ordered round-210/217 program. No new
scientific choice is made. A missing daily record triggers an acquisition
request; it is not replaced by interpolation or by the ten-step record.
