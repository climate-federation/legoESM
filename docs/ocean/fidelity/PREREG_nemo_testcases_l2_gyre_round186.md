# Preregistration — round 186, pre-day-180 process ranking

Committed before admitting or scoring the Round-185 pre-day-180 record.  The
operator's acquisition reached `STOP 0`, wrote all 1,080 expected process
frames, and produced byte-identical day-30 and day-180 restarts.  Its own
admission then refused because the generalized process-record gate still
required the Round-123 run's resolved `nn_itend=1440` instead of the requested
interval end `1080`.  Round 186 repairs that fail-closed check, admits the
existing record without rebuilding or rerunning NEMO, extends the existing
production trace over steps 1--1080, and ranks those compiled process rows by
projection onto the fixed Round-183 day-240 temperature-error field.

Evidence lives under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round186/`.  The acquired
record remains under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round185/oracle_pre180_process/`.
No model physics, configuration value, carried-state schema, default,
stabilizer, or NEMO source is authorized to change.

## Compiled process order

The compiled Round-185 writer records stage-3 temperature before the process
chain and after advection, surface boundary, shortwave, lateral diffusion, and
the implicit vertical solve in
`GYRE_OMIP_L2_P3_SM_R185PREPROC/BLD/ppsrc/nemo/stprk3_stg.f90`; the exact line
ranges are registered only after the compiled file is read.  Its field order
and 1,415,300-byte record layout are unchanged from the admitted Round-123
writer.  The existing `nemo_testcase_l2_gyre_year_owners.py` process trace and
scorer are extended; no parallel harness is created.

## Frozen predictions and falsifiers

1. The admission defect is only the resolved-card endpoint check.  Making the
   expected `nn_itend` an explicit consequence of `end_step` admits the
   existing Round-123 interval with `1440` and the Round-185 interval with
   `1080`.  A one-step-wrong resolved endpoint must fail.  Any other admission
   failure refutes this diagnosis and stops the trace.
2. The existing NEMO record contains exactly 1,080 contiguous frames totaling
   `1,528,524,000` bytes.  Both passive restarts retain their frozen SHA-256
   values, all frames chain exactly, every registered process row is active,
   and the stamp, truncation, surface-ULP, surface-effect, and trajectory-ULP
   plants each print `STATUS PLANT-FIRED` and exit nonzero.  Failure of any row
   refuses the record; NEMO is not rebuilt or rerun.
3. The extended legoESM trace executes the production JIT step from the same
   seed-zero rest state for steps 1--1080.  Its diagnostic observer changes
   zero carried-state bytes, its day-180 T/S/u/v/SSH snapshot is bit-identical
   to the immutable Round-183 arm, and its stamp, row-ULP, and production-effect
   plants fire.  A moved state byte or snapshot bit stops the ranking.
4. The earlier-interval budget preserves the initial-state row separately,
   telescopes all six compiled process rows plus rounding to the independently
   evaluated day-180 error array, and reconstructs it with maximum wet-cell
   residual no larger than `4e-15 K`.  Each row is then projected onto the
   fixed Round-183 day-240 endpoint, whose RMS must remain exactly
   `6.5861718814795174e-05 K`.  Closure failure or endpoint drift refuses the
   ranking.
5. Frozen magnitude prediction: shortwave is the largest physical row over
   steps 1--1080; surface boundary is its largest cancelling partner.  This is
   falsified if any other compiled physical row has a larger absolute signed
   day-240 projection.  The measured ordering wins and the failed prediction
   remains in the receipt.
6. Every physical row is partitioned into the existing ten-day blocks, with
   exact signed projection for every block.  The receipt reports both the
   earliest bitwise-nonzero block and the strongest block; it introduces no
   post-hoc magnitude threshold and does not relabel a projection as a source
   statement.
7. If the winning row's existing developed-state operand records match the
   landed trajectory, the first non-bit compiled statement is named and cited.
   Otherwise the round stops for the specific missing record rather than
   promoting a downstream row or using a stale pre-landing record.
8. A landing remains subject to the complete Decision-43/45/55/59 ladder,
   month, year, card-census, DINO, tank, generic-card, plant, citation, and
   review gates.  Without those gates no physics lands this round.

The final diff receives a separate read-only Codex review.  A `DO NOT SHIP`
verdict blocks a landing.  No configuration decision is expected.

## Frozen addendum: magnitude-bearing shortwave statement record

The completed ranking promotes shortwave penetration, but the compatible
`oracle_qsr_stage3_kt00000001.bin` record contains only the surface flux and
completed increment at the smooth first step.  It does not contain the live
developed-state optical operands or statement boundaries, so it cannot name a
first non-bit statement for the magnitude-bearing developed flow.

Before requesting a new measurement, freeze the next record as follows.
Record stage-3 `qsr_2BD` at step 1080, in the strongest measured ten-day
interval (days 170--180), from the unchanged admitted year card.  Store the
surface `qsr`, `r3t(Kmm)`, `gdepw_1d`, `e3t_3d`, `tmask`, `wmask`, the actual
post-minus-pre `Krhs` increment, and a post-call replay of the compiled
statements.  The replay is diagnostic only and runs after the production call.

Predictions and falsifiers:

1. The step-180 and step-1080 restarts remain byte-identical to the admitted
   uninstrumented record; either mismatch **REFUTES** passivity.
2. The post-call replay is bit-identical to the actual increment on every
   stored cell.  Any unequal cell **REFUTES** calibration and the record is not
   admitted.
3. A one-ULP mutation of the stored actual increment makes admission exit
   nonzero with `STATUS PLANT-FIRED`; a green mutation **REFUTES** control
   non-vacuity.
4. The record is evidence for a subsequent compiled-order statement walk, not
   a candidate and not a landing.  No internal shortwave statement is claimed
   in round 186 without this record.
