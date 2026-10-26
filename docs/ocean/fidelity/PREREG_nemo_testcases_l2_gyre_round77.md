# Preregistration: NEMO-testcases L2 GYRE round 77 acquisition admission repair

Date: 2026-09-12. Frozen after reading the operator's round-76 acquisition log,
the admitted source and candidate records, and the candidate's compiled source,
but before editing or exercising the replacement acquisition card or admission
checker.

## Observed stop and immutable scientific boundary

The operator's round-76 target built and ran successfully. Its dedicated record
gate printed `ROUND76_UAMID_RECORD_AT_BAR`: the U-midpoint stream is exactly
1,127,856 bytes and all 50 compiled midpoint associations replay bit-for-bit.
Thus the earlier full-domain size concern is **RETRACTED/RESOLVED** by the
already committed reduced-domain writer and reader; changing their valid field
count, extent, precision, or association would be an unsupported new arm.

The run stopped later at exit 2 in the inherited-record admission checker. Its
decisive result is `unregistered record magic 'NEMO_L2_BTADV_2'` for the
round-75 kt=2 transport-mean stream. The compiled candidate opens that inherited
stream, writes its six-integer header and fixed arrays at
`GYRE_OMIP_L2_P3_SM_R76UAMID4/BLD/ppsrc/nemo/dynspg_ts.f90:442-453`, then writes
the ten substep fields in their branch-resolved order at the same compiled
source's `:588-607`. The new reduced-domain midpoint stream is separately opened
and headed at `:454-463`, and it records `un_e`, `ub_e`, `ubb_e`, and `ua_e`
immediately after the compiled association at `:496-510`.

Round 73's first robust scientific boundary remains the kt=2 stage-1 `un_adv`
input: 580 of 580 wet U cells differ, with maximum absolute difference
`0.00012029895814569258`. No new scientific measurement or production change
is eligible in this acquisition-only round.

## Frozen repair and falsifiers

Extend only the shared acquisition admission checker with a strict comparator
for `NEMO_L2_BTADV_2`, composing the existing round-14 reader rather than
inventing another layout. Compare both headers, the scalar divisor, all 50
weights, both reciprocal metrics, all ten named fields at every substep, and
all four final pre/post-boundary fields. Scalars must be bit-identical; arrays
must be bit-identical on the explicit owned horizontal domain after removing
the two-cell halo printed by the run. Every raw difference outside that owned
projection must be counted and reported as halo, never silently ignored.

Prediction: replaying the already completed round-76 source/candidate pair with
the repaired checker passes all inherited consumed fields and the exact final
restart and mesh identity. A one-ULP plant must land in an owned cell of the
transport-mean record, make the checker fail, and exit nonzero. This prediction
is falsified if the repaired checker finds any owned difference, any header or
layout discrepancy, any other unregistered changed record, or if the plant does
not fire. A falsifier is recorded as **REFUTED**; no readiness path is emitted
until it is reconciled.

Prepare a replacement operator card with new target
`GYRE_OMIP_L2_P3_SM_R77UAMID5` and new evidence directory
`round77/oracle_uamid_kt2`, both absent at preregistration. It must clone
GYRE_PISCES, copy the unchanged R75 EXP00 and MY_SRC files one by one, preserve
the source namelist byte-for-byte, apply only the already validated WRITE-only
round-76 patch, and stamp the clean committed checkout. The writer and strict
reader must independently retain the corrected 32 by 22, four-field,
1,127,856-byte layout. Exact preprocessing with the R75 keys and includes plus
`gfortran -fsyntax-only` must exit zero with empty output. The layout,
resolved-row, header, truncation, replay-ULP, stamp, admission-consumed, and
admission-inventory plants must all exit nonzero.

## Rule 12 and exclusions

No numerical statement changes, so GYRE kt=1--10 and days 1--30 are
**UNREACHED / unchanged** against the recorded before arm. LOCK_EXCHANGE and
OVERFLOW execute no changed production statement. DINO remains **UNREACHED**
with explicit shared-midpoint and 96--98% per-row cancellation risk. ORCA2
remains **UNMEASURED WITH SPEC**: resolve its compiled external-mode card;
record every entry, coefficient, midpoint operand/result, metric transport,
reciprocal metric, accumulator exit, normalization, and boundary handoff for
kt=1--10; replay in compiled order; register every moved row; preserve every
AT-BAR row; and forbid an earlier first-over-bar boundary.

No configuration/default, carried state, stabilizer, production ocean
statement, canonical NEMO source/build/run, year harness, reconciliation gate,
freshwater pair, #1484 guard, or held manifest may change. The round-70 patch
remains held. The round stops for operator acquisition after review, citations,
focused checks, and a clean committed handoff.
