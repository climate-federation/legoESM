# Preregistration — NEMO testcase L2 GYRE round 148

Date: 2026-09-22

Incoming lane tip: `973ec1c51ca1880ef6c158cbbe3c6a7282e9315a`.
Evidence lives under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round148/`.  This document is
frozen before adding or measuring the developed-state LDF instrument.

## Question and existing-record audit

Round 147 showed that the LDF family alone removes 68.9461721587972% of the U
and 75.4145976079978% of the V incoming slow-forcing maximum at NEMO's exact
day-180 state.  Its adjacent cumulative snapshots cannot distinguish the LDF
operator from the in-place accumulator rounding.

The compiled GYRE program forms the F-point curl intermediate first, then the
T-point divergence intermediate at
`GYRE_OMIP_L2_P3_SM_R146RHSFAM/BLD/ppsrc/nemo/dynldf_lev.f90:121-130`, and
adds their gradients to the shared U/V `Krhs` accumulator at
`GYRE_OMIP_L2_P3_SM_R146RHSFAM/BLD/ppsrc/nemo/dynldf_lev.f90:132-140`.
The admitted Round-140 and Round-146 records contain the completed RHS and
cumulative family boundaries, but no `zwf`/`zwt`, live `e3t/e3u/e3v/e3f`, or
pre/post-LDF accumulator pair.  Therefore no internal statement can be scored
from the existing records without inventing an oracle by subtraction.

## Passive acquisition

Clone the admitted Round-146 producing card under the new target
`GYRE_OMIP_L2_P3_SM_R148LDF`.  Add a WRITE-only instrument to the existing
`dynldf_lev_lap` routine at `kt=1081`.  It records:

1. NEMO's exact U/V `Kbb` inputs, T/U/V/F masks, `ahmt`/`ahmf`, the six live
   Kbb/Kmm thickness operands, and every horizontal metric used by the compiled
   statement;
2. the compiled `zwf` and `zwt` values over exactly the indices assigned by the
   first loop, once per active level; and
3. the U/V `Krhs` arrays immediately before and after the in-place additions.

Admission requires byte identity with Round 146 for both restarts, the
Round-140 developed RHS, the Round-146 family record, and every inherited
developed-state record.  It also requires a closed field census, exact byte
count, finite registered values, and source-layout proof from the compiled
writer.  Header, truncation, missing-field, direct-intermediate ULP,
post-accumulator ULP, and inherited-restart-byte plants must each print
`STATUS PLANT-FIRED` and exit nonzero.  Any moved inherited scientific byte or
owned restart byte refuses passivity and requires a new target.

## Frozen operand prediction and falsifiers

The frozen prediction is that the first source-order non-bit operand is the
live F-point thickness multiplying the curl bracket on compiled line 123.
The production model derives the F-point thickness from its one cell-centre
`h_k` field; NEMO reads its QCO live `e3f_3d * (1+r3f*fe3mask)` field.  Confirm
only if the recorded F-point thickness differs and every earlier registered
input is bit-exact.  Refute if the F-point thickness is bit-exact, or if an
earlier input differs.

Given the recorded operands, extend the existing Round-50 literal LDF walker;
do not add a second numerical transcription.  The source-order table is:
inputs, `zwf`, `zwt`, isolated U/V term reconstructed from recorded
intermediates, then the pre/post `Krhs` accumulation.  A row is exact only at
zero unequal owned cells.  The prediction that the thickness factor owns the
family gap is refuted if both compiled intermediates are bit-exact while only
the pre/post addition differs.

The production-JIT plant changes one consumed recorded operand and must move a
registered intermediate or post-accumulator row.  Eager and isolated-JIT rows
may be reported but cannot substitute for the production step.

## Candidate and trajectory bar

No candidate exists until the direct record names a source-exact statement.
Only such a candidate may proceed through the full Decision-43/45 gate:
kt=1..10 ladder, day 1..30, day 30/60/90/120/180/240/300/360, first-over-bar,
all kt1 AT-BAR rows, and a closed moved-row registry.  DINO must be measured
because its Euler lane executes the same LDF operator.  LOCK_EXCHANGE and
OVERFLOW resolve LDF off; ORCA2 remains `UNMEASURED-WITH-SPEC` until its own
developed-state record exists.

No configuration, default, carried state, scheme, stabilizer, canonical NEMO
source, production physics, or immutable before arm changes in this acquisition
round.  Expected status is `STOPPED_FOR_RECORD`.
