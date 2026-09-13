# Preregistration: NEMO-testcases L2 GYRE round 76 kt=2 transport-mean walk

Date: 2026-09-12. Frozen after the operator admitted the round-75 record and
after reading the new target's compiled source, but before running a live
legoESM comparison or inspecting any unreported record field.

## Boundary and admitted evidence

Round 73 stopped at the kt=2 stage-1 `un_adv` input: all 580 wet U cells were
non-bit, with maximum absolute difference `0.00012029895814569258`. The
operator has now admitted the missing producer record at commit
`6ca9a954e84ae9cdef4885ae5c5b4f1e8fb68cf6`. Its acquisition gate reports the
exact 3,789,976-byte layout and bit-exact replay of all 50 U rows, all 50 V
rows, and both normalizations. The clean record SHA-256 is
`47ba91919df3a9a1c55be7fd886d620c721b2f175a737f32a05d1be2498ddf4c`.

The compiled round-75 target sets both transport accumulators to zero at
`GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/dynspg_ts.f90:377-378`. At every
external substep it forms `zhU=e2u*ua_e*zhup2_e` at `:538-544`, assigns
`za2=wgtbtp2(jn)`, and left-associates the U and V accumulator statements at
`:571-580`. It divides the completed accumulators once at `:814-818`, then
applies the active U/V boundary exchange and hands off the post-boundary values
at `:823-834`.

## Frozen measurement and falsifiers

Add one committed round-76 probe by composing, not reimplementing, the existing
strict round-14 record reader and round-72 oracle-seeded kt=2 context. Run the
ordinary production-jitted barotropic solver with its existing WRITE-only
substep trace. Require fp64 policy, x64, JIT, clean commit stamp, the admitted
producer commit and record digest, the round-75 record gate, the historical
round-64/66 seeded-context gates, and byte-identical ordinary seeded state.

For U, walk the compiled statements in this frozen order: zero accumulator;
substep entry; raw `wgtbtp2`; `zhU`; fixed reciprocal `r1_e2u`; the exact
left-associated increment; and substep exit. Stop the live-path verdict at the
first non-bit statement. Only if `zhU` is first non-bit, compare its already
recorded `ua_e` and `zhup2_e` inputs as a discriminant; do not interpret a
downstream accumulator result as an independent owner. V is withheld until U
is owned.

Prediction: the zero seed, substep-1 entry, and all 50 raw weights are bit-exact;
the first live U mismatch is substep-1 `zhU`, and its recorded `ua_e` input is
already non-bit. The U wet-cell mismatch count at that first statement is
predicted to be 580 of 580. This is falsified if any earlier row is non-bit, if
`zhU` is exact, if its mismatch count differs from 580, or if `ua_e` is exact.
Any such outcome is recorded as REFUTED and replaces the predicted boundary.

Independently feed only NEMO's recorded inputs through the one shared
`nemo_literal_accumulate_transport` statement and require its U result to be
bit-exact at every substep before making any production eligibility claim. A
one-ULP change to the first nonzero recorded U exit must make the exact replay
fail and exit nonzero. A null-live-`zhU` plant must move the first live mismatch
past `zhU` or the control is vacuous. Every record array, live trace array, and
geometry array used in a claim must print float64.

If the shared accumulator is exact on NEMO inputs but live `zhU` is not, no
accumulator edit is eligible: the first non-bit producer input owns the next
walk. If the shared accumulator itself is non-bit, no production change may be
written until the exact association difference is named from the compiled
statement. If the first non-bit producer lacks recorded operands, prepare a
new WRITE-only acquisition with a new target name and stop for the operator.

## Rule 12 preregistration

No production statement is assumed eligible. If and only if a shared
production statement is bit-exact on NEMO inputs, a later code edit in this
round must preregister every moved GYRE row, preserve every AT-BAR row, and keep
the first-over-bar boundary no earlier across kt=1--10 and days 1--30 against
the recorded before arm. LOCK_EXCHANGE and OVERFLOW must execute the same
statement or prove it is inactive. DINO retains explicit shared-accumulator
risk and 96--98% cancellation risk. ORCA2 remains UNMEASURED WITH SPEC: resolve
its compiled external-mode card; record every substep entry, weight, transport,
reciprocal metric, exit, normalization, and boundary handoff for kt=1--10;
replay in compiled order; register every moved row; preserve every AT-BAR row;
and forbid an earlier first-over-bar boundary.

No configuration/default, carried state, stabilizer, NEMO source/build/run,
year harness, reconciliation gate, freshwater pair, #1484 guard, or held
manifest may change. The round-70 patch remains held.
