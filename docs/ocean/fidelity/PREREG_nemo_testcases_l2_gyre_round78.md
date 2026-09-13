# Preregistration: NEMO-testcases L2 GYRE round 78 U-midpoint ownership

Date: 2026-09-12. Frozen after the operator admitted the round-77 acquisition
and before inspecting any unreported midpoint field, running a live legoESM
comparison, or editing the shared midpoint arithmetic.

## Admitted boundary and compiled statement

The operator's round-77 card completed at clean producer commit
`6c0fe440340c1c1c7ad16d8cbf547d93264bee26`. Its dedicated gate reports an
exact 1,127,856-byte, float64 U-midpoint record with SHA-256
`efff2a6ab74770890221790ac7b3c7d01d07bbf9ee3a520fdad1d99041be8a0b`, all 50
midpoint replays bit-exact, byte-identical source/target restart and mesh, and
full inherited-record admission. Every record-dependent plant exited nonzero.

The compiled round-77 target selects the forward coefficients for the first
two initializing substeps and the AB3-AM4 coefficients afterward at
`GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:481-489`. It then
forms the U midpoint in the written left association
`ua_e = za1*un_e + za2*ub_e + za3*ubb_e` at the same compiled source's
`:496-502`. The WRITE-only record serializes the coefficient triple and those
four owned 32 by 22 fields immediately afterward at `:506-509`.

Round 73's first robust boundary remains the final kt=2 stage-1 `un_adv`: all
580 wet U faces differ, with maximum absolute difference
`0.00012029895814569258`. Round 76 showed that the shared downstream metric and
accumulator statements are bit-exact on NEMO inputs, so the midpoint producer
is next in compiled order. V remains withheld until U is owned.

## Frozen measurement, predictions, and falsifiers

First extract the existing production midpoint body into one module-level
shared helper without changing a single operation, association, rounding
barrier, caller, argument, or configuration. The production solver and the
round-78 probe must both call that helper. This is an instrumentation refactor,
not a numerical arm. It is accepted only if the ordinary production-jitted
state and every returned barotropic substep field are bit-identical to the
pre-refactor implementation; any moved bit refutes the extraction and no
scientific result from it is cited.

Compose the admitted round-76 strict reader and round-72 oracle-seeded kt=2
context. Require fp64 policy, x64, JIT, clean commit stamps, the exact producer
commit and record digest above, the round-77 record gate, the 45/69 exact and
24/69 classified inherited-record census with 281 admitted representatives,
and byte-identical ordinary seeded state.

For every substep, compare in compiled order: the three coefficients, `un_e`,
`ub_e`, `ubb_e`, and `ua_e`. Stop the live ownership verdict at the first
non-bit row. Independently call the one shared production helper on only NEMO's
recorded coefficients and operands for all 50 substeps.

Prediction: all coefficients and all 50 shared-helper results on NEMO inputs
are bit-exact. The first live non-bit row is substep-1 `un_e`, at exactly 2 of
580 wet U faces and maximum absolute difference
`2.117582368135751e-22`; this follows the already recorded round-76 two-face
`ua_e` result and the compiled initializing coefficients `(1,0,0)`. It is below
the independently demonstrated `8.881784197001252e-16` cross-trace compilation
floor, so it is recorded as exactness debt and may not justify a numerical
edit. The prediction is falsified by any earlier coefficient mismatch, a
different first row/count/magnitude, or any non-bit shared-helper result.

A live-`un_e` null plant must substitute only the oracle comparison target at
the predicted row and move the first live boundary. A one-ULP plant in the
first nonzero recorded `ua_e` must make the shared-helper exactness gate fail.
Both plants must exit nonzero. Every compared record, live trace, and helper
array must report float64. Failed predictions remain **REFUTED** in the receipt.

If the first live non-bit row is an input, no midpoint edit is eligible and the
next walk follows that input's compiled producer, using an existing admitted
record if complete or preparing a new WRITE-only acquisition otherwise. If the
inputs are exact but the shared result is not, freeze a separate numerical
candidate preregistration before any arithmetic edit.

## Rule 12 and exclusions

No numerical statement is preregistered to change. The helper extraction must
be bit-identical on the ordinary production path; therefore GYRE kt=1--10 and
days 1--30, LOCK_EXCHANGE, OVERFLOW, and DINO are unchanged and **UNREACHED**
unless a later separately preregistered numerical candidate exists. DINO keeps
explicit shared-midpoint and 96--98% per-row cancellation risk.

ORCA2 remains **UNMEASURED WITH SPEC**: resolve its compiled external-mode
card; record every entry, coefficient, midpoint operand/result, metric
transport, reciprocal metric, accumulator exit, normalization, and boundary
handoff for kt=1--10; replay in compiled order; register every moved row;
preserve every AT-BAR row; and forbid an earlier first-over-bar boundary.

No configuration/default, carried state, stabilizer, NEMO source/build/run,
year harness, reconciliation gate, freshwater pair, #1484 guard, or held
manifest may change. The round-70 patch remains held.
