# ORCA2 round 66 preregistration — Decision 52 independent start

Date frozen: 2026-09-28

Base: `7ab42435b2f56f8c2a97cedee5bb2ea3fd1737d4`

Claim label: **independent**.  legoESM starts from its own card state and is
compared only with NEMO's own from-rest trajectory.  No NEMO entry field is
loaded into legoESM.  This table is separate from every earlier
"given NEMO's recorded entry" result.

Sea ice remains out of scope.  The card's six-entry `unmeasured_features`
tuple, all selectors, and the one-category SI3 state remain frozen.

## Source-first boundary

The compiled ORCA2 program initializes T/S through `dta_tsd`, sets u/v to
zero, and copies the before level at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/istate.f90:93-140`.
The active ORCA_R2 T/S hand alterations and final mask are at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dtatsd.f90:217-254,307-310`.
Round 7 already gated legoESM's independent T/S/u/v entry bit-exact.

NEMO then adjusts both SSH time levels for its five-category snow/ice mass at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/iceistate.f90:440-465`.
That statement belongs to the explicitly out-of-scope sea-ice identity.  This
round neither transcribes it nor substitutes its result.  The known SSH entry
difference is therefore retained as part of the independent comparison.

The existing ORCA2 ladder already owns record admission, the exact external
surface operands, the production JIT/fp64/libm step, and all 40 entry/stage
checkpoints through kt=10.  It will be extended with an explicit independent
mode whose default remains the Decision-52 SSH bridge used by earlier twins.
No second ladder implementation is authorized.

## Frozen predictions and falsifiers

1. **Entry state.** At kt=1, independent T, S, u, and v predict zero unequal
   cells.  Independent SSH predicts exactly `16,433 / 26,640` unequal cells
   with maximum absolute difference `0.015479333813968585 m`, reproducing the
   already-admitted round-7 entry comparison.  Any changed count or maximum
   is REFUTED and must be reconciled before interpreting the trajectory.
2. **No bridge.** The executed state at kt=1 must be byte-identical to the
   card's own initial state in all five fields.  A control that applies the
   historical SSH replacement must make the round-66 classifier refuse.
3. **First non-bit statement.** The first non-bit checkpoint predicts
   `kt=1 entry SSH`, owned by the compiled ice-mass adjustment above.  If any
   earlier field differs, or if SSH is exact, the prediction is REFUTED.
4. **Complete ladder.** The production run predicts 40 checkpoints
   (`kt=1..10`, entry plus stages 1/2/3), 200 exact field rows, all finite and
   float64.  The gate will report unequal count, maximum absolute difference,
   and RMS for every row, plus the complete kt=10 five-field ranking.  A stop,
   missing row, non-finite value, wrong dtype, or record-admission failure
   refutes completion.
5. **Disposition.** This is a measurement-only round.  No model statement is
   predicted to land.  If predictions 1-4 hold, the independent ladder becomes
   the lane's headline short-horizon metric and the next round ranks ORCA2
   month-scale errors by magnitude.  If they do not, the round is HELD with
   the first failed predicate named.

Failed predictions remain **REFUTED** in the receipt.  Bit identity, unequal
count, maximum absolute difference, and RMS remain separate fields.

## Required controls and validation

- One-ULP mutation of an exact independent T entry cell must refuse.
- Applying the old SSH bridge must refuse the independent classifier.
- Removing one checkpoint or changing one expected entry count must refuse.
- The compiled SSH citation must pass the citation gate and a rigid shift must
  fail.
- Run focused tests, the 170-test card battery, the single permitted
  `tests/ocean/fidelity -n 12` battery, the default citation audit, and the
  separate read-only Codex review.  With no `packages/` change, GYRE and other
  trajectory gates are not applicable.

ASKED: execute Decision 52's independent-start ORCA2 ten-step ladder.

UNASKED: configuration, selector, threshold, forcing, stabiliser, carried
state, NEMO arithmetic, sea ice, and the held shared tracer QCO/RK candidate.
