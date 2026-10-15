# Preregistration — ORCA2 round 284 round-282 review corrections

Date: 2026-10-15

## Scope

Disposition the independent review of round 282 before new statement work.
This round restores the frozen round-276 Decision-96 classifier, corrects the
landed np_CRV citation to the EEN routine ORCA2 actually executes, runs the
previously omitted `[19,33,3]` after-VOR falsifier, hardens the selector tests,
and measures the grid-ladder's omitted top-F-row U-halo statement as a separate
default-off arm.  Decision 126, sea ice, NEMO decks, and NEMO sources remain
untouched.

## Compiled statements

The shipped ORCA2 deck selects EEN.  Its executing `vor_een` np_CRV branch is
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90:762-766`, inside
`SUBROUTINE vor_een` at line 694; the F-cell reciprocal is built from
`e1f*e2f` at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/domhgr.f90:155-156`.
The previous `dynvor.f90:648-651` citation is in `vor_ens`, a non-executing
scheme for this deck, and is retracted.

At the top F row, `vor_een` reads the north `e1u*pu` operand at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90:764-766`; NEMO's U-fold halo supplies
that operand.  The grid-ladder implementation is imported as an independently
selectable, default-false arm so the area statement and the halo statement are
not confounded.

## Frozen predictions and falsifiers

1. **R284-P1 — classifier restoration.** Reclassifying round 282's frozen
   artifacts through the unchanged round-276 classifier leaves statement A
   eligible, while B and A+B refuse on OMT-1..3 when their first-non-bit
   checkpoint moves.  Any different result is retained as a refutation.
2. **R284-P2 — citation context.** The corrected 762–766 anchor passes only
   inside `vor_een`.  A planted `vor_ens` enclosing-routine expectation must
   fail even though an np_CRV text match exists elsewhere in the file.
3. **R284-P3 — missing round-282 falsifier.** On the matched stage-2 after-VOR
   boundary, statement A moves the `[19,33,3]` residual toward NEMO.  Equal or
   larger absolute error refutes round 282's claimed confirmation; the receipt
   is corrected loudly either way.
4. **R284-P4 — selector binding.** With `een_nemo_relative_vorticity=True`,
   forcing the EEN NEMO-zeta call-site argument false must fail a direct test.
   Selecting the area arm with `een_metric_weighting != "nemo"` must raise.
   The imported helper tests must retain their west-cyclic, south-row, and
   missing-operand refusals.
5. **R284-P5 — top-F-row candidate.** The separate halo arm changes only the
   north U operand used by np_CRV, moves the matched top-F after-VOR row toward
   NEMO, and is scored singly over rung 0, OMT-1..4, and rung 10 under both
   labels.  It lands only if the restored Decision-96 classifier passes every
   card with no exact-row loss and unchanged-or-toward first-over-bar row.

All failed predictions remain in the receipt.  Any package landing also owes
the GYRE year, DINO, tanks, citation gate and context plant, focused tests, the
single permitted fidelity battery, and independent operator review.

## Evidence target

Evidence goes under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round284/`.
The receipt is
`docs/ocean/fidelity/testcases/nemo_testcases_l4_orca2_round284_round282_review_receipt.md`.
