# ORCA2 round 194 preregistration — atomic V reciprocal and transport unit

Date: 2026-10-09. Frozen base:
`9c2c05b740d86c473210f2232fab1085c500aebb`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round194/`.

Every rung-0 and month number is **independent hierarchy rung 0**: the card
starts from its corrected climatological T/S, zero velocity and zero sea
surface. Rung-7 numbers, if reached, are **given NEMO's entry**. The two claim
labels are never mixed. The shipped ORCA2 card, sea ice, its six selectors and
its `unmeasured_features` tuple remain unchanged unless the entire unit passes
the standing landing gates; no partial operand may land.

## Frozen atomic unit and source order

Reapply round 192's indivisible private unit: NEMO's source-associated
completed-RHS depth average, raw reference face depth, no extra compact V
transport mask, separately materialised `zhV`, and seven-array external-mode
association. Add exactly one newly named statement from round 193: NEMO builds
`r1_e1v = 1/e1v` without a mask at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/domhgr.f90:152` and consumes it in the
substep transport accumulation at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:600-608`.

The candidate uses the geometric reciprocal wherever `e1v` is nonzero. The
existing zero-width legoESM halo sentinel remains zero; this is representation
plumbing, not a stabiliser or a NEMO configuration choice. The ordinary path
is unchanged unless the private hook is selected. If the atomic unit passes
all gates, its cited NEMO statements land together; otherwise the package tree
is restored to the frozen base.

Before any trajectory score, admit the round-96 rank-complete record and prove
the candidate V transport accumulator bit-exact against NEMO after each of all
65 external substeps. The record's substep index, incoming accumulator,
materialised `zhV`, weight and completed accumulator must each be present in
source order. A one-ULP accumulator plant and a masked-reciprocal control must
both refuse.

The primary trajectory gate is the corrected-entry independent rung-0
kt=1..10 ladder: 40 checkpoints, 200 field rows. Decision 96 counts only rows
whose RMS score moves: strict majority toward NEMO, first-over-bar row toward
or equal, no bit-exact row lost, and kt=10 stage-3 SSH maximum not worse.
Bit-moved/RMS-equal rows are registered but do not vote. A terminal refusal is
an automatic hold.

Only if rung 0 is eligible, run the given-entry rung-7 ladder under the same
predicate, then the independent month boundary, GYRE's certified ladder/year,
DINO, tanks, citation gates and focused/full fidelity batteries.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R194-P1 | The added reciprocal closes the complete external accumulator sequence. | All 65 completed V accumulators are bit-exact against the admitted round-96 record, with exact incoming accumulator, `zhV` and weight at every substep. | Any missing stream, non-exact prerequisite or residual: **REFUTED**; stop without a trajectory claim. |
| R194-P2 | The enlarged atomic unit removes round 192's terminal and completes the rung-0 ladder. | All 40 checkpoints and 200 rows are finite through kt=10 stage 3. | Any refusal/non-finite row: **REFUTED**; retain the first terminal and restore production. |
| R194-P3 | The complete unit is a Decision-96 net improvement on rung 0. | RMS toward exceeds away, first debt is toward/equal, no exact row is lost, and kt10 stage-3 SSH maximum does not increase. | Any predicate fails: **REFUTED**; keep the unit private and name the first failing row. |
| R194-P4 | Given-entry rung 7 remains eligible. | Its 200-row ladder completes, no exact row is lost, first-over-bar is not earlier, and SSH maximum is not worse. | Any failure: **REFUTED** and hold. |
| R194-P5 | If both ladders pass, the independent month advances beyond the registered step-96 boundary. | Step 96 completes finitely or the first refusal is later. | Step 96 or earlier refuses: **REFUTED**; retain the exact boundary. |
| R194-P6 | Every control binds. | Masked-reciprocal, accumulator-ULP, exact-row-loss, false-majority, score-equal-vote, entry-bit and candidate-identity plants each refuse. | Any plant stays green: invalid gate; report no landing verdict. |

If P1, P2 or P3 fails, downstream gates cannot reverse the dispositive rung-0
result and are not run. Failed predictions remain in the receipt.

ASKED choices: round 193's OPEN atomic unit and Decision 96's fixed predicate.  
UNASKED choices: empty.
