# ORCA2 round 271 preregistration — written-order EEN transfer

Date: 2026-10-13.  Base: `5ab4e9eebce9aabe88787b00d9a361d6a6bb158e`.
This file is frozen before importing either grid-ladder round-7 statement or
measuring the matched ORCA2 stage-2 vorticity boundary.

## Authorized unit

Measure the grid-ladder round-7 implementations without re-derivation:

1. `c1b4d0d69`, NEMO's written operation order for the EEN vorticity trend;
2. `9852cfd5d`, NEMO's interpolated F-point live-thickness ratio at RK3
   stages 2 and 3.

Until those commits appear on the landed
`fidelity/nemo-testcases-orca2-gridladder` branch, they are a private
measurement arm only.  Select both statements on OMT-1..4, rung 0 and shipped
rung 10, whose resolved cards execute the vector-invariant EEN path.  Defaults,
OMT-0's flux-form branch, GYRE, DINO, tanks, decks, initial/carried state,
sea-ice selectors and the ORCA2 `unmeasured_features` tuple remain unchanged.

The compiled statements are the ORCA2 record build's EEN order at
`dynvor.f90:720`, `:727`, `:767-768`, `:772-775`, `:779-783` and the stage
ratio interpolation at `stprk3_stg.f90:218`, `:239`.  These are arithmetic
transcriptions, not stabilisers.

## Frozen predictions and falsifiers

- R271-P1 — import identity.  The private package commits are byte-identical
  to `c1b4d0d69` and `9852cfd5d`, and the ORCA2/OMT edits only select their two
  appended-default-false fields.  Any semantic reimplementation or unrelated
  selector movement refutes.
- R271-P2 — matched stage-2 EEN closure.  On the landed six-statement tree,
  the after-VOR U maximum under both **independent** and **given NEMO's entry**
  labels falls from `1.2863307536456779e-8 m/s2` to the campaign floor
  (`<=2e-10 m/s2`), with the candidate replay reproducing the completed live
  accumulator first.  A residual above the floor refutes closure; the next
  source-ordered arm is live `e3f_0vor*(1+r3f)` only after confirming the
  written-order selectors executed.
- R271-P3 — Decision-96 census.  Every executing OMT card and rung 0 has a
  strict majority of RMS-moving rows toward NEMO, no exact-row loss and a
  first-over-bar row toward or unchanged.  The rung-0 stage-3 SSH maximum is
  reported but is not a veto.  A majority failure, exact-row loss or first-row
  movement away holds the transfer.
- R271-P4 — shipped/shared isolation.  Shipped rung 10 passes the same
  Decision-96 predicate.  GYRE's certified ladder/year, DINO month and the
  tank/generic-card gates are byte-identical because their cards leave both
  selectors off.  Any unexplained movement is red.
- R271-P5 — controls.  The direct written-order arithmetic and F-ratio plants
  fire; the Decision-96 exact-row-loss and first-row-away plants refuse; the
  matched-stage one-bit plant refuses.  A plant staying green invalidates its
  gate.
- R271-P6 — branch provenance.  A landing is permitted only after the same
  round-7 implementation is present on the grid-ladder branch.  If the branch
  has not landed it, this round remains HELD even if the private arm closes
  ORCA2.

## Measurement order

1. Read the compiled ORCA2 statements and the grid-ladder round-7 diff; import
   the two pinned commits privately and select them only on the executing ORCA2
   and OMT cards.
2. Run the matched stage-2 after-VOR replay under both labels, including
   candidate self-replay and the planted violation.
3. If it closes, run the same-tree selector-off Decision-96 ladders for
   OMT-1..4, rung 0 and shipped rung 10, then GYRE, DINO, tanks, citation gates
   and tests.  Land only the grid-ladder-landed implementation.
4. If it does not close, leave the private arm out of production and report
   the first surviving source-ordered operand; do not stack a third statement
   without a new preregistration.

No NEMO acquisition or configuration decision is required.
