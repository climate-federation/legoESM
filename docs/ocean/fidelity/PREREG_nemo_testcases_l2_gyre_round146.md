# Preregistration — NEMO testcase L2 GYRE round 146

Date: 2026-09-21

Incoming lane tip: `39ea1c80863d8924b5b2fd71a0a6ec99c6970f0e`

Evidence lives under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round146/`.  This document is
frozen before acquiring or scoring a new RHS-family record.

## Question and compiled order

Round 142 measured that replacing the complete developed three-dimensional
momentum RHS removes 68.946% of the U and 75.415% of the V incoming
slow-forcing maximum.  Round 145 showed that the downstream wind residue is
source-exact when given NEMO's QCO reciprocal but fails the month/year gate.
This round ranks the larger upstream RHS families at NEMO's day-180 entry.

The admitted compiled program calls HPG, LDF, VOR, KEG, and ZAD in that order
at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:142-169`, then consumes
the completed `Krhs` in the depth average at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:210-228`.  The existing
Round-140 record contains only the completed RHS, so it cannot distinguish the
five families.

## Passive acquisition

Before any physics change, clone the admitted Round-140 card under the new
target `GYRE_OMIP_L2_P3_SM_R146RHSFAM`.  An additive WRITE-only source card
records the native U/V `Krhs` arrays after each of the five compiled calls at
step 1081.  It must not assign, reorder, or add a call.  Admission requires:

1. the step-1080 and step-1081 restarts and all inherited Round-140 records
   are byte-identical to the admitted Round-140 run;
2. the new record has exactly the registered header, field census, extents,
   and byte count;
3. its final ZAD boundary is bit-identical to Round 140's completed RHS;
4. a one-ULP record plant, a missing-field plant, a truncation plant, and a
   passive-admission plant each print `STATUS PLANT-FIRED` and exit nonzero.

Any moved inherited artifact, unmatched final boundary, or absent plant marker
REFUTES passivity and stops the round without interpreting family rows.

## Directed ranking

For each family, preserve the model's other four live production terms and
replace only that family's U/V addend by the difference between NEMO's adjacent
cumulative boundaries.  HPG uses its first cumulative boundary against exact
zero.  Every arm runs through the complete production-JIT step from NEMO's
admitted day-180 state, retains unowned boundary faces, and must remain
bit-identical to a plain call carrying the same private override.

The frozen prediction is that HPG is the largest family: it removes more than
half of the ordinary incoming U and V maximum errors.  The falsifier is any
other family producing the greatest reduction in either face, or HPG reducing
either maximum by at most 50%.  A family is called the local owner only if the
same family gives the largest reduction on both faces.  Otherwise the result
is split ownership and no single family is promoted.

A closed registry records ordinary plus five directed arms for incoming and
final U/V, with unequal cells, maximum and RMS absolute differences, and first
unequal index.  Its omission plant and a one-ULP family-input plant must fire
through production JIT.

## Year sensitivity and landing bar

Only the largest unambiguous family proceeds to a one-family source-exact
candidate and the full Decision-43/45 gate.  Frozen candidate prediction:
day-30, day-240, and day-360 T3D RMS all decrease; first-over-bar is not
earlier; no kt1 AT-BAR row leaves; kt2 T/S remain AT-BAR; and every moved
ladder/year row is registered.  DINO is measured if its resolved recipe
executes the statement.  A failed condition keeps the candidate held and
restores production.  If the local ranking does not identify one family, the
year sensitivity is UNMEASURED this round rather than guessed.

No configuration, default, carried state, scheme, stabilizer, canonical NEMO
source, or immutable before arm changes.  ORCA2 remains
`UNMEASURED-WITH-SPEC` pending its own developed-state native forcing record.
The expected final status is `HELD` unless one source-exact family passes every
Decision-43/45 condition.
