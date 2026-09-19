# Preregistration — NEMO testcase L2 GYRE round 119

Date: 2026-09-19

Incoming lane tip: `4d250301588d3ed0ad83fb20d6bf520e175d576e`

This document is frozen before any Round-119 scientific measurement.  The
unchanged production anchors are kt2 T/S/U/V maximum error
`1.4210854715202004e-14` / `2.1316282072803006e-14` /
`2.7377110452773967e-12` / `3.2849219221489645e-12`, kt3 T/S maximum error
`8.600419718618468e-7` / `6.979441735666114e-8`, and day-30 T rms
`6.890484901489568e-5 K`.  They are inherited comparison anchors, not new
measurements.  Evidence will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round119/`.

## P1 — shared-gate production-association discriminator

Round 118 closed NEMO's direct pre-loop producer chain bit-for-bit, but its
isolated NEMO-order accumulation still differed from the production live total
on 6,882 U and 6,566 V cells at one last bit.  Round 119 extends the existing
Round-83 producer gate; it does not create a second harness.  From the same
recorded kt2 stage-1 entry it will score three explicitly labelled executions:

1. the ordinary full production step under JIT, using the accumulator values
   actually produced at its HPG/KEG, VOR, ZAD and LDF barriers;
2. a private full-production-step NEMO-order arm that consumes the identical
   production operands and materializes HPG -> LDF -> VOR -> KEG -> ZAD; and
3. the existing isolated-closure JIT reconstruction.  Production eager is
   reported separately and cannot certify a production row.

The exact compiled program is
`GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/stp2d.f90:144-179`:
`dyn_hpg` writes `Krhs` at `:147`, `dyn_ldf` adds at `:151`, `dyn_vor` adds at
`:155`, `dyn_keg` adds at `:174`, and `dyn_zad` adds at `:177`.  The lateral
operator's in-place `Krhs` writes are
`GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynldf_lev.f90:121-141`.

Frozen predictions:

- the ordinary production-JIT final accumulator remains bit-for-bit equal to
  Round 118's direct live total;
- every raw HPG/LDF/VOR/KEG/ZAD operand is bit-for-bit identical between the
  ordinary and NEMO-order production executions;
- the NEMO-order production arm is bit-for-bit equal, boundary by boundary, to
  the existing isolated NEMO-order JIT reconstruction; and
- changing only the association from the model's actual combined
  `(HPG + KEG) -> VOR -> ZAD -> LDF` program to the compiled NEMO order accounts
  for the Round-118 closure: ordinary versus NEMO-order final differs on exactly
  6,882/17,400 U and 6,566/17,100 V cells with maximum
  `8.470329472543003e-22` in each component.

A different raw operand, a NEMO-order production/isolated mismatch, failure to
reproduce the registered cell counts and maximum, or failure of the ordinary
trace to close its live total refutes the corresponding prediction.  In those
cases the first mismatching production boundary owns the next walk; no source
association claim or candidate is allowed.

The production control changes one active HPG accumulator word by one ULP
inside the full step.  It must move that boundary and every later boundary that
consumes the word, leave all untargeted raw operands unchanged, print
`STATUS PLANT-FIRED`, and exit nonzero.  An isolated-only plant is not accepted.

## P2 — first non-bit statement and magnitude ownership

If P1 closes, the first different executed statement is the shared `Krhs`
association itself: legoESM's current combined HPG/KEG expression and later LDF
addition do not reproduce NEMO's compiled in-place sequence.  This establishes
attribution only.  It is not automatically a landing candidate because the
Round-118 cumulative table predicts much larger ZAD error: HPG stays BIT; the
first NEMO cumulative non-BIT boundary after LDF remains
`2.5292467120726215e-14` U / `3.502735092670824e-14` V within `1e-26`, while
the ZAD incremental maximum remains `1.9220297482797664e-9` U /
`1.966061294804274e-9` V within `1e-21`.

The gate will therefore re-score the current production ZAD operands.  The
frozen magnitude prediction, inherited from the admitted Round-86/87 walk, is
that W remains the largest non-BIT ZAD operand while Kmm velocity, face
thickness and metrics remain BIT.  If that reproduces, the next candidate must
start at W's first current-tip producer statement; it may not spend a
trajectory run on the last-bit association alone.  If a different ZAD operand
now dominates, that measured operand becomes the OPEN walk instead.

## P3 — landing and blast-radius stop

No production edit is preregistered in this base document.  If P1-P2 expose a
source-exact production candidate, a separate committed addendum must freeze
its exact before/after prediction before production is edited.  Only then may
it spend Decision 43's full kt=1..10 ladder and days 1-30 run against a before
arm measured at this incoming commit: day-30 T rms must decrease, first-over-bar
must not move earlier, no kt=1 AT-BAR row may leave the bar, every moved row
must be registered, and every card sharing the statement must be measured.

The diagnostic hook and returned trace are private and unconstructible by any
recipe, so the expected blast radius for this measurement-only round is zero:
GYRE, the generic NEMO-GYRE recipe, DINO, LOCK_EXCHANGE and OVERFLOW cannot
execute it.  DINO nevertheless shares the production tendency statement and
is explicitly AT RISK for any later landing; its cancellation-sensitive rows
must be measured before such a landing.  ORCA2 remains
`UNMEASURED-WITH-SPEC`: independently record the same stage-entry operands,
score ordinary versus compiled-order production JIT/eager/isolated boundaries
with a production plant, then run its certified trajectory for any shared
candidate.  No stabilizer, configuration default, coefficient, carried state,
restart representation, year harness, reconciliation gate, freshwater pair,
#1484 guard, or held patch changes in this round.
