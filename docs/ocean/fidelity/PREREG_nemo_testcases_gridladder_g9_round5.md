# PREREG — grid ladder G9, round 5 (gridladder_r50)

Frozen before any round-50 measurement.  Record: round 46's
(`gridladder_rounds/round46/oracle_g9_r46/twin_a`), unchanged; no acquisition.
Card at `075369b2e` (round 49 landed).

## The statement

Round 49 named the first differing stage-3 statement inside NEMO's FCT.  The
compiled G5 source forms the first upstream U/V flux products directly from
the volume transports and Kbb tracer
(`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/traadv_fct.f90:501-504`),
differences the integrated U/V/W fluxes, multiplies that sum by `r1_e1e2t`,
and divides the resulting rate by the live Kmm thickness
(`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/traadv_fct.f90:530-538`).
It repeats the transport products on the midpoint tracer, averages the first
and second upstream fluxes, and writes the upstream Krhs rate in that same
order (`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/traadv_fct.f90:568-609`).
The high-order fluxes, `nonosc`, and final correction then execute in compiled
order (`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/traadv_fct.f90:193-199`,
`:743-938`, `:322-330`).

The landing will be a selectable literal arithmetic form inside the existing
shared FCT kernel, default off and selected only by G9.  The existing fold
statements remain the boundary operations around those same flux arrays.  No
second FCT implementation and no other card selection.

## Frozen predictions and falsifiers

- P1 — kt 1 stage-3 after-advection Krhs: T 404,775 / S 415,895 unequal wet
  cells become 0 / 0.  REFUTED by at least one unequal wet cell.
- P2 — kt 1 stage-3 pre-implicit content: T 1,226 / S 762 unequal wet cells
  become 0 / 0.  REFUTED by at least one unequal wet cell.
- P3 — kt 1 step-end tracer rows: T 1,076 / S 587 unequal wet cells become
  0 / 0.  REFUTED by at least one unequal wet cell.  This is the requested
  closure prediction; the refuting number is therefore 1 for either tracer.
- P4 — round 46's stage-3 operands before `tra_adv` remain exact or AT-BAR;
  after `tra_adv`, the literal FCT Krhs is bit-equal.  REFUTED if an upstream
  operand is newly over the bar or the after-advection Krhs has any unequal
  wet cell.
- P5 — G5 and G8 traced-step jaxprs, with addresses masked, retain their
  round-49 hashes `339a8016d262c4f6` and `03ccdb94113eab61`.  REFUTED by
  either hash moving.

The record content test becomes value-based: stage-3 FCT Krhs, pre-implicit
content, and step-end T/S are bit-equal to NEMO.  Its wiring-removal plant must
fail and the restored test must pass under jit with traced inputs and fp64.

## Certification criterion

Score the unchanged independent and given-NEMO-entry labels at kt 1..10.
G9 CERTIFIED iff every row is exact or AT-BAR (normalised error <= 1e-15;
D127 flushed subnormals counted separately).  If certified, the ladder result
also certifies the ORCA2 fold import's stage-1 V transport mask, round-46
exchanges, round-47 FCT fold statements, and round-48 end-of-stage fold and
stage pair.  Otherwise the first remaining statement is named.
