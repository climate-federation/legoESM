# PREREG — grid ladder G9, round 6 (gridladder_r51)

Frozen before any round-51 numerical measurement.  The only oracle record is
round 46's `gridladder_rounds/round46/oracle_g9_r46/twin_a`; no new
acquisition is authorised unless that record cannot decide the registered
questions.  Card at `c818101382a0` (round 50 landed).

## Registered split: inherited residue or kt-2 statement

The primary prediction is that independent kt 2 stage 1 amplifies kt 1's
AT-BAR residue.  Starting kt 2 from NEMO's recorded step-entry prognostics and
supplying NEMO's recorded kt-2 barotropic stage-1 handoff
(GIVEN-NEMO-ENTRY) will make all seven stage-1 rows exact or AT-BAR
(normalised error <= 1e-15).  A single stage-1 row above `1e-15` refutes the
carry-over prediction and establishes a kt-2-only statement or missing entry
operand.

The first statement-level suspect is the low-order FCT divergence at kt 1
stage 3.  Compiled NEMO differences the integrated U, V, and W upstream fluxes
in source order, adds the horizontal divergences before W, multiplies by the
reciprocal cell area, and negates the result
(`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/traadv_fct.f90:530-533`).
The prediction is that a compiled, live-operand walk first differs at this
statement through XLA reciprocal-multiply or contraction/materialisation:
at least one wet-cell bit is unequal after every preceding statement is
equal.  **Zero unequal wet cells in the literal first-divergence result is the
registered numerical refuter.**  If refuted, the walk continues in compiled
order to the first later differing NEMO statement; no earlier statement will
be relabelled post hoc.

## Frozen measurements and falsifiers

- P1 — GIVEN-NEMO-ENTRY kt 2 stage 1: all seven rows are exact or AT-BAR.
  REFUTED by any normalised error greater than `1e-15`.
- P2 — real-record literal first divergence: at least one unequal wet cell,
  with zero unequal wet cells the refuting number.  The walk records the
  unequal count after each source statement and names the first nonzero one.
- P3 — after changing only the first differing statement, both G9 labels at
  kt 1..10 contain no DEBT row.  REFUTED by one row above `1e-15`.
- P4 — all 26 non-G9 traced-step jaxprs are byte-for-byte unchanged after
  address masking.  REFUTED by one changed jaxpr.

## Round-50 control repair

The literal-FCT value fixture will carry nonzero, nondegenerate U, V, and W
transports and pass cell area as a traced argument under jit and fp64.  Its
NumPy oracle transcribes NEMO's lines exactly.  A horizontal/W regrouping
plant and a direct-divide-for-reciprocal-multiply plant must each fail; either
plant passing refutes the control repair.

## Certification criterion

G9 is CERTIFIED only if every independent and GIVEN-NEMO-ENTRY row at kt
1..10 is exact or AT-BAR, every new exact-value test passes with jit, traced
inputs, and fp64, both wiring-removal plants fail, and the 26 non-G9 jaxprs do
not move.  Certification covers the ORCA2 fold import's stage-1 V transport
mask, round-46 exchanges, round-47/48 fold statements and stage pair,
round-49 stage-3 RHS order, and round-50 literal FCT.
