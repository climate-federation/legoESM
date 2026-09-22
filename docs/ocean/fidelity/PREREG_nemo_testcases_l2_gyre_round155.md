# Preregistration — NEMO testcase L2 GYRE round 155

Date: 2026-09-22

Incoming lane tip: `d06ddf3e4f14a02b71208b339f8d7833b7c8dd53`.
Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round155/`.  This document is
frozen before parsing or comparing the acquired Round-154 developed transport
record.

## Oracle statement and order

The only oracle used by this walk is the compiled source of the record's own
target, `GYRE_OMIP_L2_P3_SM_R154TRPWALK`.  Stage 3 binds Kmm to N+1/2 at
`BLD/ppsrc/nemo/stprk3_stg.f90:261-277`.  The active `np_HYB` arm writes the U
barotropic correction as
`un_adv * (r1_hu_0 / (1 + r3u(Kmm))) - uu_b(Kmm)` at
`BLD/ppsrc/nemo/stprk3_stg.f90:300-310`.  It then writes
`zFu = e2u * (e3u_3d * (1 + r3u(Kmm)*umask)) *
(uu(Kmm) + zub*umask)` at
`BLD/ppsrc/nemo/stprk3_stg.f90:313-315`.  The writer reads those operands and
completed values only after the loops at
`BLD/ppsrc/nemo/stprk3_stg.f90:317-321`.

The comparison order is therefore:

1. `un_adv`, `r1_hu_0`, `r3u(Kmm)`, the written live inverse depth, and
   `uu_b(Kmm)`;
2. the written `zub` correction;
3. `e2u`, reference `e3u_3d`, `r3u(Kmm)`, `umask`, and the written live
   `e3u(Kmm)`;
4. `uu(Kmm)`, the written corrected velocity, and the completed `zFu`.

The existing Round-152/154 production-step gate will be extended.  It loads
NEMO's exact day-180 entry, executes one ordinary production step under JIT,
and observes the shared `_nemo_ws_stage_transport` path.  It will also report
production eager and isolated-closure JIT, without relabelling either as the
production result.  No second transport implementation is permitted.

## Frozen admission and measurement predictions

1. The acquired record has exactly 20 registered finite fields, its seven
   restarts plus `mesh_mask.nc` are byte-identical to the Round-153 baseline,
   and the stamp, truncation, restart-byte and operand-ULP plants all print
   `STATUS PLANT-FIRED` and exit nonzero.  Any failure refuses the record.
2. Reference geometry (`e2u`, `e3u_0`, `umask`, `r1_hu_0`) is bit-exact.  A
   non-bit reference-geometry row refutes this prediction and becomes the
   first boundary if it is first in compiled order.
3. The first production-JIT non-bit U row is `un_adv`, inherited from the
   external/barotropic step.  A bit-exact `un_adv`, or any earlier unequal
   operand, refutes this prediction and remains in the receipt.
4. At least one later Kmm state operand (`r3u(Kmm)`, `uu_b(Kmm)`, or
   `uu(Kmm)`) is also non-bit at the developed stage.  If all three are bit,
   this prediction is refuted.
5. A one-ULP perturbation to a nonzero observed production-JIT `un_adv` cell
   must change the registered written-correction or completed-transport row,
   print `STATUS PLANT-FIRED`, and exit nonzero.  The observed step result must
   remain byte-identical to the ordinary step when no plant is active.

The first production-JIT unequal input owns the next upstream walk.  A local
transport statement is a candidate only if every preceding operand is bit and
its written output is non-bit.  Without that condition no physics lands and
no ladder, month, year, or DINO candidate arm is run.

No physics, configuration, default, carried state, restart schema, stabilizer,
year harness, reconciliation gate, freshwater pair, or #1484 guard changes in
this round.  ORCA2 remains **UNMEASURED-WITH-SPEC**: repeat the developed-entry
stage-3 transport operand walk on the ocean-only ORCA2 card before transferring
a statement verdict.

## Addendum — directed operand attribution (frozen before measuring)

Measured at tip `7a15317a9` and frozen before the substitution runs.  The
compiled-order walk above names the first unequal INPUT.  Decision 43 ranks
owners by MAGNITUDE, so the completed stage-3 U transport difference is also
attributed operand by operand: the two written statements at
`GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:300-309` and
`:313-315` are re-evaluated in one isolated JIT closure with ONE operand
replaced by NEMO's recorded value, and the completed transport is scored
against NEMO's own recorded `zFu`.  Isolated arms are never relabelled as
production.

Frozen predictions:

A1. The calibration arm, in which EVERY operand is NEMO's recorded value,
    rebuilds NEMO's `zFu` bit for bit (0 of 21,780 cells unequal).  If it does
    not, the statement association itself differs and becomes the candidate.
A2. Substituting NEMO's `uu(Kmm)` alone removes more than 90% of the
    production baseline maximum difference `1.4246544619672932`.
A3. Substituting NEMO's `un_adv` alone removes less than 1% of it.
A4. The ranked magnitude owner of the developed stage-3 U transport
    difference is therefore `uu(Kmm)`, the stage-2 velocity, NOT the
    compiled-order first input `un_adv`.

Falsifier: if `un_adv`'s alone-arm reduction is greater than or equal to
`uu(Kmm)`'s, A2 and A4 are REFUTED and `un_adv` remains both the
compiled-order and the magnitude owner.  A refuted prediction stays in the
receipt.  No physics, configuration, or carried state changes either way.
