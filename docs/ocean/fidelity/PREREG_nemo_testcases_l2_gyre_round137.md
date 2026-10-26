# Preregistration — NEMO testcase L2 GYRE round 137

Date: 2026-09-21

Incoming lane tip: `e4854dfe7a5f4e2bdd4f949e10d5e01dc040b448`

This document is frozen before the Round-137 NEMO acquisition or developed
external-step measurement. Evidence will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round137/`.

Round 136 drove legoESM from NEMO's admitted completed-step-1080 state and
found the first recorded non-bit stage-3 temperature boundary before
advection: the QCO geometry built from the external solve. Entry `q_Kbb` was
BIT, while `q_Kmm` and `q_Kaa` differed in 599 of 600 wet columns. The stored
records do not contain the `pssh` operand or a developed-state external-mode
trace. Round 137 therefore stays upstream of tracer advection and brackets the
one external solve that produces step-1081 `pssh`.

## P0 — existing-record inventory and minimum acquisition

The admitted Round-81 external-step stream covers only from-rest `kt=2`.
Round 123 records step-1081 stage-3 `r3t(Kbb/Kmm/Kaa)` but not the `pssh`
operand or the 50 external substeps. Round 132 records the accepted restart at
completed step 1080, including SSH and all six absolute barotropic histories,
but not the next solve. No existing stream brackets the developed solve, so a
new passive record is required rather than inferring `pssh` from `r3t`.

The acquisition extends the existing Round-81 record layout and gate; it does
not create a second external-step harness. Its source card is the admitted
Round-123 configuration, whose compiled step program calls `stp_2D` before
stage 1 at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3.f90:187-201`.
The executing dispatch reaches `dyn_spg_ts` at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stp2d.f90:288-301`.

One additive `dynspg_ts.F90` writer at `kstp=1081` records all 50 substeps in
the established source order: entry and b/bb histories, midpoint values,
face depths and transports, continuity forcing/divergence/SSH, backward SSH,
pressure gradients, Coriolis/drag trends, slow forcing, exit values, and
post-swap values. It additionally records final `pssh(:,:,Kaa)` after the
compiled finalization at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/dynspg_ts.f90:797-845`.
A same-step additive stage-1 writer records the exact full-step `ssha`, stored
`r1_ht_0`, and resulting `r3ta` around the executing QCO call at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:142-195`.

The run changes only `nn_itend` from 1440 to 1081. It must reproduce the
admitted Round-123 step-1080 restart and step-1081 process record bit for bit.
The writer is refused as non-passive if either differs. The compiled source,
binary, source-card manifest, byte layouts, record counts, and fp64 scalar
kind are fail-closed. Every nonzero shell exit must first print `REFUSE:`.

Frozen acquisition predictions:

1. exactly one external-step record and one QCO-operand record are produced;
2. the step-1080 restart and existing step-1081 process record remain
   byte-identical to Round 123;
3. the QCO record's `ssha` equals the external record's final `pssh` bit for
   bit on every stored cell;
4. replaying `r3ta = ssha*r1_ht_0` is bit-exact on all 600 wet columns.

Any failed identity, missing field, extra byte, non-finite value, or non-bit
QCO replay refutes the record and stops the round. Layout, truncation,
header-step, final-`pssh` one-ULP, QCO one-ULP, and passive-admission plants
must each exit nonzero and print `STATUS PLANT-FIRED` or a named `REFUSE`.

## P1 — production-JIT developed external-step walk

Extend the existing Round-82 external-step walk and reuse Round 136's complete
restart bridge; do not add an isolated model stepper. The production call is
`LatLonCGridOceanModel.step -> self._step_jitted` from the admitted
completed-step-1080 NEMO state. The trace is captured from the existing
barotropic substep seam. An independently compiled ordinary call must return
the same state bit for bit.

For each of 50 substeps, compare every registered boundary in compiled order.
Also compare the final `pssh` directly and the stage-1 QCO multiplication.
Report production JIT as the claim-bearing row; production eager and isolated
record replay are labeled controls and cannot substitute for it. A one-ULP
change to a consumed wet entry SSH and a one-ULP change to recorded final
`pssh` must move a registered row and exit nonzero.

Frozen scientific predictions and falsifiers:

1. model and NEMO final `pssh` differ in exactly 599 of 600 wet columns, the
   same set as Round 136's `q_Kaa`; a different count or set refutes the
   claim that the mismatch is already present before QCO;
2. given each model's own `pssh` and NEMO's stored `r1_ht_0`, the QCO
   multiplication is BIT; any unequal replay cell makes the multiplication,
   not merely its input, a candidate;
3. the first production-JIT non-bit external boundary is in substep 1 at the
   frozen slow momentum forcing (`slow_u` or `slow_v`), after exact entry,
   history, midpoint, continuity, pressure, Coriolis, and drag rows; an earlier
   unequal row or exact slow forcing refutes this ordering;
4. the SSH mismatch first appears no earlier than substep 2, because substep-1
   continuity precedes the first mismatched velocity update. An unequal
   substep-1 continuity SSH refutes that mechanism.

The receipt registers every moved boundary, the first non-bit statement,
the first substep on which each final-`pssh` unequal column appears, and whether
the 599-column set is complete before finalization. Spatial overlap alone is
not ownership.

## P2 — outcome and campaign gates

If final `pssh` is already non-bit, the downstream
`pr3t = pssh*r1_ht_0` statement is exonerated only if its exact-operand replay
is BIT; the walk continues upstream and no ratio rewrite lands. If `pssh` is
BIT but the multiply is not, a candidate may be built only from the cited
compiled association and must pass the full Decisions 43+45 ladder, month,
year, and shared-card gates before landing. No candidate is expected this
round.

A separate read-only Codex command must try to refute the source-card ancestry,
record passivity, byte layout, restart timing, complete restart bridge,
production-JIT status, source ordering, set-equality claim, QCO replay,
plants, and OPEN boundary. A `DO NOT SHIP` verdict blocks the receipt.
Every cited statement must be mapped by the receipt citation gate; its shifted
citation plant must exit nonzero.

No physical card, default, carried production state, stabilizer, NEMO source,
DINO behavior, ORCA2 selector, LOCK_EXCHANGE, OVERFLOW path, or immutable
before arm changes. Expected status is `HELD`; `DECISION_NEEDED` is `NONE`.
The acquisition is run in-round if the sandbox permits `makenemo` and
`mpirun`; otherwise the committed run script is reported under
`ACQUISITION_NEEDED`.
