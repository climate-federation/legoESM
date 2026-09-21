# Preregistration — NEMO testcase L2 GYRE round 140

Date: 2026-09-21

Incoming lane tip: `63ef82a7f7f5d6519df2cb4f5a6a6e1d5c8cecae`

This document is frozen before parsing the acquired Round-139 scientific
operands or running a developed-state production comparison. Evidence will
live under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round140/`.

Round 138 established that the day-180 production-JIT external solve is BIT
through pressure, Coriolis, and drag, then first differs at the frozen slow
momentum forcing: 580 / 580 wet U faces differ by at most
`4.2854247978022983e-13`, and 570 / 570 wet V faces differ by at most
`4.4333086294645174e-13`. Round 139 committed the minimum operand writer but
stopped before acquisition. The operator has now reported
`ROUND139_DEVELOPED_SLOW_FORCING_READY`; no scientific field from that record
has been read for this preregistration.

## P0 — fail-closed Round-139 readmission

The executing compiled target copies the incoming `Ue_rhs`/`Ve_rhs`, calls
`dyn_cor_2D`, writes its returned operands, subtracts them through the native
masks, and writes the final fields at
`GYRE_OMIP_L2_P3_SM_R139SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:292-348`.
The four-point Coriolis statements are
`GYRE_OMIP_L2_P3_SM_R139SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:1361-1382`.

Before any comparison, admit the exact 33,852-byte record at producer commit
`63ef82a7f7f5d6519df2cb4f5a6a6e1d5c8cecae`. Require its stamp, fp64 header,
six-field census, no trailing bytes, and bit-exact replay. Require the inherited
step-1080 restart, step-1081 process stream, external stream, and QCO stream to
remain byte-identical to Round 137. Require the acquisition completion marker,
closed output manifest, and the header, truncation, replay-ULP, record-stamp,
and passive-admission plant markers. Any failure stops the round without a
scientific claim.

## P1 — developed production-JIT operand split

Extend the existing Round-83 slow-forcing walk; do not create another model
stepper. Reuse the Round-138 bridge from NEMO's exact completed-step-1080
restart and its certified step-1081 forcing. Drive
`LatLonCGridOceanModel.step -> self._step_jitted` with the existing live-stage
operand hook and register, in compiled order:

1. incoming `Ue_rhs`, then incoming `Ve_rhs`;
2. returned initializing Coriolis U, then V;
3. native U mask, then native V mask; and
4. final frozen forcing U, then V.

Every row reports scored cells, cells unequal, maximum absolute difference,
RMS difference, and first unequal index with signed-zero sensitivity. The
traced production step and an independently compiled ordinary production step
must return bit-identical pytrees. A registry plant that removes one row and a
one-ULP change to one consumed wet incoming-U value must each exit nonzero with
`STATUS PLANT-FIRED`.

Frozen predictions inherited from Round 139:

1. incoming U is the first non-bit operand and differs on 580 / 580 wet faces;
2. incoming V is next and differs on 570 / 570 wet faces;
3. both initializing Coriolis fields and both masks are BIT;
4. replacing only the incoming pair by NEMO's pair makes the production-JIT
   final frozen forcing BIT on all 580 U and 570 V wet faces; and
5. the ordinary unhooked step reproduces Round 138's final U/V rows and the
   live-stage hook does not move any returned-state cell.

An exact incoming pair, a non-bit earlier Coriolis or mask input, any different
face count, a non-exact record replay, a moved returned state, or a non-exact
directed arm is `REFUTED` and remains in the receipt. The final subtraction by
itself never establishes an upstream owner.

## P2 — conditional incoming-RHS continuation

If P1 confirms incoming forcing as first, continue the same compiled-order
walk. NEMO builds the three-dimensional momentum RHS through HPG, LDF, VOR,
WZV, KEG, and ZAD at
`GYRE_OMIP_L2_P3_SM_R139SLOW/BLD/ppsrc/nemo/stp2d.f90:139-175`, depth-averages
it at `GYRE_OMIP_L2_P3_SM_R139SLOW/BLD/ppsrc/nemo/stp2d.f90:192-204`, then
adds drag and wind at
`GYRE_OMIP_L2_P3_SM_R139SLOW/BLD/ppsrc/nemo/stp2d.f90:210-230` before passing
the completed pair to the external solve at
`GYRE_OMIP_L2_P3_SM_R139SLOW/BLD/ppsrc/nemo/stp2d.f90:290-301`.

The acquired record contains no internal boundary before the completed
incoming pair. Therefore, only after P1 confirms that branch, prepare and run
a new passive GYRE target that adds a step-1081 instance of the existing
`NEMO_L2_SLOW_2` field list without changing or removing any executing
statement. It records the 3-D RHS operands, completed depth mean, post-drag
value, wind operands, and post-wind incoming pair. The run must preserve every
Round-139 inherited output byte for byte, carry a new target name, print a
named refusal before every nonzero exit, use bash timing, syntax-prove the
source card, and run record/replay/passivity plants.

Frozen conditional prediction: the geometry, masks, and reference-depth
reciprocals are BIT, while the first upstream non-bit row is the completed
three-dimensional `Krhs` momentum RHS (U before V). An exact `Krhs`, an earlier
geometry/mask row, or a first mismatch in depth averaging, drag, or wind
`REFUTES` this prediction. If `Krhs` is first but the new record does not split
its operator accumulations, the receipt names the minimum next passive record;
it does not infer an HPG/LDF/VOR/KEG/ZAD owner.

If P1 instead finds Coriolis first, do not acquire the RHS record. Continue
through the existing Coriolis operand registry and name its first direct
non-bit input or statement. If acquisition is blocked, report
`STOPPED_FOR_RECORD` with the committed run script.

## P3 — landing, review, citations, and scope

No landing is expected unless one shared statement is directly proven wrong
under production JIT and can clear Decisions 43 and 45. A diagnostic outcome
changes no production physics, so the ladder, month, year, DINO,
LOCK_EXCHANGE, OVERFLOW, tanks, and ORCA2 are not rerun. ORCA2 remains
`UNMEASURED-WITH-SPEC`: its ocean-only card needs the same native operand
registry before an identity claim.

A separate read-only Codex pass must try to refute record ancestry, source
order, passivity, restart timing, production-JIT status, ordinary-state
identity, registry completeness, directed substitution, conditional record
scope, and every plant. A `DO NOT SHIP` verdict blocks the round. Every
compiled-source citation is mapped by the receipt citation gate, and its
shifted-citation plant must exit nonzero.

No configuration, default, carried state, scheme, stabilizer, canonical NEMO
source, or immutable before arm changes. Expected status is `HELD`, or
`STOPPED_FOR_RECORD` only if the conditional admitted record cannot be
produced. `DECISION_NEEDED` is `NONE`.
