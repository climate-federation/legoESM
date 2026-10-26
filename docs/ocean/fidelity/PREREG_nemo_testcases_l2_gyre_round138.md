# Preregistration — NEMO testcase L2 GYRE round 138

Date: 2026-09-21

Incoming lane tip: `4be746b6f8022672458a8349624cbb2f65b497f2`

This document is frozen before parsing the Round-137 developed external record
against a legoESM production step. Evidence will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round138/`.

Round 136 found the first recorded developed-state tracer discrepancy before
advection: `q_Kbb` was BIT, while `q_Kmm` and `q_Kaa` differed in the same 599
of 600 wet columns. Round 137 acquired the missing developed external solve.
The operator reported `ROUND137_DEVELOPED_EXTERNAL_READY`; its scientific
predictions remain the frozen predictions below until this round measures the
legoESM production closure.

## P0 — fail-closed record readmission

The as-built target's compiled dispatch calls the external solver at
`GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/stp2d.f90:291-298`. The executing
external program initializes its frozen SSH and momentum forcing at
`GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90:289-325`, advances
the 50 substeps at
`GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90:462-844`, and
finalizes `pssh(:,:,Kaa)` at
`GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90:857-890`.

Before any model comparison, re-admit the exact 10,444,796-byte external
record and 22,508-byte QCO record at producer commit
`4be746b6f8022672458a8349624cbb2f65b497f2`. Require their stamps, fp64
headers, exact field census, no trailing bytes, every record-local replay,
step-1080 restart and step-1081 process passivity identities, and all eight
embedded Round-137 plant markers. The external final `pssh`, final substep
SSH, and stage-1 `ssha` must be BIT; the recorded NEMO replay
`r3ta = ssha*r1_ht_0` must be BIT. Any failure stops the round without a
scientific claim.

## P1 — one developed production step and complete external table

Extend the existing Round-82 external-step walk; do not create another
stepper. Reuse Round 136's admitted step-1080 restart bridge, including T/S,
u/v, independent depth means, six absolute barotropic histories, SSH, TKE
state and coefficients, and the restart's pre-stage QCO scratch. Drive
`LatLonCGridOceanModel.step -> self._step_jitted` with the certified step-1081
forcing and capture the existing shared barotropic substep trace. An
independently compiled ordinary production call must return a bit-identical
state.

For all 50 substeps, score every registered boundary in the compiled order:
entry and histories; midpoint extrapolation
(`dynspg_ts.f90:474-518`); face depths, transports, continuity and fresh SSH
(`dynspg_ts.f90:529-591`); backward SSH, pressure, Coriolis, drag and frozen
forcing (`dynspg_ts.f90:631-684`); the velocity update
(`dynspg_ts.f90:697-760`); and the history rotation
(`dynspg_ts.f90:810-840`). Report cells unequal, maximum absolute difference,
RMS difference, and first unequal index for every row. Signed zero counts as
non-bit. A registry plant removing one boundary must fail.

Frozen predictions inherited from Round 137:

1. final model `pssh` differs from NEMO in exactly 599 of 600 wet columns,
   with exact set equality to Round 136's `q_Kaa` unequal-column set;
2. the first production-JIT non-bit external boundary is substep 1
   `slow_u` or `slow_v`; every entry, history, midpoint, continuity, pressure,
   Coriolis and drag boundary before it is BIT;
3. SSH is BIT through substep-1 continuity and first becomes non-bit no earlier
   than substep 2;
4. every final-`pssh` unequal column has an explicitly registered first
   appearance substep, and all 599 have appeared before finalization.

A different count or set, an earlier boundary, an exact slow-forcing pair, an
unequal substep-1 SSH, or a column first appearing only during finalization is
`REFUTED` and remains in the receipt.

## P2 — production QCO exact-input proof

The stage program calls QCO with the completed external `ssha` at
`GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/stprk3_stg.f90:176-205`; the
compiled T-point statement is
`GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/domqco.f90:237-258`.

Run two private, production-JIT, one-field directed arms through the same
step-entry closure: stage-1 QCO receives (a) legoESM's captured final `pssh`
and (b) NEMO's recorded final `pssh`, with every other input unchanged. The
ordinary, externally traced, and QCO-directed returned states are compared so
the observer cannot silently alter the external result.

Frozen prediction: the T-point QCO result is BIT against `pssh*r1_ht_0` for
each arm, and the NEMO-`pssh` arm is BIT against recorded NEMO `r3ta` on all
600 wet columns. A non-bit exact-input arm makes the multiplication the
candidate; otherwise it exonerates that statement and leaves the first
external non-bit operand upstream. A one-ULP change to a consumed wet entry
SSH and a one-ULP change to the recorded final `pssh` must each move a
registered row and exit nonzero with `STATUS PLANT-FIRED`.

## P3 — verdict, citations, and scope

No landing is expected. If the first non-bit row is frozen slow momentum
forcing, the receipt names the executing assignment and the first demonstrated
non-bit operand, but does not infer which upstream momentum operator produced
it without an admitted boundary. The OPEN item must name the next existing
record or the minimum passive acquisition needed to split that operand.

A separate read-only Codex pass must try to refute record passivity, source
ordering, restart timing, full state mapping, production-JIT status, ordinary
state identity, all 50 rows, set equality, QCO exact-input proof, and plants.
A `DO NOT SHIP` verdict blocks the receipt. Every compiled-source citation is
mapped by the receipt citation gate and its shifted-citation plant must exit
nonzero.

No production physics, card, default, carried state, stabilizer, NEMO source,
immutable before arm, DINO behavior, LOCK_EXCHANGE, OVERFLOW, tank, or ORCA2
path changes. Expected status is `HELD`; `ACQUISITION_NEEDED` and
`DECISION_NEEDED` are both `NONE`.
