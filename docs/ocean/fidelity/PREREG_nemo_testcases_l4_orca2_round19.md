# NEMO testcase Lane 4 — ORCA2 card round 19 preregistration

Date: 2026-09-25

Parent: `16608cd9a31edaec93a0227944d411e4d0b1b8f2`

Status: **PREREGISTERED BEFORE ROUND-19 SCIENTIFIC SCORING.**

The operator completed round 18's acquisition, but its admission refused both
rank streams at 2,042,100 bytes against an expected 2,057,588 bytes.  This
instrument reconciliation precedes preregistration; no recorded ocean operand
has yet been scored.  The record's compiled source declares eight written
arrays on the full 94 by 152 local domain and `zu_frc` alone on the 90 by 148
interior data domain.  That mixed extent predicts 2,042,100 bytes exactly.
The corrected acquisition admission changes only this count and will be run
against the existing record; there will be no rebuild or rerun.

Every solver number in this round is labelled **given NEMO's entry**.  The
whole-card ladder remains separately labelled **independent with Decision-52
SSH**.  The six sea-ice selectors and the card's `unmeasured_features` tuple
are frozen.  Decisions 54, 57, and 58 remain pending and untouched.

## Compiled statements and record

The existing record is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round18/acquisition/orca1ice_u_history_ranked_np2`.
The executing source is the record's compiled
`ORCA2_ORCA1ICE_OMIP_L4_R18UHIST/BLD/ppsrc/nemo/dynspg_ts.f90`.
It declares the full-domain and interior update operands at lines 180–182,
opens and identifies one rank-tagged stream at lines 458–467, records the
history inputs and midpoint at lines 501–513, evaluates the vector update at
lines 688–711, exchanges its result at lines 754–770, records the update
operands and exchanged exit at lines 814–822, and closes the stream at lines
866–870.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R19-P1 | The existing run is admissible without rebuild or rerun. | Both rank streams are exactly 2,042,100 bytes; their headers name distinct ranks and the first two substeps; run-completion and split rank markers pass. | Any different byte count, header, rank, substep, completion stamp, or marker. |
| R19-P2 | The inherited boundary reproduces before the U-history record is used. | Substep 1 remains bit-exact in the canonical stream; substep-2 `continuity_du` remains first non-bit at 64 / 8,794 wet T cells, maximum `7.705384632572532e-07`; direct substep-2 midpoint `ua_e` remains non-bit on 64 / 128 adjacent faces, maximum `1.2223159767330016e-15`. | Any changed order, count, or maximum; stop for instrument drift. |
| R19-P3 | The recorded midpoint statement replays its own target exactly. | For both substeps, `(za1*un_e + za2*ub_e) + za3*ubb_e` is bit-exact against recorded midpoint `ua_e` on the scored rank-0 window. | Any replay difference; stop because the record cannot support a statement walk. |
| R19-P4 | No source-ordered U field is non-bit before the substep-1 exchanged exit. | On the 128 faces adjacent to the inherited continuity support, substep-1 coefficients, `un_e`, `ub_e`, `ubb_e`, midpoint `ua_e`, `rDt_e`, `zu_spg`, `zu_trd`, and `ssumask` are bit-exact; all comparable owned `zu_frc` cells are also bit-exact. | The first earlier non-bit field owns the walk and later predictions are not promoted. |
| R19-P5 | The first direct non-bit boundary is substep-1 post-exchange `ua_e`, and it becomes substep-2 `un_e` and midpoint `ua_e`. | The exchanged exit differs on the rank-boundary support; the next substep's recorded `un_e` is bit-identical to that exit and its forward midpoint replay is bit-exact. | The exit is exact, the mismatch appears before exchange, or history rotation does not preserve it. |
| R19-P6 | The existing record can localize the birth to NEMO's exchange boundary but cannot attribute an exchange statement. | The recorded vector expression replays the owned update exactly, while the non-bit support is outside `zu_frc`'s recorded interior extent; stop for a pre/post-exchange record rather than infer halo arithmetic. | A fully recorded earlier operand or replay differs on the support; name that earlier statement instead. |
| R19-P7 | Admission and arithmetic controls are non-vacuous. | A swapped-rank header, one-ULP midpoint target, and one-ULP exchanged exit each make their corresponding checks fail. | Any plant stays green. |
| R19-P8 | No production statement lands without a single cited shared-model statement becoming bit-exact under the full landing gate. | Otherwise no `packages/` diff; the independent ORCA2 ladder remains `LADDER_MEASURED`, with kt=10 entry T maximum `3.9430791763114783` on 430,552 cells. | Any ungated model change, earlier ladder boundary, moved unregistered row, or changed GYRE output. |

Failed predictions remain **REFUTED** in the receipt; none is rewritten after
measurement.  The walk always stops at the first non-bit statement in compiled
source order.

## Controls and stop rules

- Admission uses the existing run only.  No rebuild, rerun, or artifact repair
  is allowed.
- Rank-local arrays are mapped from recorded layout metadata and the compiled
  declarations; no halo value is inferred from a neighboring field.
- The midpoint and vector expressions use the compiled association order.
- `zu_frc` is scored only on its declared interior extent.  Its absent halo is
  never treated as zero or as evidence.
- Any production change requires the full ORCA2 and GYRE landing gates.
- The receipt citation gate must pass, and a rigid two-line shift of every
  rendered compiled citation must fail.
- No configuration choice, carried-state change, stabilizer, sea-ice change,
  or action on pending Decisions 54, 57, or 58 is allowed.

## Choices

ASKED: reconcile the U-history byte count against the compiled writer, admit
the existing record, and walk histories, coefficients, and the ordered vector
update.

UNASKED: none.
