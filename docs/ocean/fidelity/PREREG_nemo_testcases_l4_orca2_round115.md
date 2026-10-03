# Preregistration — ORCA2 round 115 EEN product signed-zero walk

Date: 2026-10-03. Base: `239ee70072c8d2178bd254be3581aa25a51cc4ce`.
Scope is one hierarchy-rung-0 arithmetic boundary, measured **given NEMO's
recorded entry**. No configuration, forcing, initial state, carried state,
threshold, stabilizer, sea-ice selector, or `unmeasured_features` entry may
change.

## Executed oracle path

Round 114 closed every recorded operand, denominator, quotient, and ordered
sum in `zpvo_nw`. The compiled rung-0 U loop next stores the product of the
live U thickness, live V thickness, neighboring V mask, and `zpvo_nw`, then
records the accumulator before and after the same source recurrence
(`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1263-1270`).
Round 109 measured two signed-zero-only differences in the stored product,
1,618 in the accumulator before, and 5,195 after, but correctly stopped at the
then-earlier non-bit quotient.

The source statement contains one left-associated four-factor product. The
first arm will compare the current replay, which materializes each binary
product, with the literal source expression materialized only at the stored
assignment. No downstream accumulator convention may be changed in that arm.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R115-P1 | With the round-114 associations active, `mbku`, `zpvo_nw`, both live thicknesses, and the neighboring mask are bit-exact; the stored product is first with exactly two signed-zero-only differences and no magnitude difference. | The rank-complete per-level gate reproduces those counts and names `term_nw` first. | Any earlier bit or moved census means the baseline changed; stop and reconcile before interpreting an arm. |
| R115-P2 | NEMO's single stored product assignment, without extra materialization between its four written factors, closes both product sign bits. | The literal one-statement arm makes `term_nw` bit-exact while every preceding row remains exact. | Any remaining product bit refutes this interpretation; retain the result and name the first discriminating operand/sign association. |
| R115-P3 | Product closure does not by itself close the carried recurrence: `acc_before` remains the next non-bit row. | `term_nw` is exact and `acc_before` still has at least one signed-zero difference. | If both accumulator rows also close, record that result but do not infer an addition owner without a planted recurrence control. |
| R115-P4 | The statement executes on ORCA2, VORTEX, VORTEX_VEC, and the two NEMO-faithful DINO recipes; GYRE and the flux-form tanks do not execute it. | Resolved-card routing equals this set and its planted mutation fires. | Any additional card executes: add its binding gate before a landing. |

## Measurement and landing bar

Extend the admitted round-107 per-level record and the round-114 exact
fraction association. Compare only executed levels on both ranks under
production JIT on CPU with fp64/x64/libm. Oracle-bit, candidate-bit, and route
plants must fire. The gate must print every operand's shape and dtype before
indexing it.

This round may land a model statement only if the recorded product and its
downstream recurrence are bit-exact given NEMO's recorded operands on every
executing card, both ORCA2 ladders satisfy their frozen predicates, and all
shared-card, citation, and push gates pass. Otherwise it is HELD at the first
non-bit accumulator row. No fold, scale, later-substep, configuration, or
independent-month change is in scope.
