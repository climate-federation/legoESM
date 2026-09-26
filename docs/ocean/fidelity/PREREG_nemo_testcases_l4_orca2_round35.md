# NEMO testcase Lane 4 — ORCA2 card round 35 preregistration

Date: 2026-09-26

Parent: `d28759f36fe032526eb78a253f593c49e56663e2`

Status: **PREREGISTERED BEFORE ROUND-35 SCIENTIFIC SCORING.**

Round 34 selected NEMO's carried `hf_0`, but the given-entry
lateral-diffusion replay remains non-bit at
`3.181628207426175e-09` / `2.9702048395431957e-09` m/s2.  The next
compiled-order statement is the F-point free-surface ratio `r3f`.

Direct-operator results are **given NEMO's entry (kt=2 recorded state)**.
Trajectory results are **independent with Decision-52 SSH**.  These labels
will not be mixed in one table.  The six sea-ice selectors and the card's
`unmeasured_features` tuple remain frozen.  Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round35/`.

## Compiled statements

The executing build materialises `e1e2f=e1f*e2f`, then stores
`r1_e1e2f=1/e1e2f`, at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domhgr.f90:154-156`.
The executing QCO routine forms `r3f` by multiplying the bracketed four-cell
surface sum, stored `r1_hf_0`, and stored `r1_e1e2f`, in that order, at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domqco.f90:273-286`.
The executing lateral-diffusion consumer reads the resulting live F
thickness at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:123`.

legoESM currently forms the same F-area product but divides the partially
formed `r3f` numerator by it.  This round tests only that arithmetic boundary;
all values, masks, indexing, forcing, and state remain fixed.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R35-P1 | The first remaining differing `r3f` statement is division by the F-area product instead of multiplication by NEMO's stored reciprocal. | The F-area product itself is bit-identical to `e1f*e2f`, while a one-variable source-order arm changes at least one `r3f` value and no earlier operand. | The area product differs, the source-order arm is bit-identical, or an earlier operand differs. |
| R35-P2 | Replacing only that division with NEMO's stored-reciprocal construction closes the given-entry LDF replay bit-exactly. | U and V each have zero unequal scored cells, while the parent reproduces Round 34's registered maxima and the plant fires. | Either field retains an unplanted unequal cell, the parent maxima or score sets move, or the plant is silent. |
| R35-P3 | If implemented, the independent ORCA2 ladder completes kt=1..10, loses no exact/AT-BAR row, and keeps the first non-bit statement at kt=1 stage-1 temperature. | 40/40 checkpoints, empty AT-BAR-loss list, every moved row registered, and unchanged first statement. | A refusal, an unregistered move, an exact row leaving its bar, or an earlier first statement. |
| R35-P4 | GYRE remains byte-identical through ten steps and 30 days. | Zero differing certified rows, `np.array_equal` residual arrays, and byte-identical daily snapshots. | Any content or snapshot difference; because GYRE executes this shared statement, that result stops the landing for a user decision. |
| R35-P5 | DINO, lock-exchange, overflow, focused, citation, and push gates remain green, and every planted violation fires. | Required shared-card tests pass and each plant exits nonzero. | Any new failure or passing plant. |

Failed predictions remain **REFUTED** and are not rewritten.

## Order and landing rule

1. Commit this preregistration before any new round-35 score.
2. Extend the existing round-32/34 replay; do not build a second LDF replay.
3. Expose the parent and reciprocal-order `r3f` values and substitute only the
   compiled reciprocal spelling.
4. Implement only if R35-P1 is confirmed.  Land only if R35-P2 reaches zero,
   ORCA2 loses no exact/AT-BAR row, GYRE is byte-identical, shared-card gates
   pass, and every plant fires.  Otherwise retain the measurement and hold.

No stabilizer, clipping, configuration choice, carried-state change, NEMO
source edit, sea-ice edit, score change, or acquisition is authorized.

## Choices

ASKED: Round 34's OPEN section authorizes the compiled-order one-variable
walk through the remaining `r3f`/metric operands.

UNASKED: none.
