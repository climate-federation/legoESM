# Preregistration — ORCA2 round 122 northern F-grid association

Date: 2026-10-03. Base: `b5c7227e0f`. Every hierarchy-rung-0 number is
**independent** because rung 0 starts from NEMO's own from-rest state. No
configuration choice, forcing, initial state, carried state, stabilizer,
sea-ice selector, or `unmeasured_features` entry may change.

## Admitted boundary and compiled statement

Round 121 found the first northern-V EEN difference at `ff_f`: 1,431
magnitude-unequal executed cells for northeast fraction 1 and northwest
fraction 3, all on global row `j=147`. The record is passive, rank-complete,
and arithmetically sufficient.

The executed deck reads `ff_f` as an F-point field with sign `+1` and
`jpfillcopy` (`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/domhgr.f90:233-237`).
For a T-pivot fold, the compiled double-precision F-point branch overwrites the
northern row from the preceding row with reversed inner-domain longitude
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/lbcnfd.f90:722-747`). Stripping
NEMO's two halos predicts the native-grid association
`north_ff[i] = ff_f[j=146, i=179-i]`, sign `+1`.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R122-P1 | The compiled permutation is the full reverse `179-i`, sourced from row 146 with sign `+1`. | All 180 source indices are unique, span `0..179`, and the mapped values equal NEMO's recorded northern `ff_f` values bitwise. | Preserve **REFUTED**; test no-shift and one-cell-shift alternatives only as controls, not as selectable physics. |
| R122-P2 | One association closes both first boundaries. | NE fraction-1 and NW fraction-3 `ff_f` scores move from 1,431/1,431 to 0/0 bit/magnitude unequal. | Stop at the observed residual; do not change production. |
| R122-P3 | The next source-ordered operands remain `e3f_0vor`. | Candidate first differences are 514 cells for NE and 521 for NW, with the round-121 supports unchanged. | Name the earlier surviving operand and stop there. |
| R122-P4 | The association is shared only by literal EEN cards carrying recorded operands. | Resolved census remains ORCA2, both DINO recipes, VORTEX, and VORTEX_VEC true; GYRE and both tanks false. | Any scope movement refuses the arm. |
| R122-P5 | If the single statement lands, no exact ORCA2 row leaves the bar and GYRE remains within its certified gate. | Rung-0 and rung-7 comparators report no exact-row loss or earlier first debt; GYRE/DINO/tank/card gates pass. | Hold the statement and name the first red row. |

## Measurement and landing bar

The measurement reuses the committed round-120 self-describing reader and
round-121 fraction assembly. It runs production JIT on CPU under fp64/libm,
refuses a dirty or wrongly stamped tree, prints the resolved card census, and
has independent oracle-bit, candidate-bit, permutation-shift, and scope-route
plants. A production edit is eligible only if the mapped numerator is bit-exact
for both paths and every shared landing gate passes. The later `e3f_0vor`,
frozen-mask, northern accumulator/final-scale pair, and 68-cell substep-2 U
residual remain downstream.
