# Preregistration — ORCA2 round 137 stage-3 FCT/content walk

Date: 2026-10-04. Base: `76c6f95e9`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round137`.
All step-36 values are **independent**: the rung-0 card starts from its own
climatological T/S, zero velocity, and zero sea surface. Given-entry ladder
populations remain separate.

Round 136 proved that the implicit tracer solve first returns non-finite T at
`(j,i,k)=(86,159,0)`, while its pre-ZDF content is already non-finite at level
`k=3`. This round returns to the already-admitted passive stage-3 FCT trace and
walks the exact upstream column cell `(86,159,3)` in compiled source order.
NEMO constructs the two-step upstream predictor at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:495-610`, forms and limits the
antidiffusive flux at `:193-200,260-330,743-938`, and calls `tra_zdf` only
after the explicit tracer RHS is complete at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:734-749`.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R137-P1 | The round-135 passive FCT observer remains admissible at the level-3 target. | Every ordinary state leaf and the independent repeat are bit-identical; the target/source-order/passivity plants fire. | Reject all FCT values and stop at the observer defect. |
| R137-P2 | The first target-cell non-finite lies in the corrected FCT divergence/final RHS, after the low-order midpoint and averaged-upstream fluxes. | `first_u/v/w`, `first_div`, `midpoint`, `average_u/v/w`, `upstream_div`, and `rhs_after_up` are finite at `(86,159,3)`; a later limiter/final-divergence row is first non-finite. | Mark **REFUTED** and retain the earliest measured source row; do not skip it. |
| R137-P3 | The non-finite explicit FCT RHS is the value carried into the pre-ZDF content at `(86,159,3)`. | The FCT trace's final RHS and caller advection content are non-finite at the target, and the pre-ZDF content census reproduces the same target boundary. | Mark **REFUTED** and walk the first finite-to-non-finite association between the two values. |
| R137-P4 | No production statement, configuration, deck, selector, carried state, stabilizer, sea-ice field, or `unmeasured_features` entry changes. | The round is a committed measurement/gate/receipt only; ORCA2 rung-0/rung-7, GYRE, tanks, and DINO retain their certified rows. | Hold at the first moved shared-gate row. |

## Round bar

The first source-ordered target row owns the walk. A statement is named only
from the compiled ORCA2 branch and the admitted passive trace. The distinct
global averaged-upstream-flux overflow at `(87,160,5)` remains separate unless
this target-cell census mechanically joins the two. No physics landing, card
change, stabilizer, threshold relaxation, or sea-ice change is authorized.
