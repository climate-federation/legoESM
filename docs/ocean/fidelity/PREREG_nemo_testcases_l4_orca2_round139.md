# Preregistration — ORCA2 round 139 FCT bound-input walk

Date: 2026-10-04. Base: `3ad32e934`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round139`.
All step-36 values are **independent**: the rung-0 card starts from its own
climatological T/S, zero velocity, and zero sea surface. Given-entry ladder
populations remain separate.

Round 138 proved that the first non-finite adjacent beta operand at
`(j,i,k)=(87,159,3)` is the seven-neighbour `zup` maximum. This round extends
the admitted private passive FCT trace with the seven source-ordered `zbup`
inputs and the selected input's `pbef`/`paft` pair. The compiled ORCA2 source
builds `zbup = MERGE(MAX(pbef,paft),-zbig,tmask==1)` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:816-845` and consumes centre,
west, east, south, north, above, below at `:853-856`.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R139-P1 | The new stencil trace is passive and source ordered. | Its two ordinary FCT outputs and every ordinary step-state leaf are bit-identical to separately compiled unobserved runs; registry, passivity, source-order, association, and plant controls fire. | Reject every stencil number and stop at the observer defect. |
| R139-P2 | The first non-finite `zbup` member is east, `(87,160,3)`. | The seven-member census is finite through centre/west and first non-finite at east. | Mark **REFUTED** and retain the actual earliest source-ordered member. |
| R139-P3 | The selected member's `pbef` is finite and `paft` is non-finite. | The selected coordinate has finite Kbb/base and non-finite provisional low-order update; `MAX(pbef,paft)` reproduces the selected `zbup`. | Mark **REFUTED** and walk the actual first non-finite input or association. |
| R139-P4 | No production statement, configuration, deck, selector, carried state, stabilizer, sea-ice field, or `unmeasured_features` entry changes. | Only a private default-off trace, its gate/tests, receipt, and citation map change; all standing shared-card gates retain their certified rows. | Hold at the first moved shared-gate row. |

## Round bar

The first source-ordered non-finite stencil member and then its first
non-finite `pbef`/`paft` input own the walk. A statement is named only from the
compiled ORCA2 branch and an admitted passive trace. Round 135's distinct
averaged-upstream-flux overflow remains separate unless the coordinate/input
association joins it mechanically. No physics landing, card change,
stabilizer, threshold relaxation, or sea-ice change is authorized.
