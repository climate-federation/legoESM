# Preregistration — ORCA2 round 168 kt=8 exit-depth operand split

Date: 2026-10-07. Frozen base: `4adbb02ce`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round168/`.

Every ORCA2 number in this round is **independent**: hierarchy rung 0 starts
from its own climatological T/S, zero velocity and zero sea surface. No
given-NEMO-entry rung-7 number is mixed into the result. Sea ice, all six
sea-ice selectors and the shipped card's `unmeasured_features` tuple remain
unchanged.

## Frozen source order and scope

The admitted round-166 record remains the only oracle input. Its compiled
program advances sea surface from the U/V transports at
`ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:585-595`, constructs
the area-weighted U-face sea surface `zsshu_a` at `dynspg_ts.f90:633-636`,
and forms `hu_e = hu_0 + zsshu_a` at `dynspg_ts.f90:763`. The round-167
registered set is fixed at 41 unique native U faces corresponding to the 42
compact non-finite reciprocals at kt=8 external substep 2.

The walk first scores the carried raw `hu_0` and the produced U-face SSH
separately. It then performs both isolated sum replays: candidate raw depth +
recorded `j002_sshu_a`, and carried raw depth + candidate face SSH. If face
SSH is first non-bit, the walk scores candidate `ssha_e` and replays recorded
`j002_ssha_e` through legoESM's face-average statement. If `ssha_e` is first
non-bit, it continues backward in compiled order through the already exposed
midpoint transports, their U/V differences, divergence, forcing, RHS,
increment and after-SSH update. A row is named only from the first non-bit
statement in this order.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R168-P1 | The raw `hu_0` operand is not the round-167 zero-depth owner. | The carried raw depth is bit-exact on all 41 registered faces and candidate-raw + recorded face SSH reproduces recorded `hu_e` there. | Any registered raw-depth bit differs or the recorded-face-SSH replay fails there. |
| R168-P2 | Candidate `zsshu_a`, not raw `hu_0`, owns the zero exit-depth sum. | Candidate face SSH differs on all registered faces; substituting recorded `j002_sshu_a` closes all 41 exit depths and candidate face SSH reproduces the zero sum with the unchanged raw depth. | Candidate face SSH is exact at a registered face or recorded face SSH does not close the exit depth. |
| R168-P3 | The face-average arithmetic is faithful given NEMO's after-SSH; its input `ssha_e` is already non-bit. | Replaying recorded `j002_ssha_e` through the literal five-operation U-face expression gives recorded `j002_sshu_a` at every registered face, while candidate after-SSH differs there or in their two-cell stencil. | The recorded-after-SSH replay remains non-bit, or candidate after-SSH is exact across every contributing stencil. |
| R168-P4 | The first upstream non-bit row is already present among the recorded transport/continuity operands. | The source-ordered walk names one of transport U/V, U/V difference, divergence, forcing, RHS, increment or after-SSH and reports its magnitude. | Every upstream row is bit-exact, so the exact next missing operand is named without a physics landing. |
| R168-P5 | This remains a measurement-only hold. | No `packages/`, recipe, selector, forcing, carried-state or sea-ice change lands; the adverse atomic halo/V-transport unit remains private. | A cited single source unit closes the walk and passes every Decision-96 landing gate in this round. |

## Controls and terminal rule

The gate must reject independent plants in the admitted record census, the
41-face registry, the source-order registry and each replay predicate. It
must retain full-domain and active-mask bit counts and must prove that every
registered face is exercised by a nonzero raw depth. No stabiliser, clip,
configuration choice, bar relaxation or partial atomic-unit landing is
permitted. Failed predictions remain in the receipt.

ASKED choices: continue round 167's compiled-source operand walk. UNASKED
choices: empty.
