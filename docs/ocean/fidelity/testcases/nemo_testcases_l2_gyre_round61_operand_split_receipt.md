# GYRE L2 round 61 — kt=2 carried-operand split

## Round 61 — result

Date: 2026-09-11. Preregistered parent: `4baf0d012bda`. Final clean
measurement: `6934ecec8ec1`; admitted producer: `1a695951be13`. Evidence:
`round61/kt2_operand_split_with_velocity_v6.json`.

The preregistered 744-row premise is **REFUTED on the restored tip**: the
actual all-model baseline is 5,325/17,400 unequal, not 744. Therefore Rule 10
forbids naming a sole operand, and Rule 12 forbids a physics/configuration
change. The 744 row belonged to the held round-60 association arm and cannot
be combined with restored-tip carries.

| kt=2 operand | unequal | max abs |
|---|---:|---:|
| sh2 | 17,400 | 4.103e-8 |
| rn2 / rn2b | 13,299 / 13,299 | 1.746e-17 / 1.746e-17 |
| entry en | 654 | 3.469e-18 |
| entry avm | 5,159 | 2.776e-17 |
| entry avt | 4,905 | 1.735e-18 |
| entry dissl | 15,387 | 1.214e-17 |
| taum | 263 | 2.776e-17 |
| e3t_Kmm / e3w_Kmm / masks | 0 / 0 / 0 | 0 / 0 / 0 |

Single NEMO substitutions (compiled first-consumption order; format
`operand: unequal`) were `taum:5325`, `rn2b:5325`, `e3w:5325`, `sh2:5211`,
`avm:5325`, `e3t:5325`, `dissl:5325`, `en:5325`, `avt:5325`, `rn2:4155`,
`masks:5325`. None reached zero; the complete cumulative walk ended at 3,606.
Thus the primary sh2 prediction and the entry-carry fallback are both
**REFUTED** under their frozen falsifiers.

The compiled call creates `sh2` before the TKE closure at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfphy.f90:317-337`. Its live branch
uses carried avm, Kmm×Kbb face differences, qco face-W metrics, masks, and the
coastal combine at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90:83-114`. Replaying that
shared statement with NEMO velocities is 0/17,400 unequal. Replacing only Kmm
with legoESM gives 17,400 unequal, max 7.588e-19; however the current state has
no Kbb face-velocity carry and selects centered-current shear. This explains
why a 744 attribution is not mechanically available; it does not authorize a
new carry or card choice.

No production fix landed. Consequently GYRE kt3 T/S, day 30,
LOCK_EXCHANGE, OVERFLOW, DINO, and ORCA2 reruns are not applicable. The
one-ULP e3t operand plant exited nonzero (1 unequal); focused tests pass 11/11.

ASKED: split, source trace, gated eligibility decision. UNASKED: adding a Kbb
carry or changing GYRE shear configuration. Open: acquire a same-program
actual-carry record whose baseline is 744, or separately preregister the
state/configuration expansion.
