# ORCA2 round 198 preregistration — pressure-gradient and trend cancelling unit

Date: 2026-10-09. Frozen base: `f8c5a1b57`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round198/`.

Every number is **independent hierarchy rung 0**. The card starts from its
corrected climatological T/S, zero velocity and zero sea surface. No
given-NEMO-entry or rung-10 number may enter this round. The shipped ORCA2 card,
sea ice, its six selectors and its `unmeasured_features` tuple remain unchanged.
This is an offline, measurement-only split: no executable observer, model or
configuration change, state change, stabiliser or tolerance change is allowed.

## Frozen source order and protocol

Round 197 proved that the substep-2 vector V update closes bit-for-bit only when
both recorded `zv_spg` and completed `zv_trd` replace the candidate operands.
Neither half lands independently. This round keeps that pair atomic while it
walks each half in NEMO's compiled order.

NEMO first forms the half-step-back sea surface from four coefficients and four
SSH levels at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:642-650`, then evaluates
the V surface-pressure gradient from its north difference and `r1_e2v` at
`dynspg_ts.f90:652-660`. It next calls the EEN barotropic Coriolis operator at
`dynspg_ts.f90:663-670`, and, because this deck has no tides or wetting/drying,
adds the explicit drag product at `dynspg_ts.f90:680-702`. The completed pair
is consumed at `dynspg_ts.f90:715-727`.

Extend round 197's already-passive measurement context and reuse the production
literal pressure-gradient and EEN helpers. Do not transcribe a second boundary
or Coriolis operator. For substep 2 compare, in source order: the four
back-interpolation coefficients; after/current/two older SSH operands; each
weighted term and partial sum; completed back-interpolated SSH; the north
difference; `zldg`; `r1_e2v`; and completed `zv_spg`. Substitute recorded
operands one at a time and cumulatively. Then compare midpoint U, the eight
literal EEN coefficients, completed `cor_v`, drag entry V, inverse depth and
drag coefficient/product, and completed `zv_trd`. Reconstruct NEMO's drag only
as its recorded `trd_v - cor_v`, and prove that reconstruction is non-vacuous.
Finally replace the source-exact pressure and trend halves together in round
197's vector-update replay and require the same bit-exact target.

The gate must parse the admitted round-96 record from both ranks exactly once,
require every stream and shape, reproduce round 197's 1,226-cell pressure and
16,055-cell trend debts, and preserve the passive traced/untraced state. Plants
must reorder the source registry, perturb one nonzero back coefficient by one
ULP, make the derived drag identity vacuous, and remove one required stream.
Every plant must refuse before a number is cited.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R198-P1 | The admitted record is sufficient; no acquisition is needed. | Every registered SSH, pressure, Coriolis and drag operand exists on both ranks and both offline replays close their recorded outputs. | Missing target or replay residual: **REFUTED**; write a fail-closed acquisition and stop. |
| R198-P2 | The first non-bit pressure input is the back-interpolated SSH, carried from the already-non-bit after-SSH. | Coefficients and older registered SSH levels are exact before the first non-bit SSH operand/back-SSH row. | An earlier coefficient or history is non-bit: **REFUTED**; stop at that input. |
| R198-P3 | Given NEMO's recorded back-interpolated SSH, legoESM's literal V-gradient statement is bit-exact. | Substituting only recorded `j002_sshp2_bck` closes `zv_spg` on all 26,640 cells. | Residual remains: **REFUTED**; continue through `zldg`, `r1_e2v`, subtraction and product association and name the first non-bit operand/statement. |
| R198-P4 | The EEN Coriolis half is exact on active faces; the explicit drag/addition is the first magnitude debt in `zv_trd`. | `cor_v` is exact on active V faces before a non-bit drag operand, drag product or completed sum. | Active `cor_v` is non-bit: **REFUTED**; stop at its first source-ordered midpoint/coefficient operand. |
| R198-P5 | Making both pressure and completed trend source-exact closes the vector update bit-for-bit, while either half alone does not. | The cumulative pair reaches 0 / 26,640 unequal and both single halves remain non-exact. | Pair residual: **REFUTED**; the record/replay mapping is incomplete and no attribution is reported. |
| R198-P6 | Controls bind. | Registry, coefficient-bit, drag-identity and missing-stream plants each refuse. | Any plant stays green: invalid instrument; report no operand claim. |

No trajectory census follows a held operand result. If both halves become
source-exact and the atomic private unit is eligible for a trajectory score,
that score belongs to the next preregistered round; this round does not land
physics.

ASKED choices: round 197's OPEN source-ordered split of the pressure-gradient
and EEN-plus-drag cancelling pair.  
UNASKED choices: empty.
