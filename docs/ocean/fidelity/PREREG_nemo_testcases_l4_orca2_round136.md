# Preregistration — ORCA2 round 136 shared-control repair and ZDF walk

Date: 2026-10-04. Base: `37bb6855e`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round136`.
All step-36 numbers in this round are **independent**, starting from the
rung-0 card's own climatological T/S, zero velocity, and zero sea surface.
No given-entry and independent populations will be mixed.

The round first clears round 135's inherited shared-gate blocker. The existing
VORTEX control exposes a live stage-1 RHS in one compiled hook program and
feeds that host-materialized array into a different hook program. The merged
tree now gives those two programs different fusion/rounding, so that comparison
does not test an identity substitution in one arithmetic graph. The replacement
control will apply an identity transform to the live RHS at the existing
override boundary in one graph and pair it with a non-identity plant.

Only after that control is green, the independent headline cell
`(j,i,k)=(86,159,0)` is walked after stage-3 FCT. The compiled rung-0 program
calls `tra_ldf` before `tra_zdf` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:734-749`. The implicit solver
selects `avt` at `trazdf.f90:180-216`, builds its tridiagonal at `:218-235`,
factors it at `:268-273`, forms the forward RHS at `:283-291`, and back-solves
at `:293-299`.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R136-P1 | Round 135's VORTEX red is a cross-program known-answer defect, not a moved production statement. | An identity transform of the live stage-1 RHS in the same traced program reproduces every ordinary state leaf bit-for-bit; a `1e-7` relative transform moves the step. | Mark **REFUTED** and retain the first unequal leaf; do not proceed to the ORCA2 walk. |
| R136-P2 | The repaired private control is inert at defaults. | VORTEX/GYRE/ORCA2 production paths construct no transform, and the certified ladders move zero rows versus base. | No landing; name the first moved row. |
| R136-P3 | The headline T cell stays finite through the complete stage-3 explicit content and first becomes non-finite inside the literal implicit tracer solve. | The existing source-ordered trace is finite at the target through pre-ZDF content and has a first non-finite row among matrix factor, forward RHS, or back-solve/solution. | Mark **REFUTED** and name the observed earlier or later boundary without skipping it. |
| R136-P4 | The first target-cell non-finite inside ZDF is in the forward RHS/recurrence rather than coefficient construction. | `effective_K`, `e3t`, `e3w`, wet mask, and all three matrix diagonals are finite at the target; `content_T` or a later recurrence is the first non-finite row. | Mark **REFUTED** and retain the first source-ordered non-finite coefficient row. |
| R136-P5 | The ZDF trace is an admissible observer. | Its `state_after` is bit-identical to an ordinary step-36 replay in all ordinary leaves, including NaN payloads; a trace-field plant makes the checker refuse. | Reject every ZDF number and stop at the observer defect. |

## Round bar

No configuration, deck, selector, carried-state definition, stabilizer,
threshold, sea-ice field, or `unmeasured_features` entry may change. A model
file change lands only if the repaired exact control, ORCA2 rung-0 and rung-7
ladders, the shared GYRE gate, citation gates and plants, focused tests, and the
required fidelity battery all satisfy their standing predicates. Otherwise
the round is HELD at the first red row.
