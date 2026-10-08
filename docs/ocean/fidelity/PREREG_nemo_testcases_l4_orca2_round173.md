# Preregistration — ORCA2 round 173 kt=8 HPG operand walk

Date: 2026-10-08. Frozen base: `b4fbf34f8`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round173/`.

Every scientific number in this round is **independent**: hierarchy rung 0
starts from its own climatological T/S, zero velocity and zero sea surface. No
given-NEMO-entry rung-7 number is mixed into the result. Sea ice, the six
sea-ice selectors and the shipped ORCA2 card's `unmeasured_features` tuple are
unchanged.

## Frozen record and compiled order

The oracle input is the operator-admitted rank-complete record at
`orca2_rounds/round172/acquisition/orca2_rung0_hpg8_ranked_10step_np2`.
Admission must be repeated from both self-describing headers and must again
prove exactly-once 148 x 180 coverage plus byte-identical kt=1..10 terminal
restarts against the round-170 producer. No scientific row is emitted if that
admission or any of its six plants fails.

The executing rung-0 program forms `rhd` with `eos`, then calls `hpg_sco` first
in the three-dimensional RHS at
`ORCA2_OMIP_L4_R172HPG8/BLD/ppsrc/nemo/stp2d.f90:145-149`. The compiled HPG
forms the surface `zhpi/zhpj`, `zuap/zvap` and their sums at
`ORCA2_OMIP_L4_R172HPG8/BLD/ppsrc/nemo/dynhpg.f90:386-408`, then advances the
same boundaries level by level at
`ORCA2_OMIP_L4_R172HPG8/BLD/ppsrc/nemo/dynhpg.f90:411-440`.

The registered walk order is:

1. `rhd`;
2. `e3w(Kmm)`;
3. `gdept_z0(Kmm)`;
4. `r1_e1u` / `r1_e2v`;
5. `zhpi_u` / `zhpi_v`;
6. `zuap_u` / `zuap_v`;
7. `sum_u` / `sum_v`.

Each boundary is scored on its own active T/U/V domain over the exactly-once
assembled global field. U precedes V within a paired row. A difference on a
dry or redundant face cannot select the first boundary.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R173-P1 | The operator-produced HPG8 record is admissible without reinterpretation. | Both rank records parse from their headers, cover the global domain exactly once, all eleven registered fields are finite on owned cells, and all twenty restarts are byte-identical to the round-170 producer. | Any header, payload, field registry, placement, finiteness, coverage or restart comparison fails. |
| R173-P2 | The record is an unperturbed statement instrument. | Replaying the recorded operands through the already-certified literal HPG implementation reproduces every recorded `zhpi`, `zuap` and sum field bit-for-bit on active cells. | Any active-cell bit differs; no candidate comparison is emitted. |
| R173-P3 | `rhd` is the first non-bit kt=8 HPG operand and already carries the explosive scale. | Candidate `rhd` differs first in the registered order and its maximum error is at least `1e20`; all earlier registered rows (none) are exact. | `rhd` is exact, its error stays below `1e20`, or an admission/self-replay prerequisite fails. |
| R173-P4 | Replacing only candidate `rhd` by NEMO's recorded `rhd` removes the HPG explosion. | With all candidate geometry and metrics unchanged, the literal HPG U/V sums both fall below `1e20` and their maximum error decreases by at least `1e12`. | Either sum remains explosive or improves by less than `1e12`. |
| R173-P5 | This is measurement-only unless a single source statement becomes eligible under the complete B57 atomic-unit gates. | No `packages/`, card, configuration, carried-state, stabiliser or sea-ice change lands; a non-bit input routes the next walk to its producer. | Physics/configuration changes without the full atomic-unit gates, or any stabiliser is added. |

Failed predictions remain in the receipt. The first non-bit selector is
independent of the explosive classifier: a finite-scale earlier difference
still owns the walk.

## Controls and terminal rule

The gate must reject planted rank placement, record self-replay, first-boundary
selection and explosive classification changes. A one-ULP active-cell plant
must move the selected row. Candidate and oracle arrays, geometry and metrics
must print `float64`; the JAX policy is fp64/libm and the backend CPU.

If the first non-bit row is an operand, stop at that producer boundary and
name the next source-ordered acquisition or existing passive record needed to
split it. Do not infer a statement below a non-bit operand. If all operands are
bit-exact but a recorded statement is not, only that cited statement may be
considered for the B57 atomic unit. No configuration choice, bar relaxation or
partial halo-unit landing is permitted.

ASKED choices: continue round 172's independent rung-0 HPG walk.  
UNASKED choices: empty.

## Instrument correction after the first refused run

The first committed measurement reached kt=8 and refused before writing a
scientific report: the separately compiled candidate-input reconstruction did
not reproduce the existing offline HPG component boundary bit-for-bit. Thus
the intended operand walk is not yet an admitted instrument and no value from
that run is citable. R173-P2's record-side replay remains unclassified because
the candidate-side identity prerequisite stops first.

The next run first prints the exact U/V identity rows while retaining the same
hard refusal. If the residual is a layout error, correct the association and
repeat. If the separately compiled graph itself changes arithmetic, replace it
with an operand exposure from the already-admitted standalone component graph
and require the exposed and unexposed HPG U/V outputs to be array-identical
before reading any operand. The frozen walk order, predictions, thresholds and
terminal rule above do not change.
