# Preregistration — ORCA2 round 170 kt=8 slow-forcing producer walk

Date: 2026-10-08. Frozen base: `665afb4ce`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round170/`.

Every ORCA2 number in this round is **independent**: hierarchy rung 0 starts
from its own climatological T/S, zero velocity and zero sea surface. No
given-NEMO-entry rung-7 number is mixed into the result. Sea ice, all six
sea-ice selectors and the shipped card's `unmeasured_features` tuple remain
unchanged.

## Frozen record, source order and scope

The only new oracle input is the operator-admitted round-169 record under
`orca2_rounds/round169/acquisition/orca2_rung0_slow8_ranked_10step_np2`.
Its checker reports two self-describing rank records, exactly-once global
coverage and 20 terminal restarts byte-identical to the round-166 baseline.

The record executable first publishes the three-dimensional face thickness,
momentum RHS and mask operands, then forms NEMO's vector-invariant vertical
average at
`ORCA2_OMIP_L4_R169SLOW8/BLD/ppsrc/nemo/stp2d.f90:215-227`. It calls
`dyn_drg_init` and publishes the post-drag boundary at
`ORCA2_OMIP_L4_R169SLOW8/BLD/ppsrc/nemo/stp2d.f90:244-253`, then applies the
explicit surface-stress increment in written multiplication order and
publishes the post-wind boundary at
`ORCA2_OMIP_L4_R169SLOW8/BLD/ppsrc/nemo/stp2d.f90:255-274`. The rung-0
namelist has no atmospheric-pressure, embedded-ice or wave-load forcing, so
the post-wind and final boundaries must be identical; the measurement will
instantiate and print those resolved switches rather than infer them.

The walk scores both U and V in that compiled order. For the depth reduction
it compares each operand, replays the literal ordered reduction from all
recorded operands, then substitutes one candidate operand at a time. Drag and
wind are scored at their recorded boundaries and replayed with recorded
operands where the production hooks expose them. A statement is named only
at the first finite-to-explosive or first non-bit source boundary.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R170-P1 | The operator's round-169 record is admissible without reinterpretation. | Two rank files give exactly-once coverage, all 23 self-described fields have the registered shapes, and all 20 terminal restarts are byte-identical to the additions-only baseline. | Any coverage, header, payload, field registry or restart comparison fails. |
| R170-P2 | NEMO's literal vertical-reduction arithmetic is bit-exact against its recorded post-depth boundary. | Replaying `SUM(e3*rhs*mask)*r1_h0` in source order from recorded operands gives 0 unequal active U and V cells. | Any active bit remains unequal. |
| R170-P3 | The explosive candidate slow forcing is carried into `stp2d` by the three-dimensional momentum RHS, not created by the vertical reduction, drag or wind statements. | The candidate RHS is the first non-bit operand and already has explosive magnitude; substituting recorded RHS closes the candidate reduction, while recorded reduction/drag/wind replays remain exact. | Candidate RHS is bit-exact or finite-scale, recorded RHS does not close the reduction, or an earlier thickness/mask/reciprocal or later drag/wind statement is the first finite-to-explosive boundary. |
| R170-P4 | Rung 0 executes no live wind/load increment after drag. | The instantiated card prints `surface_stress_implicit=True` but zero stress for the zero-flux forcing, and the admitted `drag`, `wind` and `final` arrays are bit-identical for both faces; all resolved load switches are false. | Any recorded or candidate post-drag boundary moves through wind/load, or any load switch resolves true. |
| R170-P5 | The round remains measurement-only and the complete halo unit remains private. | No `packages/`, card, selector, forcing, carried-state, stabiliser or sea-ice change lands. | A cited statement closes the complete unit and passes every Decision-96 landing gate in this round. |

## Controls and terminal rule

The gate must reject independent plants in rank placement, the recorded
literal reduction, one-variable RHS substitution, source ordering and the
zero-wind/load invariant. Failed predictions remain in the report. The
candidate observer must reproduce the unobserved kt=8 trajectory bit-for-bit.
No stabiliser, clip, bar relaxation, configuration choice or partial atomic
halo-unit landing is permitted.

If the three-dimensional RHS is the first unresolved boundary and the
admitted files do not split its compiled stage-1 terms at kt=8, write a new
fail-closed, self-describing per-rank acquisition for precisely that boundary
and stop with `ACQUISITION_NEEDED`; do not attribute from candidate-only
diagnostics.

ASKED choices: continue round 169's compiled-source independent rung-0 walk.
UNASKED choices: empty.
