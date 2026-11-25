# Preregistration — ORCA2 round 156 complete-association finite-growth bracket

Date: 2026-10-05. Frozen base: `b0f6f56c1`.
Every ORCA2 number in this round is **independent**: hierarchy rung 0 starts
from its own climatological T/S, zero velocity, and zero sea surface. No
configuration, initial state, forcing, carried-state form, stabiliser,
sea-ice selector, or `unmeasured_features` entry may change.

## Source order and question

The corrected round-155 private arm applies NEMO's complete external-mode
source unit: raw face depths at compiled
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:538-545`, the stored
unmasked V transport at `dynspg_ts.f90:568-570`, its later continuity use at
`dynspg_ts.f90:584-591`, and the seven-field boundary association at
`dynspg_ts.f90:761-779`. It is exact at kt=1 substep 2 but makes the rung-0
trajectory non-finite at kt=8 stage-1 T.

After the external solve, NEMO first consumes its returned sea surface and
depth-mean velocities while constructing the RK stage time levels at compiled
`stprk3_stg.f90:150-180`, then consumes the completed transport average and
carried depth-mean velocity in `zub/zvb` at `stprk3_stg.f90:265-283`. This
round asks where the complete arm first departs from unchanged production and
which one of the seven associated fields first carries that departure through
the compiled consumers. Partial associations are measurement arms only; the
seven-field call remains the source unit.

## Frozen protocol

1. Run unchanged production and the corrected complete private arm in lockstep
   from the same rung-0 entry, forcing, precision policy, JIT path, and card.
   At every kt=1..10 entry and stage boundary, compare arrays directly between
   the two executions before comparing either with NEMO. Record finiteness,
   unequal-cell count, maximum absolute difference, and argmax for T/S/u/v/ssh
   plus carried uu_b/vv_b.
2. Require the unchanged arm to reproduce the admitted round-155 rung-0
   ladder. Locate the first finite complete-arm departure and retain every
   later row through the first non-finite boundary. No NaN-hiding reduction is
   permitted.
3. Starting at that first departure, activate exactly one associated field at
   a time in compiled call order: U velocity, V velocity, U depth, V depth,
   U inverse depth, V inverse depth, sea surface. Each arm retains raw reference
   depth, unmasked V transport, and V-transport materialisation. Compare each
   partial arm with unchanged production at the same boundary and run it only
   far enough to determine whether it reproduces the complete arm's first
   finite departure or its kt=8 refusal.
4. Production remains unchanged in this round. Any later landing still owes
   both ORCA2 ladders, the salinity veto, GYRE's full shared gate, DINO,
   VORTEX, tanks, generic cards, citations, and the push gate.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R156-P1 | The new pair instrument is passive on the production arm. | Its production trajectory is array-identical to round 155 through kt=10 and retains first debt at kt=1 stage-1 T. | Any production row moves, refuses, or changes first debt. |
| R156-P2 | The corrected complete arm first departs finitely before its kt=8 non-finite boundary. | A finite arm-vs-production row is non-bit at kt<=7, with the exact kt/stage/field recorded. | All compared rows are exact until the first non-finite row, or any oracle/reference row is non-finite. |
| R156-P3 | The first moved stage boundary is downstream of the external solve: u, v, ssh, uu_b, or vv_b moves before T or S. | The earliest unequal row belongs to that set and T/S remain exact at the immediately preceding boundary. | T or S is the earliest unequal row, or the departure predates the first external solve. |
| R156-P4 | At least one one-field association arm is non-vacuous and distinguishes the complete arm from production at the first finite boundary. | A one-field arm moves at least one bit there while a planted no-op field stays exact. | Every one-field arm stays exact, every arm duplicates the complete result, or the control cannot fail. |
| R156-P5 | The complete arm reproduces round 155's terminal boundary. | First non-finite is kt=8 stage-1 T. | The boundary moves or the arm completes kt=1..10. |

## Controls and terminal rule

The comparison reduction must detect signed-zero and one-ULP plants and must
refuse candidate or reference NaNs before any maximum/RMS reduction. A field
selector outside the seven-name registry must refuse. A plant that requests a
field whose associated value equals production at the measured cell must be
rejected as vacuous, not credited as a control.

This round is **HELD** after naming the first finite boundary and its first
one-field carrier. No production physics lands from a partial association, and
no candidate that becomes non-finite is promoted. If the complete arm's first
finite departure cannot be observed without changing its returned trajectory,
the round stops with the instrument defect named.

ASKED choices: none. UNASKED choices: empty.
