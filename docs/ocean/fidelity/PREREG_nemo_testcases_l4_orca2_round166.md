# Preregistration — ORCA2 round 166 kt=8 external-substep boundary

Date: 2026-10-07. Frozen base: `a3b92b9d2`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round166/`.

Every ORCA2 number in this round is **independent**: hierarchy rung 0 starts
from its own climatological T/S, zero velocity and zero sea surface. No
given-NEMO-entry rung-7 number is mixed into the table. Sea ice, all six
sea-ice selectors and the shipped card's `unmeasured_features` tuple remain
unchanged.

## Frozen source order and protocol

Round 165 proves that kt=8 enters with finite SSH but that the saved
barotropic after-SSH is already non-finite before stage-1 interpolation. The
compiled rung-0 oracle forms extrapolated velocity/SSH and face depths at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:503-559`, forms the
metric transports at `dynspg_ts.f90:564-570`, and applies their divergence to
SSH at `dynspg_ts.f90:580-595`. The later pressure, Coriolis/drag, velocity and
seven-array association statements are at `dynspg_ts.f90:630-790`.

The round reuses the existing 65-substep production trace registry and the
same complete four-statement private arm: raw reference face depth,
seven-array external-mode association, unmasked V transport and materialised
completed `zhV`. A wrapper requests the already-built trace, publishes only
scalar finite/non-finite summaries through an ordered host callback, and
returns the solver's ordinary state and averages unchanged.

1. Run the complete arm without the observer and require round 165's terminal:
   kt=7 complete, kt=8 stages 1-2 exposed, stage 3 absent, and the exact raw-
   mesh `e3w_int` refusal.
2. Repeat with the passive substep observer. Require the same terminal, then
   report the first substep and source-ordered boundary among entry, midpoint,
   face depths, transports, divergence, after-SSH, pressure/trend, exit
   velocities and exit depths that becomes non-finite.
3. Prove observer passivity by requiring every returned kt=1..7 checkpoint and
   the kt=8 terminal class to match the unobserved run. If the existing NEMO
   record does not cover the named kt=8 substep, write a content-pinned,
   rank-complete, self-describing acquisition and stop for that record; do not
   infer the oracle operand.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R166-P1 | The passive substep observer reproduces round 165's terminal boundary. | Both runs complete kt=7, expose kt=8 stages 1-2, and refuse on the same `raw-mesh e3w_int` invariant before stage 3 returns. | The observer changes any completed checkpoint, advances/retards the terminal, or changes its class. |
| R166-P2 | Within the kt=8 stage-1 external solve, entry/midpoint/face-depth/transport operands remain finite and `after_ssh` is the first non-finite boundary. | The first non-finite row is `after_ssh`, sourced by the compiled transport divergence update at `dynspg_ts.f90:580-595`; its substep and first cell are recorded. | Any earlier source-ordered operand is non-finite, or every substep boundary stays finite. |
| R166-P3 | The existing admitted NEMO external-substep record is insufficient for the named kt=8 boundary. | No rank-complete kt=8 frame covers the named substep, so a new acquisition is emitted and no oracle attribution is claimed. | An admitted self-describing kt=8 frame already covers the boundary and passes its checker. |
| R166-P4 | Localization alone leaves production unchanged and the atomic unit HELD. | No package/configuration change lands; the receipt names either the next recorded-operand walk or the exact missing record. | One cited statement both closes the complete arm and passes every landing gate in this round. |

## Terminal rule

An observer that changes the terminal or any completed checkpoint is rejected.
No stabiliser, clip, configuration choice, carried-state change, sea-ice
change, bar relaxation or partial source-unit landing is permitted. Failed
predictions remain in the receipt.

ASKED choices: continue round 165's passive barotropic walk. UNASKED choices:
empty.
