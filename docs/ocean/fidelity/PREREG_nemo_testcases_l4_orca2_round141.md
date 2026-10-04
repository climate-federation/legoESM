# Preregistration — ORCA2 round 141 finite-magnitude growth record

Date: 2026-10-04. Base: `770095c16`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round141`.
Every trajectory value in this round is **independent**: hierarchy rung 0
starts from its own climatological T/S, zero velocity, and zero sea surface.
Given-entry ladder populations remain separate.

Round 140 proved that the stage-3 upstream predictor at wet T cell
`(j,i,k)=(87,159,4)` remains finite through a numerator of
`3.815621519141334e307`, then overflows only when divided by an after thickness
of `0.00573471208449003 m`.  The division is the messenger.  This round first
acquires the missing NEMO step-30..36 entry and stage-1 transport record, then
compares the same source-ordered quantities at the target column and its
incident faces before taking any arm.

The compiled rung-0 program calls the split-explicit solve before RK3 stage 1
at `ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/stprk3.f90:201-215`.  The solver
starts from `ssh`, `uu_b`, `vv_b`, and the live `r3u/r3v` thickness ratios at
`dynspg_ts.f90:357-375`, and produces the carried external velocities, sea
surface, and depth-mean transports at `dynspg_ts.f90:888-960`.  Stage-1 tracer
transport then consumes the face thicknesses, velocities, and thickness ratios
at `traadv.f90:170-235`, followed by `wzv` and the vertical transport at
`traadv.f90:267-316`.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R141-P1 | The admitted rung-0 records contain no NEMO step-30..36 entry/barotropic/stage-1 transport stream. | A content census finds terminal step 240 plus step 1..10 restarts/frames only; the new acquisition is required. | Name the existing complete stream and admit it instead of running NEMO. |
| R141-P2 | legoESM first differs from NEMO by more than the `2e-10` floor no later than step 30 at the target-column source-order ledger. | The first unequal row is at step `<=30`. | Mark **REFUTED** and retain the actual first step in 30..36, or report that the earlier step remains unmeasured and acquire it before attribution. |
| R141-P3 | The first over-floor quantity is in the sea-surface/barotropic group before the stage-1 `zFu/zFv/zFw` group. | Entry `ssh/r3t` or the split-explicit external fields first exceed the floor. | Mark **REFUTED** and walk the actual first source-ordered stage-1 transport operand. |
| R141-P4 | The NEMO recorder is additions-only. | Both ranks emit one self-describing record per step 30..36, all parser/header/name/payload plants fire, and kt=10 terminal restarts are byte-identical to the admitted rung-0 build. | Reject every recorded number and repair the instrument under a fresh target. |
| R141-P5 | No model, card, deck physics, selector, carried state, stabilizer, sea-ice field, or `unmeasured_features` entry changes. | The round contains only the committed recorder/acquisition/checker, diagnostics, receipt, and citation map. | Hold at the first production or shared-gate movement. |

## Round bar

The first step and first source-ordered quantity exceeding the frozen floor own
the next walk.  The acquisition must be rank-complete, self-describing, and
content-pinned; its checker derives names and payload lengths from each header.
No clipping, extra damping, configuration choice, Decision-94 action, sea-ice
change, or inference from legoESM alone is authorized.
