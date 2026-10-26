# Preregistration — ORCA2 round 91 rung-0 frame admission

Date: 2026-10-01. Base: `a62f67376`. All labels are **independent**: this
round concerns the rung-0 from-rest NEMO hierarchy deck, not the shipped-card
given-entry twin.

## Frozen measurement

Before this file was committed, only the top-level record census was inspected:
the operator-created round-90 directory contains 80 entry/stage frame files,
20 terminal restart shards, one self-describing surface-input record, and no
legacy fixed-schema stage-1 runoff/WZV record. No frame payload, admission JSON,
plant log, restart comparison, or stage-to-stage value has been read.

The round will produce:

1. The committed round-90 admission gate's verdict for all 80 self-describing
   frames and the surface record, including every planted violation.
2. A byte comparison of all 20 terminal restart shards against the admitted
   round-83 rung-0 record.
3. A rank-complete kt=1 stage census of T, S, u, v, and ssh at stage 0 and the
   first later stage that differs, with the compiled rung-0 stage-program call
   separating those frames cited.
4. A disposition of the stage-0 frame as the true rung-0 step-entry state. If
   the record admits, this is the only state eligible to initialize the rung-0
   card and the later given-entry ladder.

## Predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R91-P1 | The operator record passes the existing frame and surface admission gates without modification. | 80 frames parse to physical EOF with the exact rank/step/stage inventory; all frame and surface plants fire; owner-off surface fields are ABSENT. | Any gate refusal: keep its exact failed invariant, do not edit the checker post hoc, and stop for a replacement record or named instrumentation repair. |
| R91-P2 | The round-90 repair is observational. | Every one of the 20 terminal restart shards is byte-identical to its round-83 counterpart. | Any byte moves: **REFUTED**; the instrumentation repair cannot be admitted or used for physics claims. |
| R91-P3 | The rank-complete stage-0 frame at kt=1 is NEMO's true rung-0 step-entry ocean state. | Both rank headers identify the before-level selected by the compiled entry call, T/S are finite, and u/v/ssh carry the initialized entry values before the first stage routine. | Header/level disagreement, missing rank, or an earlier compiled state mutation: **REFUTED**; name the earlier operand stream still required. |
| R91-P4 | The first state movement is between stage 0 and stage 1, after the compiled stage-1 program. | At least one owned wet value differs at stage 1 while stage 0 remains the cited entry snapshot. | Stage 1 is byte-identical: continue in compiled order to stage 2/3; no post-hoc boundary is skipped. |

## Landing and refusal bar

No legoESM physics or hierarchy-card statement is authorized in this round.
The shipped ORCA2 card and its sea-ice `unmeasured_features` tuple remain
immutable. Admission requires every corruption plant to fire, all 80 frames,
and 20/20 byte-identical terminal restarts. A failed prediction is retained as
**REFUTED**. A missing operand stream or failed invariant yields
`ACQUISITION_NEEDED`; a configuration ambiguity yields `DECISION_NEEDED`.
