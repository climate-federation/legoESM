# ORCA2 round 187 preregistration — month growth record admission and walk

Date: 2026-10-08. Base: `6120711eeb07a772cccd9a1d71b264cfff8d220d`.
Every number in this round is **independent**: the rung-0 card starts from its
own corrected, bit-exact initial state. No rung-7 or given-entry result is mixed
into the tables.

## Frozen scope

Round 187 first audits the existing round-186 restart files against the
compiled restart writer, repairs the admission gate's time convention only if
the payload is sound, and runs `run.sh --admit-existing`; it never rebuilds or
re-runs the existing NEMO targets. A restart with an empty field or a time
header inconsistent with the compiled executed path is not admissible. If the
terminal checkpoint was overwritten, the nine intact steps remain citable but
the record is partial, and a fresh fail-closed acquisition is written under a
new target name.

The independent production month is then replayed unchanged and scored at
each admissible checkpoint. The fixed growth selector follows round 174: the
maximum absolute T/S/u/v/SSH error is compared with the previous checkpoint's
maximum (the first checkpoint uses the `2e-10` floor), and growth is a strict
ratio greater than 10. At the first selected checkpoint, the existing
rank-complete stage frames are used in stage order; operators are replayed
offline from ordinary completed states only. No in-executable observer is
authorised.

The search before implementation found and reuses the round-186 record gate,
month runner, restart readers and score helpers, the round-174 growth selector,
and the rung-0 stage-frame ladder. No second trajectory runner or scientific
implementation is authorised.

No model file, physical configuration, forcing, carried state, stabiliser,
public selector, sea-ice selector, or `unmeasured_features` entry changes.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R187-P1 | The existing step-95 files are ordinary NEMO restarts whose stored time follows the compiled writer's convention and whose T/S/u/v/SSH payload is complete. | Both ranks and twins have the compiled expected time value, non-empty finite fields, and are bit-identical; all ten steps admit without a NEMO rerun. | A zero/other time value, empty/non-finite field, or twin mismatch: **REFUTED**. Retain the exact header/payload census, admit only the intact subset, and write a new-target acquisition rather than weakening the gate. |
| R187-P2 | The nine earlier checkpoints (10..90) are sound even if step 95 is not. | Every rank/twin has `kt == filename step`, complete finite T/S/u/v/SSH, twin array identity, and step-10 calibration identity. | Any earlier checkpoint fails: **REFUTED** and stop the month walk at that exact record defect. |
| R187-P3 | The first coarse >10x error-growth boundary is step 10 relative to the fixed floor. | The mechanically derived table selects step 10, with the field, magnitude, cell, and ratio printed. | Any later checkpoint or no crossing: **REFUTED**; retain the table and walk the selected result instead. |
| R187-P4 | Within the selected checkpoint, the first stage leaving the floor is stage 1 and the first source-ordered operator is HPG. | Rank-complete stage rows select stage 1; offline operator replay selects HPG; the replay reproduces the completed candidate accumulator within the fixed floor before any operand claim. | A different stage/operator or a replay that does not close: **REFUTED**; name the measured boundary or stop as instrument-invalid. |
| R187-P5 | No scientific landing is eligible this round. | The walk stops on a missing operand record, an unresolved cancelling unit, or the incomplete step-95 record; packages remain unchanged. | A cited one-statement owner with all operands and full landing gates available: **REFUTED**; only then run the full shared landing protocol. |
| R187-P6 | Every new classifier is fail-closed. | Plants for terminal truncation, earlier-step header, missing rank, one-ULP twin movement, step-10 calibration, growth selection, stage order, and operator closure all fire. | Any plant stays green: instrument invalid; quote no scientific verdict. |

The compiled source citations and exact executed branch are recorded only after
the source audit. Failed predictions remain in the receipt. The round ends
`STOPPED_FOR_RECORD` if step 95 is not recoverable from the existing record,
even if the intact checkpoints support a provisional earlier boundary.
