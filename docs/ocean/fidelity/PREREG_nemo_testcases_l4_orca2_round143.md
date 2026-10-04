# Preregistration — ORCA2 round 143 finite-growth walk

Date: 2026-10-04. Base: `312db78c1`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round143`.
Every trajectory value is **independent**: hierarchy rung 0 starts from its own
climatological T/S, zero velocity, and zero sea surface. No Decision-52 entry
operand is used. Sea ice and the card's `unmeasured_features` stay unchanged.

The operator reports that the committed round-141 launcher exited zero. This
round first re-runs its admission checker against the existing files, then
compares NEMO and legoESM at column `(j,i)=(87,159)` and incident faces for
steps 30 through 36 in the recorded source order.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R143-P1 | The existing record is admissible without changing the round-141 checker or files. | The checker prints `PASS_R141_GROWTH_RECORD`; all admission plants still fire. | Reject every science value; repair only a demonstrated checker defect or request a fresh target. |
| R143-P2 | Both NEMO and legoESM are finite at the target through step 35; legoESM first loses finiteness during step 36 while NEMO stays finite. | Source-ordered rows reproduce the round-140 step-36 boundary and contain no earlier non-finite target value. | Retain the result as REFUTED and make the earlier step/value the walk boundary. |
| R143-P3 | The first target row beyond the frozen `2e-10` floor is the step-30 entry sea surface, inherited before that step's barotropic solve. | `abs(legoESM-NEMO) > 2e-10` first occurs at `kt=30`, row `ssh_entry`, while all earlier rows in the registered window are at or below the floor. | Retain REFUTED; the earliest observed step and source-ordered row own the walk. If the crossing predates the record window, write a rank-complete acquisition for the missing earlier steps rather than infer it. |
| R143-P4 | The collapse is already visible in the sea-surface/thickness group before the stage-1 transport group. | At the first over-floor step, an entry or post-barotropic `ssh/r3t` row exceeds the floor before `un_adv/vn_adv/zFu/zFv/zFw`. | Retain REFUTED and walk the first transport row if it precedes the thickness rows. |
| R143-P5 | This is a measurement-only round unless one compiled, one-variable statement closes the named boundary under all standing gates. | No `packages/`, card, deck, carried-state, stabilizer, sea-ice, or `unmeasured_features` change lands without that proof. | Revert the unproved change and finish HELD or STOPPED_FOR_RECORD. |

## Round bar

The analysis must parse the self-describing record, prove exact global/rank
coverage, run CPU fp64 with production JIT, reproduce an unobserved ordinary
trajectory bit-for-bit, and print every registered target row for steps 30..36.
The first row beyond `2e-10` is selected mechanically in compiled source order.
No stabilizer or configuration choice is permitted.
