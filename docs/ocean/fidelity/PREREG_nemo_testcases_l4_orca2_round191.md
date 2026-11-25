# ORCA2 round 191 preregistration — exact-forcing external-mode walk

Date: 2026-10-09. Base: `7b7c8371be36339d60cab8facde34eb4b4507d64`.
Every trajectory number is **independent hierarchy rung 0**: the card starts
from its corrected initial state, not from NEMO's recorded entry. The shipped
rung-10 card, sea ice and its `unmeasured_features` tuple remain untouched.

## Frozen scope and statistic

Continue round 190's OPEN downstream of the exact completed U/V slow forcing.
Reuse the admitted round-96 rank-complete 65-substep record and round 178's
four-arm offline walker. Extend that walker only to publish its already-computed
source-ordered rows for the `slow_only` and `slow_and_history` arms. No new
executable observer, model path, configuration, threshold, field, weighting or
NEMO acquisition is authorised.

The walk scores the complete recorded domain in compiled `dynspg_ts` order:
entry forcing and carried histories; midpoint U/V/SSH and face depths; metric
transports; continuity SSH; transport accumulators and face SSH; backward SSH,
pressure gradient, Coriolis, completed trends, exit U/V/depth/inverses; then the
weighted external-mode endpoint. It stops at the first row whose maximum
absolute difference exceeds the frozen `2e-10` floor. Bit identity is uint64
identity, so signed zero remains visible but does not alone cross the floor.

The exact-slow-forcing arm replaces only the completed U/V forcing by the
recorded NEMO values. The history arm additionally replaces only NEMO's six
carried U/V/SSH before/before-before arrays. This is an offline replay from the
passive completed state; the traced and untraced pure solver endpoints must be
array-identical before any scientific row is read.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R191-P1 | The round-96 record and round-178 instrument remain admissible on the round-190 tip. | Both rank shards cover 148x180 exactly once, declare 65 substeps, every passivity row is bit-exact, and all inherited plants fire. | Any census, header, passivity or plant failure: instrument invalid; quote no science and stop for the record/tool repair. |
| R191-P2 | Exact slow forcing moves the first over-floor boundary downstream to substep-1 `ssh_after`. | Every exact-forcing source row through substep-1 U/V metric transport is at the floor; `ssh_after` is the first strict over-floor row. | Any earlier row: **REFUTED**; name that row and stop there without changing order or floor. |
| R191-P3 | The six carried external histories are still a null arm. | `slow_only` and `slow_and_history` trace digests and endpoint arrays are bit-identical. | Any moved bit: **REFUTED**; histories own the first changed row and must be split in written U-b/U-bb/V-b/V-bb/SSH-b/SSH-bb order. |
| R191-P4 | The first over-floor `ssh_after` is produced by NEMO's continuity recurrence from a non-bit transport operand, not by forcing or history. | Exact forcing/history leave the boundary unchanged; the last exact prefix and first debt reproduce at substep 1 in `dynspg_ts.f90:564-591`. | If the histories close it, R191-P3 is refuted; if all recorded transport operands are exact but `ssh_after` differs, name the recurrence association itself as the next statement. |
| R191-P5 | No standalone statement lands this round. | The first named downstream unit is still compensated under the ten-step ladder or needs an operand split. | A cited one-statement arm becomes bit-exact and passes Decision 96 plus every shared gate: land it and report all moved rows. |
| R191-P6 | All controls are non-vacuous. | Rank-placement, record-bit, source-order, arm-identity, endpoint-ULP and exact-arm row-selection plants each refuse. | Any plant stays green: invalid instrument; report no statement claim. |

If no complete cited unit passes the landing predicate, the round ends
**HELD** with the exact first downstream row and its next offline operand split.
