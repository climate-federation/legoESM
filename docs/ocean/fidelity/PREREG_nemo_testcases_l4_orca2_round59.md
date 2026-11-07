# ORCA2 round 59 preregistration — byte-identical-GYRE UP3 landing

Date frozen: 2026-09-28  
Base: `360ba71cef`  
Cards: `orca2_vector_een_c2` plus every shared executing card  
ORCA2 claim label: **given NEMO's entry** (Decision 52)  
GYRE/OVERFLOW/tank claim label: **independent**, except the direct UP3
statement replay, which is **given NEMO's recorded operands**

## Authorization and one-statement scope

Round 57 named the first non-bit OVERFLOW kt=3 stage-2 UP3 statement: the U
T-face flux differed on 282 / 17,000 contributing faces, comprising four
wet/dry-boundary mask omissions and 278 wet-interior association differences.
Round 58 measured a one-statement candidate that closed that replay to 0 /
17,000 unequal while leaving all 40 ORCA2 checkpoints, 70 GYRE ladder rows,
210 GYRE residual arrays, and 360 GYRE daily snapshots byte-identical.  It was
reverted only because its preregistration incorrectly required a strict GYRE
day-30 improvement even when the candidate did not move GYRE.

Operator note B12 supplies the corrected predicate: strict day-30 improvement
applies only when GYRE moves.  A candidate whose certified GYRE ladder,
day-30/240/360 snapshots, and residual digests are byte-identical passes the
shared-code gate.  This round retries the same measured statement without
changing its implementation.

The executing compiled branch forms masked horizontal curvature at
`OVERFLOW_OMIP_L1_P3_R56UP3/BLD/ppsrc/nemo/dynadv_up3.f90:157-166`, selects
curvature from the sign of the Kmm velocity pair at `:182-192`, and multiplies
the transport sum by the source-ordered face value at `:194-195`.  The
unpatched NEMO 5.0.2 source carries the same statement order at
`src/OCE/DYN/dynadv_up3.F90:145-177`; the cited compiled record is authoritative
for this card.

The candidate changes only the existing NEMO-UP3 same-direction T-point flux
spelling.  It adds no scheme, selector, configuration field, default, carried
state, score domain, threshold, stabiliser, or sea-ice field.  The
Oceananigans UP3 arm and first-order/centred arms retain their existing
arithmetic.  ORCA2's six sea-ice selectors and `unmeasured_features` tuple stay
unchanged at `STOP_SELECTOR_GAP`.  The held QCO/RK candidate remains absent.

Repository and test search found the existing `_up3_reconstruct` production
path, its four call sites, the round-57 header-derived statement gate, the
round-58 measured implementation and landing gate in repository history, and
the existing UP3 unit/selector/differentiability tests.  This round reuses
those exact changes; it adds no second operator or diagnostic copy.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R59-P1 | The clean base reproduces round 57 before editing. | Direct statement gate reports 282 / 17,000 unequal, partitioned as four boundary and 278 wet-interior cells, maximum `0.005124451203774175`; its one-ULP plant adds exactly one refusal. | Any count, partition, maximum, provenance, or plant differs: stop and reconcile before implementation. |
| R59-P2 | The unchanged round-58 source-order candidate closes the direct statement given NEMO's recorded operands. | Earlier curvature/pair/selector rows remain 0 unequal and the production U T-face flux becomes 0 / 17,000 unequal, including empty mismatch partitions. | Any earlier row moves or any flux cell remains unequal: do not land. |
| R59-P3 | The implementation remains one binding, differentiable production statement. | Direct eager/JIT values agree; gradients are finite and non-zero; independent association and omitted-mask controls fail; replacing the production helper makes the binding test fail. | Any eager/JIT mismatch, zero/non-finite gradient, dead control, or revert-green binding test: do not land. |
| R59-P4 | ORCA2 retains its admitted ten-step policy given NEMO's entry. | All 40 checkpoints complete; no AT-BAR row leaves; first non-bit remains kt=1 stage-1 T; every moved row is registered. | Any refusal, AT-BAR loss, earlier first row, or unregistered move: hold. |
| R59-P5 | The corrected two-branch GYRE predicate admits the candidate. | If the trajectory is byte-identical, all 70 ladder rows, 210 residual arrays, and 360 daily snapshots match the base byte-for-byte and day-30/240/360 values are unchanged. If it moves, the full Decision 43/45/55/59 movement predicate passes, including strict day-30 improvement. | Byte-identical branch: any byte or digest moves. Movement branch: any binding Decision 43/45/55/59 row fails. Hold and name the first failure. |
| R59-P6 | The statement does not regress other executing cards. | OVERFLOW has no earlier first-over-bar, no kt=1 AT-BAR loss, and no row worsens beyond its strict 2-ULP bar; LOCK_EXCHANGE, DINO, tanks, generic cards, and card census pass. | Any card refusal or unregistered movement: hold and name the first failing row. |
| R59-P7 | The one statement lands. | R59-P2 through P6 pass; default and round citation gates have zero unmapped citations with plants firing; required test batteries have no new red; independent review has no unresolved blocker. | Any prerequisite fails: revert the model candidate and finish `HELD` with the failing row registered. |

Failed predictions remain in the receipt as **REFUTED**.  No post-hoc pair is
eligible to land in this round.

## Required measurements and landing rule

1. Re-run the round-57 direct statement gate and plant at the frozen base.
2. Reapply the exact round-58 source-order statement, tests, instrument, and
   citation re-anchors; then run the direct statement gate and plant.
3. Re-run the controlled ORCA2 ladder, GYRE ten-step and 360-day gates, and
   compare their arrays and digests against the clean base.  Apply the
   byte-identical branch only if all certified artifacts are exactly equal;
   otherwise apply the standing movement predicate.
4. Run OVERFLOW, LOCK_EXCHANGE, DINO, tank, generic-card, census, focused,
   push-gate-equivalent, and 170-test card batteries named by round 58.
5. Run a separate read-only `codex exec` review of the diff.  If it cannot
   initialize, record `independent review unavailable in-sandbox` exactly.
6. Run default and round receipt citation gates with a real shifted-line plant
   and one `tests/ocean/fidelity -n 12` battery after confirming no other
   pytest battery is active.

## Frozen OPEN

If the statement lands, retry the separately held source-ordered QCO/RK
statement only in a new preregistered round.  After the step-level walk closes,
perform Decision 52's owed **independent-start** ORCA2 ladder and rank
month-scale magnitudes.  Sea ice remains out of scope at `STOP_SELECTOR_GAP`.

ASKED: retry the measured UP3 statement under the corrected byte-identical-GYRE gate.  
UNASKED: none.
