# ORCA2 hierarchy decks round 17 preregistration — admit rung 3, stage rung 2

Date: 2026-10-02

Scope: repair the checker-only inherited-result path that refused the completed
Decision-83 rung-3 run, admit that existing independent record without
rerunning NEMO, then prepare rung 2's already-authorized one-line
`nn_havtb=1 -> 0` replacement.  No NEMO source, legoESM model, card, recipe,
physics, threshold, run protocol, or admitted scientific record may change.

All run claims in this round are **independent**: NEMO starts from each rung's
own from-rest initialization.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| HD17-P1 failure ownership | The completed rung-3 run failed only because the round-16 checker calls the historical round-9 rung-3 validator, whose inherited rung-6 predicate requires the superseded `nn_havtb=1`; the replacement upper chain separately requires `nn_havtb=0`. | Reusing the historical rung-3 surface-only predicates while retaining the replacement upper-chain validator makes the unchanged record pass after every real plant fires. | Any record, resolved-namelist, frame, restart, month, inventory, or additional checker predicate still fails. |
| HD17-P2 plant binding | The `surface-consequence` and `resolved-havtb` plants remain effective through the separated surface and replacement-upper paths. | Clean validation passes and both plants print `STATUS PLANT-FIRED` with nonzero exit. | Either plant stays green, or the repair skips its owning predicate. |
| HD17-P3 rung-3 admission | The operator-created rung-3 record is complete and finite: 480 self-describing frames, eight month products, two step-240 ocean restart shards, exact-zero surface inputs, and a complete SHA inventory. | The unchanged record admits with the independent label and Decision-83 uniform background. | Any count, finiteness, restart step, deck, zero-input, producer, or inventory check differs. |
| HD17-P4 rung-2 one-line delta | Decision 83 changes only `namzdf.nn_havtb: 1 -> 0` between the superseded and replacement rung-2 decks; the replacement rung-3 to rung-2 boundary remains exactly the already source-resolved GM eddy-induced velocity and mixed-layer eddy module. | Parsed assignment and physical-line diffs show exactly the authorized background line, while the adjacent-rung diff contains only the named rung-3 module assignments and established run-protocol lines. | A second Decision-83 deck assignment changes, or an unrelated adjacent-rung physical assignment appears. |
| HD17-P5 acquisition disposition | No admissible `nn_havtb=0` rung-2 record exists before handoff. | A fail-closed launcher preserves the complete value-1 record and passes all preflight plants for a fresh operator acquisition. | An admissible replacement record already exists, or preflight cannot uniquely pin the old record, repaired recorder, deck, build, inputs, and compiled branch. |

## Required gates

1. Add a direct regression test that distinguishes the historical rung-3
   surface predicates from its superseded inherited rung-6 predicate; prove
   both the surface and replacement-background plants fire.
2. Run every real rung-3 admission plant, then clean admission against the
   existing record.  Do not invoke NEMO.
3. If rung 3 admits, reuse the established hierarchy helpers to build and gate
   rung 2's one-line replacement and fail-closed acquisition launcher.
4. Run the complete hierarchy focused battery, both citation gates and a real
   citation plant, then the permitted ocean-fidelity battery once if no other
   pytest process is active.
5. Request read-only independent review.  If unavailable in the sandbox,
   record that exact verdict.

## Hold conditions

Hold without scientific admission if any rung-3 record predicate fails after
the checker repair.  Stop for a record after staging rung 2.  Stop for a user
decision if rung 2 cannot be expressed by the already-authorized one-line
Decision-83 change while retaining its previously compiled module boundary.
