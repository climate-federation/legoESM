# ORCA2 hierarchy decks round 16 preregistration — admit rung 4, stage rung 3

Date: 2026-10-02

Scope: repair the checker-only inherited-result lookup that refused the
completed Decision-83 rung-4 run, admit that existing independent record
without rerunning NEMO, then prepare rung 3's already-authorized one-line
`nn_havtb=1 -> 0` replacement.  No NEMO source, legoESM model, card, recipe,
physics, threshold, run protocol, or admitted scientific record may change.

All run claims in this round are **independent**: NEMO starts from each rung's
own from-rest initialization.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| HD16-P1 failure ownership | The completed rung-4 run failed only because the round-15 checker reads `nn_havtb_uniform` from the rung-5 result's top level even though the round-14 checker stores it in the nested rung-6 result. | A checker-only nested lookup makes the unchanged existing record pass after every real plant fires. | Any record, resolved-namelist, frame, restart, month, inventory, or additional checker predicate still fails. |
| HD16-P2 plant binding | The `resolved-havtb` plant remains effective through the inherited rung-5/rung-6 validator chain. | Clean validation passes and that plant prints `STATUS PLANT-FIRED` with nonzero exit. | The plant stays green, or the repair bypasses the rung-6 result rather than consuming it. |
| HD16-P3 rung-4 admission | The operator-created rung-4 record is complete and finite: 480 self-describing frames, eight month products, two step-240 ocean restart shards, and a complete SHA inventory. | The unchanged record admits with the expected independent label and Decision-83 uniform background. | Any count, finiteness, restart step, deck, producer, or inventory check differs. |
| HD16-P4 rung-3 one-line delta | Decision 83 changes only `namzdf.nn_havtb: 1 -> 0` between the superseded and replacement rung-3 decks; the replacement rung-4 to rung-3 boundary remains exactly the already source-resolved bulk-forcing/restoring/freshwater-budget module. | Parsed assignment and physical-line diffs show exactly the authorized background line, while the adjacent-rung diff contains only the named rung-4 module assignments and run-protocol lines already classified by the rung-3 gate. | A second Decision-83 deck assignment changes, or an unrelated adjacent-rung physical assignment appears. |
| HD16-P5 acquisition disposition | No admissible `nn_havtb=0` rung-3 record exists before handoff. | A fail-closed launcher preserves the complete value-1 record and passes all preflight plants for a fresh operator acquisition. | An admissible replacement record already exists, or preflight cannot uniquely pin the old record, repaired recorder, deck, build, inputs, and compiled branch. |

## Required gates

1. Add a direct regression test for the clean nested lookup and its
   `resolved-havtb` plant; show the plant fires.
2. Run every real rung-4 admission plant, then the clean admission against the
   existing record.  Do not invoke NEMO.
3. If rung 4 admits, reuse the established hierarchy helpers to build and gate
   rung 3's one-line replacement and fail-closed acquisition launcher.
4. Run the complete hierarchy focused battery, both citation gates and a real
   citation plant, then the permitted ocean-fidelity battery once if no other
   pytest process is active.
5. Request read-only independent review.  If unavailable in the sandbox,
   record that exact verdict.

## Hold conditions

Hold without scientific admission if any rung-4 record predicate fails after
the lookup repair.  Stop for a record after staging rung 3.  Stop for a user
decision if rung 3 cannot be expressed by the already-authorized one-line
Decision-83 change while retaining its previously compiled module boundary.
