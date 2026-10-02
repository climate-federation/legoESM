# ORCA2 hierarchy decks round 19 preregistration — admit rung 1 and close the NEMO-side hierarchy

Date: 2026-10-02

Scope: admit the operator-completed Decision-83 rung-1 replacement record,
recheck its exact boundary against the main lane's pinned rung-0 deck, and
inventory the complete NEMO-side hierarchy.  No NEMO source, legoESM model,
card, recipe, physics, threshold, run protocol, or scientific record may
change.

All run claims are **independent**: NEMO starts from each rung's own from-rest
initialization.  The operator's acquisition and its admission output predate
this preregistration; their existing result is not used as a fresh prediction.
This round independently reruns the committed mechanical checks before reading
or reporting their detailed payload results.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| HD19-P1 rung-1 plants | Every named rung-1 record plant fires through the committed round-18 gate. | Each plant exits nonzero and prints `STATUS PLANT-FIRED`. | Any plant stays green, crashes without the status marker, or reaches a different predicate. |
| HD19-P2 rung-1 admission | The operator-created rung-1 record is complete and finite: 480 self-describing frames, exact-zero surface input, eight month products, two step-240 ocean restart shards, uniform background, and a complete SHA inventory. | Clean validation returns `PASS_RUNG1_HAVTB0_RECORD` without rerunning NEMO. | Any frame, payload, input, month, restart, resolved-namelist, producer, or inventory predicate differs. |
| HD19-P3 authorized replacement | The admitted replacement differs from the preserved rung-1 record only by Decision 83's `namzdf.nn_havtb: 1 -> 0`. | Parsed and physical-line diffs contain exactly that row; the preserved record and admission retain their pinned hashes. | Any second assignment changes or any preserved evidence differs. |
| HD19-P4 rung boundaries | Replacement rung 2 to rung 1 remains BBL/geothermal-only, and replacement rung 1 to pinned main rung 0 remains damping plus run-protocol/file-name rows only. | The committed complete semantic diffs equal those two registered sets. | Any unrelated physical assignment appears or a named module selector is absent. |
| HD19-P5 hierarchy completion | Rungs 1 through 10 each have an admitted independent 240-step record at the authorized deck; rungs 1 through 6 use `nn_havtb=0`, while rungs 7 through 10 retain the shipped `nn_havtb=1`. | Every admission file reports its rung-specific PASS status and every canonical record/inventory exists; rung 1's clean gate closes the last replacement record. | Any rung lacks canonical admitted evidence, has the wrong background selector, or fails its own committed clean gate. |

## Required gates

1. Run every real round-18 rung-1 record plant, then clean validation against
   the existing record.  Do not invoke NEMO.
2. Re-run the exact rung-2/rung-1 and rung-1/main-rung-0 semantic diffs using
   the committed gate; do not infer either boundary from prose.
3. Inventory canonical admission files and records for rungs 1 through 10,
   preserving all superseded `nn_havtb=1` evidence for rungs 1 through 6.
4. Run the complete hierarchy focused battery, both citation gates and a real
   citation plant, then the permitted ocean-fidelity battery once if no other
   pytest process is active.
5. Request a separate `codex exec --sandbox read-only` review of the round
   diff.  If it cannot initialize, record
   `independent review unavailable in-sandbox`.

## Hold conditions

Hold without completion if any record plant, clean admission, rung boundary,
or rung inventory predicate fails.  Request no acquisition unless the existing
rung-1 run is incomplete.  Request no user decision unless the measured
boundary exceeds Decisions 80 and 83.
