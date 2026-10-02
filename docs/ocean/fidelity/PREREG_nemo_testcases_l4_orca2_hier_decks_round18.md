# ORCA2 hierarchy decks round 18 preregistration — admit rung 2, stage rung 1

Date: 2026-10-02

Scope: admit the operator-completed Decision-83 rung-2 replacement record,
then prepare rung 1's already-authorized one-line `nn_havtb=1 -> 0`
replacement.  No NEMO source, legoESM model, card, recipe, physics, threshold,
run protocol, or admitted scientific record may change.

All run claims in this round are **independent**: NEMO starts from each rung's
own from-rest initialization.  The 05:30 top-level acquisition log predates
the rung-2 record and admission written at 06:41 and is not evidence for this
round.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| HD18-P1 rung-2 admission | The operator-created rung-2 record is complete and finite: 480 self-describing frames, exact-zero surface input, eight month products, two step-240 ocean restart shards, and a complete SHA inventory. | Every real plant fires and clean admission returns `PASS_RUNG2_HAVTB0_RECORD` without rerunning NEMO. | Any frame, payload, input, month, restart, resolved-namelist, producer, or inventory predicate differs. |
| HD18-P2 inherited routing | Rung 2's resolved checks use the replacement rung-3 upper chain and the rung-2-only GM/MLE block, never the superseded full rung-2 chain. | Existing routing tests pass and the `resolved-havtb` and `gm-mle-consequence` plants both fire. | A superseded inherited predicate is called, or either owning plant stays green. |
| HD18-P3 rung-1 one-line replacement | Decision 83 changes only `namzdf.nn_havtb: 1 -> 0` between the superseded and replacement rung-1 decks. | Parsed assignment and physical-line diffs contain exactly that authorized row. | Any second assignment or physical line changes. |
| HD18-P4 rung-1 module boundary | With both adjacent rungs at `nn_havtb=0`, replacement rung 2 to replacement rung 1 differs only by the already source-resolved BBL and geothermal selectors. | The complete parsed boundary equals the admitted value-1 rung-1 boundary: `ln_trabbl` and `ln_trabbc` true to false, with run-protocol lines unchanged. | Any unrelated physical assignment appears or either named selector is absent. |
| HD18-P5 acquisition disposition | No admissible `nn_havtb=0` rung-1 record exists before handoff. | A fail-closed launcher preserves the complete value-1 record and passes every preflight plant for a fresh operator acquisition. | An admissible replacement record already exists, or preflight cannot uniquely pin the old record, repaired recorder, deck, build, inputs, and compiled branch. |

## Required gates

1. Run every real rung-2 admission plant, then clean admission against the
   existing record.  Do not invoke NEMO.
2. Reuse the established hierarchy helpers to build and gate rung 1's
   one-line replacement; add a direct test that rejects the superseded
   inherited chain and proves both the BBL/geothermal and background plants
   fire.
3. Run the complete hierarchy focused battery, both citation gates and a real
   citation plant, then the permitted ocean-fidelity battery once if no other
   pytest process is active.
4. Request a separate read-only review.  If it cannot initialize in the
   sandbox, record `independent review unavailable in-sandbox`.

## Hold conditions

Hold without scientific admission if any rung-2 record predicate fails.  Stop
for a record after staging rung 1.  Stop for a user decision if rung 1 cannot
be expressed by the already-authorized one-line Decision-83 replacement while
retaining the source-resolved BBL/geothermal-only boundary.
