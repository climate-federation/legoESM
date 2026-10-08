# Preregistration — ORCA2 round 103 rung-0 ladder

Date: 2026-10-02. Base: `04396546bc37aea5346fb621509a7b8a0f00d793`.
This round closes round 102's two gate-only OPEN items. It changes no model
physics, production card, deck, carried state, threshold, stabilizer, sea-ice
selector, or ORCA2 `unmeasured_features` entry.

The rung-0 measurement is **independent**: it starts from the admitted NEMO
rung-0 stage-0 entry in the round-90 two-rank record. The rung-7 measurement
is **given NEMO's recorded entry** under Decision 52. The two claim classes
will not be mixed in one result table.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R103-P1 | The admitted round-90 rung-0 record is sufficient for a complete ten-step ladder. | The gate parses exactly 80 self-describing shards (2 ranks x 10 steps x 4 boundaries), assembles each boundary with exact once-only global coverage, and emits 200 field rows (10 steps x 4 boundaries x 5 fields). | Any missing, malformed, overlapping, or uncovered shard: refuse and name the exact stream; request acquisition only if the admitted record truly lacks it. |
| R103-P2 | The rung-0 replay preserves the admitted entry and carries the existing first non-bit boundary forward. | The kt=1 stage-0 bridge is 0/3,223,440 unequal; the first non-bit checkpoint is kt=1 stage 1; the source-ordered statement remains the already measured `depth_u` boundary rather than being reassigned from field order. | Any entry bit moves or the first checkpoint changes: reconcile before recording a ladder. |
| R103-P3 | The gate reaches kt=10 without inventing an exact drag or another stabilizer. | All 40 checkpoints are finite and scored while the measurement card continues to declare `linear_implicit_bottom_drag` unbuilt. | A non-finite or refusal before kt=10: keep HELD and report the exact step/field; do not stabilize it. |
| R103-P4 | A one-bit corruption of a recorded rung-0 stage frame makes the gate refuse. | The planted run exits nonzero at the altered checkpoint. | Plant stays green: the gate is invalid and no ladder number is citable. |
| R103-P5 | The existing complete rung-7 record supports the shipped-card kt=1..10 replay after the round-102 dry-temperature signed-zero precondition fix. | The gate reaches `LADDER_MEASURED`, emits all 200 trajectory rows, and changes no executed bit predicate. | Missing rank/surface/stage stream or an unregistered refusal: name it exactly and report `ACQUISITION_NEEDED` only if no admitted complete record exists. |

## Landing bar

The round lands the rung-0 gate only if R103-P1 through P4 pass mechanically,
focused tests pass, the citation gate and its plant fire, and the independent
review is recorded. Rung-7 is a read-out: failure there does not relax or
rewrite the rung-0 predicate. No `packages/` change is planned, so the GYRE
year and ladder remain unchanged by construction; any unexpected model-file
diff is a hard refusal.
