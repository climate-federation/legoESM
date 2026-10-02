# ORCA2 hierarchy decks round 12 preregistration — rung-1 admission and rung-0 handoff

Date: 2026-10-02

Base: `5b80c1b970` (`fidelity/nemo-testcases-orca2-decks`).

Scope is NEMO-side admission and hierarchy-deck reconciliation only.  The
operator reports that the committed round-11 launcher completed successfully;
this round independently reruns the committed admission against that existing
target, source-classifies every non-protocol rung-1/rung-0 assignment, and
closes or stops the side lane.  No NEMO run occurs in the sandbox.  No legoESM
package, card, recipe, physics, configuration, carried state, sea-ice
declaration, NEMO source, or oracle record is changed.

Every run claim is labelled **independent**: NEMO starts from each rung's own
from-rest initialization.

The compiled source was read before freezing the reconciliation predictions.
In particular, `nn_havtb` is consumed unconditionally while initializing the
background tracer diffusivity (`zdfphy.f90:205-228`); the other extra selectors
are owner-off or resolve to their reference defaults.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| HD12-P1 rung-1 record | The operator-produced target is complete and admissible in place. | The committed round-11 gate fires all 20 admission plants, then prints `PASS_RUNG1_RECORD` over 480 self-describing frames, finite month products, two finite fp64 step-240 ocean restarts, the exact-zero surface input, and a complete SHA-256 inventory. | Any stream is absent, malformed, non-finite, unpinned, or the clean gate refuses; retain the record and stop at its first refusal. |
| HD12-P2 admission non-vacuity | Every round-11 admission predicate can fail on its named planted violation. | Every registered plant prints `STATUS PLANT-FIRED` before the clean pass. | A plant stays green, crashes outside the expected gate error, or mutates the clean target; repair only the instrument under a firing regression before admitting. |
| HD12-P3 active cross-lane difference | Of the ten non-protocol/non-file/non-damping differences, `nn_havtb=1` in rung 1 versus `0` in main-lane rung 0 is physics-active even with constant vertical mixing. | Compiled `zdf_phy_init` applies the selector to `avtb_2d` and the resolved outputs preserve the 1-versus-0 difference; therefore rung 1 to rung 0 is not a damping-only edge and this side lane makes no deck choice. | The selector is owner-off, both resolved outputs agree, or the resulting background tracer diffusivity is bit-identical. |
| HD12-P4 remaining cross-lane differences | The other nine extra differences do not change the resolved physics of these two decks. | `ln_spc_dyn` is absent from the no-AGRIF compiled source; `ln_sssr_bnd` and `nn_chldta` sit behind false module owners; and the six explicit false vertical-mixing selectors equal the reference defaults used when absent. | Any compiled active consumer, different resolved value, input read, allocation, or tendency is found; name it separately and retain the edge as unresolved. |

Failed predictions remain in the receipt as **REFUTED**.

## Controls and landing predicate

The existing rung-1 target is read-only except for the launcher's admission
JSON, plant logs, and checksum ledger.  Self-describing headers determine field
names, ranks, dimensions, and payload lengths.  Rung 1 is admitted only after
every plant fires and the clean gate passes.  The side lane is complete only as
a deck/record producer for rungs 1 through 10; it does not call rung 1 to rung 0
a one-module edge while an active non-damping difference remains.  GYRE is
byte-identical by construction because no model file changes.
