# ORCA2 hierarchy decks round 11 preregistration — rung-2 admission and rung-1 preflight

Date: 2026-10-01

Base: `713a3e86b1` (`fidelity/nemo-testcases-orca2-decks`).

Scope is NEMO-side acquisition and admission only.  This round inspects the
operator-produced rung-2 target, admits it in place only through the committed
round-10 gate, then constructs rung 1 by removing only the bottom-boundary-layer
and geothermal-heating module fixed by Decision 80.  No legoESM package, card,
recipe, physics, configuration, hierarchy threshold, carried state, or sea-ice
declaration changes.  No NEMO run occurs in the sandbox.

Every run claim is labelled **independent**: each NEMO rung starts from its own
from-rest initialization.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| HD11-P1 rung-2 record | The operator-produced target is complete and admissible in place. | The committed round-10 gate fires every plant, then prints `PASS_RUNG2_RECORD` over 480 self-describing frames, finite month products, two finite fp64 step-240 ocean restarts, the exact-zero surface input, and a complete SHA-256 inventory. | Any stream is absent, malformed, non-finite, unpinned, or the clean gate refuses; retain the record and stop at its first refusal. |
| HD11-P2 admission non-vacuity | Every round-10 admission predicate can fail on its named planted violation. | Every registered plant prints `STATUS PLANT-FIRED` before the clean pass. | A plant stays green, crashes outside the expected gate error, or mutates the clean target; repair only the instrument under a firing regression before admitting. |
| HD11-P3 rung-1 boundary | Rung 1 is a namelist-only removal of bottom-boundary-layer transport and geothermal heating from admitted rung 2. | The compiled source proves the two selectors independently guard all allocations and runtime calls; the complete parsed deck delta contains only `ln_trabbl=true→false` and `ln_trabbc=true→false`, with retained parameters inert. | A CPP-key/rebuild is required, another physical assignment must change, an owner-off recorder field lacks an ABSENT rule, or two source readings are defensible; report `DECISION_NEEDED`. |
| HD11-P4 rung-1 preflight | The reused repaired binary can stage rung 1 on a fresh target without running NEMO. | Source/build/input/deck pins, exact two-assignment delta, upper-record admission, header contract, run protocol, and planted violations pass; the launcher prints its named preflight-ready line. | Any pin, plant, or protocol check fails; do not request acquisition. |

Failed predictions remain in the receipt as **REFUTED**.

## Controls and landing predicate

Existing records are read-only during diagnosis and admission.  Any rung-1
target is distinct and refuses pre-existing run output.  Self-describing headers
determine field names, ranks, dimensions, and payload lengths; no hand-predicted
byte count substitutes for the header.  Every new check ships a planted
violation.  GYRE remains byte-identical by construction because no model file
changes.
