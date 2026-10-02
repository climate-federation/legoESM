# ORCA2 hierarchy decks round 9 preregistration — rung-4 admission and rung-3 boundary

Date: 2026-10-01

Base: `d8edcb850` (`fidelity/nemo-testcases-orca2-decks`).

Scope is NEMO-side acquisition and admission only.  The operator reports that
the committed round-8 launcher completed the rung-4 run and printed
`PASS_RUNG4_RECORD`; this round independently runs the committed admission
gate on that existing record, freezes its evidence, and only after admission
reads the compiled surface-boundary dispatch needed to define rung 3.  If the
compiled source establishes one side-effect-free namelist-only unforced arm,
the round stages the exact rung-3 deck and a fail-closed acquisition.  If it
does not, the round stops for `DECISION_NEEDED`.  No legoESM package, card,
recipe, physics, configuration, hierarchy switch, threshold, carried state,
or sea-ice declaration changes.

All run claims are labelled **independent**: each NEMO rung starts from its own
from-rest initialization.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| HD9-P1 rung-4 record | The operator-produced record is exactly the round-8 preflight deck and complete version-2 schema. | The clean gate admits 480 frames with eight finite PRESENT and two runoff-owned ABSENT fields per frame, finite month products, two finite fp64 terminal restarts, and a complete SHA-256 inventory. | Any deck pin, frame header/payload/EOF, resolved consequence, month product, terminal restart, or inventory check fails. |
| HD9-P2 admission non-vacuity | Every round-8 admission plant binds on the existing record. | All fifteen plants report `PLANT-FIRED` before the clean admission passes. | Any plant stays green or fails outside its intended mutation. |
| HD9-P3 unforced boundary | The compiled surface dispatcher and reference namelist establish exactly one namelist-only, side-effect-free way to remove bulk forcing, restoring, and freshwater budgeting while retaining every unnamed switch. | One source-cited selector set is reachable with the reused CPP keys/build and its parsed rung-4-to-rung-3 delta contains only the Decision-80 rung-3 module. | Both `ln_usr` and `ln_flx` are defensible, neither is side-effect-free, a required input is unavailable, or a CPP-key/build change is required; stop for `DECISION_NEEDED`. |
| HD9-P4 rung-3 preflight | If HD9-P3 confirms, the derived deck is exact and the existing repaired recorder remains safe with its declared owners. | The gate pins the upper-rung admission, compiled dispatch, deck delta, resolved consequences, binary/build/inputs, and firing plants; the launcher reaches `PREFLIGHT_READY` without invoking NEMO. | Any pin, owner, consequence, delta, plant, or launcher check fails. |

Failed predictions remain in the receipt as **REFUTED**.

## Controls and landing predicate

The existing rung-4 run is never rerun.  Its committed gate and all plants run
before any rung-3 deck is constructed.  Source reading begins only after that
admission.  Rung 3 changes only the user-requested surface-forcing/restoring/
freshwater-budget module; every other parsed namelist assignment, input,
CPP key, binary, rank count, step count, and from-rest protocol remains exact.
The checker continues deriving payload lengths from self-describing headers
and reaches physical EOF.  No NEMO run is attempted in the sandbox.
