# ORCA2 hierarchy decks round 15 preregistration — repair inherited plants, admit rung 5, stage rung 4

Date: 2026-10-02

Base: `3731c75a0` (`fidelity/nemo-testcases-orca2-decks`).

Scope is NEMO-side only.  The operator ran Decision 83's replacement rung-5
deck to completion, but admission stopped because the inherited round-2
`terminal-nonfinite` plant raised through the round-14 gate without printing
the required `STATUS PLANT-FIRED` marker.  This round repairs that gate
contract once in the module that raises, verifies every declared record plant
of every hierarchy gate reports the marker through its command-line entry
point, admits the existing rung-5 run without rerunning NEMO, and only after
admission prepares rung 4's one-assignment Decision-83 replacement.  No
legoESM package, card, recipe, physics, threshold, or carried state changes.
Every run claim is labelled **independent**: NEMO starts from the rung's own
from-rest initialization.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| HD15-P1 failure ownership | The existing rung-5 run is complete and finite; only the inherited terminal plant's reporting contract stopped admission. | The clean gate admits the existing record unchanged after the reporting repair, while `terminal-nonfinite` exits nonzero and prints `STATUS PLANT-FIRED`. | The clean record fails any payload, finiteness, restart, resolved-deck, month, or inventory predicate. |
| HD15-P2 single repair point | The round-2 terminal validator is the raising module and can report a planted terminal failure before re-raising through any descendant gate. | The round-2 direct CLI and every descendant CLI that exposes `terminal-nonfinite` print the marker and exit nonzero without modifying data. | Any descendant still emits only a traceback, or a clean gate is reclassified as a plant. |
| HD15-P3 complete plant contract | Every non-`none` plant declared by every hierarchy gate reports `STATUS PLANT-FIRED` at its CLI boundary. | A parameterized unit test drives every declared plant name through every gate's `main()` with the relevant evaluation path replaced by a deterministic gate failure, and all print the marker with exit 1. | Any declared plant lacks the marker or exits zero. |
| HD15-P4 rung-5 admission | The existing replacement resolves `nn_havtb=0` and satisfies the inherited complete-record contract. | Clean admission reports 480 self-describing frames, finite month products, finite fp64 step-240 restarts, complete SHA inventory, and every real plant fires. | Any record product, resolved value, payload, terminal state, inventory entry, or real plant fails. |
| HD15-P5 rung-4 one-line delta | Rung 4's replacement deck differs from its admitted superseded deck only by `namzdf.nn_havtb: 1 -> 0`; its boundary from replacement rung 5 remains shortwave-only. | Parsed assignments and physical lines report exactly that replacement, while the rung-5/rung-4 boundary changes only the named shortwave module assignments. | Any second Decision-83 deck change or any unrelated boundary assignment appears. |
| HD15-P6 acquisition disposition | No replacement value-0 rung-4 record exists before handoff. | Preflight passes and the round ends with a committed fail-closed launcher that preserves the old record. | An already-existing admissible value-0 rung-4 record is found. |

Failed predictions remain in the receipt as **REFUTED**.

## Controls and landing predicate

The fixed plant must fail through the production CLI and print the marker; a
clean invocation must still fail normally without the marker if its record is
bad.  Rung-5 admission must rerun every real synthetic violation against the
existing record before clean admission.  Rung-4 preflight must refuse a second
deck delta, missing Decision-83 line, changed retained coefficient, changed
source/build/input/CPP/run pins, or unpreserved superseded evidence.  The
launcher must use `--admit-existing` for rung 5, never rerun it, and must not
invoke `makenemo` for rung 4.

This round lands checker/tests, rung-5 admission, and rung-4 preregistered
acquisition tooling only if all predicates pass.  Rung 4 remains UNMEASURED
until the operator runs it.  No GYRE run is required because no model file
changes.
