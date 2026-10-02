# ORCA2 hierarchy decks round 10 preregistration — rung-3 acquisition triage

Date: 2026-10-01

Base: `582485ac9` (`fidelity/nemo-testcases-orca2-decks`).

Scope is NEMO-side acquisition and admission only.  The operator reports that
the committed round-9 rung-3 launcher exited 1 at its line 108 and printed
`REFUSE: ORCA2 hierarchy rung-3 acquisition failed`.  This round classifies
that failure from the frozen launcher, full operator log, and target-directory
products before changing anything.  Complete products are admitted in place;
an acquisition-harness defect is repaired under firing controls and requests a
fresh target only when existing products cannot be admitted.  No legoESM
package, card, recipe, physics, configuration, hierarchy switch, threshold,
carried state, or sea-ice declaration changes.

All run claims are labelled **independent**: each NEMO rung starts from its own
from-rest initialization.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| HD10-P1 failure boundary | The reported line-108 failure has one decisive command and one earlier NEMO or staging error in the full log; it is not classified from the wrapper REFUSE line alone. | The launcher line, command status, first causal error, and target census agree on one boundary. | Logs and products disagree, the decisive subprocess lacks output, or multiple changes are required; stop without attribution. |
| HD10-P2 existing-record disposition | Existing rung-3 products are admitted without rerunning NEMO if and only if the committed gate's self-describing frame, resolved-deck, exact-zero-input, month, restart, and SHA checks all pass. | All plants fire and the clean gate prints its named PASS line. | Any required stream is absent, partial, malformed, non-finite, or unpinned; do not admit it. |
| HD10-P3 repair scope | If HD10-P1 identifies an acquisition-only defect, the smallest repair changes only the launcher/checker boundary that failed and preserves the exact rung-3 physical deck and reused binary. | A regression control fails on the old behavior, passes on the repair, all prior hierarchy tests stay green, and preflight reaches its named ready line. | The repair changes a physical namelist assignment, CPP key, binary, model source, record schema, or admission threshold; stop for review or decision. |
| HD10-P4 rung-2 boundary | Only after rung 3 is admitted, rung 2 differs from it solely by removing the shipped GM eddy-induced velocity and mixed-layer eddy module named by Decision 80. | Compiled source and resolved namelist establish a namelist-only one-module delta, with every differing line printed and a fail-closed fresh-target launcher. | The module cannot be disabled without a CPP-key/build change, two readings are defensible, or another physical assignment must change; report `DECISION_NEEDED`. |

Failed predictions remain in the receipt as **REFUTED**.

## Controls and landing predicate

The existing target is read-only during diagnosis.  The round does not invoke
`mpirun` or `makenemo`.  Any new acquisition target is distinct and refuses
pre-existing output.  Self-describing headers determine field names, ranks,
dimensions, and payload lengths; no predicted byte count substitutes for the
header.  Every new or repaired check ships a planted violation.  GYRE remains
byte-identical by construction because no model file changes.
