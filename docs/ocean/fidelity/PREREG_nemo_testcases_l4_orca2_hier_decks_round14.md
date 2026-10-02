# ORCA2 hierarchy decks round 14 preregistration — admit rung 6, stage rung 5

Date: 2026-10-02

Base: `e4a959788` (`fidelity/nemo-testcases-orca2-decks`).

Scope is NEMO-side only.  Decision 83 fixes `nn_havtb=0` on hierarchy rungs
0 through 6.  The operator has run round 13's replacement rung-6 acquisition;
this round admits that existing record and, only after it passes, prepares the
same one-assignment replacement and independent reacquisition for rung 5.  No
legoESM package, card, recipe, physics, threshold, or carried state changes.
Every run claim is labelled **independent**: NEMO starts from the rung's own
from-rest initialization.

## Frozen oracle reading

The compiled vertical-physics initializer reads `nn_havtb`, initializes the
horizontal multiplier uniformly, and enters the equatorial shape only for
selector value one
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfphy.f90:140-149`,
`:205-228`).  Decision 83 therefore predicts the zero arm retains the existing
constant-mixing closure and coefficients while making the background uniform.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| HD14-P1 rung-6 admission | The operator-run replacement resolves `nn_havtb=0` and passes the inherited complete-record contract. | Clean admission reports 480 self-describing frames, finite month products, finite fp64 step-240 restarts, complete SHA inventory, and every named plant fires. | Any product, resolved value, payload, terminal state, inventory entry, or plant fails. |
| HD14-P2 rung-6 preservation | The admitted value-1 evidence was moved without alteration to the required superseded paths. | The old admission, exact decks, 480 frames, restarts, month products, and complete inventory retain their pinned digests. | Any old evidence is absent or differs. |
| HD14-P3 rung-5 one-line delta | The replacement rung-5 deck differs from its admitted superseded deck only by `namzdf.nn_havtb: 1 -> 0`. | Parsed assignments and physical lines report exactly that single change; build, inputs, CPP keys, run protocol, and all other assignments are identical. | Any second assignment, physical line, build/input/CPP pin, or protocol row changes. |
| HD14-P4 rung-6 to rung-5 edge | With both rungs at `nn_havtb=0`, the complete active module boundary remains runoff only. | Parsed boundary changes only `namsbc_rnf.ln_rnf: true -> false`; the background selector is equal. | Any additional active assignment differs. |
| HD14-P5 acquisition disposition | No replacement rung-5 record exists before handoff. | Preflight passes and the round ends with a committed fail-closed launcher targeting a fresh replacement while preserving the old record. | An already-existing admissible value-0 rung-5 record is found. |

Failed predictions remain in the receipt as **REFUTED**.

## Controls and landing predicate

Admission must refuse malformed headers, wrong field names or payload lengths,
missing/non-finite frames, non-finite or wrong-step terminal states, changed
resolved settings, incomplete inventory, or any non-firing plant.  Rung-5
preflight must refuse a second deck delta, a missing Decision-83 line, a
changed retained coefficient, changed source/build/input/CPP/run pins, or an
unpreserved superseded record.  The launcher must preserve, never delete, the
admitted value-1 rung-5 evidence and must not invoke `makenemo`.

This round lands the rung-6 admission record and rung-5 preregistered
acquisition tooling only if those predicates pass.  Rung 5 remains
**UNMEASURED** until the operator runs and the record is admitted.  No GYRE run
is required because no model file changes.
