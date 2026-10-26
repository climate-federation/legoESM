# ORCA2 round 56 preregistration — OVERFLOW stage-2 UP3 boundary walk

Date frozen: 2026-09-27  
Base: `8bf64602ab505715d26f3b71518b2e04f7221d3c`  
ORCA2 claim label: **given NEMO's entry** (Decision 52; held QCO arm only)  
OVERFLOW claim label: **given NEMO's recorded operands**

## Frozen question and scope

Round 55 excluded the stage-2 ENS signed-zero addition as the compensating
owner of round 48's held source-ordered tracer QCO statement.  This round
continues the compiled kt=3 stage-2 program from the admitted `after_vor`
endpoint through flux-form UP3 momentum advection to the admitted `after_adv`
endpoint.  It first scores that boundary bitwise.  If the boundary is non-bit,
it inventories whether the admitted record contains every operand needed to
walk `dynadv_up3` in source order.  Missing operands produce one additions-only,
self-describing acquisition under a new target; they are not reconstructed or
inferred.

No model, card, selector, carried state, threshold, mask, stabiliser, or sea-ice
field may change in this round.  In particular the ORCA2 card's six ice
selectors and `unmeasured_features` tuple remain untouched at
`STOP_SELECTOR_GAP`.

## Compiled branch read before prediction

The producing configuration calls `dyn_adv` after the recorded `after_vor`
boundary and before `after_adv` at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3_stg.f90:350-363`.
The compiled dispatch selects `dyn_adv_up3` at `dynadv.f90:134-145`.  The
executed UP3 routine consumes explicit `pFu/pFv/pFw`; it forms horizontal
curvatures at `dynadv_up3.f90:150-159`, face fluxes at `:174-200`, horizontal
RHS additions at `:202-220`, vertical face fluxes at `:278-345`, and the last
level additions at `:349-360`.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R56-P1 | The round-50 parent record remains admissible and its compiled/running configuration selects the flux-form UP3 branch with explicit stage transports. | Existing admission is `AT_BAR`; compiled sentinels and resolved namelist select `np_FLX_up3`, and the kt=3 stage-2 headers retain the registered slots, shape, origin, and fp64 width. | Any admission, provenance, branch, slot, shape, or precision failure: `STOPPED_FOR_RECORD`; quote no boundary number. |
| R56-P2 | UP3 changes at least one active-u bit between the recorded stage-2 `after_vor` and `after_adv` endpoints; the one-row OVERFLOW card still has no active v face. | Active-u bit census is nonzero and active-v classification is `UNMEASURED_NO_ACTIVE_FACE`. | Zero active-u changes: **REFUTED**; advection is excluded as the downstream owner and the walk advances past it without acquisition. |
| R56-P3 | The admitted kt=3 record is insufficient for a source-order UP3 replay because it lacks the exact stage-2 `zFu/zFv/zFw` transports and the internal curvature/face-flux/divergence boundaries. | Schema and file census find the two RHS endpoints but no kt=3 stage-2 transport or UP3-internal stream. | All required streams already exist: **REFUTED**; admit and walk them without rebuilding NEMO. |
| R56-P4 | If P2 and P3 confirm, an additions-only writer can capture the missing executed arrays without moving any inherited consumed field. | Preflight proves zero removed source lines; the committed parser derives names, ranks, dimensions, and payload lengths from the self-describing header; one-payload-ULP and producer-stamp plants refuse; restart/mesh and every inherited consumed field remain exact. | Any source replacement, schema ambiguity, passive-stream movement outside admission policy, or plant staying green: reject the instrument and request no run. |
| R56-P5 | No scientific statement is eligible to land before the new record exists. | Final `packages/` diff is empty and disposition is `STOPPED_FOR_RECORD` with one fail-closed operator `run.sh`. | An existing exact record supports the full walk, or P2 refutes the boundary premise: continue in-round under the corresponding branch instead of stopping. |

Failed predictions remain in the receipt as **REFUTED**.  Any statistic not
listed above is labelled post-hoc.  The boundary gate must plant a real active-u
cell and add exactly one refusal.

## Required verification

1. Admit the parent and score `after_vor` versus `after_adv` on the card masks.
2. Run the active-u endpoint plant and require exactly one additional unequal
   cell.
3. Search existing committed instruments before adding a writer; extend the
   round-53 acquisition pattern rather than inventing another format.
4. If acquisition is needed, commit every patch/module/parser/name list used by
   `run.sh`, use repository-relative paths, a fresh target name, explicit
   `REFUSE` lines, no `/usr/bin/time`, and do not run `makenemo` or `mpirun` in
   the sandbox.
5. Run focused controls, the 170-test shared-card battery, the citation gate
   with a firing plant, one ocean-fidelity battery, and a separate read-only
   `codex exec` review (or retain the mandated unavailable wording).

## Frozen OPEN if stopped for record

Operator runs the reported round-56 acquisition.  The next round admits its
record and walks, in compiled order, the stage transport, curvature, selected
face value, flux divergence, and RHS addition to the first non-bit statement.
Only after that statement is measured may the held QCO pair be retried.

