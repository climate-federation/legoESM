# ORCA2 round 63 preregistration — dyn_zdf internal walk

Date frozen: 2026-09-28  
Base: `6ac4ecd1b8d76eb35b48888b28892fdec78aa4e8`  
Card exercised by the shared-statement walk: `OVERFLOW-zps`  
Claim labels: OVERFLOW trajectory **independent**; NEMO internal frames
**given NEMO's recorded operands**

## Frozen scope and source order

Round 62's operator acquisition reports `CONSUMED_FIELD_ADMISSION PASS`,
`exact=30/33`, `changed=3`, `admitted=16`, and the ready root
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round62/acquisition/oracle_overflow_dynzdf_internals`.
This round first re-runs the committed admission gate against that record and
its round-50 parent.  It then adds one private WRITE-only production observer
for the four already-materialized stage-3 momentum boundaries and compares the
admitted base and held source-ordered UP3 arms from the same round-60
hash-bound independent kt=3 entry.

The executing acquired source forms the explicit velocity at
`OVERFLOW_OMIP_L1_P3_R62ZDF/BLD/ppsrc/nemo/dynzdf.f90:139-159`, subtracts the
barotropic velocity at `:165-171`, adds its explicit bottom/top drag at
`:172-192`, and finishes the U/V implicit solves at `:343-362` and `:512-531`.
The deck takes `ln_dynadv_vec`, `ln_drgimp`, and `ln_dynspg_ts`; the observer
must expose exactly those executed branches and must not add a selector,
configuration field, state field, stabiliser, forcing, or sea-ice change.

Repository search found and will extend the existing round-60/61 controlled
entry, direction classifier, sidecar, and raw-Kaa observer; the round-62
self-describing parser remains the only oracle-record reader.  The held UP3
arm is the already-measured round-59 candidate, reconstructed exactly from its
committed diff and removed again before final disposition.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R63-P1 | The operator-produced record is admissible without reinterpretation. | Round-62 admission returns `AT_BAR`; U/V implicit endpoints are array-equal to the admitted round-50 raw-Kaa endpoints; payload, stamp, and inherited-consumption plants refuse. | Any admission or plant fails: stop for record; quote the failed invariant and do not score payloads. |
| R63-P2 | One passive observer can expose all eight legoESM arrays from the production-jitted step without moving its returned T/S/u/v/ssh. | Ordinary and observed complete states are array-equal; trace has exactly the ordered eight fields, fp64 shapes match the oracle, and a one-ULP active-U plant adds one refusal. | Any state moves, field is absent/duplicated, dtype/shape differs, or plant is inert: fix the instrument before quoting a science number. |
| R63-P3 | The held UP3 arm remains strictly TOWARD through the explicit update, barotropic subtraction, and drag; the implicit solve is the first TOWARD-to-MIXED boundary seen at round 61's raw-Kaa endpoint. | First three U rows classify TOWARD and the fourth MIXED under the frozen L-inf+L2 rule. | Record the actual first direction change.  A strict AWAY row names a candidate compensating statement; MIXED never does. |
| R63-P4 | The NEMO internal endpoints reconcile with the older instruments. | Oracle explicit input is source-consistent with the stage-3 pre-ZDF record; oracle implicit solve is bit-identical to round-50 raw Kaa; legoESM implicit observer matches round 61's raw observer under each arm. | Reconcile layouts/time levels/instruments before recording either number. |
| R63-P5 | No scientific statement lands this round unless the walk names one strict source-order compensating owner and its measured pair passes the complete shared gate. | Final package diff is observer-only; held UP3 and QCO/RK changes are absent. | If a strict owner is named, attempt only that already-authorized measured pair under ORCA2, GYRE Decision 43/45/55/59, OVERFLOW, DINO, tanks, generic-card, citation, and push gates. |
| R63-P6 | The private observer is inert on GYRE. | Base and final GYRE ten-step reports have 70 unchanged rows and array-equal residuals; 30 daily snapshots are byte-identical. | Any movement: remove the observer and finish HELD with the failed row named. |

Failed predictions remain **REFUTED** in the receipt.  Direction uses exact
per-cell movement plus the frozen aggregate rule; a smaller scalar norm alone
does not name an owner.  All runs are CPU fp64/libm with production JIT.

## Required gates and OPEN

Commit this preregistration before parsing the new payload or running either
model arm.  Run one-variable base and held-UP3 arms from the same controlled
entry, the observer non-interference and plant controls, the round-62 admission
and plants, the GYRE ten-step and day-30 byte-identity gates for the final
observer tree, a separate read-only Codex review, citation audit plus real
shifted-line plant, focused tests, and the single permitted ocean-fidelity
`-n 12` battery.  Re-anchor the default receipt by the required SequenceMatcher
map if the model-file edit moves its citations.

After this walk, keep both physics candidates held unless a strict owner passes
the complete gate.  Decision 52's independent-start ORCA2 ladder and the
month-scale ORCA2 ranking follow the step-level walk.  Sea ice remains unchanged
at `STOP_SELECTOR_GAP`.

ASKED: admit and walk the four `dyn_zdf` internal boundaries.  
UNASKED: configuration, carried state, stabiliser, forcing, and sea-ice changes.
