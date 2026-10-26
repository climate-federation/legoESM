# ORCA2 round 62 preregistration — acquire the dyn_zdf internal walk

Date frozen: 2026-09-28  
Base: `121bcd1c3d513f6f75bd97cf3a12369e65ee0038`  
Card exercised by the shared-statement safety walk: `OVERFLOW-zps`  
Claim labels: OVERFLOW trajectory **independent**; oracle frames **given NEMO's recorded operands**

## Frozen scope and compiled source order

Round 61 localised the first stage-3 TOWARD-to-MIXED change to the executing
`dyn_zdf`.  The compiled routine forms the explicit Kaa velocity at
`tests/OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynzdf.f90:139-153`,
subtracts the split-explicit barotropic velocity at `:159-163`, adds its
top/bottom drag contribution at `:164-182`, and applies the U and V implicit
tridiagonal solves at `:188-351` and `:357-519`.  The caller records only the
pre-routine RHS and final Kaa endpoints at
`tests/OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3_stg.f90:429-433`.

This round first inventories every admitted self-describing OVERFLOW record.
If none contains all four internal boundaries, it commits the smallest
additions-only writer, parser/admission gate, and operator-run acquisition.
The writer targets only `(kt, stage) = (3, 3)`, accumulates complete U/V rows
at the four cited seams, and emits one self-describing record.  It changes no
NEMO statement, namelist, legoESM model file, configuration, carried state, or
sea ice.  Repository search found and reuses the round-50 parent record,
round-53 additions-only writer/parser pattern, and inherited-record admission.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R62-P1 | No admitted record carries all four `dyn_zdf` internal velocity frames. | Header-driven inventory finds only pre-ZDF and raw-Kaa endpoints, with no explicit-update, barotropic-subtraction, drag, or solve sequence. | Any admitted record declares all eight required U/V arrays: use it, do not create an acquisition. |
| R62-P2 | The additions-only patch lands exactly four executed observation seams without changing an existing source line. | Dry patch has zero removed/replaced lines; compiled sentinels occur after the four cited statements on the active branch. | A source line is removed, a sentinel is ambiguous/dead, or syntax fails: acquisition refuses. |
| R62-P3 | The record is self-describing and physically complete. | Magic/version, `(kt,stage)=(3,3)`, `Kbb/Kmm/Krhs/Kaa`, origin, dimensions, fp width, ordered field names, each payload length, and physical EOF are parsed from the header; arrays are finite. | Any hand-predicted byte count/header tuple, missing/reordered field, non-finite value, or trailing/truncated payload: refuse. |
| R62-P4 | The new run is observationally identical to its round-50 parent. | Restart and mesh are byte-identical, all inherited streams are admitted, only the named record is new, and payload/stamp/consumption plants each fail. | Any inherited-state movement or unregistered output: refuse and do not cite the record. |
| R62-P5 | The existing sandbox cannot run the acquisition because PMIx socket creation is forbidden. | Commit a fail-closed `run.sh` under a new target and finish `STOPPED_FOR_RECORD` with its absolute path; do not invoke it here. | If a complete existing record refutes R62-P1, admit it and continue the walk in this round. |
| R62-P6 | No production physics lands in round 62. | `git diff <base> -- packages` is empty; held UP3 and QCO/RK remain absent. | Any package/configuration/state/sea-ice diff remains: do not finish. |

Failed predictions remain **REFUTED** in the receipt.  A MIXED boundary is not
an owner.  The eventual walk must compare base and held-UP3 arms from the same
hash-bound independent entry and must plant one active U value before quoting
any direction.

## Required output and OPEN

Commit this preregistration before inventorying record headers.  If acquisition
is required, commit every input it reads by repository-relative path, run its
preflight and parser tests, perform separate read-only Codex review, run the
citation gate with a real shifted-line plant, and report the operator path.
No `mpirun` attempt is permitted under operator note B10.

After admission, the next round adds the corresponding passive legoESM
observers and walks explicit update -> barotropic subtraction -> drag ->
implicit solve.  Decision 52's independent ORCA2 ladder and month-scale
ranking remain after the step walk.  Sea ice remains unchanged at
`STOP_SELECTOR_GAP`.

ASKED: inventory/acquire the source-ordered `dyn_zdf` internal frames.  
UNASKED: configuration, carried-state, stabiliser, physics, and sea-ice changes.
