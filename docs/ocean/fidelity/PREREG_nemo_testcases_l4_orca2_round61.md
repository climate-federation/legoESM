# ORCA2 round 61 preregistration — close the stage-3 raw-Kaa boundary

Date frozen: 2026-09-28  
Base: `ef132df0e58ec0a273448a35751a7f8cb39896c9`  
Cards: `OVERFLOW-zps`; the held shared UP3 statement remains a controlled arm  
Claim labels: OVERFLOW trajectory **independent**; statement records **given NEMO's recorded operands**

## Frozen scope and source order

Round 60 left one missing production boundary between stage-3 vertical mixing
and the barotropic replacement.  The compiled record writes `raw_kaa`
immediately after `CALL dyn_zdf` and before the depth-mean correction at
`tests/OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3_stg.f90:430-448`.
The executed vertical solve first forms the explicit stage value and then
applies its implicit matrix at
`tests/OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynzdf.f90:139-170`.

The legoESM program already defers the stage-3 barotropic correction until
after its implicit vertical solve.  This round adds one private WRITE-only
observer for the state at that exact seam, extends the committed round-60 walk,
and reruns the same controlled kt=3 entry for the base and exact held UP3 arm.
It does not change public configuration, physics, carried state, thresholds,
or sea ice.  Repository search found the existing `_NEMOWSRK3TestHooks`, the
post-solve `_ws_stage3_correction` seam, the late diagnostic substitutions,
and the round-60 sidecar comparator; those are extended rather than duplicated.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R61-P1 | The new observer is passive and names exactly NEMO's stage-3 post-`dyn_zdf`, pre-barotropic `raw_kaa` boundary. | Its U/V capture is taken after the production implicit solve and before `_ws_stage3_correction`; ordinary T/S/ssh and unexposed final U/V are array-equal; construction refuses combining it with any other momentum exposure. | Any downstream read, public selector, ambiguous combination, or ordinary-state move: instrument invalid; do not measure. |
| R61-P2 | The admitted round-50 record and frozen round-60 controlled entry reproduce without drift. | Admission is `AT_BAR`, entry hash is unchanged, and the one-ULP active-U plant adds exactly one refusal. | Any stamp/header/hash/count drift: stop and reconcile. |
| R61-P3 | The held UP3 arm remains TOWARD NEMO through stage-3 `raw_kaa`; the following barotropic replacement is the first change to MIXED. | `s3.raw_kaa.u` is measured TOWARD, while the already measured postbar row remains MIXED. | `raw_kaa` is MIXED or AWAY: prediction REFUTED; name `dyn_zdf` as the first direction-changing boundary, and name it a compensating owner only if the frozen strict TOWARD-to-AWAY predicate fires. |
| R61-P4 | Closing the missing frame does not establish a strict compensating owner. | No measured row after stage-2 ADV is strictly AWAY, so `compensating_owner` remains null. | `s3.raw_kaa.u` is AWAY: prediction REFUTED and `dyn_zdf` is the strict owner. |
| R61-P5 | Default production arithmetic is unchanged by the false-by-default private observer. | Base-vs-tip GYRE ten-step residual arrays are array-equal with zero differing rows, and day-30 snapshots are byte-identical; focused default-path tests pass. | Any default trajectory move: revert the observer and finish HELD with the first moved row. |
| R61-P6 | No physics lands and neither held candidate is retried. | The only package diff is the private observer seam; source-ordered UP3 and QCO/RK remain absent. | Any physics/configuration/state/sea-ice diff remains: do not finish. |

Failed predictions remain **REFUTED** in the receipt.  MIXED is never promoted
to an owner.  The observer's test must fail when its capture or late
substitution is removed, and its noninterference check must compare arrays,
not printed summaries.

## Required output and OPEN

Commit this preregistration before adding the observer or running the new
measurement.  Then implement the smallest hook extension, run focused tests,
the controlled base/candidate walk and plant, the required GYRE base/tip
trajectory plus day-30 comparison, separate read-only Codex review, citation
gates with a real shifted-line plant, and the single ocean-fidelity battery.

If the raw boundary is strictly AWAY, the next round preregisters the UP3 +
`dyn_zdf` pair.  Otherwise retain both held candidates and continue from the
first mechanically unresolved statement.  Decision 52's independent ORCA2
ladder and month-scale ranking remain after the step walk.  Sea ice remains
unchanged at `STOP_SELECTOR_GAP`.

ASKED: close the missing stage-3 post-vertical-mixing/pre-barotropic boundary.  
UNASKED: configuration, carried-state, stabiliser, physics, and sea-ice changes.
