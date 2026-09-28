# Round 190 receipt — production QSR observer association

**Status: HELD.**  Round 189's `9,679/18,000`-cell,
`2.467770444880557e-06 K` actual-row/direct-rebuild split reproduces exactly,
but it is an **observer classification artifact**, not a model shortwave
statement.  The process observer recomputes a second Kbb shortwave field that
differs from the production Kbb field in `10,200/18,000` wet cells, maximum
`1.713729456486868e-10 K/s`, and assigns that difference to its QSR bucket.
Associated through one `14,400 s` step, that observer-only difference carries
the full `2.467770444880557e-06 K` maximum.  The first non-bit boundary is the
observer recomputation; no compiled NEMO or executable legoESM statement is
named, and no physics, configuration, carried state, default, or certified
trajectory changed.

Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round190.md`, commit
`cc72e49e8`.  Final evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round190/association_split_final.json`,
commit `bd225cab0`.

## Compiled order and executed scope

The record's compiled stage-3 program calls advection and surface boundary
first, calls `tra_qsr` once into the same `Krhs` accumulator, and then calls
lateral mixing at
`GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/stprk3_stg.f90:911-950`.
Inside that one QSR call, NEMO evaluates the two-band attenuation and adds the
result directly to `pts(...,Krhs)` at
`GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/traqsr.f90:616-645`.
Thus NEMO has no second diagnostic Kbb evaluation whose difference belongs in
the physical QSR bucket.

The existing legoESM observer re-runs the combined physics callable, names that
output `_process_qsr_kbb`, subtracts it from the complete Kbb tendency to form
the surface row, and assigns the remainder to QSR at
`ocean_model_latlon_cgrid.py:6917-6965`.  The Round-190 extension returns the
already-materialized operands and outputs through the same production-JIT
step; the scorer does not recreate them before comparison.

## Frozen predictions and results

The Round-189 split reproduced exactly, so extending the return graph did not
move the claimed boundary.  The frozen prediction that the observer's Kbb
field would be BIT with the production Kbb field is **REFUTED**.

| production-JIT row | unequal / 18,000 | maximum | disposition |
|---|---:|---:|---|
| observer Kbb QSR vs production Kbb QSR | 10,200 | `1.713729456486868e-10 K/s` | first non-bit boundary |
| observer surface-rate bridge reconstruction | 132 | `1.6940658945086007e-21 K/s` | downstream rounding floor |
| observer QSR-rate bridge reconstruction | 28 | `8.470329472543003e-22 K/s` | downstream rounding floor |
| returned cumulative `Bqsr` rebuild | 4,584 | `7.105427357601002e-15 K` | returned-boundary rounding floor |

The observer excess associated through the live QCO weights differs from the
production-QSR-only association in `9,340` cells with maximum
`2.467770444880557e-06 K`, exactly the Round-189 split maximum.  Reconstructing
the observer QSR rate from that excess leaves only `1.2705494208814505e-21 K/s`.
This closes the magnitude while preserving the first-row rule: the large row
is born in the observer recomputation, before the bridge's last-bit residuals.

## Retractions and campaign consequence

Round 188's statement that the first inherited input was the live stretch was
retracted by Round 189's direct substitution.  Round 189's surviving
"source-to-process association" boundary is now narrowed further: the model's
source association is not the owner; the observer's second Kbb evaluation
misallocates equal-and-opposite content between the surface-boundary and QSR
buckets.  Therefore Round 186's ranking of shortwave as the largest physical
pre-day-180 carrier is not usable until those two buckets are rescored from the
already-materialized production Kbb QSR field.  The combined surface-plus-QSR
source is unaffected by this classification defect.

## Controls, scope, and certified rows

The production association plant changed one active source cell by
`2^-40 K/s`, moved one returned `Bqsr` cell, printed
`STATUS PLANT-FIRED: production-qsr-association; unequal=1`, and exited 1.
The wrapper accepted only that nonzero exit and marker.

Only private WRITE-only trace output and its scorer changed.  Ordinary model
construction cannot select the trace, and the trace's returned prognostic state
still comes from the separately compiled ordinary production call.  The
certified values therefore remain kt2 T/S/U/V
`1.4210854715202004e-14`, `2.1316282072803006e-14`,
`8.326672684688674e-17`, `9.714451465470120e-17`; kt3 T/S
`4.9403105251144552e-07`, `4.0085410546453204e-08`; and day-30/day-240/day-360
T3D RMS `2.3276772050683987e-06`, `6.5861718814795174e-05`,
`2.6709923853294689e-03 K`.  First over bar remains kt3.

GYRE is the only card exercised by the diagnostic hook.  Generic NEMO-GYRE,
DINO, LOCK_EXCHANGE, OVERFLOW, and ORCA2 execute no changed production
statement.  ORCA2 remains **UNMEASURED-WITH-SPEC**: a native developed process
trace must expose its already-materialized Kbb shortwave and cumulative source
boundaries before transferring this observer result.

## Review and verification

The required independent Codex command was attempted with `--sandbox
read-only`; it produced no scientific verdict because its app-server could not
initialize.  Its decisive output is quoted verbatim:

> `Error: failed to initialize in-process app-server client: Read-only file system (os error 30)`

Accordingly this diagnostic receipt remains **UNREVIEWED** under the campaign's
two-reviewer rule.  Nothing is landing, so this does not authorize a physics
claim or candidate.

The receipt citation gate passed all three cited ranges with zero unmapped,
failed, or map-audit-failure rows.  The cumulative default-receipt audit passed all
274 citations.  Shifting the compiled `traqsr` citation by two lines changed
the receipt gate to FAIL and exited 1.  The focused battery
`test_nemo_testcase_l2_gyre_round186_qsr_walk.py`,
`test_nemo_testcase_receipt_citation_gate.py`, and `test_nemo_recipe.py`
reported `49 passed in 309.14s (0:05:09)`.  No broad ocean battery is required
because no executable production path, recipe, or physics gate changed.

## OPEN — maintenance debt after Decision 63

Decision 63 declares Round 190 the last GYRE batch round and moves the active
budget to ORCA2.  If GYRE maintenance resumes, correct the existing process
observer, not model physics: classify surface and QSR using the
already-materialized production `_qsr_b` field instead of the second
combined-physics evaluation.  Show the old observer arm reproduces this
round's equal-and-opposite transfer, the corrected arm keeps every cumulative
total and ordinary carried-state byte BIT, and the observer plant fires.  Then
re-run the pre-day-180 owner ranking: surface boundary and QSR must be reported
separately and as their invariant sum.  Resume the magnitude walk from the
largest corrected physical row.  No acquisition or configuration decision is
needed.
