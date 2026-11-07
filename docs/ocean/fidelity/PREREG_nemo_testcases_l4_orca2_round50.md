# ORCA2 round 50 preregistration — complete the OVERFLOW kt=3 acquisition

Date frozen: 2026-09-27  
Base: `6a2df3730`  
ORCA2 claim label: **given NEMO's entry** (Decision 52)  
OVERFLOW claim label: **independent**

## Frozen scope

Round 49 bounded the held source-ordered QCO/RK candidate's first OVERFLOW
trajectory movement to kt=3, but deliberately stopped before writing the
NEMO instrument.  The operator's attempted acquisition exited 66 at that
deliberate refusal; no target configuration or run was created.  This round
completes only the missing write-only instrument, strict parser/admission gate,
and operator-executed acquisition script under a new target name.  It does not
change `packages/`, a NEMO configuration value, model state, a selector, a
threshold, a stabiliser, or the ORCA2 sea-ice debt tuple.

The executing OVERFLOW source orders stage-2/3 momentum as EOS, HPG,
vorticity, then advection at
`OVERFLOW_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:327-354`; stage 3 then adds
lateral mixing and calls vertical mixing/time integration at
`stprk3_stg.f90:378-422`, followed by the barotropic correction at
`stprk3_stg.f90:425-438`.  Tracers are zeroed and receive advection and the
surface source at `stprk3_stg.f90:480-505`; stages 1/2 then execute the QCO/RK
assignment at `stprk3_stg.f90:512-524`, while stage 3 applies its remaining
operators at `stprk3_stg.f90:530-567`.

Repository search found the existing general consumed-field admission gate,
the round-63 self-describing recorder pattern, and the round-123 source-order
tracer snapshots.  This round extends those patterns rather than creating a
second trajectory or admission implementation.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R50-P1 | One additions-only patch can expose every kt=3 source-order boundary needed by round 49. | Applying the patch removes/replaces zero source lines, preprocessing/Fortran syntax checks pass, and compiled-source sentinels cover every requested boundary. | Any source deletion, syntax failure, or missing sentinel: fix the instrument before acquisition; do not request a run. |
| R50-P2 | The record schema is mechanically closed. | Synthetic records parse to physical EOF; wrong magic/version/step/stage/slot/dimension/fp width/order/count, duplicate/missing fields, non-finite payload, truncation, trailing bytes, stamp mismatch, and a one-ULP payload plant all refuse. | Any malformed record is admitted: hold and repair the gate. |
| R50-P3 | The write-only build changes no consumed NEMO state. | The operator run reproduces the pinned restart and every inherited record under raw identity or the existing consumed-field admission, and the admission plant refuses. | Any owned consumed value or restart differs: reject the acquisition. |
| R50-P4 | No scientific statement lands in this round. | Final `packages/` diff is empty and disposition is `STOPPED_FOR_RECORD` with the new `run.sh`. | An admitted record already exists: parse it and continue the round-49 walk instead. |

Failed predictions remain in the receipt.  The operator-run record is not
claimed until the admission gate passes.

## Required verification

1. Prove the patch is additions-only against the record's actual `MY_SRC`
   source and preprocess it with that configuration's compiled keys.
2. Unit-test the parser with valid synthetic stage-1/2/3 records and every
   frozen refusal above, including a non-vacuous payload plant.
3. Run the schema gate in preflight mode, citation gates with a real plant,
   focused tests, the shared-card battery, and the ocean-fidelity battery once.
4. Run a separate read-only `codex exec` review; record its verdict or the
   mandated unavailable wording.
5. Write the acquisition under round 50 with a new target configuration and
   run directory.  Per the withdrawn in-sandbox-run note, do not invoke
   `makenemo` or `mpirun`; return its absolute path to the operator.

## Frozen OPEN

After the record is admitted, compare held candidate and base at kt=3 in the
compiled order, name the first unequal statement, and test that statement
bit-exact before forming a cancelling pair.  Then return to ORCA2's kt=1
stage-1 T producer, the independent Decision-52 initial state/year, and the
round-20 slow forcing.  Sea ice remains out of scope at `STOP_SELECTOR_GAP`.

ASKED: complete the acquisition whose intentional round-49 preflight refusal
the operator reached.  
UNASKED: none.
