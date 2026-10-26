# Preregistration: NEMO-testcases L2 GYRE round 93 stage-twin consolidation

Date: 2026-09-14. Frozen at incoming tip `84a1689dee72` before changing the
stage instrument or running any Round-93 comparison. Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round93/`.

## Question and governing decision

Decision 41 makes the RK3 stage, rather than an isolated operator, the unit of
proof. This instrumentation round extends the existing Round-46/51 stage gate;
it does not create a second harness and it does not propose a physics landing.
For kt=1 and kt=2 it will score every recorded field handed to the next stage
twice: once after driving the stage from NEMO's recorded entry, and once in the
ordinary chained legoESM trajectory. The first stage in compiled execution
order with a non-bit output in the given-NEMO-entry table is the first owned
stage. A field exact in that table but non-bit in the chain is inherited.

The compiled program runs the external solve, stage 1, the first pointer swap,
stage 2, the second pointer swap, stage 3, and the final swap at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3.f90:190-215`. Its three
stage-level coefficient and time-level selections are at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:138-146,196-200,236-244`.
The active stage builds velocity transports and W before the dynamics walk at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:431-480`, performs
stage-3 LDF before ZDF at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:690-712`, and applies
the external-mode correction at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:728-759`.

## Frozen output contract

The consolidated JSON has `given_nemo_entry` and `chained` tables for kt=1
and kt=2. Each row records stage, field, unequal cells, maximum absolute error,
and `BIT`, `AT-BAR`, `DEBT`, or `UNMEASURED_WITH_SPEC`. Rows cover the recorded
state outputs T/S/U/V/SSH, stage geometry/thickness, transports/W, barotropic
targets and histories wherever the next stage consumes them. A missing record
is printed explicitly and cannot be classified BIT. The report separately
names the earliest owned non-bit stage and the first non-bit field in its
compiled handoff order.

The plant changes one finite recorded stage-entry value by exactly one ULP,
runs the same stage comparison, and must flip at least its dependent row from
the unplanted result. A plant that does not flip a row, a malformed/truncated
record, a changed producer stamp, or a mismatched input identity exits nonzero.

## Frozen predictions and falsifiers

1. The ordinary kt=1 chained state repeats the certified Round-85 values:
   stage-1 U/V maxima `5.421010862427522e-20`/`5.421010862427522e-20`, stage-2
   U/V `1.0842021724855044e-19`/`8.131516293641283e-20`, and stage-3 U/V
   `2.7377110452773967e-12`/`3.284922138989399e-12`; stage-1 T/S/SSH remains
   bit-exact. Any changed printed value is retained as a refutation.
2. Given NEMO's external and stage entry, kt=1 stage 1 is predicted to be the
   first owned non-bit stage, with its U/V rows non-bit and its T/S/SSH rows
   bit-exact. If every stage-1 output is bit-exact, or an earlier external row
   is owned non-bit, that ownership prediction is refuted and the measured
   boundary replaces it.
3. At least one one-ULP stage-entry plant changes a dependent output row and
   exits nonzero. A zero exit or an unchanged table refutes instrument
   sensitivity and forbids a stage verdict.
4. Because no production candidate lands, the certified headline values stay
   kt2 T/S/U/V `1.4210854715202004e-14`, `2.1316282072803006e-14`,
   `2.7377110452773967e-12`, `3.284922138989399e-12`; kt3 T/S
   `1.627497246303733e-4`, `6.327735185607253e-6`; day-30 T RMS
   `1.2397011295506804e-2 K`. Any rerun difference refutes non-interference.

## Rule-12 and card dispositions

| lane | frozen disposition |
|---|---|
| GYRE stage twin | kt=1 and kt=2, given-NEMO-entry and chained tables; all missing fields fail closed |
| GYRE kt=1..10 and days 1..30 | No physics landing; retain the immutable Round-85 comparator and rerun the certified ladder if production semantics move |
| LOCK_EXCHANGE-zco and OVERFLOW-zps | The new seam is private test instrumentation and is not invoked; no tank physics claim |
| DINO | Shared WS-RK3 stage risk is explicit; no numerical neutrality or regional-cancellation claim |
| ORCA2 | **UNMEASURED-WITH-SPEC:** record native stage entries and all next-stage-consumed fields for its selected integrator, require the same two tables and ULP plants, and refuse any missing row, AT-BAR loss, or earlier first-over-bar |

No configuration, coefficient, timestep, stabilizer, carried-state policy,
year harness, reconciliation gate, freshwater pair, #1484 guard, canonical
NEMO source, or NEMO executable changes are allowed. Held manifest patches are
not applied because Decision 41 permits their evaluation only when their owned
stage is reached by the consolidated walk.
