# Preregistration: NEMO-testcases L2 GYRE round 94 stage-twin closure

Date: 2026-09-14. Frozen at incoming tip `7e60f5099304` before any Round-94
scientific comparison. Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round94/`.

## Question and existing instrument

Decision 41 makes the complete RK3 stage program the unit of proof. Round 93
extended the existing Round-46/51 gate but correctly refused ownership because
48 output rows were still unmeasured and its five-field stage-entry seam did
not close every stage-local input. Round 94 extends that same gate; it does not
create a second harness and it does not propose a production-physics landing.

The pre-implementation search found the admitted Round-81 external-step reader,
the Round-71 tracer-stage reader, the Round-46 stage reader, and the existing
live production-JIT trace. Those are reused. No numerical statement or reader
is duplicated.

The compiled GYRE program calls vertical physics before the external solve
at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/stprk3.f90:168-190`, then runs
stages 1, 2, and 3 with their pointer swaps at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/stprk3.f90:195-222`.
The external solver rotates the absolute current/b/bb histories after each
substep at
`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:834-846`.
The executing stage constructs horizontal transports at
`GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/stprk3_stg.f90:278-324`, then
`tra_adv_trp` initializes or updates all three effective tracer transports at
`GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/stprk3_stg.f90:792-819`.

## Frozen measurement contract

1. External rows will consume the admitted kt=1 absolute-history endpoint and
   Round-81's kt=2 substep record. The model candidate is the actual returned
   `bt_hist` carry, never a reconstructed deviation. Each of the six endpoint
   histories is scored cellwise in both given-NEMO-entry and chained tables.
2. Stage transports will consume the admitted Round-71 kt=1/2 stage-1/2
   tracer records and the direct kt=1 stage-3 tracer record. The existing
   Round-46 momentum-side stream is not used for `zFw`, because its compiled
   write precedes `tra_adv_trp` on the vector branch. Any kt=2 stage-3 field
   lacking a direct post-`tra_adv_trp` record stays
   `UNMEASURED_WITH_SPEC`.
3. The stage-entry identity table will add the stage QCO ratios and the
   step-local TKE/viscosity/diffusivity carry which the executing stage
   consumes. The given-entry arm uses the corresponding admitted NEMO bundle;
   the chained arm uses the model's own carry. Shape, dtype, finite-value,
   stage, time-level, producer, and digest checks remain fail closed.
4. The output tables will list the same stage context wherever it is carried
   unchanged to the next stage. A row may be `BIT`, `AT-BAR`, `DEBT`, or
   `UNMEASURED_WITH_SPEC`; missing coverage can never become `BIT`.

## Frozen predictions and falsifiers

1. The 24 missing external-history rows close. Given NEMO's kt=1 entry, all
   six kt=1 endpoint histories are predicted bit-exact. At kt=2, at least the
   already measured external U/V/SSH debt propagates into one endpoint-history
   row. Any absent endpoint, wrong source time level, or mismatched endpoint
   identity refutes this prediction and leaves the row unmeasured.
2. Direct tracer-stage records close `zFu/zFv/zFw` at kt=1 stages 1--3 and
   kt=2 stages 1--2. Their overlapping kt=1 zFu/zFv rows reproduce Round 93
   exactly. Because the kt=1 stage-1 W row differs by
   `3.5937485546815465e-8`, its post-`tra_adv_trp` `zFw` is predicted non-bit.
   A bit-exact zFw row or any changed overlapping zFu/zFv value refutes the
   prediction and is retained.
3. Given the admitted NEMO stage context, all newly injected stage-entry
   context rows are predicted bit-exact. A one-ULP plant in one finite,
   nonzero context value must make its own row unequal and exit nonzero. A
   green plant or unchanged planted row refutes instrument sensitivity.
4. The existing record inventory cannot directly close kt=2 stage-3 effective
   transports after `tra_adv_trp`; all six paired-table rows for that triplet
   are predicted to remain `UNMEASURED_WITH_SPEC`. The overall gate therefore
   remains `UNMEASURED`, no first owned stage is promoted, and no physics may
   land. Discovery of an already-admitted direct record refutes this inventory
   prediction and the direct record must be used.
5. Instrumentation is WRITE-only. The certified trajectory remains kt2
   T/S/U/V `1.4210854715202004e-14`, `2.1316282072803006e-14`,
   `2.7377110452773967e-12`, `3.284922138989399e-12`; kt3 T/S
   `1.627497246303733e-4`, `6.327735185607253e-6`; and day-30 T RMS
   `1.2397011295506804e-2 K`. Any changed headline value refutes
   non-interference.

## Rule-12 and testcase dispositions

| lane | frozen disposition |
|---|---|
| GYRE stage twin | Extend the single gate; every missing field remains loud and prevents ownership |
| GYRE kt=1..10 and days 1..30 | No candidate; preserve and, if production semantics move, rerun the immutable Round-85 comparator |
| LOCK_EXCHANGE-zco and OVERFLOW-zps | Instrument hooks remain private and off; no tank-physics claim changes |
| DINO | Shared WS-RK3 stage risk remains explicit; no numerical-neutrality or regional-cancellation claim |
| ORCA2 | **UNMEASURED-WITH-SPEC:** record native stage entries, TKE/context carries, transports, absolute histories, and next-stage outputs; require paired bitwise tables and red plants; reject missing rows, AT-BAR loss, or earlier first-over-bar |

No configuration, coefficient, timestep, stabilizer, carried-state policy,
year harness, reconciliation gate, freshwater pair, #1484 guard, held manifest,
NEMO source, or NEMO executable changes are allowed.
