# Preregistration: NEMO-testcases L2 GYRE round 96 stage-one ownership audit

Date: 2026-09-14. Frozen at incoming tip `771d5ab2b9c3` before any Round-96
scientific comparison. Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round96/`.

## Question and existing instrument

Decision 41 makes the whole stage program the unit of proof. Round 95 named
kt=1 stage-1 U as the first owned non-bit output, but the stage-entry table did
not score the incoming momentum RHS which that stage consumes. The compiled
driver calls `stp_2D`, writes that RHS, and only then enters stage 1 at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3.f90:188-202`.
Inside the compiled stage, vector-form stage 1 adds no new momentum operator
at `GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:365-371`,
because GYRE selects vector form at
`round46/oracle_kt2_stage/ocean.output:797-803`. Its first momentum statement
is therefore the vector RK assignment at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:664-673`,
followed by the depth-mean correction and final add at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:724-760`.

The pre-implementation search found the existing Round-46 named stage record,
the existing Decision-41 stage-twin gate, the production-JIT live-stage RHS,
raw-velocity and output trace, and the shared RK assignment and barotropic
correction helpers. Round 96 extends those tools in place. It does not create a
second stage harness or duplicate a numerical method.

## Frozen measurement contract

1. Add `rhs_entry_u/v` to every Decision-41 stage-entry table. Score the
   production-JIT RHS actually consumed by each stage against the admitted
   Round-46 record, on native wet U/V cells. A one-ULP plant in kt=1 stage-1 U
   RHS must flip that row and exit nonzero.
2. For kt=1 stage 1, walk the compiled statements in order using NEMO's own
   recorded operands: incoming RHS, vector RK assignment, depth-weighted
   products, ascending-level sum, target-minus-mean correction, and final add.
   Use direct NEMO boundaries where recorded; label derived-only boundaries
   as transcriptions, never as direct ownership proof.
3. Prove the private trace is WRITE-only by comparing its ordinary T/S/U/V/SSH
   output bitwise with the untraced path. Every row must have a nonempty native
   wet mask, binary64 candidates and references, and finite values.
4. Preserve every Round-95 stage-twin row exactly. A changed overlap, missing
   new RHS row, green plant, or commit-stamp mismatch that reaches record
   consumption refutes the instrument.

## Frozen predictions and falsifiers

1. The kt=1 stage-1 incoming RHS is predicted non-bit and therefore inherited
   from `stp_2D`, which precedes the stage. If it is BIT, Round 95's stage-owned
   classification survives and the walk proceeds to the assignment.
2. Given NEMO's own Kbb velocity, RHS, `rDt=rn_Dt/3`, and mask, the shared
   vector assignment is predicted bit-exact against NEMO's direct
   `post_update` output. Any unequal cell names that assignment as the first
   owned non-bit statement.
3. Given NEMO's direct `post_update`, Kaa reference thickness, reciprocal
   depth, target barotropic velocity, and mask, the shared barotropic
   correction is predicted bit-exact against direct `post_baro`. Any unequal
   cell after a BIT assignment names the first non-bit correction boundary;
   without a direct intermediate NEMO value, no narrower arithmetic statement
   will be claimed.
4. If the incoming RHS is non-bit, Round 95's claim that kt=1 stage-1 U is
   stage-owned is **REFUTED** and no production candidate is eligible this
   round. The first owned stage must be recomputed only after the upstream
   external/`stp_2D` program is added to the stage contract.
5. With no eligible production candidate, the certified trajectory remains
   kt2 T/S/U/V `1.4210854715202004e-14`, `2.1316282072803006e-14`,
   `2.7377110452773967e-12`, `3.284922138989399e-12`; kt3 T/S
   `1.627497246303733e-4`, `6.327735185607253e-6`; and day-30 T RMS
   `1.2397011295506804e-2 K`. Any changed headline refutes non-interference.

## Rule-12 and testcase dispositions

| lane | frozen disposition |
|---|---|
| GYRE stage twin | Extend the one gate with the omitted consumed RHS and the kt1-stage1 compiled-order walk; inherited debt revokes stage ownership |
| GYRE kt=1..10 and days 1..30 | No candidate unless a direct stage-owned statement is named and made BIT; otherwise preserve the immutable Round-85 comparator |
| LOCK_EXCHANGE-zco and OVERFLOW-zps | No production statement is proposed; private hooks remain off and focused shared-path tests must pass |
| DINO | Shared WS-RK3 instrumentation risk is explicit; no numerical-neutrality or regional-cancellation claim |
| ORCA2 | **UNMEASURED-WITH-SPEC:** record the selected integrator's pre-stage momentum RHS as well as native stage entries, outputs, histories, transports and closure carries; require paired bitwise tables and red plants |

No production configuration, coefficient, timestep, stabilizer, carried-state
policy, year harness, reconciliation gate, freshwater pair, #1484 guard, held
manifest, NEMO source, or NEMO executable may change.
