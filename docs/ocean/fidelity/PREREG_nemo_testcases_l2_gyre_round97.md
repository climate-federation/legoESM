# Preregistration: NEMO-testcases L2 GYRE round 97 full stage-one RHS

Date: 2026-09-14. Frozen at incoming tip
`db6e5b6dde1122fab0a857bb0970316d7c0f1b5e` before any Round-97 candidate
measurement. Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round97/`.

## Question, magnitude, and existing instrument

Decision 41 makes the stage program the unit of proof. Round 96 corrected the
first owned boundary: legoESM's five stage-one operators reproduce NEMO's full
momentum accumulator bit-for-bit given NEMO operands, but the live stage then
hands the RK assignment a depth-mean-removed perturbation RHS. The mismatch is
`2.0121494123449567e-8 m s-2` in both components. This is upstream of the
kt=1 stage-1 U/V outputs and therefore blocks every later stage even though the
headline magnitude targets remain kt3 T `1.627497246303733e-4 K` and day-30 T
RMS `1.2397011295506804e-2 K`.

The compiled GYRE program accumulates HPG, LDF, VOR, KEG and ZAD directly in
`Krhs` at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:141-176`, computes the
separate depth mean `Ue_rhs/Ve_rhs` without modifying `Krhs` at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:202-213`, passes the same
`Nrhs` slot from `stp_2D` to stage 1 at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3.f90:190-202`, and consumes it
in the selected vector-form assignment at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:664-673`. The
compiled run selects vector form in `round46/oracle_kt2_stage/ocean.output:
797-805`.

The pre-implementation search found the existing Decision-41 stage-twin gate,
its live full-accumulator trace, the shared RK assignment, the separate
barotropic slow forcing, and the shared stage-mean correction. Round 97 changes
only the stage-1 assignment input from `du_dt_pert/dv_dt_pert` to the already
computed full `du_dt/dv_dt`; it does not add a second harness, numerical helper,
configuration option, state field, forcing, or stabilizer.

## Frozen candidate and local proof

1. The shared NEMO-identity WS-RK3 stage 1 receives the full operator
   accumulator. The separate `F_slow_u/F_slow_v` remains the barotropic
   external-step forcing and the stage correction target remains the external
   solver's barotropic velocity.
2. Given NEMO's recorded kt=1 stage entry and external-step output, both
   `momentum_rhs_u/v` entry rows and all eight live stage-one RHS-walk rows must
   be BIT. Any unequal wet cell refutes local exactness.
3. Given that same entry, kt=1 stage-1 U and V outputs must be BIT after the
   source assignment and barotropic correction. An exact RHS with either
   output still non-bit moves ownership to the direct update/correction
   boundary and holds this candidate.
4. A one-ULP perturbation to an otherwise exact stage-one RHS reference must
   flip exactly one wet cell and the gate must exit nonzero. The existing
   commit-stamp plant must fail before record consumption.

## Frozen trajectory predictions and falsifiers

The source change is a depth-uniform component that NEMO's all-stage
barotropic correction removes at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:732-758`.
Accordingly, the registered prediction is that the stage boundary becomes BIT
while the complete-step trajectory retains the Round-85 comparator values:
kt2 T/S/U/V `1.4210854715202004e-14`, `2.1316282072803006e-14`,
`2.7377110452773967e-12`, `3.284922138989399e-12`; kt3 T/S
`1.627497246303733e-4`, `6.327735185607253e-6`; and day-30 T RMS
`1.2397011295506804e-2 K`.

The prediction is REFUTED if any headline changes. The landing is refused if
any of the 954 registered rows that was AT-BAR leaves the bar, if first-over-
bar moves earlier than kt2 U/V, or if any moved row is absent from the full
Rule-12 table. A harmless or favorable movement is retained and reported but
does not replace the exact stage-row requirement. The day-30 run is required
even if the ladder rejects the candidate, because magnitude remains part of
the round record.

## Rule-12 and testcase dispositions

| lane | frozen disposition |
|---|---|
| GYRE stage twin | Require BIT full RHS and BIT kt1-stage1 U/V given NEMO entry; red one-ULP and stamp plants |
| GYRE kt=1--10 | Compare every registered row with the immutable Round-85 after arm; no AT-BAR loss and no earlier first-over-bar |
| GYRE days 1--30 | Run a fresh member and score every day against `year_owners`; report day-30 T RMS before/after |
| LOCK_EXCHANGE-zco | The shared WS-RK3 statement is constructible; require focused card and integrator tests, with no tank-fidelity claim from GYRE operands |
| OVERFLOW-zps | Same shared-path requirement; partial-cell construction and focused integrator tests must pass |
| DINO | Shared WS-RK3 statement executes; local GYRE proof does not establish DINO trajectory neutrality, and the 96--98% regional-cancellation warning remains explicit |
| ORCA2 | **UNMEASURED-WITH-SPEC:** record the selected integrator's full unprojected momentum RHS, separate vertical mean, native stage entries/outputs, histories, transports, closure carries, and red plants before a bitwise claim |

No production configuration, coefficient, timestep, stabilizer, carried-state
policy, year harness, reconciliation gate, freshwater pair, #1484 guard, held
manifest, NEMO source, or NEMO executable may change.
