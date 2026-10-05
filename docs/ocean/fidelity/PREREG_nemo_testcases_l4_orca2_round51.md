# ORCA2 round 51 preregistration — admit and walk the OVERFLOW kt=3 pair record

Date frozen: 2026-09-27  
Base: `932cbfa9ec2f2fbcbf51a03ca8e46e5b39b78c62`  
ORCA2 claim label: **given NEMO's entry** (Decision 52)  
OVERFLOW claim label: **independent**

## Frozen question and scope

Round 48's source-ordered QCO/RK tracer assignment made the direct ORCA2
statement bit-exact but worsened five later OVERFLOW U rows.  Rounds 49-50
bounded that movement to kt=3 and produced the missing source-order record.
The operator reports that record admitted with 24/27 inherited streams exact,
three calibrated streams changed, 16 inherited differences admitted, and both
restart and mesh unchanged.  This round first reproduces those admissions,
then compares the held candidate and base through kt=3 in compiled order.

The record's executing source zeros the tracer RHS, calls advection and the
surface source, and materializes the stage-1/2 QCO assignment at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3_stg.f90:492-541`.
The following stage calls EOS and then HPG before vorticity and advection at
`stprk3_stg.f90:327-363`.  The selected EOS polynomial is
`eosbn2.f90:684-718`; the selected SCO HPG recurrence is
`dynhpg.f90:341-419`.  These are compiled lines in the configuration that
produced the admitted record.

No configuration, selector, default, carried-state field, threshold, score
domain, stabiliser, or sea-ice field may change.  The ORCA2 card's six-item
`unmeasured_features` tuple stays frozen.  The held package change is restored
only for the controlled candidate measurement and is removed unless a complete
two-statement landing clears every shared-card gate.

## Existing implementation and instrument

Repository search found the existing round-50 self-describing parser, the
round-49 OVERFLOW trajectory comparison, and the shared compiled
`expose_live_stage_operands` trace.  The trace already returns all three stage
states, operator operands, stage RHS values, raw stage velocities, stage
outputs, and the separately compiled ordinary prognostic result.  This round
extends those instruments; it does not create a second model ladder or
reimplement an operator.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R51-P1 | The operator-produced record reproduces round 50's admission. | Schema/stamps pass at producer `932cbfa9e`; restart and mesh are byte-identical; the inherited comparison reports `exact=24/27`, `changed=3`, `admitted=16`; payload, stamp, and inherited-field plants all refuse. | Any count/hash/status differs: stop and reconcile the record; do not score it. |
| R51-P2 | The live trace is passive and the controlled pair starts from one identical kt=3 state. | For each arm, trace `state_after` is bit-identical to an ordinary compiled step; base and candidate kt=3 input T/S/u/v/ssh are array-equal; resolved config and forcing receipts are identical. | Any observer or input difference: reject the comparison as confounded. |
| R51-P3 | The first base/candidate movement inside kt=3 is the stage-1 Kaa tracer assignment. | Stage-1 entry, momentum boundaries, tracer `rhs_entry`, `after_adv`, and `after_sbc` are array-equal; stage-1 `Kaa_T` or `Kaa_S` is the first unequal recorded boundary. | Any earlier boundary moves: name it and retract the QCO propagation order before further attribution. |
| R51-P4 | With the source-ordered candidate, stage-2 EOS `rhd` is bit-exact on NEMO operands and the first downstream non-bit statement is the SCO HPG accumulator. | Candidate stage-1 Kaa T/S equal NEMO; candidate stage-2 Kmm T/S and recomputed `rhd` equal NEMO bitwise; `after_hpg_u` or `after_hpg_v` is the first later unequal row. | If `rhd` is unequal, EOS owns the next walk; if HPG is exact, continue to VOR/ADV in recorded order.  Record the actual first row without changing the order or bar. |
| R51-P5 | The admitted record names a second statement but does not by itself authorize a landing. | Final package diff equals the base and disposition is HELD unless one second statement is independently source-exact and the pair passes the complete ORCA2/GYRE/tank/generic/DINO gate. | A source-exact pair clears every landing gate: land it and report every moved row. |

Failed predictions remain in the receipt as `REFUTED`; no threshold or scored
cell set changes after measurement.

## Required measurement and landing rule

1. Re-run the round-50 record admission and all three non-vacuous plants.
2. Reconstruct candidate `1722ef29a3c01b094af757520109e55694063d95`
   as the single production QCO association change and verify its package diff.
3. Run base and candidate from the identical OVERFLOW entry through kt=3 with
   fp64/libm and production JIT.  Score every recorded stage boundary in source
   order with `np.array_equal`, bitwise unequal counts, and maximum magnitude.
4. For the first downstream unequal row, replay only the compiled statement
   from NEMO-recorded operands through the production implementation.  A
   one-representable-value plant must make that exact row refuse.
5. A two-statement pair may land only if each statement is independently
   bit-exact given NEMO inputs and the full shared-statement gate named in
   round 49 passes.  Otherwise remove the candidate and hold with the named
   next statement.

## Frozen OPEN

After the pair boundary is named, return to ORCA2's whole-card first non-bit
checkpoint (kt=1 stage-1 T), the independent Decision-52 initial state/year,
and round-20 slow forcing.  Sea ice remains out of scope at
`STOP_SELECTOR_GAP`.

ASKED: continue the admitted cancelling-pair walk under the standing shared-
statement landing rule.  
UNASKED: none.
