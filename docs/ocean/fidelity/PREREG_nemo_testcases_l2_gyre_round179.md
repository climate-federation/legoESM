# Preregistration — round 179, developed `ldf_slp` causal-input record

Committed before building, parsing, or interpreting the Round-179 record.
Round 178 established that the first returned developed tracer-LDF row is
`uslp`: 16,820 of 17,400 active U-face cells differ, with maximum absolute
difference `1.84356055604884082e-04`, under both the complete production JIT
step and complete eager execution.  The admitted Round-177 stream contains
that output but not the ordered inputs and intermediates that produce it.
This round therefore extends the existing developed tracer-LDF walk with one
passive `ldf_slp` record.  It does not create a second model-side harness.

No production physics, configuration, carried state, restart schema, card
default, or immutable Round-163 trajectory changes.

## Compiled program and record

The acquired Round-177 build proves the active path.  The step computes
`rhd` and calls `ldf_slp(kstp,rhd,rn2b,Nbb,Nbb)` at
`GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/stprk3.f90:173-180`.
Inside the compiled routine, the ordered program is:

1. read `nmln` and form the live mixed-layer depths at
   `GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/ldfslp.f90:156-180`;
2. form masked horizontal density gradients and initialise the descending
   vertical recurrence at `ldfslp.f90:183-213`;
3. form the U/V density-gradient, limiter, mixed-layer selector, live-depth,
   raw-slope and recurrence rows at `ldfslp.f90:222-257`; and
4. apply the Shapiro stencil to write `uslp`/`vslp` at
   `ldfslp.f90:260-276`.

The new target is `GYRE_OMIP_L2_P3_SM_R179SLPWALK`.  It copies the admitted
Round-132 daily-restart source card file by file, adds only an `ldfslp.F90`
observer, and runs through step 1440.  At step 1081 the observer records the
direct `prd`, `pn2`, `nmln`, live QCO ratios, masks and metrics, followed by
full-level copies of every ordered U/V intermediate required to reach the
filtered slopes: `zhmlpt`, inverse mixed-layer depths, `zgru/zgrv`, `zdzr`,
`zau/zav`, the raw and limited denominators, `iku/ikv`, selectors, live face
depths, raw `zwz/zww`, the mixed-layer accumulators before and after each
level, and `uslp/vslp`.

## Frozen predictions and falsifiers

1. The existing records do not contain those causal rows.  Finding an
   admitted record with all of them **REFUTES** the need for acquisition and
   stops the new writer before a NEMO build.
2. The first non-bit causal row is predicted to be the developed mixed-layer
   index `nmln`, the first state-dependent branch input read by the compiled
   routine.  A bit-exact `nmln`, or any earlier non-bit direct input in the
   compiled table, **REFUTES** this prediction; the failed prediction remains
   in the Round-180 receipt.
3. `prd` and `pn2` are predicted non-bit but downstream of `nmln` in compiled
   execution order.  Either row being bit-exact **REFUTES** its individual
   prediction and is retained.
4. The observer is passive only if the candidate step-1080 and step-1440
   restart files are byte-identical to the uninstrumented Round-132 daily
   restarts.  Any moved byte refuses the record; an inherited stream from a
   differently instrumented build is informational only under the developed-
   record admission policy.
5. The fixed header, field inventory, dimensions, time levels, dtype and byte
   count are derived from the literal compiled writer.  A one-byte magic
   corruption must print `STATUS PLANT-FIRED` and exit nonzero.  Removing one
   registered intermediate from the dry source must print a named refusal and
   exit nonzero.
6. The stored filtered `uslp` and `vslp` are compared with the admitted
   Round-177 values over their active faces for information.  Because those
   streams come from differently instrumented builds, their last-bit
   difference is not an admission refusal.  Binding passivity is restart
   identity; binding in-run calibration rebuilds the Shapiro outputs from the
   new record's own raw slopes, masks, and literal compiled association.
7. Round 180 extends `developed_tracer_ldf_statement_walk` itself, executes
   the complete production JIT step and complete eager step from the same
   NEMO day-180 entry, and scores the record in compiled order.  A one-ULP
   plant on the named first causal row must move its registered consumer while
   all earlier rows stay unchanged.

No landing is eligible from this acquisition-only round.  If Round 180 names
a one-variable NEMO-exact statement, it must pass the full Decisions
43/45/55/59 trajectory, card, tank, and DINO gates before landing.
