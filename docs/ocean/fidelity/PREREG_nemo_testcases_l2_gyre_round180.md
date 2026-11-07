# Preregistration — round 180, developed `ldf_slp` causal walk

Committed before admitting, parsing, or scoring the acquired Round-179 record.
The operator reports NEMO `STOP 0`, byte-identical step-1080 and step-1440
restarts, and `ROUND179_SLOPE_WALK_READY`, but this round independently runs
the committed fail-closed admission and corruption plant before interpreting
any scientific row.

No production physics, configuration, carried state, restart schema, card
default, or immutable trajectory changes unless the walk names a
one-variable NEMO-exact statement and that candidate passes the complete
Decision 43/45/55/59 trajectory and card gate.

## Compiled order

The acquired build calls `ldf_slp(kstp,rhd,rn2b,Nbb,Nbb)` at
`GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/stprk3.f90:173-180`.
The compiled routine reads `nmln`, forms the live mixed-layer depths and
inverse depths at
`GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/ldfslp.f90:187-229`,
forms the descending horizontal and vertical density gradients at
`ldfslp.f90:231-261`, and then forms the face gradients, limiter, mixed-layer
selectors, live depths, raw slopes, recurrence, and Shapiro output at
`ldfslp.f90:270-361`.  The score table follows that order and distinguishes
direct inherited inputs from statements owned by `ldf_slp`.

## Frozen predictions and falsifiers

1. The committed `--admit-existing` path passes the 7,324,076-byte record,
   both restart comparisons, current-commit stamp, integer selector checks,
   and exact in-run Shapiro reconstruction.  Any failure stops scientific
   interpretation.
2. `--plant-admission` corrupts one magic byte, prints
   `STATUS PLANT-FIRED: record-magic`, and exits nonzero.  A zero exit or a
   failure before the corrupt byte is checked invalidates the admission.
3. The ordinary production step and each diagnostic production step differ
   in zero carried-state bytes under both complete JIT and complete eager
   execution.  Any moved byte invalidates that execution mode.
4. The first non-bit causal row is predicted to be inherited `nmln`, the first
   state-dependent branch input read by the compiled routine.  A bit-exact
   `nmln`, or an earlier non-bit row in compiled order, **REFUTES** this
   prediction.
5. Inherited `prd` and `pn2` are each predicted non-bit.  Either bit-exact row
   **REFUTES** its individual prediction.  They remain classified as inherited
   inputs even if their first compiled use occurs after `nmln`.
6. Production JIT is authoritative.  Complete eager execution is predicted
   to name the same first causal row; disagreement retains both tables and
   adopts the production-JIT result.
7. The first non-bit owned statement is predicted to be downstream of the
   first non-bit inherited input, rather than an independently wrong local
   transcription.  If any owned row is non-bit while all of its registered
   operands are bit-exact, this prediction is **REFUTED** and that statement
   becomes the candidate.
8. A one-ULP production-step perturbation to the named first causal row must
   move its registered immediate consumer while every preceding registered
   row remains unchanged.  A blind plant or upstream movement invalidates the
   instrument.

If the first causal row is inherited, Round 180 names it and stops at that
boundary without changing a downstream slope statement.  A source-exact
one-variable statement may land only after the complete trajectory/card/tank
gate; otherwise the round remains HELD.
