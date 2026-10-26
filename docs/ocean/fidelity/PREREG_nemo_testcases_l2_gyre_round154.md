# Preregistration — NEMO testcase L2 GYRE round 154

Date: 2026-09-22

Incoming lane tip: `d4e238abbed5bb0e702795a1981d9ab239be5db1`.
Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round154/`.  This document is
frozen before admitting or comparing the acquired developed-state FCT record.

## Refused acquisition and mandatory discrimination

The operator ran the Round-153 acquisition to `STOP 0`, but its post-run gate
refused `oracle_developed_rhs_kt00001081.bin`.  Before this preregistration,
the mandatory passive-instrument discrimination established these fixed input
facts:

* all seven NEMO restart files, including steps 1080 and 1081, are byte
  identical between the admitted Round-148 run and the Round-153 run;
* eleven inherited private diagnostic streams differ, while the admitted
  process-budget stream is byte identical; and
* the compiled `traadv_fct` diff adds observation calls and two temporary
  arrays, with assignments inside the production loops at
  `GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo/traadv_fct.f90:113-114`,
  `:332-341`, `:498-634`, and `:924-953`.  The observed NEMO statements are
  unchanged from the Round-148 compiled source at
  `GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/traadv_fct.f90:164-330`,
  `:495-610`, and `:849-936`.

This is the operator-note AO case in which the restart is identical but a
writer changes materialisation of private streams.  Round 153's prediction
that every inherited stream would remain byte identical is **REFUTED** and
will remain visible in the receipt.  It is not relabelled as confirmed.

## Frozen admission correction

The record is admitted only against its own producing target.  The corrected
gate must:

1. require all seven restart files and `mesh_mask.nc` to be byte identical to
   Round 148;
2. enumerate every inherited private stream, report the complete changed and
   unchanged sets, and require the changed set to equal the eleven files
   already observed during discrimination;
3. require the new 61-field FCT record, every private stream consumed by the
   Round-154 comparison, and the step-1081 restart to come from the same
   Round-153 run and binary;
4. retain the exact schema, finite-payload, limiter-activity, stamp,
   truncation, missing-field, coefficients-one, and byte-change controls; and
5. add a restart-byte plant that changes one restart digest, prints
   `STATUS PLANT-FIRED`, and exits nonzero.

Any restart difference, any unregistered changed stream, a zero T or S limiter
coefficient census, or a failed plant refuses the record.  No changed private
stream is compared numerically across the two builds or used as an oracle for
the FCT walk.

## Developed FCT walk

After admission, extend the existing Round-111/112 production-step instrument;
do not create a second FCT implementation.  Drive the production step from
NEMO's exact day-180 entry and expose the values already formed by the shared
FCT implementation.  Compare, in compiled order, the common inputs and both
tracers' first upwind faces, first divergence, midpoint, averaged faces,
upstream divergence and direct RHS write, pre-limiter faces, sign-selected
coefficients, post-limiter faces, final divergence, divisor, and final direct
RHS write.

Report cells unequal and maximum absolute difference under production JIT,
production eager, and isolated-closure JIT.  The production-JIT result is
authoritative.  A one-ULP perturbation to a nonzero recorded transport input
must flip its production-JIT row, print `STATUS PLANT-FIRED`, and exit nonzero.
Every record field is either mapped to a scored model value or explicitly
classified as a context row observed at the caller; an unclassified field is
a hard failure.

Frozen predictions are:

1. both NEMO tracers have at least one directly recorded limiter coefficient
   different from exact one;
2. the first production-JIT non-bit row occurs no later than the first
   horizontal upwind faces, as it did from rest in Round 112; a later or BIT
   result refutes this prediction and is retained;
3. at least one developed-state active-limiter row differs from NEMO, because
   Round 152 measured a non-bit completed FCT contribution; and
4. the first non-bit row, not the largest downstream row, owns the next
   compiled statement walk.

No physics, configuration, default, carried state, restart schema, stabilizer,
year harness, reconciliation gate, freshwater pair, or #1484 guard changes in
this round.  Without a single source-exact production-JIT statement there is
no candidate and no month/year arm.  ORCA2 remains
`UNMEASURED-WITH-SPEC`: repeat this developed-entry active-branch FCT walk on
the ocean-only ORCA2 card before transferring a statement verdict.
