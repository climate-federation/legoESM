# NEMO testcase lane 2 GYRE — phase-3 round-14 preregistration

Date: 2026-09-04
Session: `01a05cb9-7625-7f40-9e83-b3fa767b1945`
Reviewed baseline: `b947a570e3abb01252e57eed7c34db882dec49b3`

## Asked cross-card criterion change

The user changed the shared-code cross-card admission rule on 2026-09-04.
This is an ASKED policy change, not an operator hypothesis.  The shared
`--compare-to` path will compare each scored cell's absolute legoESM-to-NEMO
residual before and after a change.  It will fail when any cell worsens by
more than two float64 ulps of the oracle value at that cell, any row changes
AT-BAR to DEBT, or `first_over_bar` moves to an earlier step.  Improvement
toward NEMO is unrestricted.  Previous-legoESM movement remains measured and
reported but is not an admission criterion.  Every scored residual field will
be stored in a compressed, hashed NPZ sidecar so a scalar reduction cannot
hide compensating cells.

Three controls are registered before implementation:

1. increasing one cell's residual by three local oracle ulps MUST fail on the
   cell criterion;
2. decreasing one cell's residual MUST pass when every other input is held
   fixed; and
3. changing one row from AT-BAR to DEBT MUST fail even when its stored fields
   are unchanged.

The current OVERFLOW and LOCK_EXCHANGE stage and `kt=1…10` trajectory reports
will be compared with oracle-relative residual fields from clean
`57429ecf5f3` runs.  DINO will be run only if source search finds a CPU
single-step term gate whose documented/observed wall clock is below 20
minutes; otherwise its absence or cost is recorded as NOT RUN.

## Round-12 follow-ups

No physics change is registered here.  A no-`/data` CI test will pin uint64
hex patterns for fixed GYRE oracle T/S/depth inputs through the shared `prd`
chain.  The EOS docstring will say what the recorded optimized HLO actually
shows: the explicit `optimization_barrier` operations are eliminated (105 to
0), while the finite-select IEEE identity remains.  The isomorphism map will
list, as PLAUSIBLE debt only, the named shared recurrences that currently rely
on `optimization_barrier` alone.  None will be modified in this round.

## Next measured boundary: stage-1 `un_adv` / `vn_adv`

Round 13 leaves the first non-bit-exact primitive input to the stage-1 native
transport at `un_adv`/`vn_adv`, with maxima `1.20e-13` U and `5.50e-14` V.
Before changing arithmetic, the config-local WRITE-only NEMO record will dump
the external-mode substep transports, their two weight operands, each
source-ordered accumulated partial sum, and the final value handed to
`stprk3_stg`.  Shipped NEMO source remains untouched.

The preregistered candidate order is:

1. **Weighted-mean accumulation association/order.**  CONFIRM if all dumped
   substep values and weights match independently but the first accumulated
   partial sum differs, and a one-variable source-order arm makes the final
   `un_adv`/`vn_adv` bit-exact (or moves it at residual scale).  The 800
   previously matched barotropic frames certify states, not this mean.
2. **Weight construction/value.**  CONFIRM if the first non-bit-exact operand
   is `wgtbtp1` or `wgtbtp2`, traced back to its NEMO construction rather than
   inferred from the final mean.
3. **Substep transport.**  CONFIRM only if a dumped transport entering the
   sum is first non-bit-exact.  A matching state frame does not exonerate a
   separately formed transport operand.
4. **Mean-to-stage hand-off.**  CONFIRM if the accumulated mean is bit-exact
   at `dynspg_ts` exit but differs when read by the stage transport statement.

Attribution requires scaling before an owner label and a private one-variable
arm.  A shared external-mode fix lands only when NEMO's statement is reproduced
without a GYRE switch.  Then stage-1 tracers, stage-2 Kaa, GYRE `kt=1…10`, and
the new oracle-relative OVERFLOW/LOCK_EXCHANGE gates are rerun.  If no
candidate clears the boundary, it remains honest DEBT and the register advances
only after its enumerated boundaries are exhausted.  No independent review of
this preregistration has occurred.
