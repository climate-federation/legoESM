# NEMO testcase lane 2 GYRE — stage-3 WZV SSH operand preregistration

Date: 2026-09-03  
Parent register: stage-3 WZV input-transport arm measured.

The source-faithful shared-transport landing reduces the stage-3 `pFw`
normalized residual from `2.4980249777374853e-13` to
`2.3530283179984255e-13`.  Replacing only `pFu/pFv` with the oracle transport
reduces it further to `1.011809215412126e-14`, so horizontal transport is
causal but not the sole owner.  The next input group in the literal recurrence
is the SSH triplet: `Kbb=N`, `Kmm=N+1/2`, and `Kaa=N+1` are fixed by
`stprk3_stg.F90:218-235`; `wzv_RK3_t` forms the free-surface term from
`r3t(Kaa)-r3t(Kbb)` while `div_hor_RK3_t` divides and remultiplies by
`e3t(Kmm)` (`sshwzv.F90:317,332-336`; `divhor.F90:116-123,140-142`).

## Fixed arm and outcomes

Run one private CPU/fp64 arm that keeps the faithful candidate `pFu/pFv` but
substitutes the oracle stage-2 SSH for `Kmm` and oracle stage-3 SSH for `Kaa`;
the pinned initial SSH remains `Kbb`.  Score `ww` before/after the inactive
adaptive partition and final `pFw` pointwise at the immutable normalized
`1e-15` bar.  Compare arm movement with the faithful residual before assigning
any owner label.

- If the arm becomes AT-BAR and moves at least 99% of the faithful residual,
  label the SSH operand triplet CONFIRMED_CAUSAL_OWNER.
- If it moves at residual scale but remains DEBT, label it NOT_SOLE_OWNER and
  advance to the next source-ordered operand.
- If movement is below 1% of the faithful residual, label it NEAR-NULL / NO
  DISCRIMINATING POWER, not exoneration.

A planted wet-cell perturbation must fail by at least `0.9`.  This diagnostic
arm is private and cannot select a public card.
