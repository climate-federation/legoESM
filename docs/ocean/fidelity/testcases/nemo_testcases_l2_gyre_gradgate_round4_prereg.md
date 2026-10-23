# Certified GYRE reverse-mode gradient gate — round 4 preregistration

Date frozen: 2026-10-11

Incoming lane tip: `d678430f3e473286e6f15c41e4bbadc856aa57db`.

This file is frozen before the round-4 pre-fix trajectory control, code change,
or new gradient measurement.  The numerical protocol and bars remain those in
`nemo_testcases_l2_gyre_gradgate_round1_prereg.md`: the certified `GYRE-zco`
card, CPU fp64/libm, production `LatLonCGridOceanModel._step_jitted`, the
four-field live-volume-normalized after-state loss, the registered PRNG-0
evaluation point and PRNG-42 direction, reverse mode versus forward mode and
central finite differences, reverse/forward relative error at most `1e-9`,
reverse/FD relative error at most `0.05`, and finite nonzero derivatives.  No
card configuration, objective, direction, finite-difference step, threshold,
clipping, stabilizer, or `stop_gradient` changes are permitted.

## Authorized change

Round 3 localized the first non-finite transpose to both live-`e3t(Kmm)`
divisions in `_nemo_cen2_tracer_rhs`: 3,120 level-cells on the 104-cell dry
outer perimeter.  NEMO executes those statements only on its owned tracer-cell
loop domain (`traadv_cen.F90:137-149,203-216`); legoESM stores the cropped
perimeter with zero thickness and masks it after the literal division.

The authorized change uses the real `e3t_kmm` on active tracer cells and a
benign nonzero denominator on inactive cells, then explicitly selects exactly
zero for the inactive contribution.  The same double-`where` construction is
applied independently to the horizontal and vertical divisions.  It changes no
active-cell arithmetic, physics, configuration, or wet-cell primal value.

## Registered predictions and falsifiers

1. **Pre-fix control remains DEBT.**  Before the change, the one-step row will
   retain reverse `NaN`, forward `-3.884979632703889e-06`, and central FD
   `-3.884977393501259e-06` (within normal deterministic fp64 representation).
   A finite reverse value, changed forward/FD row, or different first owner
   refutes this control.
2. **Active-cell primal identity.**  The ten-step GYRE trajectory gate run
   before and after the change in the same CPU fp64/libm environment will have
   identical row names, statuses, and `absolute_max` values; every compared
   trajectory field will be byte-identical.  Any changed bit refutes the
   authorized construction and holds the round.
3. **The first reverse NaN is removed.**  After the change, the one-step
   reverse, forward, and central-FD directional derivatives will all be finite
   and nonzero, with reverse/forward relative error at most `1e-9` and
   reverse/FD relative error at most `0.05`.  Any non-finite value, zero,
   out-of-bar comparison, or new first non-finite statement refutes this
   prediction and holds the round.
4. **The plant still fires.**  The deliberately doubled custom VJP will turn
   the otherwise confirmed one-step classifier row into `REFUTED` and a CLI
   plant run will exit nonzero.  A plant that remains confirmed invalidates the
   instrument.
5. **Direct inactive-cell transpose control.**  With zero `e3t_kmm` and zero
   tracer mask on an inactive cell, both CEN2 contributions and every operand
   cotangent there will be finite and exactly zero.  The same test must retain
   bitwise equality with the existing literal-source result on positive active
   cells.  Failure of either half rejects the change.
6. **Six-step CPU remains a compiler OPEN item.**  The pytest parameter stays
   skipped before compilation with Round 3's exact LLVM section-memory reason;
   no successful six-step gradient or resource value is predicted in this
   round.  The one-step production row is the binding CPU gradient gate.

The certified-card regression battery includes the GYRE ten-step trajectory
control, card recipe/hash tests for GYRE, TSUNAMI, VORTEX, and VORTEX_VEC, the
focused LOCK_EXCHANGE/OVERFLOW/DINO/ORCA2 tests already provided by the
repository, the full CPU step-gradient matrix, and the testcase receipt
citation gate.  Existing NEMO-comparison DEBT remains DEBT; only a changed
trajectory row or test status attributable to this edit is a regression.

## Required verdict

Round status is **LANDED** only if the pre/post GYRE trajectory is byte-identical,
the one-step production row clears both unchanged agreement bars, and the plant
still exits nonzero.  Otherwise it is **HELD**.  GPU rows, six-step CPU resource
rows, 360-step memory extrapolations, and the additional certified-card gradient
selectors remain OPEN unless independently measured; no value is inferred from
a skip or failed compile.
