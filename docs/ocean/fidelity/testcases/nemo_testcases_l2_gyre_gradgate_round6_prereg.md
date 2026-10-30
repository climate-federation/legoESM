# Preregistration — certified GYRE reverse-mode gradient gate, round 6

Date: 2026-10-12

Incoming lane tip: `8b59d6657337e30ae2d218819ab5fe01e127e740`.

This document is frozen before any Round 6 GYRE gradient, timing, memory, or
compiler measurement. Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gradgate_rounds/round6/`.
No NEMO executable will be run. No physics, card configuration, timestep,
objective, perturbation, direction, finite-difference step, tolerance,
stabilizer, clipping, guard, or `stop_gradient` change is permitted.

## Fixed program and source identity

The program remains `build_nemo_testcase_card("GYRE-zco")` from its certified
day-0 state. Every modeled step enters production
`LatLonCGridOceanModel._step_jitted`; CPU rows use the certified fp64/libm
policy. The objective remains the four-field live-volume-normalized after-state
loss reused from the GYRE gradient twin (`901cfe0d8`), evaluated at the fixed
PRNG-0 `1e-3 K` wet-cell perturbation in the fixed PRNG-42 unit direction.

The reverse/forward relative-error bar remains `1e-9`; the reverse/central-FD
bar remains `0.05`, with central-FD step `1e-3`. Round 5's accepted one-step
row was reverse `-3.8849796327038826e-06`, forward
`-3.884979632703889e-06`, central FD `-3.884977393501259e-06`, reverse/forward
relative error `1.7442211333597704e-15`, and reverse/FD relative error
`5.763743532444348e-07`.

The compiled NEMO GYRE program disables FCT in RK3 stages 1 and 2 at
`cfgs/GYRE_OMIP_L2_P3_SM/WORK/traadv.F90:295-298`, dispatches those stages to
`tra_adv_cen` at `:374-380`, and divides tracer flux divergences inside its
owned-cell loops at `cfgs/GYRE_OMIP_L2_P3_SM/WORK/traadv_cen.F90:137-149,203-216`.
The landed active-cell denominator treatment is unchanged. The NEMO TKE
dissipation statement remains `cfgs/DINO/MY_SRC/zdftke.F90:832-837`; its
landed primal-identical reverse guard is unchanged.

## R6-P1: cold-step/warm-loop segmentation

The user-approved program makes step 1 a separate adjoint segment, then applies
the selected shared checkpoint schedule only to steps 2 through N. This is
required because the certified cold step replaces three `None` leaves with the
warm fixed pytree; JAX scan carries require identical input/output structure.
The cold segment and every warm iteration still call production
`_step_jitted`. No scan spans the cold-to-warm transition.

Prediction: for one and six steps, each of `none`, `uniform`, and `binomial`
produces a primal state byte-identical to the unchanged production Python loop.
A changed state leaf, a schedule that bypasses `_step_jitted`, a cold transition
inside `lax.scan`, or a schedule-dependent objective REFUTES P1. A direct
synthetic cold-to-warm pytree test must fail for the old unsegmented loop and
pass for all three segmented schedules.

## R6-P2: CPU agreement and resource rows

Prediction: all three segmented six-step CPU fp64/libm rows compile and return
finite, nonzero reverse, forward, and central-FD directional derivatives that
clear the unchanged `1e-9` and `0.05` bars. Agreement is expected across
schedules to those bars; last-bit equality between schedules is not required.
Any non-finite or zero derivative, failed compilation, primal mismatch, or
relative error above either bar REFUTES the corresponding row. If LLVM section
allocation still aborts, retain the exact message and measure the largest
step count that compiles, without relabeling it as a six-step result.

For each schedule, peak host RSS is sampled during a blocked warm reverse pass;
the JSON records sampled baseline, absolute peak, increment, sample count, cold
compile-plus-reverse wall time, and warm reverse wall time. A dirty/stale
commit, non-fp64 operand, wrong backend/policy, non-finite gradient, sampler
failure, or schedule primal mismatch invalidates the resource row. The
deliberately doubled-VJP plant must remain REFUTED while the unplanted synthetic
control remains CONFIRMED.

## R6-P3: GPU rows and projections

If the operator's Round 5 `gpu1_*.json` files exist, copy no artifact into git;
verify their clean-commit provenance, explicit
`PrecisionPolicy.fp64(transcendentals="native")`, physical GPU-1 execution,
finite gradients, and label them **GPU, XLA transcendentals — not the certified
bit-exact policy**. Missing or refused JSON remains UNMEASURED.

The projections are frozen before reading any GPU number. For each schedule let
`B` be the six-step sampled baseline and `P6` its sampled peak. The 60-step
projection is `B + 10*(P6-B)` for `none`, `P6` for `uniform`, and
`B + (P6-B)*ceil(sqrt(60))/ceil(sqrt(6)) = B + (P6-B)*8/3` for `binomial`.
The 360-step projection is `B + 60*(P6-B)` for `none`, `P6` for `uniform`, and
`B + (P6-B)*ceil(sqrt(360))/ceil(sqrt(6)) = B + (P6-B)*19/3` for `binomial`.
These are schedule-complexity extrapolations from measured six-step peaks, not
allocations or successful long-window runs. No projection is reported without
a valid six-step GPU resource JSON for that schedule.

## Scope and required verdict

Round 6 does not add VORTEX or TSUNAMI gradient rows. The reviewer-requested
cross-card trajectory pairs are reported only if an allowed pre-fix source is
available; construction fingerprints are not relabeled as trajectory evidence.
SWG remains OPEN until its certified card exists.

Round 6 is LANDED only if the segmentation's direct tests pass, the exact
three-file CPU battery remains green, and every successfully compiled six-step
CPU row is classified fail-closed against the registered bars. Failed
predictions and compiler failures remain in the receipt.
