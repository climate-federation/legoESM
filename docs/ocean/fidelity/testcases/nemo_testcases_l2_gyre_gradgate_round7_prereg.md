# Preregistration — certified GYRE reverse-mode gradient gate, round 7

Date: 2026-10-12

Incoming lane tip: `2b83ae6a986e5a67acc8eb2d4f16a36f59ed7989`.

This document is frozen before any Round 7 GYRE gradient, timing, memory, or
compiler measurement. Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gradgate_rounds/round7/`.
No NEMO executable will be run. No physics, card configuration, timestep,
objective, perturbation, direction, finite-difference step, tolerance,
stabilizer, clipping, guard, or `stop_gradient` change is permitted.

## Fixed program and imported instrument

The only measured card remains `build_nemo_testcase_card("GYRE-zco")` from its
certified day-0 state. Every modeled step enters production
`LatLonCGridOceanModel._step_jitted`; CPU rows use the certified fp64/libm
policy. The objective remains the four-field live-volume-normalized after-state
loss reused from the GYRE gradient twin (`901cfe0d8`), evaluated at the fixed
PRNG-0 `1e-3 K` wet-cell perturbation in the fixed PRNG-42 unit direction.

The reverse/forward relative-error bar remains `1e-9`; the reverse/central-FD
bar remains `0.05`, with central-FD step `1e-3`. Round 6's accepted one-step
row was reverse `-3.8849796327038826e-06`, forward
`-3.884979632703889e-06`, central FD `-3.884977393501259e-06`, reverse/forward
relative error `1.7442211333597704e-15`, and reverse/FD relative error
`5.763743532444348e-07`.

The per-step composition, compiled reverse timer, CPU RSS sampler, GPU allocator
instrument, and resource-suite classifier are ported from the independently
reviewed SWG gradient-gate lane, commits `79f086a51`, `d5292ab5c`,
`93c294b47`, `9259e21f3`, and `afd3350a6`. The SWG card dispatch and SWG
measurements are excluded. Review findings in
`codex_review_swggg_r3.log` require two additions here before measurement:
historical uncompiled-transform timings must be executable retractions, and
time-scaling claims must have a suite classifier plus a constant-time plant.

## R7-P1: production identity and per-step composition

Prediction: the unchanged one-step and ten-step production GYRE trajectories
remain byte-identical to the incoming tip. For one, six, and sixty steps, the
gradient composition calls the production jitted step once per modeled step in
an explicit Python loop; the zero-perturbation result is byte-identical to the
direct production Python loop. A changed trajectory row, changed residual NPZ
hash, mismatching state leaf, scan replacing the agreement composition, or any
step bypassing `_step_jitted` REFUTES P1.

The three checkpoint resource schedules retain the repository checkpoint
schedule's frozen value contract (`rtol=1e-9`, `atol=1e-11`) while every body
evaluation calls the production jitted step. Any non-finite leaf, structural
mismatch, or value outside that contract REFUTES the resource row. This
contract does not weaken the byte gate on the agreement composition.

## R7-P2: CPU agreement and cancellation scale

Prediction: the one-step and six-step CPU fp64/libm rows compile and return
finite, nonzero reverse, forward, and central-FD directional derivatives that
clear the unchanged `1e-9` and `0.05` bars. The one-step values reproduce the
Round 6 row above. The six-step CPU reverse/forward relative discrepancy is
predicted below `1e-12`; a result near the GPU's `9.7e-8` instead would support
a composition-wide cancellation explanation rather than a GPU-only one.

Every row records `sum(abs(g*d))`, `norm(g)*norm(d)`, the absolute
reverse/forward discrepancy, and `abs(reverse-forward)/sum(abs(g*d))`.
Non-finite or zero derivatives, an error above either registered bar, a
missing cancellation scale, or production-byte mismatch REFUTES the row. The
doubled-VJP plant must retain forward and FD, double reverse, classify REFUTED,
and exit nonzero.

Prediction: a sixty-step CPU agreement row compiles under the per-step
production composition and clears both registered bars. A compiler abort,
non-finite derivative, or off-bar result is retained as REFUTED or UNMEASURED;
the largest successful window is reported without extrapolating an agreement
verdict.

## R7-P3: CPU resources and executable timing claims

For `none`, `uniform`, and `binomial`, six-step and sixty-step resource rows run
in separate fresh processes. Each times only the second blocked execution of
`jax.jit(jax.value_and_grad(scalar_line))` after one cold compile-and-execute
pass, and samples absolute host peak RSS during that warm execution. A row is
VACUOUS unless it stamps that transform and timer span, has finite nonzero loss
and gradient, positive elapsed times, at least two samples, and a peak above
baseline. An old row lacking the compiled-transform/timer stamps must be
classified RETRACTED, never CONFIRMED.

Prediction: warm execution time grows from six to sixty steps for every
schedule. The executable scaling suite requires positive fitted slope and the
sixty-step time to exceed the six-step time. A planted constant-time suite must
classify REFUTED and exit nonzero. Failure of either condition REFUTES the
schedule's time-scaling claim; no prose exception can promote it.

## R7-P4: GPU carry-forward and projections

Round 5's operator one-step GPU fp64/native row remains CONFIRMED. Its six-step
row remains OFF-BAR under the unchanged relative bar: reverse
`-8.185163817998039e-08`, forward `-8.185163024909660e-08`, absolute difference
`7.931e-15`, and relative difference `9.689340321095861e-08`. These are labeled
**GPU, XLA transcendentals — not the certified bit-exact policy**.

Prediction for an operator rerun with the cancellation diagnostic: the
six-step GPU scale-aware discrepancy is below `1e-12` even though the unchanged
purely relative `1e-9` bar remains failed. A scale-aware value at or above
`1e-12`, missing provenance, wrong policy, or wrong physical GPU REFUTES this
prediction. No bar changes in this round.

For every valid measured six-step resource row, the 60-step and 360-step memory
projections use the frozen schedule-complexity formulas: linear incremental
memory for `none`, constant peak for `uniform`, and
`ceil(sqrt(target))/ceil(sqrt(6))` incremental scaling for `binomial`. Wall-time
projections use the measured per-step slope only when the executable scaling
classifier passes. Missing or refused CPU/GPU rows remain UNMEASURED; no number
is fabricated.

## Required verdict

Round 7 is LANDED only if the production trajectory and one-step gradient stay
unchanged, the six-step CPU agreement clears both registered bars, all resource
and scaling classifiers plus their plants are binding, the exact three-file
CPU battery and receipt-specific citation gate pass, and no physics or card
configuration changes. The sixty-step row and GPU resource rows may remain OPEN
only with their exact failure or operator requirement recorded.
