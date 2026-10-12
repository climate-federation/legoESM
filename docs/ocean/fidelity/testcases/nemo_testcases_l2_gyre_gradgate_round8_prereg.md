# Preregistration — certified GYRE reverse-mode gradient gate, round 8

Date: 2026-10-12

Incoming lane tip: `f04f9f1d8b3a19968113ab66746866d46a4acdd9`.

This document is frozen before any Round 8 GYRE gradient, timing, memory, or
compiler-footprint measurement. Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gradgate_rounds/round8/`.
No NEMO executable will be run. No physics, card configuration, timestep,
objective, perturbation, direction, finite-difference step, tolerance,
stabilizer, clipping, guard, or `stop_gradient` change is permitted.

## Fixed program and inherited evidence

The only measured card remains `build_nemo_testcase_card("GYRE-zco")` from its
certified day-0 state. Every modeled step enters production
`LatLonCGridOceanModel._step_jitted`. CPU uses the certified fp64/libm policy;
GPU uses fp64/native and every GPU row is labeled **GPU, XLA transcendentals —
not the certified bit-exact policy**. The reverse/forward relative bar remains
`1e-9`, the reverse/central-FD bar remains `0.05`, and the four-field objective,
PRNG keys, `1e-3 K` perturbation, direction, and central-FD step remain frozen.

The operator's six-step CPU attempt at the incoming tip is retained as
UNMEASURED: after 3837 seconds it aborted with
`execution_engine.cc:54] LLVM compilation error: Cannot allocate memory` and
`LLVM ERROR: Unable to allocate section memory!`. This round does not retry a
multi-step CPU reverse. The one-step CPU row remains the certified CPU claim.

## R8-P1: executable refusal classification

Prediction: an agreement row refused before derivatives exist carries
`status="UNMEASURED"` with reverse, forward, finite difference, and both error
fields null. A computed non-finite derivative remains `REFUTED`; it is not
reclassified as absent. Any executable that prints null derivative fields as
`REFUTED`, or a computed NaN as `UNMEASURED`, REFUTES P1. Both states require
direct tests and a nonzero CLI exit.

## R8-P2: cold-step resource segmentation

The first GYRE production step changes the state pytree by materializing TKE,
barotropic-history, and RK3 carry fields. Prediction: every resource schedule
executes that cold production step before entering the shared checkpoint loop,
then applies `none`, `uniform`, or `binomial` only to the remaining warm,
fixed-pytree steps. For six steps the schedule receives the step-1 state and a
five-step loop beginning at model step 2. The zero-step behavior remains the
unperturbed entry state; the one-step behavior is exactly the cold production
step.

A checkpoint loop receiving the day-0 pytree, any carry-structure exception,
an off-by-one forcing time, a production entry other than `_step_jitted`, or a
scheduled primal outside the existing `rtol=1e-9`, `atol=1e-11` value contract
REFUTES P2. A direct GYRE-card test must fail if the cold step is moved back
inside the checkpoint loop.

## R8-P3: explicit per-step transform footprint

Code inspection at the incoming tip finds no whole-window `jit` around the
agreement transform: it is a Python reverse loop. However, that loop creates a
new `jax.vjp` and `jax.jvp` closure for every `kt`, so six steps present six
separate transform closures to JAX. Round 8 will instead reuse one dynamically
timed compiled step VJP and one compiled step JVP. JAX may specialize each once
for the cold day-0 pytree and once for the warm fixed pytree.

Prediction: one step has one VJP/JVP structure specialization; six, sixty, and
360 steps have two, not one whole-window program and not one program per step.
Every transform still differentiates the state-dependent seasonal forcing and
the production `_step_jitted` call. A tracer-count plant must observe one cold
plus one warm trace for a multi-step synthetic program; more than two traces,
closing a static `kt` into each transform, omitting the forcing derivative, or
feeding transformed replay output into a later step REFUTES P3.

The operator log's CPU abort is therefore predicted to belong to compilation
of the warm one-step transpose (or repeated warm specializations), not a single
six-step rollout transpose. Round 8 records the structural footprint in every
agreement JSON. It does not claim LLVM object-code bytes unless an artifact
measures them.

## R8-P4: unchanged CPU row and GPU multi-step acquisition

Prediction: the one-step CPU fp64/libm row reproduces Round 7 within the fixed
bars (reverse and forward approximately `-3.88497963270388e-06`, central FD
approximately `-3.884977393501259e-06`), and the certified ten-step trajectory
remains byte-identical to the incoming tip. The doubled-VJP plant remains
REFUTED. Any changed trajectory byte, off-bar one-step row, or plant that does
not flip REFUTES P4.

The committed GPU-1 runner acquires one-, six-, and 60-step agreement rows and
attempts 360 steps only after the shorter rows produce classified JSON. It
then acquires the three six-step checkpoint resource rows after cold-step
segmentation, continuing after classified `REFUTED`, `UNMEASURED`, `VACUOUS`,
or `RETRACTED` rows and printing a named REFUSE before a final nonzero exit.

Prediction: the six-step GPU relative reverse/forward result remains off the
unchanged `1e-9` bar near `9.7e-8`, while its scale-aware discrepancy
`abs(reverse-forward)/sum(abs(g*d))` is below `1e-12`; the FD error remains
below `0.05`. Sixty and 360 steps are required to be finite and nonzero but may
fail the unchanged relative bar; their measured classification is retained.
Missing cancellation scale, wrong policy/GPU, missing provenance, or a
scale-aware discrepancy at or above `1e-12` REFUTES the cancellation
prediction. No scale-aware value changes the registered verdict.

For each valid six-step resource row, 60-step and 360-step memory projections
use the frozen formulas: linear incremental memory for `none`, constant peak
for `uniform`, and square-root incremental scaling for `binomial`. Missing or
refused resource rows leave projections UNMEASURED. No resource number is
fabricated in the agent sandbox.

## Required verdict

Round 8 is LANDED only if P1-P4's code controls pass, the one-step CPU and
trajectory controls remain unchanged, the exact three-file CPU battery exits
normally, the receipt-specific citation gate and its plant bind, and no
physics or card configuration changes. GPU measurements may remain OPEN only
through the committed operator runner with the CPU compiler limit and every
missing row stated plainly.
