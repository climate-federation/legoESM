# Preregistration — certified GYRE reverse-mode gradient gate, round 9

Date: 2026-10-24

Incoming lane tip: `3c02205c533c94e1fbc0fc3256460b9806be8441`.

This document is frozen before any Round 9 GYRE trajectory, derivative,
timing, or memory measurement. Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gradgate_rounds/round9/`.
No NEMO executable will be run. No physics, card configuration, timestep,
objective, perturbation, direction, finite-difference step, tolerance,
stabilizer, clipping, guard, or `stop_gradient` change is permitted.

## Fixed program and inherited evidence

The only measured card remains `build_nemo_testcase_card("GYRE-zco")` from
its certified day-0 state. Every modeled step enters production
`LatLonCGridOceanModel._step_jitted`. CPU uses the certified fp64/libm policy;
GPU uses fp64/native and every GPU row is labeled **GPU, XLA transcendentals —
not the certified bit-exact policy**. The reverse/forward relative bar remains
`1e-9`, the reverse/central-FD bar remains `0.05`, and the four-field objective,
PRNG keys, `1e-3 K` perturbation, direction, and central-FD step remain frozen.

Round 8 established that the separately executed production step is stable,
but JAX's VJP/JVP replay differs on last bits (maximum `1.4210854715202004e-14`
in salinity). The refused diagnostic derivatives were on bar: reverse
`-3.884979632703884e-06`, forward `-3.884979632703880e-06`, and central FD
`-3.884977393501259e-06`. The binding row is UNMEASURED until the approved
primal-preserving local AD boundary passes its byte gate.

The GPU sandbox limitation remains binding. All GPU rows are acquired only by
the committed operator runner on physical GPU 1. The Round 5 six-step GPU row
remains REFUTED by the unchanged relative bar (`9.689340321095861e-08`) until a
new row proves otherwise; its cancellation-scale metric was not measured.

## R9-P1: local primal-preserving AD boundary

The boundary takes two equal-structure pytrees: a separately executed
production primal and a differentiable replay. Its ordinary output and its
custom-JVP primal output must be the production pytree bit-for-bit, including
signed zeros. Its tangent is exactly the replay tangent, so reverse mode
transposes only the unchanged replay program. The boundary contains no
`stop_gradient`, physical operation, state change, or configuration choice.

Prediction: on a synthetic pair whose primals differ by one ULP and whose
derivatives are deliberately distinct, eager, JIT, JVP, and VJP all return the
first primal bit-for-bit while JVP/VJP select only the replay derivative. A
plant that selects the replay primal must fail the byte gate. Any changed
production bit, derivative routed through the production arm, derivative not
routed through the replay arm, or non-firing plant REFUTES P1.

## R9-P2: certified one-step CPU row and trajectory identity

The local boundary will be applied at each production-step AD boundary. Raw
transform replays remain diagnostic evidence, but the binding byte gate scores
the boundary output actually composed into the gradient program.

Prediction: the certified CPU fp64/libm one-step row becomes CONFIRMED with
reverse and forward near `-3.88497963270388e-06`, relative disagreement below
`1e-9`, and central-FD disagreement below `0.05`. The doubled-VJP plant remains
REFUTED. The separately executed ten-step certified trajectory has the same
scientific JSON and residual NPZ SHA-256 as the incoming-tip control. Any
changed trajectory byte, off-bar derivative row, boundary output mismatch, or
plant that does not flip REFUTES P2.

## R9-P3: GPU agreement at training precision

The committed operator runner first remeasures one day (six 14400-second
steps), then measures 60 steps (ten days) and 360 steps (60 days) under
fp64/native. Every row records reverse, forward, central FD, the unchanged
relative bars, `sum(abs(g*d))`, the scale-aware reverse/forward discrepancy,
policy, card, commit, and boundary-byte status. The one-step control is also
retained.

Prediction: every row is finite and nonzero, the one-step row reproduces the
prior GPU value within its unchanged bars, and the boundary output remains
bit-identical to the separately executed production primal. The six-step row
is predicted to retain a relative disagreement near `1e-7` but to have a
scale-aware discrepancy below `1e-12`; that prediction does not change its
binding REFUTED verdict if the `1e-9` relative bar still fails. The 60- and
360-step relative and FD verdicts are measurements, not assumed passes. A row
is admissible for training only if both unchanged agreement bars pass.

Missing output, a non-finite/zero gradient, any byte mismatch, wrong policy or
device, or a classified row hidden by the runner REFUTES P3. The runner must
continue to resource acquisition after a classified agreement failure and
finish with a named REFUSE plus nonzero exit.

## R9-P4: measured GPU memory and warm reverse cost

For each `none`, `uniform`, and `binomial` schedule, the operator runner
measures fresh-process peak device memory and warm reverse wall time at 6, 60,
and 360 steps. The cold first step remains outside the warm fixed-pytree loop.
Timing covers the blocked second execution of the compiled reverse only. JAX
allocator peak statistics and sampled `nvidia-smi` device memory are both
recorded with preallocation disabled; the platform allocator remains refused.

Prediction: valid rows have finite nonzero gradients, at least two memory
samples, nonnegative memory growth, positive warm time, and binding provenance.
`none` memory grows with window length, `uniform` is bounded relative to
`none`, and `binomial` lies below `none` at 360 steps. Exact memory and time are
not predicted. A resource ordering plant must flip the suite. No projection is
substituted for a requested 60- or 360-step measurement.

## Required verdict

Round 9 is LANDED only if P1 and P2 pass, the direct controls and exact
three-file CPU battery exit normally, the citation gate and shifted-citation
plant bind, no physics/configuration changes, and the GPU acquisition script is
committed and fail-closed. GPU agreement and resource rows remain
ACQUISITION_NEEDED until the operator runs that script; the receipt may not
promote an unacquired multi-day row to CONFIRMED.
