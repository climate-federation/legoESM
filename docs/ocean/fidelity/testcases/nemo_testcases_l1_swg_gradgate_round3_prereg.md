# Certified SWG reverse-mode gradient gate — round 3 preregistration

Date: 2026-10-12

Status: **FROZEN BEFORE ROUND-3 MEASUREMENT**. This round repairs the two
independently confirmed round-2 GPU execution defects and replaces the
retrace-contaminated CPU wall timer with a compiled-executable timer. It
changes no physics, card input, loss, random direction, finite-difference
step, agreement bar, checkpoint schedule, stabilizer, clip, guard, or
`stop_gradient`.

## Review findings fixed before measurement

The round-2 GPU resource path is unmeasured, not negative. Its forced
`XLA_PYTHON_CLIENT_ALLOCATOR=platform` makes
`jax.devices()[0].memory_stats()` return `None` on the operator's GPU. The
runner now requires `XLA_PYTHON_CLIENT_PREALLOCATE=false`, refuses the
platform allocator, and lets JAX use its default allocator. A missing stats
dictionary or either missing required counter remains a hard refusal. The JSON
records both allocator environment variables and the complete available
integer stats dictionaries.

The round-2 path also stopped its cold-pass sampler and tried to restart the
same `threading.Thread` for the warm pass. Cold and warm passes now each own a
new sampler. A direct CPU-backed control executes the GPU two-pass control flow
and requires two distinct, single-use sampler instances.

Prediction G1: with the operator environment above, every GPU resource process
reports JAX `bytes_in_use` and `peak_bytes_in_use`; missing stats are
UNMEASURED/VACUOUS. At 24, 720, and 1440 steps both the JAX allocator and the
secondary `nvidia-smi` sampler must report positive increments ordered
`none > binomial > uniform`. All conditions CONFIRM; a positive wrong ordering
REFUTES; missing or non-positive visibility is VACUOUS. The existing forced
uniform-label/none-measurement plant must make the suite REFUTED.

## CPU timer correction

The round-1/2 `warm_reverse_wall_seconds` span called an un-jitted
`jax.value_and_grad` transformation. Although its clock excluded sampler
shutdown and thread join, the second call still retraced and compiled the
transformed Python program. The published 24/720/1440 warm wall-time columns
are therefore **RETRACTED AS EXECUTION-COST MEASUREMENTS**. Their absolute
host-RSS peaks remain process-footprint observations. Their increments are not
adjoint-tape measurements because the post-cold baseline already includes
retained compiler and allocator memory.

The repaired path constructs `jax.jit(jax.value_and_grad(line))` once. The
cold call compiles and executes it. The timed warm span starts after a fresh
sampler is running, calls the already-compiled executable, blocks on its value
and gradient, and stops before sampler shutdown/join. The JSON states that
span. A trace-count control requires two calls to trace exactly once.

The registered SWG resource windows are exactly 24, 96, 360, 720, and 1440
steps; every other count is refused. Each of `none`, `uniform`, and `binomial`
runs in a separate CPU process under fp64/libm from the same admitted kt=2
stage-1 boundary. The schedule, objective, direction, and production step are
unchanged.

Prediction T1: for each schedule, an ordinary least-squares affine fit
`warm_seconds = intercept + slope * steps` over the five registered rows has a
strictly positive slope, and the directly measured 1440-step warm execution is
slower than its 720-step execution. Either non-positive result REFUTES the
per-step scaling prediction. Fit residuals and the intercept are descriptive;
no new timing acceptance bar is introduced.

Prediction T2: the uniform schedule's 1440/720 warm-execution ratio lies in
`[1.5, 2.5]`, centered on the linear-work prediction `2.0`. A value outside
that interval REFUTES the stated scaling prediction. The direct uniform
1440-step row is the reported true CPU cost of a sixty-day gradient; no
extrapolation substitutes for it.

Prediction M1: every CPU row records a finite absolute peak RSS not below its
baseline and at least two samples. Absolute peak RSS is the honest total
process footprint during the measured pass. The baseline-subtracted increment
is retained only as a diagnostic and is not called adjoint memory. The
preregistered schedule-primal contract, finite nonzero loss/gradient, and
positive cold and warm times remain unchanged.

## Retained science and composition

No directional-agreement row is changed. The certified CPU results remain
reverse/forward/central-FD agreement at 1, 24, 720, and 1440 steps, with the
same `1e-9` and `0.05` bars and the binding broken-VJP plant. GPU agreement
rows already acquired at the round-2 tip remain valid and are not resource
measurements.

Training continues to use the measured per-step Python composition through
the production jitted `swe_rk3` step. The uniform scan is retained only for
checkpoint-resource measurement. Its 24-step primal last-bit gaps remain
`8.881784197001252e-16` in SSH, `1.3877787807814457e-17` in U, and
`6.071532165918825e-18` in V; the bit-identity prediction remains REFUTED.

## Landing rule

Round 3 lands only if the sampler lifecycle and allocator controls pass, the
GPU script is executable and covers 1440 steps, all fifteen CPU timing/resource
rows complete, the timing predictions are classified without changing a bar,
the round-1/2 wall-time retraction lives in both earlier receipts, direct tests
pass, and the citation gate is run explicitly on the round-3 receipt with its
shifted-citation plant. GPU resource magnitudes remain operator work because
this sandbox has no CUDA device.
