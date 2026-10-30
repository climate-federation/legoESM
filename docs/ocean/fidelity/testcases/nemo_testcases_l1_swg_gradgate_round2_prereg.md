# Certified SWG reverse-mode gradient gate — round 2 preregistration

Date: 2026-10-12

Status: **FROZEN BEFORE ROUND-2 MEASUREMENT**. This round repairs the GPU
memory instrument, directly measures the sixty-day CPU window, and retracts
the vacuous round-1 GPU resource verdicts. It changes no physics, card input,
loss, random direction, finite-difference step, agreement bar, checkpoint
schedule, stabilizer, clip, guard, or `stop_gradient`.

## Retraction fixed before new measurement

The round-1 GPU agreement rows are valid: reverse/forward relative errors were
`2.27064171926424e-16`, `9.30692734647596e-16`, and
`2.3070259307585205e-16` at 1, 24, and 720 steps. The six GPU resource verdicts
are **RETRACTED AS VACUOUS**. They sampled `nvidia-smi memory.used` while XLA
preallocated approximately 24.7 GiB, yielding increments `(none, uniform,
binomial) = (0, 6, 4) MiB` at both 24 and 720 steps. The tool must now classify
those old files VACUOUS rather than CONFIRMED.

## Frozen GPU resource instrument

Every GPU resource row runs in a fresh process with
`XLA_PYTHON_CLIENT_PREALLOCATE=false` and
`XLA_PYTHON_CLIENT_ALLOCATOR=platform` set before Python imports JAX. Around
the blocked reverse pass, the gate records JAX's own `bytes_in_use` and
`peak_bytes_in_use` counters from `jax.devices()[0].memory_stats()`. The fresh
process is the peak-counter reset boundary. It also samples physical GPU 1
with `nvidia-smi` during the same pass. Both baselines, peaks, increments, and
sample counts remain in the JSON; neither is substituted for the other.

An individual resource row is VACUOUS unless the JAX allocator sees a positive
increment. A three-schedule suite is CONFIRMED only if every row is measurable,
both samplers see positive increments, and both resolve the preregistered CPU
ordering `none > binomial > uniform`. Missing counters, zero increments, or an
unresolved sampler are VACUOUS; wrong positive ordering is REFUTED. The plant
reuses the `none` measurement under the `uniform` label and must turn the suite
REFUTED. This introduces no numeric memory-ratio threshold.

## Frozen direct sixty-day rows

The new CPU rows use 1440 steps (60 days), each in its own process under
`PrecisionPolicy.fp64(transcendentals="libm")`, starting from the admitted
kt=2 stage-1 entry. Agreement uses the unchanged per-step Python composition
through production `LatLonCGridOceanModel._step_jitted`, the same masked
SSH/U/V loss, keys 0 and 42, perturbation norm `1e-3`, and central-FD step
`1e-3`. Reverse/forward relative error at most `1e-9` and reverse/FD relative
error at most `0.05`, with all derivatives finite and nonzero, CONFIRMS; any
violation REFUTES. No NaN is expected for the same frozen reason as round 1:
no tracer/TKE path, live ENS F-thickness zero replacement, and static free-slip
masks.

The three direct 1440-step resource rows retain the existing schedule-primal
contract (`rtol=1e-9`, `atol=1e-11`), positive finite loss and directional
reverse derivative, positive wall time, and positive measured host-RSS
increment. They replace, rather than validate, the round-1 sixty-day
extrapolations. A failed compile or allocation is UNMEASURED.

## Training composition statement

Training on this card will use the per-step Python loop measured by the
agreement gate: every modeled step crosses the production jitted `swe_rk3`
boundary and is rematerialized as one step. It will not use the retained
uniform-scan composition, whose 24-step zero-perturbation primal differed from
production by `8.881784197001252e-16` in SSH, `1.3877787807814457e-17` in U,
and `6.071532165918825e-18` in V. That scan bit-identity prediction remains
REFUTED; the last-bit gap is not relabeled as harmless equivalence.

## Landing rule

Round 2 lands only if the round-1 tool/receipt retraction is executable, the
new direct 1440-step CPU agreement and all three resource rows complete and
pass, the shared GPU-1 script covers 1440 steps and the dual-sampler ordering
plant, every changed Python file's direct test passes, and the citation gate is
run explicitly on the round-2 receipt with a shifted-citation plant. GPU
numbers remain operator work because this sandbox reports
`CUDA_ERROR_NO_DEVICE`.
